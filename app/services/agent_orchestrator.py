"""Agent orchestrator — the Epiphany autonomous loop.

This is the heart of Epiphany. A user-supplied mission ("user goal") drives an
investigation over a **real dataset**: the agent profiles the data, surfaces the
strongest measured associations, asks Gemini to form a falsifiable hypothesis,
proves or disproves it with the *right* statistical test (chosen adaptively from
the data's shape), and — when significant — trains a real predictive model,
writes a findings report, and optionally opens a GitLab Merge Request.

The 5-step loop
===============

::

    1. TRIGGER   Fivetran  — detect the synced table (or the local dataset)
    2. EXPLORE   Elastic    — discover the real schema + rank real associations
    3. REASON    Gemini     — form one falsifiable hypothesis from the profile
    4. VALIDATE  Sandbox    — run the adaptive test (χ²/t-test/ANOVA/correlation)
    5. DEPLOY    GitLab      — train a real model, write a report, open an MR

When the Google ADK is installed and Gemini is configured, an autonomous
``LlmAgent`` dynamically chooses which tools to call. Otherwise the same real
tools run as a direct pipeline. Either way every number is computed from the
data in front of the agent — nothing is fabricated. The only thing that changes
without credentials is *where the data comes from* (a local file instead of
Elastic) and *who phrases the hypothesis* (a data-driven heuristic instead of
Gemini).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings
from app.services.clients.elastic_client import ElasticClient, ElasticMcpToolset
from app.services.clients.fivetran_client import FivetranClient
from app.services.clients.gemini_client import GeminiClient
from app.services.clients.gitlab_client import GitLabClient
from app.services.data_port import DataPort
from app.services.events import AgentLogEvent, EventBus, event_bus
from app.services.model_generator import render_model_script
from app.services.model_trainer import train_model
from app.services.repository import Repository
from app.services.report import write_report
from app.services.sandbox import (
    MODEL_ALLOWED_MODULES,
    SecurityError,
    run_statistical_test_isolated,
    scan_code,
)
from app.services.statistics import choose_test

logger = logging.getLogger("epiphany.orchestrator")

# The Google Agent Development Kit (ADK) is optional at runtime. When it — and
# Gemini — is available, the orchestrator hands control to a managed
# ``LlmAgent`` that dynamically decides which tools to call. Otherwise the same
# real tools run as a direct pipeline.
try:  # pragma: no cover - import guard
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    _ADK_IMPORTABLE = True
except Exception:  # noqa: BLE001
    LlmAgent = None  # type: ignore[assignment]
    InMemoryRunner = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    _ADK_IMPORTABLE = False

# Default mission used by the autonomous background loop (no user goal supplied).
_DEFAULT_GOAL = "Find the strongest driver of the key outcome in this dataset."

_ADK_SYSTEM_INSTRUCTION = """You are Epiphany, an autonomous data scientist.
Your master objective is the user's business goal: {user_goal}

You investigate a REAL dataset dynamically through tools. Follow this protocol:
1. Call `get_database_schema` FIRST to see the real columns and their roles
   (numeric / binary / categorical / identifier / text). Never assume columns.
2. Call `rank_associations` to see which features are most strongly associated
   with a candidate target in the actual data.
3. Optionally call `aggregate` to inspect a specific dimension/metric.
4. Form ONE falsifiable hypothesis using columns that EXIST (never identifier or
   text columns), then call `validate_hypothesis` with a `feature` and `target`
   from the schema. The correct statistical test is chosen automatically.
5. If — and only if — the result is significant (p < 0.05), call `deploy_model`
   to train a real model and (optionally) open a GitLab Merge Request.
6. Finally, state in 1–2 sentences whether the hypothesis held and what it means
   for the user's goal.

Only reason over data the tools return. Never invent numbers. Be concise.
"""


@dataclass
class HypothesisResult:
    """Outcome of the reason → validate steps."""

    statement: str
    p_value: float | None = None
    statistic: float | None = None
    is_significant: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """Runs the autonomous 5-step loop and streams progress to the event bus."""

    def __init__(
        self,
        settings: Settings | None = None,
        bus: EventBus | None = None,
        user_id: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or event_bus
        self._uid = user_id
        self.fivetran = FivetranClient(self.settings)
        self.elastic = ElasticClient(self.settings)
        self.data = DataPort(self.settings, self.elastic)
        self.elastic_mcp = ElasticMcpToolset(self.data, self.settings.elastic_index)
        self.gemini = GeminiClient(self.settings)
        self.gitlab = GitLabClient(self.settings)
        self.repository = Repository(self.settings.database_path)
        self._running = False
        self.user_goal: str | None = None
        self._adk_env_ready = False

    # ── runtime reconfiguration (Connections page) ──────────────────────
    async def reconfigure(self, settings: Settings) -> dict[str, Any]:
        """Rebuild every provider client from updated settings, live.

        Called when a user connects/updates credentials from the dashboard, so
        a provider can go live without an ``.env`` edit or a server restart. The
        data cache is dropped so the next cycle reloads from the new source.
        """
        self.settings = settings
        try:
            await self.elastic.aclose()
        except Exception:  # noqa: BLE001 - old client may already be closed
            pass
        self.fivetran = FivetranClient(settings)
        self.elastic = ElasticClient(settings)
        self.data = DataPort(settings, self.elastic)
        self.elastic_mcp = ElasticMcpToolset(self.data, settings.elastic_index)
        self.gemini = GeminiClient(settings)
        self.gitlab = GitLabClient(settings)
        self._adk_env_ready = False  # re-evaluate Gemini backend env next cycle
        logger.info("Providers reconfigured from updated connection settings.")
        return {
            "fivetran": self.fivetran.enabled,
            "elastic": self.elastic.enabled,
            "gemini": self.gemini.enabled,
            "gitlab": self.gitlab.enabled,
        }

    # ── telemetry helper ────────────────────────────────────────────────
    async def _emit(
        self, stage: str, source: str, level: str, message: str, mode: str = "live"
    ) -> None:
        await self.bus.publish(
            AgentLogEvent(
                stage=stage, source=source, level=level, message=message,
                mode=mode, user_id=self._uid,
            )
        )
        await asyncio.sleep(0.4)

    # ── Step 1 — TRIGGER (Fivetran) ─────────────────────────────────────
    async def trigger_ingestion(self) -> dict[str, Any]:
        """Detect the synced table (live) or report the real local dataset."""
        profile = await self.data.profile()
        manifest = await self.fivetran.get_sync_manifest(
            local_fallback={
                "table": profile.source,
                "rows_synced": profile.row_count,
                "schema": profile.field_names(),
            }
        )
        await self._emit(
            "trigger",
            "FIVETRAN MCP",
            "info",
            f"Detected data on `{manifest['table']}`. "
            f"{manifest['rows_synced']:,} rows, {len(profile.field_names())} columns. "
            "Schema captured.",
            manifest["mode"],
        )
        return manifest

    # ── Step 2 — EXPLORE (Elastic / real aggregations) ──────────────────
    async def explore(
        self, goal: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
        """Discover the real schema and rank real feature↔target associations."""
        profile = await self.data.profile()
        field_names = ", ".join(profile.field_names())
        await self._emit(
            "explore",
            "ELASTIC MCP",
            "info",
            f"Schema discovery on `{profile.source}` → {len(profile.columns)} "
            f"fields: {field_names}.",
            profile.mode,
        )

        targets = profile.candidate_targets()
        primary_target = targets[0].name if targets else None
        associations: list[dict[str, Any]] = []
        if primary_target:
            associations = await self.data.rank_associations(primary_target)
            if associations:
                top = associations[0]
                await self._emit(
                    "explore",
                    "ELASTIC MCP",
                    "warning",
                    f"Strongest signal vs `{primary_target}`: `{top['feature']}` "
                    f"({top['kind']} strength {top['strength']}, {top['direction']}). "
                    f"Ranked {len(associations)} candidate drivers.",
                    profile.mode,
                )
        return profile.as_dict(), associations, primary_target

    # ── Step 3 — REASON (Gemini) ────────────────────────────────────────
    async def reason(
        self,
        profile_dict: dict[str, Any],
        associations: list[dict[str, Any]],
        goal: str,
        use_llm: bool = True,
    ) -> HypothesisResult:
        """Form a falsifiable hypothesis grounded in the real profile.

        ``use_llm`` controls whether Gemini phrases the hypothesis (one API
        call) or the data-driven heuristic does. The background loop sets it
        ``False`` to preserve Gemini quota for user-triggered missions.
        """
        if use_llm:
            h = await self.gemini.generate_hypothesis(profile_dict, associations, goal)
        else:
            h = self.gemini.data_driven_hypothesis(profile_dict, associations, goal)
        h["user_goal"] = goal
        await self._emit(
            "reason",
            "GEMINI" if h.get("mode") == "live" else "EPIPHANY (data-driven)",
            "hypothesis",
            f"Hypothesis: {h['statement']}",
            h.get("mode", "simulation"),
        )
        return HypothesisResult(statement=h["statement"], metadata=h)

    # ── Step 4 — VALIDATE (adaptive statistical test) ───────────────────
    async def validate(self, hypothesis: HypothesisResult) -> HypothesisResult:
        """Validate the hypothesis with the right test on real data."""
        meta = hypothesis.metadata
        profile = await self.data.profile()
        feature = meta.get("feature")
        target = meta.get("target")

        feat_col = profile.column(feature) if feature else None
        targ_col = profile.column(target) if target else None
        if feat_col is None or targ_col is None:
            hypothesis.is_significant = False
            meta["validation"] = {"error": "feature/target not in dataset"}
            await self._emit(
                "validate", "PYTHON SANDBOX", "error",
                f"Cannot validate: `{feature}`/`{target}` not found in schema.",
            )
            return hypothesis

        # Choose the correct test from the *actual* column roles.
        test = choose_test(
            feat_col.role, targ_col.role, targ_col.cardinality, feat_col.cardinality
        )
        meta["test"] = test
        threshold = meta.get("threshold")
        if threshold is None and feat_col.role == "numeric":
            threshold = feat_col.median
            meta["threshold"] = threshold

        # Security gate: screen the model script we would deploy before trusting
        # any of this run's downstream code generation.
        try:
            report = scan_code(render_model_script(meta), allowed_modules=MODEL_ALLOWED_MODULES)
            meta["security"] = report.as_dict()
        except SecurityError as exc:
            meta["security"] = {"safe": False, "reason": str(exc)}
            hypothesis.is_significant = False
            await self._emit(
                "validate", "SANDBOX SECURITY", "error",
                f"Blocked unsafe generated code before execution: {exc}.",
            )
            return hypothesis

        # Pull the real, row-aligned values and run the test in isolation.
        values = await self.data.column_values(feature, target)
        feature_values = values.get(feature, [])
        target_values = values.get(target, [])

        result = await run_statistical_test_isolated(
            test=test,
            feature_values=feature_values,
            target_values=target_values,
            alpha=self.settings.significance_alpha,
            threshold=float(threshold) if threshold is not None else None,
        )
        hypothesis.p_value = result.p_value
        hypothesis.statistic = result.statistic
        hypothesis.is_significant = result.is_significant
        meta["validation"] = result.as_dict()
        meta["data_mode"] = profile.mode

        verdict = "verified" if result.is_significant else "rejected"
        await self._emit(
            "validate",
            "PYTHON SANDBOX",
            "success" if result.is_significant else "warning",
            f"Ran {test} on {result.sample_size:,} real rows. {result.summary} "
            f"Significance {verdict} (α={result.alpha}).",
            profile.mode,
        )
        return hypothesis

    # ── Step 5 — DEPLOY (train model + report + optional GitLab MR) ──────
    async def deploy(self, hypothesis: HypothesisResult) -> dict[str, Any] | None:
        """Train a real model, write a report, and optionally open a GitLab MR."""
        meta = hypothesis.metadata
        meta["deploy_attempted"] = True

        security = meta.get("security") or {}
        if security.get("safe") is False:
            await self._emit(
                "deploy", "GITLAB MCP", "warning",
                "Deployment denied: generated code failed the security screen.",
            )
            return None

        if not hypothesis.is_significant:
            await self._emit(
                "deploy", "GITLAB MCP", "warning",
                "Hypothesis not significant (p ≥ 0.05); no model trained.",
            )
            # Still write a findings report for the negative result.
            await self._write_findings(hypothesis, model=None, mr=None)
            return None

        # 1) Train a REAL model on the real data and measure its performance.
        target = meta.get("target")
        model_dict: dict[str, Any] | None = None
        try:
            await self._emit(
                "deploy", "MODEL TRAINER", "info",
                f"Training a predictive model for `{target}` on the real dataset...",
            )
            df = await self.data.frame()
            profile = await self.data.profile()
            artifact = await asyncio.to_thread(
                train_model, df, profile, target, self.settings.artifacts_dir
            )
            model_dict = artifact.as_dict()
            meta["model"] = model_dict
            name, value = artifact.headline_metric()
            await self._emit(
                "deploy", "MODEL TRAINER", "success",
                f"Trained {artifact.algorithm} ({artifact.task}). "
                f"{name}={value} on {artifact.metrics.get('test_rows', '?')} held-out rows. "
                f"Saved → {artifact.model_path}.",
            )
        except Exception as exc:  # noqa: BLE001 - model is best-effort, report still ships
            logger.warning("Model training failed: %s", exc)
            await self._emit(
                "deploy", "MODEL TRAINER", "warning",
                f"Model training skipped ({exc}). Proceeding with findings report.",
            )

        # 2) Optionally open a GitLab Merge Request.
        mr = await self._maybe_open_mr(hypothesis)

        # 3) Always write a local findings report (the "anyone can use" output).
        await self._write_findings(hypothesis, model=model_dict, mr=mr)
        return mr

    async def _maybe_open_mr(
        self, hypothesis: HypothesisResult
    ) -> dict[str, Any] | None:
        """Open a GitLab MR when configured + authorised; else skip cleanly."""
        if self.settings.gitlab_enabled and not self.settings.auto_deploy:
            await self._emit(
                "deploy", "GITLAB MCP", "warning",
                "Significant result ready to ship, but AUTO_DEPLOY is disabled. "
                "Set AUTO_DEPLOY=true to open the merge request.",
                "live",
            )
            return None
        if not self.settings.gitlab_enabled:
            # No GitLab configured — local outputs are the deliverable.
            return None

        await self._emit(
            "deploy", "GITLAB MCP", "info",
            f"Opening a Merge Request for the `{hypothesis.metadata.get('target')}` model...",
        )
        mr = await self.gitlab.deploy_model(hypothesis.metadata)
        hypothesis.metadata["merge_request"] = mr
        await self._emit(
            "deploy", "GITLAB MCP", "success",
            f"Committed `{mr['file_path']}` to `{mr['source_branch']}`. "
            f"Merge Request !{mr['iid']} → {mr['web_url']}",
            mr["mode"],
        )
        return mr

    async def _write_findings(
        self,
        hypothesis: HypothesisResult,
        model: dict[str, Any] | None,
        mr: dict[str, Any] | None,
    ) -> None:
        """Write the local Markdown + HTML findings report."""
        try:
            profile = await self.data.profile()
            paths = await asyncio.to_thread(
                write_report,
                self.settings.reports_dir,
                goal=hypothesis.metadata.get("user_goal", _DEFAULT_GOAL),
                profile=profile.as_dict(),
                hypothesis=hypothesis.metadata,
                validation=hypothesis.metadata.get("validation", {}),
                model=model,
                merge_request=mr,
            )
            hypothesis.metadata["report"] = paths
            if paths.get("markdown"):
                await self._emit(
                    "deploy", "REPORT", "success",
                    f"Findings report written → {paths['markdown']} (and HTML).",
                    profile.mode,
                )
        except Exception as exc:  # noqa: BLE001 - never fail a cycle on report IO
            logger.warning("Report writing failed: %s", exc)

    # ── ADK orchestration (Google Agent Development Kit) ─────────────────
    def _adk_ready(self) -> bool:
        return (
            _ADK_IMPORTABLE
            and self.settings.gemini_enabled
            and not self.settings.force_simulation
        )

    def _ensure_adk_environment(self) -> None:
        """Point the Google GenAI SDK at the right Gemini backend."""
        if self._adk_env_ready:
            return
        import os

        s = self.settings
        if s.gemini_api_key:
            os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
            os.environ.setdefault("GOOGLE_API_KEY", s.gemini_api_key)
        else:
            os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
            if s.vertex_project_id:
                os.environ.setdefault("GOOGLE_CLOUD_PROJECT", s.vertex_project_id)
            if s.vertex_location:
                os.environ.setdefault("GOOGLE_CLOUD_LOCATION", s.vertex_location)
            if s.google_application_credentials:
                os.environ.setdefault(
                    "GOOGLE_APPLICATION_CREDENTIALS", s.google_application_credentials
                )
        self._adk_env_ready = True

    def _build_adk_agent(self, goal: str, ctx: dict[str, Any]) -> Any:
        """Construct the ADK ``LlmAgent`` with real data + validation tools."""

        async def get_database_schema() -> dict[str, Any]:
            """Inspect the dataset schema. Call this FIRST. Returns the fields
            and their roles (numeric/binary/categorical/identifier/text)."""
            schema = await self.elastic_mcp.get_database_schema()
            ctx["schema"] = schema
            names = ", ".join(f["name"] for f in schema.get("fields", []))
            await self._emit(
                "explore", "ELASTIC MCP", "info",
                f"Schema discovery on `{schema.get('source')}` → {names}.",
                schema.get("mode", "local"),
            )
            return schema

        async def rank_associations(target: str) -> list[dict[str, Any]]:
            """Rank features by their real univariate association with `target`.

            Args:
                target: An existing target column from the schema.
            """
            ranking = await self.data.rank_associations(target)
            ctx["associations"] = ranking
            await self._emit(
                "explore", "ELASTIC MCP", "warning",
                f"Ranked {len(ranking)} real drivers of `{target}`.",
                self.data.mode,
            )
            return ranking

        async def aggregate(dimension: str, metric: str) -> dict[str, Any]:
            """Group `metric` by `dimension` on the real data (avg per group).

            Args:
                dimension: A categorical/binary column to group by.
                metric: A numeric column to average within each group.
            """
            result = await self.elastic_mcp.execute_aggregation(dimension, metric)
            await self._emit(
                "explore", "ELASTIC MCP", "info",
                f"Aggregated `{metric}` by `{dimension}` ({len(result.get('buckets', []))} groups).",
                result.get("mode", "local"),
            )
            return result

        async def validate_hypothesis(
            statement: str, feature: str, target: str, threshold: float | None = None
        ) -> dict[str, Any]:
            """Validate a hypothesis with the right statistical test (auto-chosen).

            Args:
                statement: One-sentence falsifiable hypothesis.
                feature: An existing feature column.
                target: An existing target column.
                threshold: Optional split point if the feature is numeric.
            """
            hypothesis = HypothesisResult(
                statement=statement,
                metadata={
                    "statement": statement, "feature": feature, "target": target,
                    "threshold": threshold, "mode": "live", "user_goal": goal,
                },
            )
            hypothesis = await self.validate(hypothesis)
            ctx["hypothesis"] = hypothesis
            return hypothesis.metadata.get("validation", {})

        async def deploy_model() -> dict[str, Any]:
            """Train a real model + write a report + (optionally) open a GitLab MR.

            Call ONLY after `validate_hypothesis` confirms significance (p<0.05).
            """
            hypothesis = ctx.get("hypothesis")
            if hypothesis is None:
                return {"status": "no_hypothesis",
                        "detail": "Validate a hypothesis before deploying."}
            mr = await self.deploy(hypothesis)
            ctx["mr"] = mr
            report = hypothesis.metadata.get("report", {})
            return {
                "status": "deployed" if hypothesis.is_significant else "skipped",
                "merge_request_url": (mr or {}).get("web_url") if mr else None,
                "report": report.get("markdown"),
                "model": hypothesis.metadata.get("model", {}).get("metrics"),
            }

        return LlmAgent(
            name="epiphany_orchestrator",
            model=self.settings.model_name,
            instruction=_ADK_SYSTEM_INSTRUCTION.format(user_goal=goal),
            tools=[
                get_database_schema, rank_associations, aggregate,
                validate_hypothesis, deploy_model,
            ],
        )

    async def _run_adk_cycle(self, goal: str) -> HypothesisResult:
        """Drive one investigation via the managed Google ADK agent."""
        self._ensure_adk_environment()
        ctx: dict[str, Any] = {}
        agent = self._build_adk_agent(goal, ctx)

        await self._emit(
            "reason", "EPIPHANY ADK", "info",
            f"Google ADK orchestrator online. Objective: “{goal}”. "
            "Selecting tools to investigate...",
            "live",
        )

        runner = InMemoryRunner(agent=agent, app_name="epiphany")
        session_id = f"mission-{uuid.uuid4().hex[:12]}"
        await runner.session_service.create_session(
            app_name="epiphany", user_id="dashboard", session_id=session_id
        )
        message = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=f"Business goal: {goal}. Begin now.")],
        )
        try:
            async for event in runner.run_async(
                user_id="dashboard", session_id=session_id, new_message=message
            ):
                text = self._final_text(event)
                if text:
                    await self._emit("reason", "GEMINI ADK", "hypothesis", text, "live")
        finally:
            close = getattr(runner, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:  # noqa: BLE001
                    pass

        hypothesis = ctx.get("hypothesis")
        if hypothesis is None:
            raise RuntimeError("ADK agent finished without validating a hypothesis.")
        return hypothesis

    @staticmethod
    def _final_text(event: Any) -> str | None:
        try:
            if not event.is_final_response():
                return None
            content = getattr(event, "content", None)
            if not content or not getattr(content, "parts", None):
                return None
            chunks = [p.text for p in content.parts if getattr(p, "text", None)]
            text = " ".join(c.strip() for c in chunks if c and c.strip())
            return text or None
        except Exception:  # noqa: BLE001
            return None

    async def _run_pipeline_cycle(
        self, goal: str, use_llm: bool = True
    ) -> HypothesisResult:
        """Direct (non-agentic) real pipeline: explore → reason → validate.

        ``use_llm`` decides whether Gemini phrases the hypothesis. The
        background loop passes ``False`` (no Gemini call); user missions that
        fall back here pass ``True``. Statistics + model are always real.
        """
        profile_dict, associations, _ = await self.explore(goal)
        hypothesis = await self.reason(profile_dict, associations, goal, use_llm=use_llm)
        hypothesis = await self.validate(hypothesis)
        return hypothesis

    async def _investigate(self, goal: str, allow_adk: bool = True) -> HypothesisResult:
        """Run the reason→validate investigation.

        ``allow_adk`` distinguishes user-triggered missions (full multi-tool
        Gemini ADK agent) from the continuous background loop, which makes NO
        Gemini calls (data-driven hypotheses) so the free-tier quota is reserved
        for live, human-triggered demos.
        """
        if allow_adk and self._adk_ready():
            try:
                return await self._run_adk_cycle(goal)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ADK orchestration failed (%s); using pipeline.", exc)
                await self._emit(
                    "reason", "EPIPHANY", "info",
                    "Switching to the direct analysis pipeline for this cycle.",
                    "live",
                )
                return await self._run_pipeline_cycle(goal, use_llm=True)
        return await self._run_pipeline_cycle(goal, use_llm=allow_adk)

    # ── Orchestration ───────────────────────────────────────────────────
    async def run_cycle(
        self, user_goal: str | None = None, allow_adk: bool = True
    ) -> dict[str, Any]:
        """Run one full autonomous pass over the real dataset."""
        if self._running:
            logger.info("run_cycle called while a cycle is already active.")
            return {"status": "busy"}

        # No dataset yet → nothing to analyse. Prompt the user to provide data.
        if not await self.data.has_data():
            await self._emit(
                "system", "ORCHESTRATOR", "warning",
                "No dataset connected yet. Upload a CSV/Parquet or connect Elastic "
                "to begin — then I'll start investigating.",
                "local",
            )
            return {"status": "idle", "detail": "no dataset connected"}

        self.user_goal = user_goal
        if user_goal:
            goal = user_goal
            logger.info("Agent mission received: %s", user_goal)
        else:
            goal = await self._default_goal()

        self._running = True
        try:
            await self.trigger_ingestion()
            hypothesis = await self._investigate(goal, allow_adk=allow_adk)

            # The ADK agent may already have deployed via its tool; only deploy
            # here when it hasn't been attempted yet.
            mr = hypothesis.metadata.get("merge_request")
            if not hypothesis.metadata.get("deploy_attempted"):
                mr = await self.deploy(hypothesis)

            hypothesis_id = await self.repository.record_hypothesis(
                hypothesis.statement,
                hypothesis.metadata,
                data_mode=hypothesis.metadata.get("data_mode", "local"),
                reason_mode=hypothesis.metadata.get("mode", "simulation"),
            )
            if mr is not None:
                deploy_row = {
                    **mr,
                    "branch": mr.get("source_branch"),
                    "title": mr.get("title") or "feat: autonomous model",
                }
                await self.repository.record_deployment(hypothesis_id, deploy_row)

            await self._emit(
                "system", "ORCHESTRATOR", "info",
                "Cycle complete. Sleeping until the next trigger...",
            )
            return {
                "status": "complete",
                "significant": hypothesis.is_significant,
                "p_value": hypothesis.p_value,
                "merge_request": mr,
                "report": hypothesis.metadata.get("report"),
            }
        except Exception as exc:  # noqa: BLE001 - report, then surface failure
            logger.exception("Agent cycle failed")
            await self._emit("system", "ORCHESTRATOR", "error", f"Cycle aborted: {exc}")
            return {"status": "error", "detail": str(exc)}
        finally:
            self._running = False
            await self.elastic.aclose()

    async def _default_goal(self) -> str:
        """Derive the autonomous mission from the dataset's detected target.

        Keeps the agent domain-agnostic: it investigates whatever outcome the
        data actually contains (churn, price, defect rate, ...), never a fixed
        domain baked into the code.
        """
        try:
            profile = await self.data.profile()
            targets = profile.candidate_targets()
            if targets:
                return (
                    f"Discover the strongest driver of '{targets[0].name}' "
                    "in this dataset."
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not derive default goal (%s).", exc)
        return _DEFAULT_GOAL

    async def _rotating_goal(self, cycle: int) -> str:
        """Vary the autonomous mission each cycle so the agent doesn't loop.

        Rotates across the dataset's candidate targets *and* a few analytical
        angles, so successive background cycles investigate genuinely different
        questions (and often pick different statistical tests) instead of
        repeating the same hypothesis forever.
        """
        try:
            profile = await self.data.profile()
            targets = [t.name for t in profile.candidate_targets()[:3]]
        except Exception:  # noqa: BLE001
            targets = []
        if not targets:
            return _DEFAULT_GOAL
        angles = [
            "Discover the strongest driver of '{t}' in this dataset.",
            "Which single feature best predicts '{t}'? Validate it statistically.",
            "Find a surprising but statistically significant relationship for '{t}'.",
        ]
        target = targets[cycle % len(targets)]
        angle = angles[(cycle // len(targets)) % len(angles)]
        return angle.format(t=target)

    async def run_forever(self, interval_seconds: float | None = None) -> None:
        """Continuously run cycles with a rotating mission between them.

        Uses the lightweight pipeline (``allow_adk=False``) so the perpetual
        loop stays within Gemini quota; user-triggered missions still get the
        full ADK agent.
        """
        interval = (
            interval_seconds
            if interval_seconds is not None
            else self.settings.autonomous_interval_seconds
        )
        cycle = 0
        while True:
            # Idle quietly until the user provides a dataset; then investigate.
            if await self.data.has_data():
                goal = await self._rotating_goal(cycle)
                await self.run_cycle(user_goal=goal, allow_adk=False)
                cycle += 1
            await asyncio.sleep(interval)
