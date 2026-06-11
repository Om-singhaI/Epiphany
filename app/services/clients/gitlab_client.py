"""Step 5 — Deploy: GitLab client.

When a hypothesis is statistically significant, generates a predictive model
script and opens a Merge Request on GitLab (the GitLab MCP server proxies the
same REST surface): create a feature branch, commit the generated asset, and
open an MR annotated with the supporting statistics for human review.

Falls back to a simulated MR descriptor when GitLab is not configured.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from app.config import Settings
from app.services.model_generator import model_filename, render_model_script

logger = logging.getLogger("epiphany.gitlab")


def _slugify(value: str, fallback: str = "feature") -> str:
    """Branch/MR-safe slug derived from an arbitrary column name."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or fallback


class GitLabClient:
    """Async-friendly wrapper around the GitLab MR workflow."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.enabled = settings.gitlab_enabled
        self._gl = None
        if self.enabled:
            self._gl = self._build_client()
            self.enabled = self._gl is not None

    def _build_client(self):
        try:
            import gitlab

            s = self._settings
            gl = gitlab.Gitlab(url=s.gitlab_url, private_token=s.gitlab_token)
            gl.auth()
            return gl
        except Exception as exc:  # noqa: BLE001
            logger.warning("GitLab init failed (%s); using simulation.", exc)
            return None

    async def deploy_model(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """Author and ship the predictive model for a validated hypothesis.

        Builds the deployment payload from the validated ``feature``/``target``,
        renders the model script, then creates a feature branch off
        ``GITLAB_TARGET_BRANCH``, commits the asset, and opens a Merge Request
        titled ``feat: autonomous-prediction-model-[feature]``. Falls back to a
        simulated MR descriptor when GitLab is not configured.

        Args:
            hypothesis: Validated hypothesis metadata (``feature``, ``target``,
                ``threshold``, ``statement``, ``validation``).

        Returns:
            A merge-request descriptor (``iid``, ``web_url``, ``source_branch``,
            ``title``, ``mode``, ...).
        """
        feature = hypothesis.get("feature", "feature")
        slug = _slugify(feature)
        branch = f"feat/autonomous-prediction-model-{slug}"
        file_path = f"models/{model_filename(hypothesis)}"
        title = f"feat: autonomous-prediction-model-{slug}"
        script = render_model_script(hypothesis)
        description = self._build_description(hypothesis, file_path)
        return await self.open_model_merge_request(
            branch, file_path, script, title, description
        )

    @staticmethod
    def _build_description(hypothesis: dict[str, Any], file_path: str) -> str:
        """Human-review MR body annotated with the supporting statistics."""
        v = hypothesis.get("validation", {})
        test = v.get("test", "statistical test")
        stat_name = v.get("statistic_name") or "statistic"
        statistic = v.get("statistic", "n/a")
        p_value = v.get("p_value", "n/a")
        sample_size = v.get("sample_size", "n/a")
        alpha = v.get("alpha", 0.05)
        model = hypothesis.get("model", {})
        try:
            p_text = f"{float(p_value):.5g}"
            n_text = f"{int(sample_size):,}"
        except (TypeError, ValueError):
            p_text, n_text = str(p_value), str(sample_size)
        perf = ""
        if model.get("metrics"):
            metric_bits = ", ".join(f"{k}={val}" for k, val in model["metrics"].items())
            perf = (
                f"\n\n**Trained model:** {model.get('algorithm')} "
                f"({model.get('task')}) \u2014 {metric_bits}."
            )
        return (
            "Generated autonomously by **Epiphany**.\n\n"
            f"**Hypothesis:** {hypothesis.get('statement', '')}\n\n"
            f"**Feature \u2192 Target:** `{hypothesis.get('feature')}` \u2192 "
            f"`{hypothesis.get('target')}`\n\n"
            f"**Validation:** {test} \u2014 {stat_name}={statistic}, p={p_text} "
            f"(n={n_text}, \u03b1={alpha}).{perf}\n\n"
            f"**Asset:** `{file_path}` \u2014 a scikit-learn model trained to "
            "predict the validated target. Please review before merging."
        )

    async def open_model_merge_request(
        self, branch: str, file_path: str, content: str, title: str, description: str
    ) -> dict[str, Any]:
        """Create a branch, commit ``content`` to ``file_path``, and open an MR."""
        if not self.enabled or self._gl is None:
            return self._simulated_mr(branch, file_path, title)

        try:
            return await asyncio.to_thread(
                self._open_mr_sync, branch, file_path, content, title, description
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GitLab MR creation failed (%s); using simulation.", exc)
            return self._simulated_mr(branch, file_path, title)

    def _open_mr_sync(
        self, branch: str, file_path: str, content: str, title: str, description: str
    ) -> dict[str, Any]:
        s = self._settings
        project = self._gl.projects.get(s.gitlab_project_id)

        # 1) Create the feature branch from the target branch. Reuse it if a
        #    prior attempt already created it (idempotent re-runs).
        try:
            project.branches.create(
                {"branch": branch, "ref": s.gitlab_target_branch}
            )
            logger.info("Created branch %s off %s.", branch, s.gitlab_target_branch)
        except Exception:  # noqa: BLE001 - branch already exists
            logger.info("Branch %s already exists; reusing.", branch)

        # 2) Commit the generated asset. Use "create" for a new file, but fall
        #    back to "update" when the asset already exists on a reused branch,
        #    so the push never fails on a second run.
        action = "create"
        try:
            project.files.get(file_path=file_path, ref=branch)
            action = "update"
        except Exception:  # noqa: BLE001 - file does not exist yet
            action = "create"
        project.commits.create(
            {
                "branch": branch,
                "commit_message": title,
                "actions": [
                    {"action": action, "file_path": file_path, "content": content}
                ],
            }
        )
        logger.info("Committed %s to %s (%s).", file_path, branch, action)

        # 3) Open the Merge Request — or reuse an open MR for this branch if one
        #    already exists (GitLab rejects duplicate MRs for the same branch).
        existing = project.mergerequests.list(
            source_branch=branch, state="opened", get_all=False
        )
        if existing:
            mr = existing[0]
            logger.info("Reusing open Merge Request !%s for %s.", mr.iid, branch)
        else:
            mr = project.mergerequests.create(
                {
                    "source_branch": branch,
                    "target_branch": s.gitlab_target_branch,
                    "title": title,
                    "description": description,
                }
            )
            logger.info("Opened Merge Request !%s for %s.", mr.iid, branch)

        return {
            "iid": mr.iid,
            "web_url": mr.web_url,
            "source_branch": branch,
            "target_branch": s.gitlab_target_branch,
            "file_path": file_path,
            "title": title,
            "mode": "live",
        }

    def _simulated_mr(self, branch: str, file_path: str, title: str) -> dict[str, Any]:
        base = (self._settings.gitlab_url or "https://gitlab.com").rstrip("/")
        return {
            "iid": 42,
            "web_url": f"{base}/mock/repo/-/merge_requests/42",
            "source_branch": branch,
            "target_branch": self._settings.gitlab_target_branch,
            "file_path": file_path,
            "title": title,
            "mode": "simulation",
        }
