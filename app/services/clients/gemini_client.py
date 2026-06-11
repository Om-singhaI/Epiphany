"""Step 3 — Reason: Gemini client (API key or Vertex AI).

Given a *real* profile of the dataset (its columns, types, and the strongest
measured feature↔target associations), Gemini proposes ONE falsifiable
hypothesis: which feature most likely drives which outcome, and why. The
orchestrator then validates it with a real statistical test.

The client prefers the low-barrier **Gemini API key** path (``GEMINI_API_KEY``,
no Google Cloud project required) and transparently uses **Vertex AI** instead
when only a project is configured — both via the unified ``google-genai`` SDK.

When Gemini is unavailable (no key/project, or a transient error), it falls back
to a **data-driven** hypothesis computed from the real association ranking — not
a hard-coded guess. The agent still investigates the strongest real signal in
the data; it simply phrases the hypothesis itself.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.config import Settings

logger = logging.getLogger("epiphany.gemini")

_SYSTEM_PROMPT = """You are Epiphany, an autonomous data scientist.
You are given a JSON profile of a real dataset: its columns with inferred roles
(numeric / binary / categorical / identifier / text), and a ranking of the
strongest measured associations between candidate features and a target.

Your job: propose ONE falsifiable, business-relevant hypothesis that the data
can actually test. Choose a `feature` and a `target` that BOTH exist in the
profile (never invent columns; never choose identifier/text columns). Prefer a
binary or categorical column as the target.

Respond with STRICT JSON only, no markdown, matching exactly:
{
  "statement": "<one-sentence falsifiable hypothesis in business terms>",
  "feature": "<an existing feature column name>",
  "target": "<an existing target column name>",
  "threshold": <number or null: a split point if the feature is numeric>,
  "rationale": "<one sentence: why this relationship is worth testing>"
}
"""


class GeminiClient:
    """Gemini reasoning client (API key preferred, Vertex AI fallback)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.enabled = settings.gemini_enabled
        self._client = None
        if self.enabled:
            self._client = self._build_client()
            self.enabled = self._client is not None

    def _build_client(self):
        try:
            from google import genai

            s = self._settings
            if s.gemini_api_key:
                return genai.Client(api_key=s.gemini_api_key)
            # Vertex path: requires a project; uses ADC / service account.
            import os

            if s.google_application_credentials:
                os.environ.setdefault(
                    "GOOGLE_APPLICATION_CREDENTIALS", s.google_application_credentials
                )
            return genai.Client(
                vertexai=True,
                project=s.vertex_project_id,
                location=s.vertex_location,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini init failed (%s); using data-driven fallback.", exc)
            return None

    async def generate_hypothesis(
        self,
        profile: dict[str, Any],
        associations: list[dict[str, Any]],
        goal: str,
    ) -> dict[str, Any]:
        """Return a structured hypothesis dict grounded in the real profile."""
        if not self.enabled or self._client is None:
            return self._data_driven_hypothesis(profile, associations, goal)

        import asyncio

        prompt = (
            f"Business goal: {goal}\n\n"
            "Dataset profile (JSON):\n"
            f"{json.dumps(profile, indent=2, default=str)}\n\n"
            "Strongest measured associations (JSON):\n"
            f"{json.dumps(associations, indent=2, default=str)}\n\n"
            "Return the hypothesis JSON now."
        )
        try:
            from google.genai import types

            resp = await asyncio.to_thread(
                self._client.models.generate_content,
                model=self._settings.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            data = json.loads(resp.text)
            data = self._validate_against_profile(data, profile, associations, goal)
            data["mode"] = "live"
            return data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Gemini generation failed (%s); data-driven fallback.", exc)
            return self._data_driven_hypothesis(profile, associations, goal)

    @staticmethod
    def _validate_against_profile(
        data: dict[str, Any],
        profile: dict[str, Any],
        associations: list[dict[str, Any]],
        goal: str,
    ) -> dict[str, Any]:
        """Guard against hallucinated columns: feature/target must exist."""
        names = {f["name"] for f in profile.get("fields", [])}
        modelable = {
            f["name"]
            for f in profile.get("fields", [])
            if f.get("role") in ("numeric", "binary", "categorical")
        }
        feature = data.get("feature")
        target = data.get("target")
        if feature not in modelable or target not in names or feature == target:
            # Repair using the strongest real association.
            fallback = GeminiClient._data_driven_hypothesis(profile, associations, goal)
            data.setdefault("statement", fallback["statement"])
            data["feature"] = fallback["feature"]
            data["target"] = fallback["target"]
            data.setdefault("threshold", fallback.get("threshold"))
            data.setdefault("rationale", fallback.get("rationale"))
        return data

    def data_driven_hypothesis(
        self,
        profile: dict[str, Any],
        associations: list[dict[str, Any]],
        goal: str,
    ) -> dict[str, Any]:
        """Public, no-network hypothesis from the real association ranking.

        Used by the autonomous background loop so it never consumes Gemini
        quota; live Gemini reasoning is reserved for user-triggered missions.
        """
        return self._data_driven_hypothesis(profile, associations, goal)

    @staticmethod
    def _data_driven_hypothesis(
        profile: dict[str, Any],
        associations: list[dict[str, Any]],
        goal: str,
    ) -> dict[str, Any]:
        """Build a hypothesis from the strongest *real* association in the data.

        This is genuine analysis, not a canned answer: it reads the measured
        association ranking and the dataset profile to pick the most promising
        feature→target relationship to test.
        """
        targets = profile.get("candidate_targets") or []
        target = targets[0] if targets else None
        fields = {f["name"]: f for f in profile.get("fields", [])}

        feature = None
        direction = "association"
        if associations:
            top = associations[0]
            feature = top.get("feature")
            direction = top.get("direction", "association")
        if feature is None:
            # No ranking available — pick any modelable column that isn't target.
            for f in profile.get("fields", []):
                if f.get("role") in ("numeric", "categorical", "binary") and f["name"] != target:
                    feature = f["name"]
                    break

        threshold = None
        feat_meta = fields.get(feature, {})
        if feat_meta.get("role") == "numeric" and feat_meta.get("median") is not None:
            threshold = round(float(feat_meta["median"]), 4)

        statement = (
            f"`{feature}` is associated with `{target}` "
            f"({direction}); this relationship is statistically significant."
            if feature and target
            else "There is a statistically significant driver of the outcome in this dataset."
        )
        return {
            "statement": statement,
            "feature": feature,
            "target": target,
            "threshold": threshold,
            "rationale": (
                "Selected as the strongest measured univariate association in the "
                "dataset; a statistical test will confirm or reject it."
            ),
            "mode": "simulation",
        }
