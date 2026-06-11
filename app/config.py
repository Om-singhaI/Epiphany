"""Application configuration for Epiphany.

All external integration settings (Fivetran, Elastic, Vertex AI / Gemini, and
GitLab) are loaded from environment variables (or a local ``.env`` file) via
``pydantic-settings``. Every provider degrades gracefully: when its required
credentials are absent, the corresponding client falls back to a deterministic
**simulation mode** so the agent loop — and the live demo — always runs.

Copy ``.env.example`` to ``.env`` and fill in the values you have.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── General ──────────────────────────────────────────────────────────
    app_env: str = Field(default="development", description="Deployment environment label.")
    # When True, the agent never calls live providers (Fivetran/Elastic/Gemini/
    # GitLab) even if creds exist. It still performs *real* analysis on the local
    # dataset — only the external network calls are disabled. Useful for offline
    # demos and CI.
    force_simulation: bool = Field(default=False)

    # Safety guard: the agent only opens *real* GitLab merge requests when this
    # is True. Defaults to False so live cycles never auto-deploy unexpectedly.
    # Local outputs (trained model + findings report) are always produced.
    auto_deploy: bool = Field(default=False)

    # Path to the local SQLite database that records every agent cycle.
    database_path: str = Field(default="epiphany.db")

    # ── Local data + outputs (bring-your-own-data) ──────────────────────
    # No dataset ships by default: a new user starts with an empty workspace
    # and must upload their own CSV/Parquet (or connect Elastic) before the
    # agent runs. Set this (or upload in the UI) to point at a file.
    data_csv_path: str = Field(default="")
    # When True, the agent analyses ``data_csv_path`` even if Elastic is
    # connected. Set automatically when a user uploads/selects a local dataset
    # so "bring your own data" always wins over the configured index.
    prefer_local_data: bool = Field(default=False)
    # Where trained model artifacts (.pkl + metrics.json) are written.
    artifacts_dir: str = Field(default="artifacts")
    # Where human-readable findings reports (.md / .html) are written.
    reports_dir: str = Field(default="reports")
    # Significance threshold for every statistical test the agent runs.
    significance_alpha: float = Field(default=0.05)

    # ── Autonomous background loop ───────────────────────────────────────
    # Run the agent continuously in the background. The loop uses the
    # lightweight pipeline (one Gemini call per cycle) to stay within quota;
    # the full multi-tool ADK agent is reserved for user-triggered missions.
    run_autonomous_loop: bool = Field(default=True)
    autonomous_interval_seconds: float = Field(default=45.0)

    # ── Auth: Clerk (frontend gate) ──────────────────────────────────────
    # Publishable key (pk_test_… / pk_live_…). Safe to expose to the browser.
    # When set, the dashboard's sign-in/up card is backed by real Clerk auth;
    # when absent, a local demo session is used so the app still runs.
    clerk_publishable_key: str | None = Field(default=None)

    # ── Step 1: Trigger — Fivetran MCP ───────────────────────────────────
    fivetran_api_key: str | None = Field(default=None)
    fivetran_api_secret: str | None = Field(default=None)
    fivetran_connector_id: str | None = Field(default=None)
    fivetran_base_url: str = Field(default="https://api.fivetran.com/v1")

    # ── Step 2: Explore — Elastic MCP ────────────────────────────────────
    elastic_cloud_id: str | None = Field(default=None)
    elastic_api_key: str | None = Field(default=None)
    elastic_url: str | None = Field(default=None)
    elastic_index: str = Field(default="prod_db.saas_telemetry")

    # ── Step 3: Reason — Gemini ──────────────────────────────────────────
    # The simplest path: a single Gemini API key (https://aistudio.google.com/
    # apikey). No Google Cloud project required — this is the recommended setup
    # for "anyone can use it".
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")

    # Optional Vertex AI path (enterprise): if you have a GCP project, the agent
    # will use Vertex instead of the API key. Either path drives the same Gemini
    # reasoning; the API key just lowers the barrier to entry.
    vertex_project_id: str | None = Field(default=None)
    vertex_location: str = Field(default="us-central1")
    # Path to a service-account JSON; if None, Application Default Credentials
    # (ADC) are used (e.g. `gcloud auth application-default login`).
    google_application_credentials: str | None = Field(default=None)

    # ── Step 5: Deploy — GitLab MCP ──────────────────────────────────────
    gitlab_url: str = Field(default="https://gitlab.com")
    gitlab_token: str | None = Field(default=None)
    gitlab_project_id: str | None = Field(default=None)
    gitlab_target_branch: str = Field(default="main")

    # ── Derived helpers ──────────────────────────────────────────────────
    @property
    def fivetran_enabled(self) -> bool:
        return bool(
            not self.force_simulation
            and self.fivetran_api_key
            and self.fivetran_api_secret
            and self.fivetran_connector_id
        )

    @property
    def elastic_enabled(self) -> bool:
        return bool(
            not self.force_simulation
            and self.elastic_api_key
            and (self.elastic_cloud_id or self.elastic_url)
        )

    @property
    def gemini_enabled(self) -> bool:
        """True when Gemini can be reached — via an API key or a Vertex project."""
        return bool(
            not self.force_simulation
            and (self.gemini_api_key or self.vertex_project_id)
        )

    @property
    def gemini_uses_vertex(self) -> bool:
        """Prefer the API key; fall back to Vertex only when no key is set."""
        return bool(not self.gemini_api_key and self.vertex_project_id)

    # Backwards-compatible alias (older code/tests referenced ``vertex_enabled``).
    @property
    def vertex_enabled(self) -> bool:
        return self.gemini_enabled

    @property
    def model_name(self) -> str:
        """The Gemini model id used for reasoning (API key or Vertex)."""
        return self.gemini_model

    @property
    def gitlab_enabled(self) -> bool:
        return bool(
            not self.force_simulation and self.gitlab_token and self.gitlab_project_id
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance.

    Persisted runtime connections (set from the dashboard's Connections page)
    are layered on top of the ``.env`` defaults, so user-supplied credentials
    take precedence without requiring an ``.env`` edit or restart.
    """
    # Imported lazily to avoid a circular import at module load time.
    from app.services.connections import apply_overrides

    return apply_overrides(Settings())


def reload_settings() -> Settings:
    """Clear the settings cache and rebuild (after a connection change)."""
    get_settings.cache_clear()
    return get_settings()
