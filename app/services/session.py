"""Per-user session isolation.

Each account (a Clerk user id, or a per-browser id in demo mode) gets its **own**
agent: its own dataset, its own results database, its own artifacts/reports, and
its own live event stream. One user never sees another user's uploaded data or
findings.

Provider credentials (Gemini / Elastic / Fivetran / GitLab) stay *global* — they
belong to whoever deployed the app — while the **data and results are isolated
per user**. The :class:`SessionManager` lazily builds and caches one
:class:`~app.services.agent_orchestrator.AgentOrchestrator` per user.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.services.agent_orchestrator import AgentOrchestrator

logger = logging.getLogger("epiphany.sessions")

USERS_ROOT = Path("data/users")
_ANON = "anonymous"


def safe_uid(uid: str | None) -> str:
    """Sanitise a user id into a safe directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(uid or "").strip()).strip("_")
    return cleaned[:80] or _ANON


class SessionManager:
    """Owns one orchestrator per user; isolates data + results."""

    def __init__(self) -> None:
        self._orchestrators: dict[str, AgentOrchestrator] = {}

    # ── paths ───────────────────────────────────────────────────────────
    def user_root(self, uid: str) -> Path:
        return USERS_ROOT / safe_uid(uid)

    def uploads_dir(self, uid: str) -> Path:
        return self.user_root(uid) / "uploads"

    def _active_file(self, uid: str) -> Path:
        return self.user_root(uid) / "active.json"

    def _read_active(self, uid: str) -> dict[str, Any]:
        f = self._active_file(uid)
        if f.exists():
            try:
                return json.loads(f.read_text() or "{}")
            except Exception:  # noqa: BLE001
                return {}
        return {}

    # ── per-user settings (global creds + per-user data/results) ────────
    def _user_settings(self, uid: str) -> Settings:
        root = self.user_root(uid)
        active = self._read_active(uid)
        # get_settings() carries the global provider creds; we overlay only the
        # per-user data + output locations on top.
        return get_settings().model_copy(update={
            "database_path": str(root / "epiphany.db"),
            "artifacts_dir": str(root / "artifacts"),
            "reports_dir": str(root / "reports"),
            "data_csv_path": active.get("data_csv_path", ""),
            "prefer_local_data": bool(active.get("prefer_local_data", False)),
        })

    # ── lifecycle ───────────────────────────────────────────────────────
    async def get(self, uid: str | None) -> AgentOrchestrator:
        """Return (creating if needed) the orchestrator for ``uid``."""
        key = safe_uid(uid)
        orch = self._orchestrators.get(key)
        if orch is None:
            self.user_root(key).mkdir(parents=True, exist_ok=True)
            orch = AgentOrchestrator(settings=self._user_settings(key), user_id=key)
            await orch.repository.init()
            self._orchestrators[key] = orch
            logger.info("Created session for user '%s'.", key)
        return orch

    async def set_dataset(
        self, uid: str, data_csv_path: str, prefer_local: bool = True
    ) -> AgentOrchestrator:
        """Point this user's agent at ``data_csv_path`` and reconfigure it."""
        key = safe_uid(uid)
        root = self.user_root(key)
        root.mkdir(parents=True, exist_ok=True)
        self._active_file(key).write_text(json.dumps({
            "data_csv_path": data_csv_path,
            "prefer_local_data": prefer_local,
        }))
        orch = await self.get(key)
        await orch.reconfigure(self._user_settings(key))
        return orch

    async def reconfigure_all(self) -> None:
        """Re-apply (new global provider creds) to every active session."""
        for key, orch in list(self._orchestrators.items()):
            try:
                await orch.reconfigure(self._user_settings(key))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Reconfigure failed for '%s': %s", key, exc)
