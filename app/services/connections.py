"""Runtime provider connections.

Lets users connect their own Gemini / Elastic / Fivetran / GitLab credentials
from the dashboard UI instead of editing ``.env`` and restarting. Connections
are persisted to a local JSON file that is layered *on top of* the ``.env``
defaults, so they survive restarts.

Security:
* The file lives next to the app and is git-ignored.
* Secret values are NEVER returned to the client — :func:`connection_status`
  reports only whether each provider is connected plus a non-secret hint
  (e.g. the index name, project id, or the last 4 characters of a key).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("epiphany.connections")

# Persisted next to the project root; layered over .env, git-ignored.
CONNECTIONS_PATH = Path("epiphany_connections.json")

# Which Settings fields belong to each provider group the UI can configure.
PROVIDER_FIELDS: dict[str, list[str]] = {
    "gemini":   ["gemini_api_key", "gemini_model"],
    "elastic":  ["elastic_api_key", "elastic_cloud_id", "elastic_url", "elastic_index"],
    "fivetran": ["fivetran_api_key", "fivetran_api_secret", "fivetran_connector_id"],
    "gitlab":   ["gitlab_token", "gitlab_project_id", "gitlab_url",
                 "gitlab_target_branch", "auto_deploy"],
    "general":  ["data_csv_path", "prefer_local_data", "force_simulation",
                 "clerk_publishable_key"],
}

# Fields whose values must never leave the server.
SECRET_FIELDS: frozenset[str] = frozenset(
    {"gemini_api_key", "elastic_api_key", "fivetran_api_secret",
     "fivetran_api_key", "gitlab_token"}
)

# Every field a connect request is allowed to set.
ALLOWED_FIELDS: frozenset[str] = frozenset(
    f for fields in PROVIDER_FIELDS.values() for f in fields
)

_BOOL_FIELDS = {"auto_deploy", "force_simulation", "prefer_local_data"}


def load_overrides() -> dict[str, Any]:
    """Load persisted connection overrides (empty dict if none)."""
    if not CONNECTIONS_PATH.exists():
        return {}
    try:
        data = json.loads(CONNECTIONS_PATH.read_text() or "{}")
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 - never let bad state break boot
        logger.warning("Could not read connections file (%s).", exc)
        return {}


def save_overrides(updates: dict[str, Any]) -> dict[str, Any]:
    """Merge ``updates`` into the persisted overrides and write them.

    Only known fields are accepted. Empty strings clear a field. Returns the
    full merged override dict.
    """
    current = load_overrides()
    for key, value in updates.items():
        if key not in ALLOWED_FIELDS:
            continue
        if key in _BOOL_FIELDS:
            current[key] = bool(value)
            continue
        text = "" if value is None else str(value).strip()
        if text == "":
            current.pop(key, None)          # clearing a field disconnects it
        else:
            current[key] = text
    try:
        CONNECTIONS_PATH.write_text(json.dumps(current, indent=2))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist connections (%s).", exc)
    return current


def apply_overrides(settings: Any) -> Any:
    """Return a copy of ``settings`` with persisted overrides layered on."""
    overrides = load_overrides()
    if not overrides:
        return settings
    valid = {k: v for k, v in overrides.items() if k in ALLOWED_FIELDS}
    return settings.model_copy(update=valid) if valid else settings


def _mask(value: str | None) -> str | None:
    if not value:
        return None
    s = str(value)
    return ("•" * max(0, len(s) - 4) + s[-4:]) if len(s) > 4 else "••••"


def connection_status(settings: Any) -> dict[str, Any]:
    """Report per-provider connection state — WITHOUT exposing any secret."""
    s = settings
    return {
        "gemini": {
            "connected": bool(s.gemini_enabled),
            "model": s.gemini_model,
            "key_hint": _mask(s.gemini_api_key),
            "uses_vertex": bool(s.gemini_uses_vertex),
        },
        "elastic": {
            "connected": bool(s.elastic_enabled),
            "index": s.elastic_index,
            "endpoint": s.elastic_cloud_id or s.elastic_url or None,
            "key_hint": _mask(s.elastic_api_key),
        },
        "fivetran": {
            "connected": bool(s.fivetran_enabled),
            "connector_id": s.fivetran_connector_id,
            "key_hint": _mask(s.fivetran_api_key),
        },
        "gitlab": {
            "connected": bool(s.gitlab_enabled),
            "project_id": s.gitlab_project_id,
            "url": s.gitlab_url,
            "target_branch": s.gitlab_target_branch,
            "auto_deploy": bool(s.auto_deploy),
            "token_hint": _mask(s.gitlab_token),
        },
        "general": {
            "data_csv_path": s.data_csv_path,
            "force_simulation": bool(s.force_simulation),
        },
    }
