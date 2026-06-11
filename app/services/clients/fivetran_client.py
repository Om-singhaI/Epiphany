"""Step 1 — Trigger: Fivetran client.

Wraps the Fivetran REST API (the Fivetran MCP server proxies the same surface)
to detect completed syncs and read the schema/row-delta of the freshly synced
telemetry tables.

When Fivetran credentials are not configured, the trigger reports the **real**
local dataset that the agent is about to analyse (its true table name, row
count, and columns) rather than a fabricated manifest — so even the offline path
describes genuine data.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import Settings

logger = logging.getLogger("epiphany.fivetran")


class FivetranClient:
    """Async client for Fivetran sync detection."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.enabled = settings.fivetran_enabled

    async def get_sync_manifest(
        self, local_fallback: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return a manifest describing the most recent completed sync.

        Returns a dict with at least ``table``, ``rows_synced``, ``schema``, and
        ``mode`` ("live" | "local"). When Fivetran is not configured,
        ``local_fallback`` (the real table/row-count/columns of the dataset the
        agent will analyse) is reported instead of a fabricated manifest.
        """
        if not self.enabled:
            return self._local_manifest(local_fallback)

        s = self._settings
        url = f"{s.fivetran_base_url}/connectors/{s.fivetran_connector_id}"
        auth = (s.fivetran_api_key or "", s.fivetran_api_secret or "")
        try:
            async with httpx.AsyncClient(timeout=20.0) as http:
                resp = await http.get(url, auth=auth)
                resp.raise_for_status()
                data = resp.json().get("data", {})
            return {
                "table": s.elastic_index,
                "connector_id": s.fivetran_connector_id,
                "succeeded_at": data.get("succeeded_at"),
                "sync_state": data.get("status", {}).get("sync_state"),
                "schema": data.get("schema"),
                "rows_synced": data.get("status", {}).get("rows_synced", 0),
                "mode": "live",
            }
        except Exception as exc:  # noqa: BLE001 - degrade to the real local dataset
            logger.warning("Fivetran API call failed (%s); reporting local data.", exc)
            return self._local_manifest(local_fallback)

    def _local_manifest(self, local: dict[str, Any] | None) -> dict[str, Any]:
        local = local or {}
        return {
            "table": local.get("table") or self._settings.elastic_index,
            "connector_id": self._settings.fivetran_connector_id or "local",
            "succeeded_at": None,
            "sync_state": "synced",
            "schema": local.get("schema") or [],
            "rows_synced": int(local.get("rows_synced") or 0),
            "mode": "local",
        }
