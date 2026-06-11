"""Step 2 — Explore: Elastic client + MCP-style database toolset.

Exposes the telemetry index as a database the agent can explore dynamically.
The client is a thin, real wrapper around Elasticsearch that fetches rows and
reads the index mapping; all profiling and aggregation happen on the resulting
:class:`pandas.DataFrame` via :class:`app.services.data_port.DataPort`, so live
Elastic data and local-file data flow through exactly the same real code path.

When Elastic is not configured, :class:`DataPort` reads the local dataset
instead — there is no fabricated schema and no synthetic aggregation. The
:class:`ElasticMcpToolset` simply surfaces the DataPort's real schema-discovery
and aggregation capabilities under MCP-style tool names for the ADK agent.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings

logger = logging.getLogger("epiphany.elastic")


class ElasticClient:
    """Async-friendly Elasticsearch row/mapping reader."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.enabled = settings.elastic_enabled
        self._client = None
        if self.enabled:
            self._client = self._build_client()
            self.enabled = self._client is not None

    def _build_client(self):
        try:
            from elasticsearch import AsyncElasticsearch

            s = self._settings
            kwargs: dict[str, Any] = {"api_key": s.elastic_api_key}
            if s.elastic_cloud_id:
                kwargs["cloud_id"] = s.elastic_cloud_id
            else:
                kwargs["hosts"] = [s.elastic_url]
            return AsyncElasticsearch(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Elastic client init failed (%s); using local data.", exc)
            return None

    async def fetch_rows(self, size: int = 10_000) -> list[dict[str, Any]]:
        """Fetch up to ``size`` source documents from the index as plain dicts.

        Uses ``search_after`` paging so we can pull more than Elasticsearch's
        default 10,000-row result window without tripping its limit.
        """
        if not self.enabled or self._client is None:
            return []
        index = self._settings.elastic_index
        page = min(size, 10_000)
        rows: list[dict[str, Any]] = []
        search_after: list[Any] | None = None
        while len(rows) < size:
            body: dict[str, Any] = {
                "size": min(page, size - len(rows)),
                "query": {"match_all": {}},
                "sort": [{"_doc": "asc"}],
            }
            if search_after is not None:
                body["search_after"] = search_after
            resp = await self._client.search(index=index, body=body)
            hits = resp["hits"]["hits"]
            if not hits:
                break
            rows.extend(h["_source"] for h in hits)
            search_after = hits[-1].get("sort")
            if search_after is None or len(hits) < body["size"]:
                break
        return rows

    async def index_mapping(self) -> dict[str, Any]:
        """Return the index's field→type mapping (best-effort)."""
        if not self.enabled or self._client is None:
            return {}
        try:
            index = self._settings.elastic_index
            mapping = await self._client.indices.get_mapping(index=index)
            first = next(iter(mapping.values()))
            return first.get("mappings", {}).get("properties", {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mapping fetch failed (%s).", exc)
            return {}

    async def row_count(self) -> int | None:
        """Return the index's document count (best-effort)."""
        if not self.enabled or self._client is None:
            return None
        try:
            resp = await self._client.count(index=self._settings.elastic_index)
            return int(resp.get("count", 0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Count failed (%s).", exc)
            return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()


class ElasticMcpToolset:
    """MCP-style database toolset over the real dataset.

    Bundles the two capabilities the ADK agent uses to explore data dynamically
    — schema discovery and aggregation — and backs them with real data via
    :class:`app.services.data_port.DataPort`. Aggregations are computed on the
    actual DataFrame, so the agent always sees genuine numbers whether the data
    came from Elastic or a local file.
    """

    def __init__(self, data_port: Any, index: str) -> None:
        self._data = data_port
        self._index = index

    @property
    def index(self) -> str:
        return self._index

    async def get_database_schema(self) -> dict[str, Any]:
        """MCP tool: list the dataset's fields and inferred analytical roles."""
        profile = await self._data.profile()
        return profile.as_dict()

    async def execute_aggregation(
        self, dimension: str, metric: str, agg: str = "mean"
    ) -> dict[str, Any]:
        """MCP tool: group ``metric`` by ``dimension`` on the real data."""
        result = await self._data.aggregate(dimension, metric, agg)
        result["mode"] = self._data.mode
        return result
