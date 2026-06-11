"""Persistence layer — records every Epiphany cycle to SQLite.

A thin, async repository (built on :mod:`aiosqlite`) that durably stores each
autonomous pass: the hypothesis, its statistical validation, and any merge
request opened as a result. This powers the dashboard's **Hypothesis Log** and
**Deployments** views and gives the agent a memory across restarts.

The schema is intentionally simple — one row per cycle plus a child row per
deployment — and is created on demand, so there is no separate migration step.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiosqlite

logger = logging.getLogger("epiphany.repository")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    statement     TEXT    NOT NULL,
    feature       TEXT,
    target        TEXT,
    threshold     REAL,
    p_value       REAL,
    statistic     REAL,
    is_significant INTEGER NOT NULL DEFAULT 0,
    sample_size   INTEGER,
    data_mode     TEXT,
    reason_mode   TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS deployments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id),
    created_at    TEXT    NOT NULL,
    mr_iid        INTEGER,
    mr_url        TEXT,
    branch        TEXT,
    title         TEXT,
    mode          TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Repository:
    """Async SQLite repository for cycle history."""

    db_path: str

    async def init(self) -> None:
        """Create tables if they do not yet exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    async def record_hypothesis(
        self,
        statement: str,
        metadata: dict[str, Any],
        *,
        data_mode: str,
        reason_mode: str,
    ) -> int:
        """Insert a hypothesis + its validation; return the new row id."""
        validation = metadata.get("validation", {})
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                """
                INSERT INTO hypotheses (
                    created_at, statement, feature, target, threshold,
                    p_value, statistic, is_significant, sample_size,
                    data_mode, reason_mode, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(),
                    statement,
                    metadata.get("feature"),
                    metadata.get("target"),
                    metadata.get("threshold"),
                    validation.get("p_value"),
                    validation.get("statistic"),
                    int(bool(validation.get("is_significant"))),
                    validation.get("sample_size"),
                    data_mode,
                    reason_mode,
                    json.dumps(metadata, default=str),
                ),
            )
            await db.commit()
            return int(cur.lastrowid)

    async def record_deployment(
        self, hypothesis_id: int, mr: dict[str, Any]
    ) -> int:
        """Insert a deployment (merge request) row; return the new row id."""
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                """
                INSERT INTO deployments (
                    hypothesis_id, created_at, mr_iid, mr_url, branch, title, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis_id,
                    _now(),
                    mr.get("iid"),
                    mr.get("web_url") or mr.get("url"),
                    mr.get("branch"),
                    mr.get("title"),
                    mr.get("mode"),
                ),
            )
            await db.commit()
            return int(cur.lastrowid)

    async def list_hypotheses(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent hypotheses, newest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM hypotheses ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = await cur.fetchall()
            return [self._hypothesis_row(dict(r)) for r in rows]

    async def list_deployments(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent deployments joined with their hypothesis."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT d.*, h.statement AS hypothesis_statement
                FROM deployments d
                JOIN hypotheses h ON h.id = d.hypothesis_id
                ORDER BY d.id DESC LIMIT ?
                """,
                (limit,),
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def metrics(self) -> dict[str, Any]:
        """Return product-level aggregate metrics for the dashboard cards."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT
                    COUNT(*)                          AS hypotheses_tested,
                    COALESCE(SUM(is_significant), 0)  AS significant_discoveries,
                    COALESCE(SUM(sample_size), 0)     AS rows_analyzed,
                    SUM(CASE WHEN data_mode = 'live' THEN 1 ELSE 0 END) AS live_cycles,
                    MAX(created_at)                   AS last_insight_at
                FROM hypotheses
                """
            )
            h = dict(await cur.fetchone())
            cur2 = await db.execute(
                "SELECT COUNT(*) AS models_deployed FROM deployments"
            )
            d = dict(await cur2.fetchone())

        tested = int(h["hypotheses_tested"] or 0)
        sig = int(h["significant_discoveries"] or 0)
        return {
            "rows_analyzed": int(h["rows_analyzed"] or 0),
            "hypotheses_tested": tested,
            "significant_discoveries": sig,
            "models_deployed": int(d["models_deployed"] or 0),
            "live_cycles": int(h["live_cycles"] or 0),
            "significance_rate": round(sig / tested, 3) if tested else 0.0,
            "last_insight_at": h["last_insight_at"],
        }

    async def latest_hypothesis(self) -> dict[str, Any] | None:
        """Return the most recent hypothesis (with metadata), or None."""
        rows = await self.list_hypotheses(limit=1)
        return rows[0] if rows else None

    async def latest_significant_hypothesis(self) -> dict[str, Any] | None:
        """Return the most recent *significant* hypothesis (with metadata).

        Powers the dynamic insight chart: the UI visualises whichever
        feature/target the agent has most recently *proven* (p < 0.05), so the
        plot reflects the agent's real discovery rather than a fixed column.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM hypotheses WHERE is_significant = 1 "
                "ORDER BY id DESC LIMIT 1"
            )
            row = await cur.fetchone()
            return self._hypothesis_row(dict(row)) if row else None

    @staticmethod
    def _hypothesis_row(row: dict[str, Any]) -> dict[str, Any]:
        row["is_significant"] = bool(row.get("is_significant"))
        meta = row.pop("metadata_json", None)
        if meta:
            try:
                row["metadata"] = json.loads(meta)
            except (TypeError, ValueError):
                row["metadata"] = None
        return row
