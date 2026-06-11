"""Epiphany backend package.

Epiphany is an autonomous AI data scientist that operates in the background to:

1. **Trigger**  — Wake on new data syncs (Fivetran MCP).
2. **Explore**  — Run high-speed aggregations on telemetry (Elastic MCP).
3. **Reason**   — Generate business hypotheses (Gemini 3).
4. **Validate** — Prove hypotheses statistically in an isolated sandbox (Python).
5. **Deploy**   — Push predictive ML models as Merge Requests (GitLab MCP).

This package contains the Phase 1 foundation: the FastAPI application, the
dashboard + WebSocket routers, and the agent orchestration service skeleton.
"""

__version__ = "0.1.0"
