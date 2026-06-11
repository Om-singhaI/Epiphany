"""Agent control router.

Exposes HTTP endpoints to inspect provider wiring and trigger an on-demand
agent cycle (e.g. the dashboard's "Force Sync" button). The autonomous loop also
runs continuously in the background (see :func:`app.main.lifespan`); this router
lets a human kick an extra cycle and observe live provider modes.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings, reload_settings
from app.services import connections
from app.services.model_generator import model_filename, render_model_script

router = APIRouter(prefix="/api/agent", tags=["agent"])


def _humanize(name: str | None) -> str:
    """Turn a raw column name into a readable label.

    e.g. ``Login_Frequency`` -> ``Login Frequency``,
    ``avg_latency_ms`` -> ``Avg Latency Ms``, ``churn_30_days`` -> ``Churn 30 Days``.
    """
    if not name:
        return "Discovered Driver"
    import re

    words = re.split(r"[\W_]+", str(name).strip())
    return " ".join(w.capitalize() for w in words if w) or "Discovered Driver"


@router.get("/status")
async def agent_status() -> JSONResponse:
    """Report which providers are live vs. running in simulation mode."""
    s = get_settings()
    return JSONResponse(
        {
            "providers": {
                "fivetran": "live" if s.fivetran_enabled else "local",
                "elastic": "live" if s.elastic_enabled else "local",
                "gemini": "live" if s.gemini_enabled else "data-driven",
                "gitlab": "live" if s.gitlab_enabled else "local",
            },
            "force_simulation": s.force_simulation,
            "auto_deploy": s.auto_deploy,
            "data_source": s.elastic_index if s.elastic_enabled else s.data_csv_path,
        }
    )


@router.get("/connections")
async def get_connections() -> JSONResponse:
    """Report per-provider connection status (never exposes secret values)."""
    return JSONResponse(connections.connection_status(get_settings()))


@router.post("/connect")
async def connect(request: Request) -> JSONResponse:
    """Save user-supplied provider credentials and reconfigure the agent live.

    Accepts a JSON body with any subset of the allowed connection fields (e.g.
    ``gemini_api_key``, ``elastic_api_key``, ``elastic_cloud_id``,
    ``fivetran_api_key``, ``gitlab_token``, ``gitlab_project_id``,
    ``auto_deploy``, ...). An empty string clears (disconnects) a field. Secret
    values are persisted server-side and never returned.
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "body must be an object"}, status_code=400)

    # Persist the provided fields, then rebuild settings + provider clients.
    connections.save_overrides(payload)
    settings = reload_settings()

    orchestrator = getattr(request.app.state, "orchestrator", None)
    applied = None
    if orchestrator is not None:
        applied = await orchestrator.reconfigure(settings)

    return JSONResponse(
        {
            "status": "connected",
            "applied": applied,
            "connections": connections.connection_status(settings),
        }
    )


# ── Dataset management (bring-your-own-data) ─────────────────────────────
_DATA_ROOT = Path("data")
_UPLOAD_DIR = _DATA_ROOT / "uploads"
_DATA_EXTS = {".csv", ".parquet", ".pq"}


def _safe_name(name: str) -> str:
    """Sanitise an uploaded filename to a basename with a safe extension."""
    base = Path(name or "dataset.csv").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "dataset"
    if not stem.lower().endswith(tuple(_DATA_EXTS)):
        stem += ".csv"
    return stem


async def _dataset_summary(orchestrator) -> dict:
    return await orchestrator.data.summary()


@router.get("/dataset")
async def dataset(request: Request) -> JSONResponse:
    """Return a domain-agnostic summary of the dataset under analysis."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    try:
        return JSONResponse(await _dataset_summary(orchestrator))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/datasets")
async def datasets() -> JSONResponse:
    """List datasets available on the server (bundled samples + uploads)."""
    items = []
    for root in (_DATA_ROOT, _UPLOAD_DIR):
        if not root.exists():
            continue
        for p in sorted(root.glob("*")):
            if p.suffix.lower() in _DATA_EXTS and p.is_file():
                items.append({
                    "name": p.name,
                    "path": str(p).replace("\\", "/"),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "uploaded": (_UPLOAD_DIR in p.parents),
                })
    return JSONResponse({"datasets": items})


async def _switch_dataset(request: Request, path: str) -> JSONResponse:
    """Point the agent at ``path`` (must live under ./data) and reconfigure."""
    target = Path(path)
    try:
        resolved = target.resolve()
        if not resolved.is_relative_to(_DATA_ROOT.resolve()):
            return JSONResponse({"error": "path must be inside ./data"}, status_code=400)
    except (ValueError, OSError):
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not resolved.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)

    connections.save_overrides({"data_csv_path": str(target), "prefer_local_data": True})
    settings = reload_settings()
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is not None:
        await orchestrator.reconfigure(settings)
        try:
            summary = await _dataset_summary(orchestrator)
        except Exception as exc:  # noqa: BLE001 - bad file, surface it
            return JSONResponse({"error": f"could not read dataset: {exc}"}, status_code=400)
        return JSONResponse({"status": "switched", "dataset": summary})
    return JSONResponse({"status": "switched"})


@router.post("/select-dataset")
async def select_dataset(request: Request) -> JSONResponse:
    """Switch analysis to an existing dataset under ./data."""
    try:
        payload = await request.json()
        path = (payload or {}).get("path")
    except Exception:
        path = None
    if not isinstance(path, str) or not path:
        return JSONResponse({"error": "provide a dataset 'path'"}, status_code=400)
    return await _switch_dataset(request, path)


@router.post("/upload-dataset")
async def upload_dataset(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Upload a CSV/Parquet file and immediately analyse it.

    The agent re-profiles the new file, picks a target, and the next cycle runs
    on it — making Epiphany work for any dataset, any domain.
    """
    name = _safe_name(file.filename or "dataset.csv")
    if Path(name).suffix.lower() not in _DATA_EXTS:
        return JSONResponse({"error": "only .csv or .parquet files are supported"}, status_code=400)

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / name
    try:
        content = await file.read()
        if not content:
            return JSONResponse({"error": "empty file"}, status_code=400)
        if len(content) > 100 * 1024 * 1024:
            return JSONResponse({"error": "file exceeds 100 MB limit"}, status_code=400)
        dest.write_bytes(content)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"upload failed: {exc}"}, status_code=400)

    # Validate it actually loads as a table before switching to it.
    try:
        import pandas as pd

        if dest.suffix.lower() in {".parquet", ".pq"}:
            pd.read_parquet(dest)
        else:
            pd.read_csv(dest, nrows=5)
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        return JSONResponse({"error": f"not a readable dataset: {exc}"}, status_code=400)

    return await _switch_dataset(request, str(dest))


@router.post("/run")
async def run_cycle(request: Request) -> JSONResponse:
    """Trigger a single agent cycle on demand (non-blocking).

    Accepts an optional JSON body ``{"user_goal": "..."}`` from the dashboard's
    Mission Control. When a goal is supplied, the agent investigates that
    specific business question; otherwise it runs its default autonomous pass
    (e.g. the header's "Force Sync" button sends no body).
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)

    user_goal: str | None = None
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            raw = payload.get("user_goal")
            if isinstance(raw, str) and raw.strip():
                user_goal = raw.strip()[:500]
    except Exception:
        # No body / invalid JSON (e.g. Force Sync) -> default autonomous cycle.
        user_goal = None

    # Fire-and-forget so the HTTP call returns immediately; progress streams
    # over the WebSocket. The orchestrator guards against overlapping cycles.
    asyncio.create_task(orchestrator.run_cycle(user_goal=user_goal))
    return JSONResponse({"status": "started", "user_goal": user_goal})


@router.get("/hypotheses")
async def list_hypotheses(request: Request, limit: int = 50) -> JSONResponse:
    """Return recorded hypotheses (newest first) for the Hypothesis Log."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    rows = await orchestrator.repository.list_hypotheses(limit=limit)
    return JSONResponse({"hypotheses": rows})


@router.get("/deployments")
async def list_deployments(request: Request, limit: int = 50) -> JSONResponse:
    """Return recorded deployments (merge requests) for the Deployments view."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    rows = await orchestrator.repository.list_deployments(limit=limit)
    return JSONResponse({"deployments": rows})


@router.get("/interventions")
async def interventions(request: Request, limit: int = 20) -> JSONResponse:
    """Return the agent's autonomous interventions for the dashboard table.

    An *intervention* is a recorded hypothesis the agent acted on: its
    discovery, the validation verdict, and — when significant — the merge
    request it autonomously opened. Composed from the live SQLite history so
    the table hydrates entirely from real agent activity (no static rows).
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)

    repo = orchestrator.repository
    hypotheses = await repo.list_hypotheses(limit=limit)
    deployments = await repo.list_deployments(limit=limit)
    by_hid = {d.get("hypothesis_id"): d for d in deployments}

    rows = []
    for h in hypotheses:
        dep = by_hid.get(h.get("id"))
        rows.append(
            {
                "id": h.get("id"),
                "created_at": h.get("created_at"),
                "feature": h.get("feature"),
                "target": h.get("target"),
                "driver": _humanize(h.get("feature")),
                "outcome": _humanize(h.get("target")),
                "statement": h.get("statement"),
                "p_value": h.get("p_value"),
                "is_significant": bool(h.get("is_significant")),
                "data_mode": h.get("data_mode"),
                "action": "Model deployed" if dep else (
                    "Validated" if h.get("is_significant") else "Rejected"
                ),
                "mr_iid": dep.get("mr_iid") if dep else None,
                "mr_url": dep.get("mr_url") if dep else None,
                "branch": dep.get("branch") if dep else None,
            }
        )
    return JSONResponse({"interventions": rows})


@router.get("/metrics")
async def metrics(request: Request) -> JSONResponse:
    """Return live aggregate metrics powering the dashboard's hero cards."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    data = await orchestrator.repository.metrics()
    return JSONResponse(data)


@router.get("/scatter")
async def scatter(request: Request) -> JSONResponse:
    """Deprecated alias for :func:`latest_insight`.

    Superseded by ``/api/agent/latest-insight``, which plots whichever
    feature/target the agent has actually validated on the real data.
    """
    response = await latest_insight(request)
    return response


@router.get("/latest-insight")
async def latest_insight(request: Request) -> JSONResponse:
    """Return the most recently *validated* discovery for the dynamic chart.

    Looks up the latest significant hypothesis (falling back to the latest
    hypothesis of any kind), then samples the real relationship between the
    proven ``feature`` and ``target`` from the dataset so the dashboard renders
    whichever driver the agent decided to validate — with axis labels and a
    title bound to that discovery rather than a fixed column.
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)

    latest = await orchestrator.repository.latest_significant_hypothesis()
    if not latest:
        latest = await orchestrator.repository.latest_hypothesis()

    if not latest:
        return JSONResponse(
            {
                "feature": None,
                "target": None,
                "points": [],
                "mode": "local",
                "is_significant": False,
                "x_label": "Discovered Driver",
                "y_label": "Outcome Probability",
                "title": "Awaiting first discovery",
            }
        )

    meta = latest.get("metadata") or {}
    feature = latest.get("feature") or meta.get("feature")
    target = latest.get("target") or meta.get("target")
    threshold = latest.get("threshold")
    if threshold is None:
        threshold = meta.get("threshold")

    dist = {"points": [], "mode": "local"}
    if feature and target:
        dist = await orchestrator.data.feature_target_points(feature, target)

    x_label = _humanize(feature)
    y_label = f"{_humanize(target)} Rate"
    return JSONResponse(
        {
            "feature": feature,
            "target": target,
            "threshold": threshold,
            "statement": latest.get("statement"),
            "p_value": latest.get("p_value"),
            "is_significant": bool(latest.get("is_significant")),
            "points": dist["points"],
            "mode": dist["mode"],
            "x_label": x_label,
            "y_label": y_label,
            "title": f"{x_label} vs {y_label}",
        }
    )


@router.get("/latest-model")
async def latest_model(request: Request) -> JSONResponse:
    """Return the model script generated for the most recent hypothesis."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    latest = await orchestrator.repository.latest_hypothesis()
    if not latest or not latest.get("metadata"):
        return JSONResponse({"script": None, "filename": None, "hypothesis": None})
    script = render_model_script(latest["metadata"])
    return JSONResponse(
        {
            "script": script,
            "filename": model_filename(latest["metadata"]),
            "hypothesis": latest.get("statement"),
            "p_value": latest.get("p_value"),
            "is_significant": latest.get("is_significant"),
        }
    )
