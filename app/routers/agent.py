"""Agent control router (per-user).

Every endpoint is scoped to the requesting user via the ``X-User-Id`` header:
each account gets its own isolated agent (its own dataset, results, and live
stream) through the :class:`~app.services.session.SessionManager`. Provider
credentials (Gemini / Elastic / Fivetran / GitLab) remain global — they belong
to whoever deployed the app — while data and findings never cross accounts.
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
from app.services.session import safe_uid

router = APIRouter(prefix="/api/agent", tags=["agent"])

_DATA_ROOT = Path("data")
_DATA_EXTS = {".csv", ".parquet", ".pq"}


# ── per-user helpers ─────────────────────────────────────────────────────
def _uid(request: Request) -> str:
    """The requesting user's id (Clerk id or per-browser id), from a header."""
    return safe_uid(request.headers.get("X-User-Id"))


async def _orch(request: Request):
    """The per-user orchestrator (created on first use), or None if unavailable."""
    sessions = getattr(request.app.state, "sessions", None)
    if sessions is None:
        return None
    return await sessions.get(_uid(request))


def _humanize(name: str | None) -> str:
    """Turn a raw column name into a readable label."""
    if not name:
        return "Discovered Driver"
    words = re.split(r"[\W_]+", str(name).strip())
    return " ".join(w.capitalize() for w in words if w) or "Discovered Driver"


def _safe_name(name: str) -> str:
    """Sanitise an uploaded filename to a basename with a safe extension."""
    base = Path(name or "dataset.csv").name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._") or "dataset"
    if not stem.lower().endswith(tuple(_DATA_EXTS)):
        stem += ".csv"
    return stem


async def _autorun(orch, n: int = 3) -> None:
    """Run a few rotating cycles for a user so their dashboard populates and the
    agent feels autonomous — right after they provide data."""
    for i in range(n):
        try:
            goal = await orch._rotating_goal(i)
            result = await orch.run_cycle(user_goal=goal, allow_adk=False)
            if result.get("status") == "idle":
                break
        except Exception:  # noqa: BLE001 - never crash the background task
            break


# ── Provider status / connections (GLOBAL — the deployer's creds) ────────
@router.get("/status")
async def agent_status() -> JSONResponse:
    """Report which providers are live vs. running in fallback mode."""
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
        }
    )


@router.get("/connections")
async def get_connections() -> JSONResponse:
    """Report per-provider connection status (never exposes secret values)."""
    return JSONResponse(connections.connection_status(get_settings()))


@router.post("/connect")
async def connect(request: Request) -> JSONResponse:
    """Save global provider credentials and reconfigure all active sessions."""
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "body must be an object"}, status_code=400)

    connections.save_overrides(payload)
    settings = reload_settings()
    sessions = getattr(request.app.state, "sessions", None)
    if sessions is not None:
        await sessions.reconfigure_all()
    return JSONResponse(
        {"status": "connected", "connections": connections.connection_status(settings)}
    )


# ── Dataset management (per-user) ────────────────────────────────────────
@router.get("/dataset")
async def dataset(request: Request) -> JSONResponse:
    """Return a summary of *this user's* dataset (or an empty state)."""
    orch = await _orch(request)
    if orch is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    try:
        return JSONResponse(await orch.data.summary())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.get("/datasets")
async def datasets(request: Request) -> JSONResponse:
    """List only the datasets *this user* uploaded."""
    sessions = getattr(request.app.state, "sessions", None)
    if sessions is None:
        return JSONResponse({"datasets": []})
    up = sessions.uploads_dir(_uid(request))
    items = []
    if up.exists():
        for p in sorted(up.glob("*")):
            if p.suffix.lower() in _DATA_EXTS and p.is_file():
                items.append({
                    "name": p.name,
                    "path": str(p).replace("\\", "/"),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "uploaded": True,
                })
    return JSONResponse({"datasets": items})


async def _switch_dataset(request: Request, path: str) -> JSONResponse:
    """Point *this user's* agent at ``path`` (must live under this user's dir)."""
    sessions = getattr(request.app.state, "sessions", None)
    if sessions is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    uid = _uid(request)
    target = Path(path)
    try:
        resolved = target.resolve()
        allowed = sessions.user_root(uid).resolve()
        if not resolved.is_relative_to(allowed):
            return JSONResponse({"error": "path outside your workspace"}, status_code=400)
    except (ValueError, OSError):
        return JSONResponse({"error": "invalid path"}, status_code=400)
    if not resolved.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)

    orch = await sessions.set_dataset(uid, str(target), prefer_local=True)
    try:
        summary = await orch.data.summary()
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"could not read dataset: {exc}"}, status_code=400)
    # Kick off a few autonomous cycles for this user, streamed live.
    asyncio.create_task(_autorun(orch))
    return JSONResponse({"status": "switched", "dataset": summary})


@router.post("/select-dataset")
async def select_dataset(request: Request) -> JSONResponse:
    """Switch analysis to one of this user's previously-uploaded datasets."""
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
    """Upload a CSV/Parquet into *this user's* private workspace and analyse it."""
    sessions = getattr(request.app.state, "sessions", None)
    if sessions is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    uid = _uid(request)

    name = _safe_name(file.filename or "dataset.csv")
    if Path(name).suffix.lower() not in _DATA_EXTS:
        return JSONResponse({"error": "only .csv or .parquet files are supported"}, status_code=400)

    up = sessions.uploads_dir(uid)
    up.mkdir(parents=True, exist_ok=True)
    dest = up / name
    try:
        content = await file.read()
        if not content:
            return JSONResponse({"error": "empty file"}, status_code=400)
        if len(content) > 100 * 1024 * 1024:
            return JSONResponse({"error": "file exceeds 100 MB limit"}, status_code=400)
        dest.write_bytes(content)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": f"upload failed: {exc}"}, status_code=400)

    # Validate it loads as a table before switching to it.
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


# ── Run a cycle (per-user) ───────────────────────────────────────────────
@router.post("/run")
async def run_cycle(request: Request) -> JSONResponse:
    """Trigger a single agent cycle for this user (non-blocking)."""
    orch = await _orch(request)
    if orch is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)

    user_goal: str | None = None
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            raw = payload.get("user_goal")
            if isinstance(raw, str) and raw.strip():
                user_goal = raw.strip()[:500]
    except Exception:
        user_goal = None

    asyncio.create_task(orch.run_cycle(user_goal=user_goal))
    return JSONResponse({"status": "started", "user_goal": user_goal})


# ── Read endpoints (per-user) ────────────────────────────────────────────
@router.get("/hypotheses")
async def list_hypotheses(request: Request, limit: int = 50) -> JSONResponse:
    orch = await _orch(request)
    if orch is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    rows = await orch.repository.list_hypotheses(limit=limit)
    return JSONResponse({"hypotheses": rows})


@router.get("/deployments")
async def list_deployments(request: Request, limit: int = 50) -> JSONResponse:
    orch = await _orch(request)
    if orch is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    rows = await orch.repository.list_deployments(limit=limit)
    return JSONResponse({"deployments": rows})


@router.get("/interventions")
async def interventions(request: Request, limit: int = 20) -> JSONResponse:
    """The agent's interventions for this user, composed from their history."""
    orch = await _orch(request)
    if orch is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)

    repo = orch.repository
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
    orch = await _orch(request)
    if orch is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse(await orch.repository.metrics())


@router.get("/scatter")
async def scatter(request: Request) -> JSONResponse:
    """Deprecated alias for :func:`latest_insight`."""
    return await latest_insight(request)


@router.get("/latest-insight")
async def latest_insight(request: Request) -> JSONResponse:
    """This user's most recently validated discovery, for the dynamic chart."""
    orch = await _orch(request)
    if orch is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)

    latest = await orch.repository.latest_significant_hypothesis()
    if not latest:
        latest = await orch.repository.latest_hypothesis()

    if not latest:
        return JSONResponse(
            {
                "feature": None, "target": None, "points": [], "mode": "local",
                "is_significant": False, "x_label": "Discovered Driver",
                "y_label": "Outcome Probability", "title": "Awaiting first discovery",
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
        dist = await orch.data.feature_target_points(feature, target)

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
    """The model script generated for this user's most recent hypothesis."""
    orch = await _orch(request)
    if orch is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    latest = await orch.repository.latest_hypothesis()
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
