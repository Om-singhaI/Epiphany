"""Epiphany FastAPI application entrypoint.

Wires together the dashboard (HTML/Jinja2), static asset serving, and the
real-time agent WebSocket stream. Run locally with:

    uvicorn app.main:app --reload

The application is intentionally thin: routing concerns live in ``app.routers``
and the agent's domain logic lives in ``app.services``.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import agent, dashboard, websocket
from app.services.agent_orchestrator import AgentOrchestrator

logging.basicConfig(level=logging.INFO)

# Resolve project-relative directories so the app runs from any CWD.
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the autonomous agent loop as a background task on boot.

    The orchestrator publishes structured events to the shared event bus, which
    the WebSocket router fans out to every connected dashboard client.
    """
    settings = get_settings()
    orchestrator = AgentOrchestrator(settings=settings)
    app.state.orchestrator = orchestrator
    await orchestrator.repository.init()
    # Start the autonomous loop unless disabled (e.g. on a constrained host
    # where cycles are triggered on demand via the dashboard instead).
    task = (
        asyncio.create_task(orchestrator.run_forever())
        if settings.run_autonomous_loop
        else None
    )
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


def create_app() -> FastAPI:
    """Application factory.

    Using a factory keeps wiring explicit and makes the app trivial to
    instantiate inside tests with isolated configuration.
    """
    app = FastAPI(
        title="Epiphany",
        description="Autonomous AI Data Scientist — Phase 2",
        version="0.2.0",
        lifespan=lifespan,
    )

    # Serve front-end assets (custom JS/CSS) from /static. Tailwind and
    # Chart.js are loaded via CDN in the template, so this is reserved for
    # any first-party assets we add later.
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # Feature routers.
    app.include_router(dashboard.router)
    app.include_router(websocket.router)
    app.include_router(agent.router)

    return app


app = create_app()
