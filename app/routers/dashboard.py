"""Dashboard router.

Serves the Epiphany single-page dashboard via a Jinja2 template at the root
``/`` endpoint, plus a lightweight ``/health`` probe for readiness checks.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.config import get_settings

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Render the main Epiphany dashboard.

    The template receives the ``request`` (required by Jinja2Templates) so it
    can later build URLs for static assets or websocket endpoints if needed.
    """
    # Modern Starlette signature: (request, name, context). Passing the name
    # first (legacy form) breaks on newer Starlette, where the context dict gets
    # misread as the template name.
    return templates.TemplateResponse(
        request,
        "epiphany_dashboard.html",
        {
            # Publishable key is safe to expose; empty string → demo login.
            "clerk_publishable_key": get_settings().clerk_publishable_key or "",
        },
    )


@router.get("/health", response_class=JSONResponse)
async def health() -> JSONResponse:
    """Readiness/liveness probe used by orchestrators and uptime checks."""
    return JSONResponse({"status": "ok", "service": "epiphany"})
