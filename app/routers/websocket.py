"""WebSocket router for the live agent stream.

Exposes ``/ws/agent-stream``. On connection the client subscribes to the shared
:class:`~app.services.events.EventBus`, receiving the *same* structured events
that the running :class:`~app.services.agent_orchestrator.AgentOrchestrator`
publishes as it executes each step of the loop. The frontend renders each event
as a colour-coded line in the "Active Agent Stream" terminal.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.events import event_bus
from app.services.session import safe_uid

logger = logging.getLogger("epiphany.websocket")

router = APIRouter()


@router.websocket("/ws/agent-stream")
async def agent_stream(websocket: WebSocket) -> None:
    """Stream live agent log events to a connected dashboard client.

    Only events belonging to the connecting user (``?uid=`` query param) are
    forwarded, so one account never sees another's agent activity.
    """
    await websocket.accept()
    uid = safe_uid(websocket.query_params.get("uid"))
    logger.info("Agent stream client connected: %s (uid=%s)", websocket.client, uid)
    try:
        async with event_bus.subscribe() as queue:
            while True:
                event = await queue.get()
                if event.get("user_id") not in (uid, None):
                    continue  # not this user's event
                await websocket.send_json(event)
    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.info("Agent stream client disconnected: %s", websocket.client)
    except Exception:  # noqa: BLE001 - log unexpected failures, then close
        logger.exception("Agent stream terminated unexpectedly")
        await websocket.close(code=1011)
