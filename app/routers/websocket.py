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

logger = logging.getLogger("epiphany.websocket")

router = APIRouter()


@router.websocket("/ws/agent-stream")
async def agent_stream(websocket: WebSocket) -> None:
    """Stream live agent log events to a connected dashboard client."""
    await websocket.accept()
    logger.info("Agent stream client connected: %s", websocket.client)
    try:
        async with event_bus.subscribe() as queue:
            while True:
                event = await queue.get()
                await websocket.send_json(event)
    except (WebSocketDisconnect, asyncio.CancelledError):
        logger.info("Agent stream client disconnected: %s", websocket.client)
    except Exception:  # noqa: BLE001 - log unexpected failures, then close
        logger.exception("Agent stream terminated unexpectedly")
        await websocket.close(code=1011)
