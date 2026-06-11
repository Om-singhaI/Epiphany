"""Agent event schema and an in-process broadcast bus.

The orchestrator and the mock log simulator both emit :class:`AgentLogEvent`
objects. The :class:`EventBus` fans those events out to every connected
WebSocket client (the dashboard's "Active Agent Stream"), so a single agent run
is observed live by all viewers.

This is a lightweight, in-process pub/sub built on :class:`asyncio.Queue`.
A future phase can swap it for Redis/PubSub without touching the orchestrator or
the WebSocket router — they only depend on the ``publish`` / ``subscribe`` API.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import AsyncIterator

logger = logging.getLogger("epiphany.events")

# Recognised loop stages (used by the frontend for colour/status mapping).
STAGES = ("trigger", "explore", "reason", "validate", "deploy", "system")


@dataclass(frozen=True)
class AgentLogEvent:
    """A single structured log line emitted by the agent.

    Attributes:
        stage: One of :data:`STAGES`.
        source: Human-readable origin label, e.g. ``"FIVETRAN MCP"``.
        level: Visual severity used by the frontend (``info``, ``hypothesis``,
            ``success``, ``warning``, ``error``).
        message: The log body text shown in the terminal.
        mode: ``"live"`` when produced by a real provider, ``"simulation"``
            when produced by a fallback. Surfaced in the UI for transparency.
    """

    stage: str
    source: str
    level: str
    message: str
    mode: str = "live"

    def to_payload(self) -> dict:
        """Serialise to a JSON-friendly dict with a server timestamp."""
        payload = asdict(self)
        payload["timestamp"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return payload


@dataclass
class EventBus:
    """Fan-out broadcaster for agent events.

    Each subscriber gets its own bounded queue. Slow subscribers drop their
    oldest event rather than blocking the publisher (back-pressure isolation).
    """

    _subscribers: set[asyncio.Queue[dict]] = field(default_factory=set)
    maxsize: int = 100

    async def publish(self, event: AgentLogEvent | dict) -> None:
        """Broadcast an event to all current subscribers."""
        payload = event.to_payload() if isinstance(event, AgentLogEvent) else event
        for queue in list(self._subscribers):
            if queue.full():
                # Drop the oldest item to make room; never block the producer.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - race-safe guard
                    pass
            queue.put_nowait(payload)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[dict]]:
        """Context-managed subscription yielding a per-client event queue."""
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=self.maxsize)
        self._subscribers.add(queue)
        logger.debug("Subscriber added (total=%d)", len(self._subscribers))
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)
            logger.debug("Subscriber removed (total=%d)", len(self._subscribers))

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Process-wide singleton shared by the orchestrator and the WebSocket router.
event_bus = EventBus()
