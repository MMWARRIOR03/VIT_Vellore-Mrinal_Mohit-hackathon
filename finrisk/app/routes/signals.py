"""
Signal endpoints.

GET  /signals          — query stored signals with optional filters
GET  /signals/stream   — SSE stream of live signals from Redis processed.signals
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.schema.risk_signal import RiskSignal
from app.streams.redis_streams import STREAM_PROCESSED_SIGNALS

logger = logging.getLogger(__name__)

router = APIRouter()


# ── GET /signals ──────────────────────────────────────────────────────────────

@router.get("", response_model=list[RiskSignal])
async def get_signals(
    request: Request,
    ticker: str | None = Query(
        default=None,
        description="Filter by ticker symbol (case-insensitive), e.g. AAPL.",
        min_length=1,
        max_length=10,
    ),
    since: datetime | None = Query(
        default=None,
        description=(
            "Return signals published at or after this timestamp. "
            "ISO 8601 format, e.g. 2024-01-15T09:00:00Z."
        ),
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of signals to return (1–500).",
    ),
) -> list[RiskSignal]:
    """
    Return stored RiskSignals, newest first.

    Filters can be combined: ``/signals?ticker=AAPL&since=2024-01-01T00:00:00Z&limit=20``
    """
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available.")

    try:
        signals = await db.get_signals(
            ticker=ticker.upper() if ticker else None,
            since=since,
            limit=limit,
        )
    except Exception as exc:
        logger.exception("Failed to fetch signals: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch signals.") from exc

    return signals


# ── GET /signals/stream (SSE) ─────────────────────────────────────────────────

@router.get("/stream", response_class=StreamingResponse)
async def stream_signals(
    request: Request,
    ticker: str | None = Query(
        default=None,
        description="If set, only forward signals that include this ticker.",
        min_length=1,
        max_length=10,
    ),
) -> StreamingResponse:
    """
    Server-Sent Events endpoint that forwards live RiskSignals from the Redis
    ``processed.signals`` stream to connected clients.

    Each SSE event has:
      - ``event: signal``
      - ``data: <RiskSignal JSON>``

    The stream starts from ``$`` (only new messages) so there is no replay of
    historical signals.  Clients that disconnect and reconnect will miss signals
    emitted while disconnected.

    Connect with::

        curl -N http://localhost:8000/signals/stream
        curl -N "http://localhost:8000/signals/stream?ticker=AAPL"
    """
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis not available.")

    ticker_filter = ticker.upper() if ticker else None

    async def _event_generator() -> AsyncIterator[str]:
        consumer = redis_client.consumer(STREAM_PROCESSED_SIGNALS, last_id="$")
        stop = asyncio.Event()

        # Send an initial heartbeat so the client knows the connection is live
        yield "event: connected\ndata: {}\n\n"

        async for msg_id, payload in consumer.iter_messages(stop_event=stop):
            # Check if client disconnected
            if await request.is_disconnected():
                logger.debug("SSE client disconnected — stopping stream.")
                stop.set()
                break

            # Optional ticker filter
            if ticker_filter is not None:
                tickers_in_signal: list[str] = [
                    t.get("ticker", "").upper()
                    for t in payload.get("tickers", [])
                ]
                if ticker_filter not in tickers_in_signal:
                    continue

            json_str = json.dumps(payload)
            yield f"event: signal\ndata: {json_str}\n\n"

        # Send a final keepalive comment so the client's parser flushes
        yield ": stream closed\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
            "Connection": "keep-alive",
        },
    )
