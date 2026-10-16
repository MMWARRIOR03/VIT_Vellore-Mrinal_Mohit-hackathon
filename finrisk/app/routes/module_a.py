"""
Module A endpoints — rebalance events and current weights.

GET /module-a/rebalance-events   — recent rebalance events from DuckDB
GET /module-a/current-weights    — current portfolio weights from live rebalancer
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/module-a", tags=["Module A"])


@router.get("/rebalance-events", response_model=list[dict[str, Any]])
async def get_rebalance_events(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Return recent rebalance events from DuckDB."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return []
    try:
        events = await db.get_rebalance_events(limit=limit)
        return events
    except Exception as exc:
        logger.error("Failed to fetch rebalance events: %s", exc)
        return []


@router.get("/current-weights", response_model=dict[str, float])
async def get_current_weights(request: Request) -> dict[str, float]:
    """Return current portfolio weights from the live IndexRebalancer."""
    # Try to get from the live rebalancer first (most up-to-date)
    module_a_runner = getattr(request.app.state, "module_a_runner", None)
    if module_a_runner is not None and module_a_runner.rebalancer is not None:
        return module_a_runner.rebalancer.get_current_weights()

    # Fallback: last persisted weights from DuckDB
    db = getattr(request.app.state, "db", None)
    if db is None:
        return {}
    try:
        events = await db.get_rebalance_events(limit=1)
        if events:
            meta = events[0].get("meta", {})
            after = meta.get("after_weights", {})
            if after:
                return after
    except Exception as exc:
        logger.error("Failed to fetch current weights: %s", exc)

    # Final fallback: equal weight over 20 tickers
    tickers = ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA",
               "JPM","BAC","GS","V","JNJ","UNH","PFE","XOM","CVX","CAT","BA","WMT","HD"]
    return {t: round(1.0 / len(tickers), 6) for t in tickers}


@router.get("/stress-runs", response_model=list[dict[str, Any]])
async def get_stress_runs(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Return recent stress runs from DuckDB (for Module B dashboard)."""
    db = getattr(request.app.state, "db", None)
    if db is None:
        return []
    try:
        return await db.get_stress_runs(limit=limit)
    except Exception as exc:
        logger.error("Failed to fetch stress runs: %s", exc)
        return []
