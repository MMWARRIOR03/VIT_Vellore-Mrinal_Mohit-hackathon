"""
Health-check endpoints.

GET /health        — liveness probe (no dependencies checked)
GET /health/ready  — readiness probe (checks Redis + DuckDB)
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_START_TIME = time.time()


@router.get("/health", response_model=None)
async def liveness() -> JSONResponse:
    """
    Liveness probe — always returns 200 if the process is alive.
    Kubernetes / load-balancers use this to decide whether to restart the pod.
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "uptime_seconds": round(time.time() - _START_TIME, 1),
        },
    )


@router.get("/health/ready", response_model=None)
async def readiness(request: Request) -> JSONResponse:
    """
    Readiness probe — checks that Redis and DuckDB are reachable.
    Returns 200 if ready, 503 if any dependency is unavailable.
    """
    checks: dict[str, Any] = {}
    all_ok = True

    # ── Redis check ───────────────────────────────────────────────────────────
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        checks["redis"] = {"status": "not_initialised"}
        all_ok = False
    else:
        try:
            # RedisStreamClient wraps the aioredis connection; ping via the
            # internal _redis attribute.
            r = redis_client._redis  # noqa: SLF001
            if r is not None:
                await r.ping()
                checks["redis"] = {"status": "ok"}
            else:
                checks["redis"] = {"status": "not_connected"}
                all_ok = False
        except Exception as exc:
            logger.warning("Redis health check failed: %s", exc)
            checks["redis"] = {"status": "error", "detail": str(exc)}
            all_ok = False

    # ── DuckDB check ──────────────────────────────────────────────────────────
    db = getattr(request.app.state, "db", None)
    if db is None:
        checks["duckdb"] = {"status": "not_initialised"}
        all_ok = False
    else:
        try:
            conn = db._conn  # noqa: SLF001
            if conn is not None:
                # Trivial query to confirm the connection is live
                conn.execute("SELECT 1").fetchone()
                checks["duckdb"] = {"status": "ok"}
            else:
                checks["duckdb"] = {"status": "not_connected"}
                all_ok = False
        except Exception as exc:
            logger.warning("DuckDB health check failed: %s", exc)
            checks["duckdb"] = {"status": "error", "detail": str(exc)}
            all_ok = False

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ok else "degraded",
            "checks": checks,
            "uptime_seconds": round(time.time() - _START_TIME, 1),
        },
    )
