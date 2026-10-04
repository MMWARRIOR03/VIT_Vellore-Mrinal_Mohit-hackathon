"""
FastAPI application factory and lifespan manager.

Startup order
─────────────
1. DuckDB store initialises schema (idempotent DDL).
2. Redis connection pool is created.
3. Fake signal generator starts its background loop (if enabled).

Shutdown order
──────────────
1. Generator loop is cancelled.
2. Redis connection is closed.
3. DuckDB connection is closed.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.storage.duckdb_store import DuckDBStore
from app.streams.redis_streams import RedisStreamClient
from app.generators.fake_signal import FakeSignalGenerator
from app.routes import health, signals, articles

logger = logging.getLogger(__name__)
settings = get_settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


# ── Global singletons (set during lifespan) ───────────────────────────────────

_db_store: DuckDBStore | None = None
_redis_client: RedisStreamClient | None = None
_generator_task: asyncio.Task | None = None
_ingestion_task: asyncio.Task | None = None


def get_db() -> DuckDBStore:
    if _db_store is None:
        raise RuntimeError("DuckDB store not initialised")
    return _db_store


def get_redis() -> RedisStreamClient:
    if _redis_client is None:
        raise RuntimeError("Redis client not initialised")
    return _redis_client


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_store, _redis_client, _generator_task, _ingestion_task

    logger.info("FinRisk API starting up …")

    # 1. DuckDB
    _db_store = DuckDBStore(settings.duckdb_path)
    await _db_store.initialise()
    logger.info("DuckDB store ready at %s", settings.duckdb_path)

    # 2. Redis
    _redis_client = RedisStreamClient(settings.redis_url)
    await _redis_client.connect()
    logger.info("Redis connected at %s", settings.redis_url)

    # 3. Attach singletons to app state so routes can access them via request.app
    app.state.db = _db_store
    app.state.redis = _redis_client

    # 4. Fake signal generator
    if settings.fake_signal_enabled:
        generator = FakeSignalGenerator(
            db=_db_store,
            redis=_redis_client,
        )
        _generator_task = asyncio.create_task(
            generator.run_forever(settings.fake_signal_interval_seconds),
            name="fake-signal-generator",
        )
        logger.info(
            "Fake signal generator running (interval=%.1fs)",
            settings.fake_signal_interval_seconds,
        )

    # 5. Ingestion pipeline (Task 2)
    if settings.ingestion_enabled:
        from app.ingestion.manager import start_ingestion
        _ingestion_task = asyncio.create_task(
            start_ingestion(db=_db_store, redis=_redis_client),
            name="ingestion-pipeline",
        )
        logger.info("Ingestion pipeline started")

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("FinRisk API shutting down …")

    if _ingestion_task is not None and not _ingestion_task.done():
        _ingestion_task.cancel()
        try:
            await _ingestion_task
        except asyncio.CancelledError:
            pass

    if _generator_task is not None and not _generator_task.done():
        _generator_task.cancel()
        try:
            await _generator_task
        except asyncio.CancelledError:
            pass

    if _redis_client is not None:
        await _redis_client.close()

    if _db_store is not None:
        await _db_store.close()

    logger.info("FinRisk API stopped cleanly.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="FinRisk Intelligence API",
        description=(
            "Real-time financial risk signal ingestion, storage, and streaming. "
            "Provides RiskSignal objects derived from news, social, and filing sources."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router, tags=["health"])
    app.include_router(signals.router, prefix="/signals", tags=["signals"])
    app.include_router(articles.router, prefix="/articles", tags=["articles"])

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _unhandled(request, exc):  # noqa: ANN001
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
