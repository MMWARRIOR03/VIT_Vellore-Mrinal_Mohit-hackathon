"""
FastAPI application factory and lifespan manager.

Startup order
─────────────
1. DuckDB store initialises schema (idempotent DDL).
2. Redis connection pool is created.
3. Fake signal generator starts its background loop (if enabled).
4. NLP worker starts (if NLP_WORKER_ENABLED=true).
5a. Replay runner starts (if REPLAY_MODE=true).
5b. Live ingestion pipeline starts (if INGESTION_ENABLED=true and not replay mode).
6. SignalExplainer singleton is attached to app.state.

Shutdown order
──────────────
All background tasks are cancelled in reverse startup order.
Redis and DuckDB are closed last.
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
from app.routes import inject as inject_router

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
_nlp_task: asyncio.Task | None = None
_replay_task: asyncio.Task | None = None
_module_a_task: asyncio.Task | None = None
_module_b_task: asyncio.Task | None = None


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
    global _nlp_task, _replay_task, _module_a_task, _module_b_task

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

    # Start both downstream consumers before producers begin publishing.  The
    # replay/NLP pipelines only publish processed signals; without these
    # runners, qualifying signals were stored but never produced stress runs.
    from module_a.runner import ModuleARunner
    from module_b.runner import ModuleBRunner

    module_a_runner = ModuleARunner(db=_db_store, redis=_redis_client)
    module_b_runner = ModuleBRunner(db=_db_store, redis=_redis_client)
    await module_a_runner.start()
    await module_b_runner.start()
    app.state.module_a_runner = module_a_runner
    app.state.module_b_runner = module_b_runner
    # The route uses these shared, stateful instances for synchronous /inject
    # responses as well.
    app.state.rebalancer = module_a_runner.rebalancer
    app.state.stress_engine = module_b_runner.engine
    _module_a_task = asyncio.create_task(
        module_a_runner.run_forever(), name="module-a-runner"
    )
    _module_b_task = asyncio.create_task(
        module_b_runner.run_forever(), name="module-b-runner"
    )
    logger.info("Module A and Module B runners started")

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

    # 5. NLP worker
    if settings.nlp_worker_enabled:
        from app.engine.worker_runner import start_nlp_worker
        # Pre-create the worker and stash it on app.state so /inject can reuse it
        from app.engine.nlp_worker import NLPWorker
        nlp_worker_instance = NLPWorker(db=_db_store, redis=_redis_client)
        app.state.nlp_worker = nlp_worker_instance
        _nlp_task = asyncio.create_task(
            nlp_worker_instance.run_forever(),
            name="nlp-worker",
        )
        logger.info("NLP worker started")

    # 6a. Replay mode
    if settings.replay_mode:
        from app.replay.replay_runner import ReplayRunner
        runner = ReplayRunner(
            sample_dir=settings.sample_data_dir,
            speed_multiplier=settings.replay_speed_multiplier,
            redis_client=_redis_client,
        )
        _replay_task = asyncio.create_task(
            runner.run_forever(),
            name="replay-runner",
        )
        logger.info(
            "Replay runner started (speed=%.1f×, dir=%s)",
            settings.replay_speed_multiplier,
            settings.sample_data_dir,
        )
    # 6b. Live ingestion pipeline
    elif settings.ingestion_enabled:
        from app.ingestion.manager import start_ingestion
        _ingestion_task = asyncio.create_task(
            start_ingestion(db=_db_store, redis=_redis_client),
            name="ingestion-pipeline",
        )
        logger.info("Ingestion pipeline started")

    # 7. LLM explainer singleton
    from app.llm.explainer import SignalExplainer
    app.state.explainer = SignalExplainer(
        provider=settings.llm_provider,
        groq_api_key=settings.groq_api_key,
        groq_model=settings.groq_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
    )
    logger.info("SignalExplainer ready (provider=%r)", settings.llm_provider)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("FinRisk API shutting down …")

    for runner in (getattr(app.state, "module_a_runner", None), getattr(app.state, "module_b_runner", None)):
        if runner is not None:
            await runner.stop()

    for task_name, task in [
        ("replay-runner", _replay_task),
        ("ingestion-pipeline", _ingestion_task),
        ("nlp-worker", _nlp_task),
        ("fake-signal-generator", _generator_task),
        ("module-b-runner", _module_b_task),
        ("module-a-runner", _module_a_task),
    ]:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.debug("%s cancelled", task_name)

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
    app.include_router(inject_router.router)

    from app.routes import module_a as module_a_routes
    app.include_router(module_a_routes.router)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def _unhandled(request, exc):  # noqa: ANN001
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
