"""
Entry point for the NLP worker background task.

Called from ``app/main.py`` lifespan when ``NLP_WORKER_ENABLED=true``.
The worker requires both the DuckDB store and the Redis client to have been
initialised before this coroutine is awaited.
"""

from __future__ import annotations

import logging

from app.engine.nlp_worker import NLPWorker
from app.storage.duckdb_store import DuckDBStore
from app.streams.redis_streams import RedisStreamClient

logger = logging.getLogger(__name__)


async def start_nlp_worker(db: DuckDBStore, redis: RedisStreamClient) -> None:
    """
    Create an NLPWorker and run it forever.

    Designed to be wrapped in an ``asyncio.create_task()`` call so the event
    loop can cancel it during shutdown.

    Example (in main.py lifespan)::

        if settings.nlp_worker_enabled:
            _nlp_task = asyncio.create_task(
                start_nlp_worker(db=_db_store, redis=_redis_client),
                name="nlp-worker",
            )
    """
    logger.info("Starting NLP worker …")
    worker = NLPWorker(db=db, redis=redis)
    await worker.run_forever()
