"""
Ingestion manager — entry point for the ingestion subsystem.

Called from ``app/main.py`` during the lifespan startup when
``INGESTION_ENABLED=true``.  Constructs the pipeline with settings from
the central config and launches it as a background asyncio Task.

Usage in main.py lifespan::

    from app.ingestion.manager import start_ingestion

    if settings.ingestion_enabled:
        ingestion_task = asyncio.create_task(
            start_ingestion(db=_db_store, redis=_redis_client),
            name="ingestion-pipeline",
        )

    # On shutdown:
    ingestion_task.cancel()
    await asyncio.gather(ingestion_task, return_exceptions=True)
"""

from __future__ import annotations

import asyncio
import logging
import os

from app.ingestion.pipeline import IngestionPipeline
from app.storage.duckdb_store import DuckDBStore
from app.streams.redis_streams import RedisStreamClient

logger = logging.getLogger(__name__)


async def start_ingestion(
    db: DuckDBStore,
    redis: RedisStreamClient,
) -> None:
    """
    Create and run the IngestionPipeline.

    All adapter configuration is read from environment variables at call time
    (rather than at import time) so that tests can set env vars before calling
    this function.

    Environment variables
    ─────────────────────
    KAGGLE_TWEETS_CSV_PATH       Path to the tweet replay CSV.
    GDELT_POLL_INTERVAL_SECONDS  How often GDELT is polled (default 900).
    EDGAR_POLL_INTERVAL_SECONDS  How often EDGAR is polled (default 1800).
    STOCKTWITS_POLL_INTERVAL_SECONDS  How often StockTwits is polled (default 300).
    INGESTION_ADAPTERS           Comma-separated list of adapters to enable.
                                 Default: "gdelt,edgar,stocktwits,kaggle".
    """
    kaggle_csv_path = os.environ.get("KAGGLE_TWEETS_CSV_PATH", "")
    kaggle_speed = float(os.environ.get("KAGGLE_REPLAY_SPEED", "10.0"))
    gdelt_interval = float(os.environ.get("GDELT_POLL_INTERVAL_SECONDS", "900"))
    edgar_interval = float(os.environ.get("EDGAR_POLL_INTERVAL_SECONDS", "1800"))
    stocktwits_interval = float(os.environ.get("STOCKTWITS_POLL_INTERVAL_SECONDS", "300"))
    adapters_env = os.environ.get("INGESTION_ADAPTERS", "gdelt,edgar,stocktwits,kaggle")
    adapters_to_run = [a.strip() for a in adapters_env.split(",") if a.strip()]

    logger.info(
        "Starting IngestionPipeline with adapters=%s, gdelt_interval=%.0fs, "
        "edgar_interval=%.0fs, stocktwits_interval=%.0fs",
        adapters_to_run,
        gdelt_interval,
        edgar_interval,
        stocktwits_interval,
    )

    pipeline = IngestionPipeline(
        db=db,
        redis=redis,
        kaggle_csv_path=kaggle_csv_path or None,
        kaggle_speed=kaggle_speed,
        gdelt_interval=gdelt_interval,
        edgar_interval=edgar_interval,
        stocktwits_interval=stocktwits_interval,
    )

    try:
        await pipeline.run(adapters_to_run=adapters_to_run)
    except asyncio.CancelledError:
        logger.info("IngestionPipeline received cancellation, shutting down cleanly")
        pipeline.cancel()
        raise
    except Exception as exc:
        logger.error("IngestionPipeline fatal error: %s", exc, exc_info=True)
        raise
