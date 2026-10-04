"""
ModuleARunner — async consumer loop that feeds signals to IndexRebalancer.

Consumption topology
────────────────────
  processed.signals  →  IndexRebalancer.on_signal  →  signals.rebalance
                                                    →  DuckDB rebalance_events

Mirrors the structure of ModuleBRunner so the two modules can be started
and stopped with the same lifecycle protocol.

Failure handling
────────────────
- Invalid JSON or Pydantic validation errors are logged and skipped.
- Unhandled exceptions in IndexRebalancer are caught and logged; the loop
  continues.
- asyncio.CancelledError propagates cleanly for graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from app.schema.risk_signal import RiskSignal
from app.storage.duckdb_store import DuckDBStore
from app.streams.redis_streams import (
    STREAM_PROCESSED_SIGNALS,
    STREAM_SIGNALS_REBALANCE,
    RedisStreamClient,
)
from module_a.rebalancer import IndexRebalancer

logger = logging.getLogger(__name__)


class ModuleARunner:
    """
    Async runner that consumes processed.signals and drives IndexRebalancer.

    Parameters
    ──────────
    db          : DuckDBStore           — rebalance events are persisted here
    redis       : RedisStreamClient     — source stream + publish target
    config_path : optional path to configs/index.yaml (None → auto-detect)
    """

    def __init__(
        self,
        db: DuckDBStore,
        redis: RedisStreamClient,
        config_path: Optional[Path] = None,
    ) -> None:
        self._db = db
        self._redis = redis
        self._config_path = config_path
        self._rebalancer: Optional[IndexRebalancer] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """
        Initialise the rebalancer and warm the price cache.

        Call this before run_forever.  Separated so tests can call start()
        and inspect the rebalancer state without running the loop.
        """
        self._rebalancer = IndexRebalancer(config_path=self._config_path)
        all_tickers = list(self._rebalancer.get_current_weights().keys())

        # Warm the price cache.  Failures are non-fatal (synthetic fallback).
        logger.info("ModuleARunner warming price cache for %d tickers…", len(all_tickers))
        self._rebalancer.price_cache.refresh(all_tickers)

        logger.info(
            "ModuleARunner started.  Universe: %d tickers.",
            len(all_tickers),
        )

    async def run_forever(self) -> None:
        """
        Consume processed.signals in a loop and run rebalance logic.

        Exits cleanly when the stop event is set or the task is cancelled.
        Always call start() before run_forever().
        """
        if self._rebalancer is None:
            raise RuntimeError(
                "ModuleARunner.start() must be called before run_forever()."
            )

        consumer = self._redis.consumer(STREAM_PROCESSED_SIGNALS, last_id="$")
        logger.info(
            "ModuleARunner consuming from stream: %s", STREAM_PROCESSED_SIGNALS
        )

        async for msg_id, payload in consumer.iter_messages(self._stop_event):
            await self._handle_message(msg_id, payload)

    async def stop(self) -> None:
        """Signal the run_forever loop to exit after the current batch."""
        self._stop_event.set()
        logger.info("ModuleARunner stop requested.")

    async def _handle_message(self, msg_id: str, payload: dict) -> None:
        """
        Deserialise a stream message into a RiskSignal and process it.

        The StreamConsumer already unwraps the outer {"data": "..."} envelope;
        payload here is the inner signal dict.
        """
        try:
            signal = RiskSignal.model_validate(payload)
        except ValidationError as exc:
            logger.warning(
                "Invalid RiskSignal (msg_id=%s): %s",
                msg_id, exc.errors(include_url=False),
            )
            return
        except Exception as exc:
            logger.warning(
                "Failed to parse stream message (msg_id=%s): %s", msg_id, exc
            )
            return

        try:
            result = await self._rebalancer.on_signal(signal)
            if result is not None:
                # Persist to DuckDB.
                await result.persist(self._db)
                # Publish to the rebalance stream for downstream consumers.
                await self._redis.publish(STREAM_SIGNALS_REBALANCE, result.to_dict())
                logger.info(
                    "Rebalance event published: id=%s trigger=%s reason=%r",
                    result.rebalance_id,
                    result.trigger_signal_id,
                    result.reason,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Unhandled error in IndexRebalancer.on_signal (event_id=%s): %s",
                signal.event_id,
                exc,
            )

    @property
    def rebalancer(self) -> Optional[IndexRebalancer]:
        """Expose the underlying IndexRebalancer (useful for the dashboard)."""
        return self._rebalancer


# ── Standalone entry point ────────────────────────────────────────────────────

async def _main() -> None:
    """Run Module A standalone."""
    from app.config import get_settings

    settings = get_settings()
    db = DuckDBStore(settings.duckdb_path)
    await db.initialise()

    redis = RedisStreamClient(settings.redis_url)
    await redis.connect()

    runner = ModuleARunner(db=db, redis=redis)
    await runner.start()

    try:
        await runner.run_forever()
    except asyncio.CancelledError:
        pass
    finally:
        await runner.stop()
        await redis.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(_main())
