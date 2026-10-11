"""
ModuleBRunner — async consumer loop that feeds signals to StressEngine.

Consumption topology
────────────────────
  processed.signals  →  StressEngine.on_signal  →  signals.stress
                                                 →  DuckDB stress_runs

The runner subscribes to the ``processed.signals`` stream (same stream the
NLP engine or FakeSignalGenerator publishes to).  It deserialises each message
into a RiskSignal and calls StressEngine.on_signal.  Non-triggering signals
are silently dropped; triggering ones are persisted and forwarded.

Failure handling
────────────────
- Invalid JSON or Pydantic validation errors are logged and skipped; the
  consumer advances its last_id so bad messages don't block the queue.
- Unhandled exceptions in StressEngine are caught and logged but don't kill
  the runner loop.
- An asyncio.CancelledError propagates cleanly for graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import ValidationError

from app.schema.risk_signal import RiskSignal
from app.storage.duckdb_store import DuckDBStore
from app.streams.redis_streams import STREAM_PROCESSED_SIGNALS, RedisStreamClient
from module_b.stress_engine import StressEngine

logger = logging.getLogger(__name__)


class ModuleBRunner:
    """
    Async runner that consumes processed.signals and drives StressEngine.

    Parameters
    ──────────
    db    : DuckDBStore  — passed through to StressEngine for persistence
    redis : RedisStreamClient — used for both consuming and publishing
    """

    def __init__(
        self,
        db: DuckDBStore,
        redis: RedisStreamClient,
    ) -> None:
        self._db = db
        self._redis = redis
        self._engine: StressEngine | None = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """
        Initialise all components.

        Call this before run_forever.  Separated so that tests can call start()
        and inspect the engine without running the loop.
        """
        self._engine = StressEngine(
            db=self._db,
            redis=self._redis,
        )
        logger.info(
            "ModuleBRunner started.  Portfolio: %d positions, NAV=$%.0f",
            self._engine.portfolio.position_count()["total"],
            self._engine.portfolio.compute_nav(),
        )

    async def run_forever(self) -> None:
        """
        Consume processed.signals in a loop and run stress tests.

        Exits cleanly when the stop event is set or the task is cancelled.
        Always call start() before run_forever().
        """
        if self._engine is None:
            raise RuntimeError("ModuleBRunner.start() must be called before run_forever().")

        consumer = self._redis.consumer(STREAM_PROCESSED_SIGNALS, last_id="$")
        logger.info("ModuleBRunner consuming from stream: %s", STREAM_PROCESSED_SIGNALS)

        async for msg_id, payload in consumer.iter_messages(self._stop_event):
            await self._handle_message(msg_id, payload)

    async def stop(self) -> None:
        """Signal the run_forever loop to exit after the current batch."""
        self._stop_event.set()
        logger.info("ModuleBRunner stop requested.")

    async def _handle_message(
        self,
        msg_id: str,
        payload: dict,
    ) -> None:
        """
        Deserialise a stream message into a RiskSignal and process it.

        The stream message has the structure emitted by FakeSignalGenerator
        or the NLP engine: ``{"data": "<json_string>"}`` which the
        StreamConsumer already decodes into a dict (the ``payload`` arg here).
        """
        # The StreamConsumer decodes the outer {"data": "..."} envelope;
        # payload is the inner JSON dict of the RiskSignal.
        try:
            signal = RiskSignal.model_validate(payload)
        except ValidationError as exc:
            logger.warning(
                "Invalid RiskSignal in stream (msg_id=%s): %s",
                msg_id, exc.errors(include_url=False),
            )
            return
        except Exception as exc:
            logger.warning(
                "Failed to parse stream message (msg_id=%s): %s", msg_id, exc
            )
            return

        try:
            result = await self._engine.on_signal(signal)
            if result is not None:
                logger.info(
                    "Stress test completed: run_id=%s  ΔNAV=$%.0f (%.2f%%)",
                    result.run_id,
                    result.delta_nav,
                    result.delta_pct,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Unhandled error in StressEngine.on_signal (event_id=%s): %s",
                signal.event_id,
                exc,
            )

    @property
    def engine(self) -> StressEngine | None:
        """Expose the underlying StressEngine (useful for the dashboard)."""
        return self._engine


# ── Standalone entry point ────────────────────────────────────────────────────

async def _main() -> None:
    """Run ModuleB standalone."""
    import os
    from pathlib import Path

    from app.config import get_settings

    settings = get_settings()
    db = DuckDBStore(settings.duckdb_path)
    await db.initialise()

    redis = RedisStreamClient(settings.redis_url)
    await redis.connect()

    runner = ModuleBRunner(db=db, redis=redis)
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
