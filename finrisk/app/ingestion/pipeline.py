"""
Ingestion pipeline: orchestrates all source adapters.

Flow
────
    SourceAdapter.fetch()
        │
        ▼
    Deduplication (InMemoryDedupStore)
        │  duplicate → discard
        ▼
    EntityLinker.link(title + body)  → augment ticker_symbols
        │
        ▼
    DuckDBStore.write_article(article_dict)
        │
        ▼
    RedisStreamClient.publish(raw.articles, payload)

Each document is processed atomically within ``process_document()``.
Failures in DuckDB or Redis are logged but do not halt the pipeline —
the ingestion loop must keep running even if storage is temporarily
unavailable.

Parallel execution
──────────────────
``run()`` launches each enabled adapter as a separate asyncio Task so that
a slow adapter (e.g. GDELT's 25-request poll cycle) doesn't block
StockTwits' faster 5-minute cycle.  The pipeline itself is a shared
resource: all adapter tasks funnel documents through the same
``process_document()`` coroutine which is safe for concurrent use because
DuckDBStore and RedisStreamClient both manage their own internal locks.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.ingestion.base import RawDocument
from app.ingestion.dedup import InMemoryDedupStore, compute_content_hash
from app.ingestion.entity_linker import EntityLinker
from app.ingestion.gdelt_adapter import GDELTAdapter
from app.ingestion.edgar_adapter import EDGARAdapter
from app.ingestion.stocktwits_adapter import StockTwitsAdapter
from app.ingestion.kaggle_tweet_replay import KaggleTweetReplayAdapter
from app.storage.duckdb_store import DuckDBStore
from app.streams.redis_streams import RedisStreamClient, STREAM_RAW_ARTICLES

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Wires adapters to storage and the Redis stream.

    Parameters
    ──────────
    db              Open DuckDBStore instance.
    redis           Connected RedisStreamClient instance.
    dedup_store     Optional shared dedup store; a new one is created if not
                    supplied.  Adapters share this store so cross-source
                    deduplication works (same story appearing on GDELT and
                    StockTwits).
    entity_linker   Optional shared EntityLinker.
    kaggle_csv_path Path to the Kaggle tweet CSV.  Falls back to env var.
    kaggle_speed    Speed multiplier for the tweet replay.
    gdelt_interval  GDELT poll interval in seconds.
    edgar_interval  EDGAR poll interval in seconds.
    stocktwits_interval  StockTwits poll interval in seconds.
    """

    def __init__(
        self,
        db: DuckDBStore,
        redis: RedisStreamClient,
        dedup_store: Optional[InMemoryDedupStore] = None,
        entity_linker: Optional[EntityLinker] = None,
        kaggle_csv_path: Optional[str] = None,
        kaggle_speed: float = 10.0,
        gdelt_interval: float = 900.0,
        edgar_interval: float = 1800.0,
        stocktwits_interval: float = 300.0,
    ) -> None:
        self._db = db
        self._redis = redis
        self._dedup = dedup_store or InMemoryDedupStore()
        self._linker = entity_linker or EntityLinker()

        # Build adapters, all sharing the same dedup + linker instances so that
        # a story appearing across multiple sources is only processed once.
        self._gdelt = GDELTAdapter(
            dedup_store=self._dedup,
            entity_linker=self._linker,
            on_document=self.process_document,
        )
        self._edgar = EDGARAdapter(
            dedup_store=self._dedup,
            entity_linker=self._linker,
            on_document=self.process_document,
        )
        self._stocktwits = StockTwitsAdapter(
            dedup_store=self._dedup,
            entity_linker=self._linker,
            on_document=self.process_document,
        )
        self._kaggle = KaggleTweetReplayAdapter(
            csv_path=kaggle_csv_path,
            speed_multiplier=kaggle_speed,
            dedup_store=self._dedup,
            entity_linker=self._linker,
            on_document=self.process_document,
        )

        self._gdelt_interval = gdelt_interval
        self._edgar_interval = edgar_interval
        self._stocktwits_interval = stocktwits_interval

        # Tracks running adapter tasks so callers can cancel them.
        self._tasks: list[asyncio.Task] = []

    # ── Core document processing ───────────────────────────────────────────────

    async def process_document(self, doc: RawDocument) -> None:
        """
        Process a single RawDocument through the full pipeline:

        1. Dedup check (the adapter may have already checked; this is a
           safety net for documents created outside of an adapter).
        2. Augment ticker_symbols via entity linking on title + body.
        3. Write to DuckDB.
        4. Publish to Redis raw.articles stream.
        """
        # 1. Dedup safety net
        combined_text = f"{doc.title} {doc.body}".strip()
        content_hash = compute_content_hash(combined_text)
        if self._dedup.is_duplicate(doc.article_id, content_hash):
            logger.debug(
                "Pipeline: duplicate skipped article_id=%s", doc.article_id
            )
            return
        self._dedup.mark_seen(doc.article_id, content_hash)

        # 2. Entity link title + body and merge results into ticker_symbols
        linked = self._linker.link(combined_text)
        existing = set(doc.ticker_symbols)
        for t in linked:
            if t not in existing:
                doc.ticker_symbols.append(t)
                existing.add(t)

        # 3. Build article_dict matching DuckDBStore.write_article() signature
        article_dict = _build_article_dict(doc)

        # 4. Write to DuckDB (non-blocking, errors logged)
        try:
            await self._db.write_article(article_dict)
        except Exception as exc:
            logger.error(
                "Pipeline: DuckDB write failed for %s: %s", doc.article_id, exc
            )

        # 5. Publish to Redis raw.articles stream
        try:
            await self._redis.publish(STREAM_RAW_ARTICLES, article_dict)
        except Exception as exc:
            logger.error(
                "Pipeline: Redis publish failed for %s: %s", doc.article_id, exc
            )

        logger.debug(
            "Pipeline: processed article_id=%s tickers=%s source=%s",
            doc.article_id,
            doc.ticker_symbols,
            doc.source,
        )

    # ── Adapter orchestration ──────────────────────────────────────────────────

    async def run(
        self,
        adapters_to_run: list[str] = ("gdelt", "edgar", "stocktwits", "kaggle"),
    ) -> None:
        """
        Start all enabled adapters as concurrent asyncio Tasks.

        ``adapters_to_run`` is a list of adapter names to activate.  Use this
        to disable individual adapters in tests or limited environments.

        This coroutine blocks until all tasks complete (i.e. until cancelled).
        Adapter tasks that raise unexpected exceptions are logged but not
        restarted automatically — wrap this in a supervisor if you need
        auto-restart semantics.
        """
        enabled = set(a.lower() for a in adapters_to_run)
        logger.info("IngestionPipeline.run() enabling adapters: %s", sorted(enabled))

        if "gdelt" in enabled:
            self._tasks.append(
                asyncio.create_task(
                    self._gdelt.run_forever(self._gdelt_interval),
                    name="adapter-gdelt",
                )
            )

        if "edgar" in enabled:
            self._tasks.append(
                asyncio.create_task(
                    self._edgar.run_forever(self._edgar_interval),
                    name="adapter-edgar",
                )
            )

        if "stocktwits" in enabled:
            self._tasks.append(
                asyncio.create_task(
                    self._stocktwits.run_forever(self._stocktwits_interval),
                    name="adapter-stocktwits",
                )
            )

        if "kaggle" in enabled:
            self._tasks.append(
                asyncio.create_task(
                    self._kaggle.run_forever(),
                    name="adapter-kaggle",
                )
            )

        if not self._tasks:
            logger.warning("IngestionPipeline: no adapters enabled; nothing to run")
            return

        # Wait for all tasks; propagate the first cancellation.
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("IngestionPipeline: cancellation received, stopping adapters")
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            # Await cancellation of remaining tasks
            await asyncio.gather(*self._tasks, return_exceptions=True)
            raise
        except Exception as exc:
            logger.error("IngestionPipeline: unexpected error in adapter task: %s", exc)
            raise

    def cancel(self) -> None:
        """Cancel all running adapter tasks."""
        for task in self._tasks:
            if not task.done():
                task.cancel()


# ── Serialisation helper ───────────────────────────────────────────────────────

def _build_article_dict(doc: RawDocument) -> dict[str, Any]:
    """
    Build the dict expected by ``DuckDBStore.write_article()``.

    DuckDB write_article() expects::

        article_id    str
        ts_published  datetime (timezone-aware)
        source        str
        url           str | None
        headline      str
        body          str | None
        tickers       list[str]
        (+ any extra keys are stored in raw_json automatically)
    """
    ts = doc.published_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    return {
        "article_id": doc.article_id,
        "ts_published": ts,
        "source": doc.source,
        "url": doc.url or None,
        "headline": doc.title,
        "body": doc.body or None,
        "tickers": doc.ticker_symbols,
        "source_weight": doc.source_weight,
        "raw_metadata": doc.raw_metadata,
    }
