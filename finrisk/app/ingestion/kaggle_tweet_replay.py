"""
Kaggle tweet replay adapter.

Reads a historical tweet CSV and replays it in chronological order,
optionally at an accelerated rate.  This is used for backtesting and demo
purposes — it simulates the social feed arriving in real time.

Expected CSV columns (flexible — column detection at load time):
  tweet_id / id           unique tweet identifier
  text / content / tweet  tweet body
  created_at / date / timestamp  publication timestamp
  ticker / stock_name / symbol   ticker symbol (optional; entity linking fills gaps)

If the CSV file is missing, the adapter logs a warning and generates synthetic
placeholder tweets for the 20-stock universe so the rest of the pipeline can
be demonstrated without a real dataset.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from app.ingestion.base import RawDocument, SourceAdapter
from app.ingestion.dedup import InMemoryDedupStore, compute_content_hash
from app.ingestion.entity_linker import EntityLinker

logger = logging.getLogger(__name__)

UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "BAC", "GS",
    "V", "JNJ", "UNH", "PFE", "XOM",
    "CVX", "CAT", "BA", "WMT", "HD",
]

# Possible column names for each logical field
_ID_COLS = ("tweet_id", "id", "tweetid", "tweet id")
_TEXT_COLS = ("text", "content", "tweet", "body", "message")
_DATE_COLS = ("created_at", "date", "timestamp", "datetime", "time", "published_at")
_TICKER_COLS = ("ticker", "stock_name", "symbol", "stock", "cashtag")

# Minimum simulated inter-tweet delay (seconds) for placeholder mode
_PLACEHOLDER_INTERVAL = 0.1


class KaggleTweetReplayAdapter(SourceAdapter):
    """
    Replays a historical tweet CSV in chronological order.

    Parameters
    ──────────
    csv_path          Path to the CSV file.  Reads from the environment variable
                      ``KAGGLE_TWEETS_CSV_PATH`` if not supplied.
    speed_multiplier  How many times faster than real time to replay.
                      1.0 = real time, 10.0 = 10× faster.
    dedup_store       Shared InMemoryDedupStore.
    entity_linker     Shared EntityLinker.
    on_document       Async callback invoked per new RawDocument.
    """

    def __init__(
        self,
        csv_path: Optional[str | Path] = None,
        speed_multiplier: float = 10.0,
        dedup_store: Optional[InMemoryDedupStore] = None,
        entity_linker: Optional[EntityLinker] = None,
        on_document: Optional[Callable[[RawDocument], Any]] = None,
    ) -> None:
        raw_path = csv_path or os.environ.get("KAGGLE_TWEETS_CSV_PATH", "")
        self._csv_path = Path(raw_path) if raw_path else None
        self._speed = max(speed_multiplier, 0.01)  # avoid divide-by-zero
        self._dedup = dedup_store or InMemoryDedupStore()
        self._linker = entity_linker or EntityLinker()
        self._on_document = on_document

    # ── SourceAdapter interface ────────────────────────────────────────────────

    async def fetch(self) -> list[RawDocument]:
        """
        Load the CSV (or generate placeholders) and return all unprocessed tweets
        as RawDocuments without any timing delays.

        This is the batch-fetch variant.  For time-accurate streaming replay,
        use ``run_forever()``.
        """
        rows = self._load_rows()
        docs: list[RawDocument] = []
        for row in rows:
            doc = self._row_to_document(row)
            if doc is not None:
                docs.append(doc)
        logger.info(
            "KaggleTweetReplayAdapter.fetch() returned %d documents", len(docs)
        )
        return docs

    async def run_forever(self) -> None:
        """
        Replay tweets in chronological order with timing proportional to
        their original timestamps divided by ``speed_multiplier``.

        After completing one pass through the CSV, waits indefinitely until
        restarted (or the task is cancelled).
        """
        logger.info(
            "KaggleTweetReplayAdapter starting replay (speed=%.1f×)", self._speed
        )
        rows = self._load_rows()

        if not rows:
            logger.warning(
                "KaggleTweetReplayAdapter: no rows to replay; sleeping forever"
            )
            await asyncio.sleep(float("inf"))
            return

        # Sort chronologically so the replay mirrors real arrival order.
        rows = sorted(rows, key=lambda r: r.get("_parsed_dt") or datetime.min)

        prev_dt: Optional[datetime] = None

        for row in rows:
            doc = self._row_to_document(row)
            if doc is None:
                continue

            # Simulate the inter-tweet delay scaled by speed_multiplier
            current_dt: datetime = doc.published_at
            if prev_dt is not None and current_dt > prev_dt:
                delta_seconds = (current_dt - prev_dt).total_seconds()
                sleep_seconds = delta_seconds / self._speed
                # Cap single sleep at 30 s to keep the loop responsive
                sleep_seconds = min(sleep_seconds, 30.0)
                if sleep_seconds > 0.001:
                    await asyncio.sleep(sleep_seconds)
            prev_dt = current_dt

            if self._on_document is not None:
                try:
                    await self._on_document(doc)
                except Exception as exc:
                    logger.warning(
                        "KaggleTweetReplayAdapter: on_document callback error: %s", exc
                    )

        logger.info("KaggleTweetReplayAdapter: replay complete; waiting for restart")
        # Wait indefinitely — run_forever is expected to be cancelled from outside.
        try:
            await asyncio.sleep(float("inf"))
        except asyncio.CancelledError:
            pass

    # ── Row loading ────────────────────────────────────────────────────────────

    def _load_rows(self) -> list[dict[str, Any]]:
        """
        Load rows from the CSV file.  Returns placeholder rows if the file
        is missing or unreadable.
        """
        if self._csv_path is None or not self._csv_path.exists():
            if self._csv_path is not None:
                logger.warning(
                    "KaggleTweetReplayAdapter: CSV not found at %s — "
                    "generating placeholder tweets for demo",
                    self._csv_path,
                )
            else:
                logger.warning(
                    "KaggleTweetReplayAdapter: no CSV path configured — "
                    "generating placeholder tweets for demo"
                )
            return _generate_placeholder_rows()

        try:
            rows: list[dict[str, Any]] = []
            with self._csv_path.open(newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for raw_row in reader:
                    row = {k.lower().strip(): v for k, v in raw_row.items()}
                    row["_parsed_dt"] = _parse_tweet_date(
                        _pick(row, _DATE_COLS, "")
                    )
                    rows.append(row)
            logger.info(
                "KaggleTweetReplayAdapter: loaded %d rows from %s",
                len(rows),
                self._csv_path,
            )
            return rows
        except Exception as exc:
            logger.error(
                "KaggleTweetReplayAdapter: failed to read CSV %s: %s",
                self._csv_path,
                exc,
            )
            return _generate_placeholder_rows()

    def _row_to_document(self, row: dict[str, Any]) -> Optional[RawDocument]:
        """Convert a CSV row dict to a RawDocument, or None if it should be skipped."""
        tweet_id = str(_pick(row, _ID_COLS, "") or "").strip()
        text = str(_pick(row, _TEXT_COLS, "") or "").strip()
        ticker_hint = str(_pick(row, _TICKER_COLS, "") or "").strip().upper()
        published_at: datetime = row.get("_parsed_dt") or datetime.now(tz=timezone.utc)

        # Ensure timezone-aware
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)

        if not text:
            return None

        # Stable article_id: prefer tweet_id, fall back to content hash prefix
        content_hash = compute_content_hash(text)
        if not tweet_id:
            tweet_id = f"tweet:{content_hash[:16]}"
        article_id = f"tweet:{tweet_id}"

        if self._dedup.is_duplicate(article_id, content_hash):
            return None
        self._dedup.mark_seen(article_id, content_hash)

        # Entity link
        linked = self._linker.link(text)
        ticker_symbols: list[str] = []
        if ticker_hint and ticker_hint in UNIVERSE:
            ticker_symbols.append(ticker_hint)
        for t in linked:
            if t not in ticker_symbols:
                ticker_symbols.append(t)

        if not ticker_symbols:
            # Can't link to any known ticker — skip to avoid noise
            return None

        return RawDocument(
            article_id=article_id,
            source="social",
            ticker_symbols=ticker_symbols,
            title=text[:280],  # tweets are max 280 chars
            body=text,
            url=f"https://twitter.com/i/web/status/{tweet_id.replace('tweet:', '')}",
            published_at=published_at,
            raw_metadata={
                "tweet_id": tweet_id,
                "source_file": str(self._csv_path) if self._csv_path else "placeholder",
            },
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pick(row: dict[str, Any], candidates: tuple[str, ...], default: Any) -> Any:
    """Return the first value in ``row`` whose key matches a candidate name."""
    for key in candidates:
        if key in row and row[key] is not None and str(row[key]).strip():
            return row[key]
    return default


def _parse_tweet_date(date_str: str) -> datetime:
    """
    Parse a tweet timestamp string to a timezone-aware datetime.

    Tries common Twitter/CSV date formats in order.
    """
    clean = str(date_str).strip()
    formats = [
        "%a %b %d %H:%M:%S %z %Y",   # Twitter API: "Mon Jan 15 12:34:56 +0000 2024"
        "%Y-%m-%dT%H:%M:%S%z",        # ISO 8601 with tz
        "%Y-%m-%dT%H:%M:%SZ",         # ISO 8601 UTC
        "%Y-%m-%dT%H:%M:%S",          # ISO 8601 naive
        "%Y-%m-%d %H:%M:%S",          # common CSV format
        "%Y-%m-%d",                    # date only
        "%m/%d/%Y %H:%M:%S",          # US format
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(clean, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return datetime.now(tz=timezone.utc)


_PLACEHOLDER_TEMPLATES = [
    "{ticker} earnings beat expectations! Strong guidance ahead.",
    "Breaking: {ticker} acquires startup — shares up in after-hours trading.",
    "${{ticker}} CEO makes surprise announcement at investor day.",
    "Watching {ticker} closely — unusual options activity detected.",
    "{ticker} Q4 results coming tomorrow. Will it beat the street?",
    "Sold my {ticker} position. Taking profits after this run.",
    "{ticker} regulatory filing reveals new product roadmap.",
    "Short squeeze incoming on {ticker}? Volume spiking.",
]


def _generate_placeholder_rows() -> list[dict[str, Any]]:
    """
    Generate a small set of synthetic tweet rows for demo use when the
    real CSV is not available.
    """
    import hashlib
    from datetime import timedelta

    rows: list[dict[str, Any]] = []
    base_dt = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)

    for i, ticker in enumerate(UNIVERSE):
        for j, template in enumerate(_PLACEHOLDER_TEMPLATES[:4]):
            tweet_text = template.format(ticker=ticker)
            tweet_id = hashlib.sha256(f"{ticker}:{j}".encode()).hexdigest()[:16]
            dt = base_dt + timedelta(minutes=(i * 4 + j) * 2)
            rows.append(
                {
                    "tweet_id": tweet_id,
                    "text": tweet_text,
                    "created_at": dt.isoformat(),
                    "ticker": ticker,
                    "_parsed_dt": dt,
                }
            )

    logger.info(
        "KaggleTweetReplayAdapter: generated %d placeholder tweets", len(rows)
    )
    return rows
