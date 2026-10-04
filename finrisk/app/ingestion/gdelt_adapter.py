"""
GDELT DOC API adapter.

Polls the GDELT v2 Document API for news articles matching each ticker in
the 20-stock universe.  One request per ticker, 1.2 s sleep between requests
to stay within GDELT's soft rate limit.  Full poll cycle runs every 15 minutes
by default.

GDELT DOC API reference:
  https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

from app.ingestion.base import RawDocument, SourceAdapter
from app.ingestion.dedup import InMemoryDedupStore, compute_content_hash
from app.ingestion.entity_linker import EntityLinker

logger = logging.getLogger(__name__)

# The 20-stock universe
UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "BAC", "GS",
    "V", "JNJ", "UNH", "PFE", "XOM",
    "CVX", "CAT", "BA", "WMT", "HD",
]

_GDELT_BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_MAX_RECORDS = 25
_TIMESPAN = "15min"
_INTER_REQUEST_SLEEP = 1.2  # seconds between ticker requests


class GDELTAdapter(SourceAdapter):
    """
    Fetches recent news articles from the GDELT DOC API for all 20 tickers.

    Each poll cycle queries each ticker individually.  GDELT returns up to
    ``maxrecords`` articles from the last ``timespan`` minutes.  Articles are
    deduplicated by SHA-256 of the normalised title so re-fetching the same
    headline within the TTL window is a no-op.

    Parameters
    ──────────
    dedup_store     Shared InMemoryDedupStore.  If None, a private one is
                    created (useful for testing).
    entity_linker   Shared EntityLinker.  If None, a private one is created.
    http_client     httpx.AsyncClient to use.  If None, a new one is created
                    and managed internally.
    on_document     Async callback invoked for each new RawDocument.  Used by
                    the pipeline to process documents as they arrive rather than
                    waiting for the full batch.
    """

    def __init__(
        self,
        dedup_store: Optional[InMemoryDedupStore] = None,
        entity_linker: Optional[EntityLinker] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        on_document: Optional[Callable[[RawDocument], Any]] = None,
    ) -> None:
        self._dedup = dedup_store or InMemoryDedupStore()
        self._linker = entity_linker or EntityLinker()
        self._client = http_client
        self._owns_client = http_client is None
        self._on_document = on_document

    # ── SourceAdapter interface ────────────────────────────────────────────────

    async def fetch(self) -> list[RawDocument]:
        """
        Query GDELT for all 20 tickers and return deduplicated RawDocuments.

        Sleeps 1.2 s between ticker requests to respect the rate limit.
        """
        client = await self._get_client()
        docs: list[RawDocument] = []

        for i, ticker in enumerate(UNIVERSE):
            if i > 0:
                await asyncio.sleep(_INTER_REQUEST_SLEEP)
            try:
                batch = await self._fetch_ticker(client, ticker)
                docs.extend(batch)
            except Exception as exc:
                logger.warning(
                    "GDELTAdapter: error fetching ticker %s: %s", ticker, exc
                )

        logger.info("GDELTAdapter.fetch() returned %d new documents", len(docs))
        return docs

    async def run_forever(self, interval_seconds: float = 900.0) -> None:
        """
        Poll GDELT every ``interval_seconds`` seconds (default 15 minutes).

        Runs until cancelled.  Transient errors are logged and the loop
        continues with the next scheduled cycle.
        """
        logger.info(
            "GDELTAdapter.run_forever() starting, interval=%.0fs", interval_seconds
        )
        try:
            while True:
                try:
                    docs = await self.fetch()
                    if self._on_document is not None:
                        for doc in docs:
                            await self._on_document(doc)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("GDELTAdapter poll cycle error: %s", exc)
                await asyncio.sleep(interval_seconds)
        finally:
            if self._owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _fetch_ticker(
        self, client: httpx.AsyncClient, ticker: str
    ) -> list[RawDocument]:
        """Fetch articles for one ticker symbol and return new RawDocuments."""
        params = {
            "query": ticker,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(_MAX_RECORDS),
            "timespan": _TIMESPAN,
        }

        try:
            response = await client.get(_GDELT_BASE_URL, params=params)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "GDELTAdapter HTTP %s for ticker %s: %s",
                exc.response.status_code,
                ticker,
                exc,
            )
            return []
        except Exception as exc:
            logger.warning(
                "GDELTAdapter request error for ticker %s: %s", ticker, exc
            )
            return []

        return self._parse_response(data, ticker)

    def _parse_response(
        self, data: dict[str, Any], hint_ticker: str
    ) -> list[RawDocument]:
        """
        Parse a GDELT artlist JSON response into RawDocuments.

        GDELT artlist format::

            {
              "articles": [
                {
                  "url": "...",
                  "title": "...",
                  "seendate": "20240115T123456Z",
                  "domain": "reuters.com",
                  "socialimage": "..."
                },
                ...
              ]
            }
        """
        articles = data.get("articles") or []
        docs: list[RawDocument] = []

        for article in articles:
            title: str = (article.get("title") or "").strip()
            url: str = (article.get("url") or "").strip()
            seendate: str = article.get("seendate") or ""
            domain: str = article.get("domain") or ""
            socialimage: str = article.get("socialimage") or ""

            if not title:
                continue

            # Dedup by SHA-256 of normalised title (GDELT has no stable article ID)
            content_hash = compute_content_hash(title)
            article_id = f"gdelt:{hashlib.sha256(url.encode()).hexdigest()[:16]}" if url else f"gdelt:{content_hash[:16]}"

            if self._dedup.is_duplicate(article_id, content_hash):
                continue
            self._dedup.mark_seen(article_id, content_hash)

            # Parse GDELT date format: "20240115T123456Z" or "20240115T123456"
            published_at = _parse_gdelt_date(seendate)

            # Entity-link the title to find all relevant tickers
            linked = self._linker.link(title)
            # Ensure the hint ticker is always present
            ticker_symbols = list(dict.fromkeys([hint_ticker] + linked))

            docs.append(
                RawDocument(
                    article_id=article_id,
                    source="news",
                    ticker_symbols=ticker_symbols,
                    title=title,
                    body="",  # GDELT artlist doesn't return body text
                    url=url,
                    published_at=published_at,
                    raw_metadata={
                        "domain": domain,
                        "socialimage": socialimage,
                        "gdelt_seendate": seendate,
                        "query_ticker": hint_ticker,
                    },
                )
            )

        return docs


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_gdelt_date(seendate: str) -> datetime:
    """
    Parse a GDELT seendate string to a timezone-aware UTC datetime.

    GDELT uses "YYYYMMDDTHHmmssZ" (sometimes without the trailing Z).
    Falls back to UTC now if parsing fails.
    """
    # Strip trailing Z for strptime compatibility
    clean = seendate.rstrip("Z").strip()
    formats = [
        "%Y%m%dT%H%M%S",
        "%Y%m%d%H%M%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            naive = datetime.strptime(clean, fmt)
            return naive.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    logger.debug("GDELTAdapter: could not parse seendate %r, using now()", seendate)
    return datetime.now(tz=timezone.utc)
