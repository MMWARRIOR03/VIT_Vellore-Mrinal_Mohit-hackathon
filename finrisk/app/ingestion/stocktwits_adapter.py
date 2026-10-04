"""
StockTwits stream adapter.

Polls the public StockTwits symbol stream endpoint for each ticker in the
20-stock universe.  No API key is required for the public stream endpoint.

Reference:
  GET https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json

The endpoint returns up to 30 messages per call.  We poll every 5 minutes
with 0.5 s between ticker requests to stay well within StockTwits' rate limits.

Sentiment hints
───────────────
StockTwits users can tag their messages as Bullish or Bearish via the
sentiment widget.  When present in ``entities.sentiment.basic``, this is
captured in ``raw_metadata["sentiment_hint"]`` for the NLP layer to use as
a prior.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

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

_STOCKTWITS_BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_INTER_REQUEST_SLEEP = 0.5  # seconds between ticker requests

# StockTwits rate-limit error code
_RATE_LIMIT_STATUS = 429
# Back-off time when rate-limited (seconds)
_RATE_LIMIT_BACKOFF = 60.0


class StockTwitsAdapter(SourceAdapter):
    """
    Fetches recent StockTwits messages for all 20 tickers.

    Each poll retrieves up to 30 messages per ticker from the public stream.
    Messages are deduplicated by their StockTwits message ID.

    Parameters
    ──────────
    dedup_store     Shared InMemoryDedupStore.
    entity_linker   Shared EntityLinker.
    http_client     httpx.AsyncClient to use.
    on_document     Async callback invoked per new RawDocument.
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
        Poll StockTwits for all 20 tickers and return deduplicated RawDocuments.
        """
        client = await self._get_client()
        docs: list[RawDocument] = []

        for i, ticker in enumerate(UNIVERSE):
            if i > 0:
                await asyncio.sleep(_INTER_REQUEST_SLEEP)
            try:
                batch = await self._fetch_ticker(client, ticker)
                docs.extend(batch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "StockTwitsAdapter: error fetching ticker %s: %s", ticker, exc
                )

        logger.info("StockTwitsAdapter.fetch() returned %d new messages", len(docs))
        return docs

    async def run_forever(self, interval_seconds: float = 300.0) -> None:
        """
        Poll StockTwits every ``interval_seconds`` (default 5 minutes).

        Backs off for 60 s when rate-limited.
        """
        logger.info(
            "StockTwitsAdapter.run_forever() starting, interval=%.0fs",
            interval_seconds,
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
                    logger.error("StockTwitsAdapter poll cycle error: %s", exc)
                await asyncio.sleep(interval_seconds)
        finally:
            if self._owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=20.0,
                headers={"Accept": "application/json"},
                follow_redirects=True,
            )
        return self._client

    async def _fetch_ticker(
        self, client: httpx.AsyncClient, ticker: str
    ) -> list[RawDocument]:
        """Fetch the latest messages for one StockTwits symbol stream."""
        url = _STOCKTWITS_BASE_URL.format(ticker=ticker)

        try:
            response = await client.get(url)
            if response.status_code == _RATE_LIMIT_STATUS:
                logger.warning(
                    "StockTwitsAdapter: rate-limited, backing off %.0fs", _RATE_LIMIT_BACKOFF
                )
                await asyncio.sleep(_RATE_LIMIT_BACKOFF)
                return []
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "StockTwitsAdapter HTTP %s for ticker %s: %s",
                exc.response.status_code,
                ticker,
                exc,
            )
            return []
        except Exception as exc:
            logger.warning(
                "StockTwitsAdapter request error for ticker %s: %s", ticker, exc
            )
            return []

        return self._parse_response(data, ticker)

    def _parse_response(
        self, data: dict[str, Any], hint_ticker: str
    ) -> list[RawDocument]:
        """
        Parse a StockTwits symbol stream JSON response.

        Response structure::

            {
              "symbol": { "symbol": "AAPL", ... },
              "messages": [
                {
                  "id": 123456789,
                  "body": "AAPL looking bullish here",
                  "created_at": "2024-01-15T12:34:56Z",
                  "user": { "id": 1, "username": "trader_joe", ... },
                  "entities": {
                    "sentiment": { "basic": "Bullish" }
                  }
                },
                ...
              ]
            }
        """
        messages: list[dict[str, Any]] = data.get("messages") or []
        docs: list[RawDocument] = []

        for msg in messages:
            msg_id = msg.get("id")
            body: str = (msg.get("body") or "").strip()
            created_at_str: str = msg.get("created_at") or ""
            username: str = (msg.get("user") or {}).get("username") or "unknown"

            if not body:
                continue

            # Stable ID from StockTwits message ID
            article_id = f"st:{msg_id}" if msg_id else f"st:{compute_content_hash(body)[:16]}"
            content_hash = compute_content_hash(body)

            if self._dedup.is_duplicate(article_id, content_hash):
                continue
            self._dedup.mark_seen(article_id, content_hash)

            # Parse sentiment hint
            entities: dict[str, Any] = msg.get("entities") or {}
            sentiment_raw: str = (entities.get("sentiment") or {}).get("basic") or ""
            sentiment_hint: Optional[str] = None
            if sentiment_raw.lower() == "bullish":
                sentiment_hint = "positive"
            elif sentiment_raw.lower() == "bearish":
                sentiment_hint = "negative"

            # Parse timestamp
            published_at = _parse_stocktwits_date(created_at_str)

            # Entity link body text
            linked = self._linker.link(body)
            ticker_symbols = list(dict.fromkeys([hint_ticker] + linked))

            url = (
                f"https://stocktwits.com/{username}/message/{msg_id}"
                if msg_id
                else f"https://stocktwits.com/symbol/{hint_ticker}"
            )

            docs.append(
                RawDocument(
                    article_id=article_id,
                    source="social",
                    ticker_symbols=ticker_symbols,
                    title=body[:280],
                    body=body,
                    url=url,
                    published_at=published_at,
                    raw_metadata={
                        "stocktwits_id": msg_id,
                        "username": username,
                        "sentiment_hint": sentiment_hint,
                        "sentiment_raw": sentiment_raw or None,
                        "query_ticker": hint_ticker,
                    },
                )
            )

        return docs


# ── Date parsing ──────────────────────────────────────────────────────────────

def _parse_stocktwits_date(date_str: str) -> datetime:
    """
    Parse a StockTwits ``created_at`` ISO 8601 string to a UTC datetime.
    """
    clean = str(date_str).strip()
    formats = [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
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
