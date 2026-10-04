"""
SEC EDGAR full-text search adapter (8-K filings).

Polls the EDGAR EFTS (full-text search) endpoint for 8-K filings matching
each ticker.  The SEC requires a descriptive User-Agent with a contact email;
requests without it return 403.

8-K item codes are parsed from the filing metadata to produce event_type_hint
values that the NLP engine can use as priors during classification.

EDGAR EFTS reference:
  https://efts.sec.gov/LATEST/search-index?q="AAPL"&forms=8-K&dateRange=custom&startdt=2024-01-15

Rate limiting
─────────────
The SEC asks for ≤ 10 requests/second.  We use 0.2 s between requests (5/s)
to stay comfortably within that limit.  The poll cycle runs every 30 minutes.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import date, datetime, timezone
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

_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_SEC_USER_AGENT = "FinRisk research@finrisk.example.com"
_INTER_REQUEST_SLEEP = 0.2  # seconds; keeps well under SEC 10 req/s limit

# Map 8-K item codes to event type hints used by the NLP scoring layer.
_ITEM_CODE_MAP: dict[str, str] = {
    "1.01": "M&A",
    "1.03": "Credit",
    "2.01": "M&A",
    "4.02": "Credit",
    "5.02": "Leadership",
    "7.01": "Earnings",
    "8.01": "Product",
}

# Regex to find item codes in filing descriptions and document names.
# Matches patterns like "Item 1.01", "item 1.01", "1.01 Entry"
_ITEM_CODE_RE = re.compile(r"\bitem\s+(\d+\.\d+)\b", re.IGNORECASE)


class EDGARAdapter(SourceAdapter):
    """
    Fetches today's 8-K filings from SEC EDGAR for all 20 tickers.

    Each RawDocument has ``event_type_hint`` in ``raw_metadata`` derived from
    the 8-K item codes found in the filing description.  If no item code is
    recognisable, event_type_hint is ``"Unknown"``.

    Parameters
    ──────────
    dedup_store     Shared InMemoryDedupStore.
    entity_linker   Shared EntityLinker.
    http_client     httpx.AsyncClient to use (managed externally).
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
        Query EDGAR for today's 8-K filings for all 20 tickers.

        Returns a deduplicated list of RawDocuments with event_type_hint.
        """
        client = await self._get_client()
        docs: list[RawDocument] = []
        today = date.today().isoformat()  # "2024-01-15"

        for i, ticker in enumerate(UNIVERSE):
            if i > 0:
                await asyncio.sleep(_INTER_REQUEST_SLEEP)
            try:
                batch = await self._fetch_ticker(client, ticker, today)
                docs.extend(batch)
            except Exception as exc:
                logger.warning(
                    "EDGARAdapter: error fetching ticker %s: %s", ticker, exc
                )

        logger.info("EDGARAdapter.fetch() returned %d new filings", len(docs))
        return docs

    async def run_forever(self, interval_seconds: float = 1800.0) -> None:
        """
        Poll EDGAR every ``interval_seconds`` (default 30 minutes).

        Runs until cancelled.
        """
        logger.info(
            "EDGARAdapter.run_forever() starting, interval=%.0fs", interval_seconds
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
                    logger.error("EDGARAdapter poll cycle error: %s", exc)
                await asyncio.sleep(interval_seconds)
        finally:
            if self._owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": _SEC_USER_AGENT},
                timeout=30.0,
            )
        return self._client

    async def _fetch_ticker(
        self, client: httpx.AsyncClient, ticker: str, today: str
    ) -> list[RawDocument]:
        """Fetch 8-K filings for one ticker filed on or after ``today``."""
        params = {
            "q": f'"{ticker}"',
            "forms": "8-K",
            "dateRange": "custom",
            "startdt": today,
        }

        try:
            response = await client.get(
                _EDGAR_SEARCH_URL,
                params=params,
                headers={"User-Agent": _SEC_USER_AGENT},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "EDGARAdapter HTTP %s for ticker %s: %s",
                exc.response.status_code,
                ticker,
                exc,
            )
            return []
        except Exception as exc:
            logger.warning(
                "EDGARAdapter request error for ticker %s: %s", ticker, exc
            )
            return []

        return self._parse_response(data, ticker)

    def _parse_response(
        self, data: dict[str, Any], hint_ticker: str
    ) -> list[RawDocument]:
        """
        Parse an EDGAR EFTS search-index response.

        The EFTS response structure::

            {
              "hits": {
                "hits": [
                  {
                    "_id": "...",
                    "_source": {
                      "file_date": "2024-01-15",
                      "period_of_report": "...",
                      "entity_name": "Apple Inc.",
                      "file_num": "...",
                      "form_type": "8-K",
                      "biz_location": "...",
                      "description": "Item 1.01 ...",
                      "accession_no": "0001234567-24-000001",
                      "display_date_filed": "January 15, 2024"
                    }
                  }
                ],
                "total": { "value": 3 }
              }
            }
        """
        hits = (data.get("hits") or {}).get("hits") or []
        docs: list[RawDocument] = []

        for hit in hits:
            source: dict[str, Any] = hit.get("_source") or {}
            hit_id: str = hit.get("_id") or ""

            accession_no: str = source.get("accession_no") or hit_id or ""
            entity_name: str = source.get("entity_name") or ""
            description: str = source.get("description") or ""
            file_date: str = source.get("file_date") or ""
            form_type: str = source.get("form_type") or "8-K"

            # Build a stable article_id from the accession number
            article_id = f"edgar:{accession_no}" if accession_no else f"edgar:{hint_ticker}:{file_date}"

            # Headline: entity name + form type + description snippet
            title = f"{entity_name} {form_type}: {description[:120]}".strip()
            if not title:
                title = f"{hint_ticker} {form_type} filing {file_date}"

            content_hash = compute_content_hash(accession_no or title)

            if self._dedup.is_duplicate(article_id, content_hash):
                continue
            self._dedup.mark_seen(article_id, content_hash)

            # Derive event_type_hint from item codes in the description
            event_type_hint = _extract_event_type_hint(description)

            published_at = _parse_edgar_date(file_date)

            # EDGAR URL to the filing index page
            if accession_no:
                cik = source.get("cik") or ""
                acc_clean = accession_no.replace("-", "")
                url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{accession_no}-index.htm"
                    if cik
                    else f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K&dateb=&owner=include&count=10&search_text="
                )
            else:
                url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K"

            # Entity-link entity name to find tickers
            linked = self._linker.link(entity_name + " " + description)
            ticker_symbols = list(dict.fromkeys([hint_ticker] + linked))

            docs.append(
                RawDocument(
                    article_id=article_id,
                    source="filing",
                    ticker_symbols=ticker_symbols,
                    title=title,
                    body=description,
                    url=url,
                    published_at=published_at,
                    raw_metadata={
                        "event_type_hint": event_type_hint,
                        "accession_no": accession_no,
                        "entity_name": entity_name,
                        "form_type": form_type,
                        "file_date": file_date,
                        "query_ticker": hint_ticker,
                    },
                )
            )

        return docs


# ── Item code helpers ──────────────────────────────────────────────────────────

def _extract_event_type_hint(description: str) -> str:
    """
    Extract the first recognised 8-K item code from ``description`` and
    return the corresponding event type hint string.  Returns "Unknown" if
    no recognised item code is found.
    """
    for match in _ITEM_CODE_RE.finditer(description):
        code = match.group(1)
        if code in _ITEM_CODE_MAP:
            return _ITEM_CODE_MAP[code]
    return "Unknown"


def _parse_edgar_date(date_str: str) -> datetime:
    """
    Parse an EDGAR date string (``"2024-01-15"`` or ``"20240115"``) to UTC.
    """
    clean = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(clean, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.debug("EDGARAdapter: could not parse date %r, using now()", date_str)
    return datetime.now(tz=timezone.utc)
