"""
Tests for the ingestion layer (Task 2).

Covers:
- InMemoryDedupStore: duplicate detection, TTL eviction, max-size eviction
- EntityLinker: company name → ticker, cashtag → ticker, multi-ticker text
- RawDocument: construction, field defaults, normalisation
- GDELTAdapter.fetch() with mocked httpx
- EDGARAdapter: item code → event_type_hint, fetch() with mocked httpx
- StockTwitsAdapter.fetch() with mocked httpx
- KaggleTweetReplayAdapter: missing-file graceful fallback, CSV parsing
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.ingestion.base import RawDocument, SOURCE_WEIGHT_DEFAULTS
from app.ingestion.dedup import (
    InMemoryDedupStore,
    compute_content_hash,
    normalize_text,
)
from app.ingestion.edgar_adapter import (
    EDGARAdapter,
    _extract_event_type_hint,
    _parse_edgar_date,
)
from app.ingestion.entity_linker import EntityLinker
from app.ingestion.gdelt_adapter import GDELTAdapter, _parse_gdelt_date
from app.ingestion.kaggle_tweet_replay import KaggleTweetReplayAdapter
from app.ingestion.stocktwits_adapter import StockTwitsAdapter


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def dedup() -> InMemoryDedupStore:
    return InMemoryDedupStore(ttl_seconds=60.0, max_entries=10)


@pytest.fixture()
def linker() -> EntityLinker:
    """Real EntityLinker backed by the actual configs/ticker_map.yaml."""
    return EntityLinker()


# ── normalize_text / compute_content_hash ─────────────────────────────────────

class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("HELLO WORLD") == "hello world"

    def test_strips_punctuation(self):
        assert normalize_text("Hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalize_text("  foo   bar  ") == "foo bar"

    def test_full_pipeline(self):
        a = normalize_text("Apple Inc. reported Q4 earnings!")
        b = normalize_text("apple inc reported q4 earnings")
        assert a == b

    def test_empty_string(self):
        assert normalize_text("") == ""


class TestComputeContentHash:
    def test_same_text_same_hash(self):
        assert compute_content_hash("Apple") == compute_content_hash("Apple")

    def test_normalised_equivalence(self):
        # Punctuation and case differences should produce the same hash.
        assert compute_content_hash("Apple Inc.!") == compute_content_hash("apple inc")

    def test_different_text_different_hash(self):
        assert compute_content_hash("Apple") != compute_content_hash("Microsoft")

    def test_returns_hex_string(self):
        h = compute_content_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex digest
        int(h, 16)  # must be valid hex


# ── InMemoryDedupStore ────────────────────────────────────────────────────────

class TestInMemoryDedupStore:
    def test_new_item_is_not_duplicate(self, dedup):
        assert not dedup.is_duplicate("id-1", "hash-abc")

    def test_marked_item_is_duplicate(self, dedup):
        dedup.mark_seen("id-1", "hash-abc")
        assert dedup.is_duplicate("id-1", "hash-abc")

    def test_same_hash_different_id_is_duplicate(self, dedup):
        dedup.mark_seen("id-1", "hash-abc")
        # Different article_id but same content hash → still a duplicate
        assert dedup.is_duplicate("id-99", "hash-abc")

    def test_same_id_different_hash_is_duplicate(self, dedup):
        dedup.mark_seen("id-1", "hash-abc")
        assert dedup.is_duplicate("id-1", "hash-xyz")

    def test_different_id_and_hash_not_duplicate(self, dedup):
        dedup.mark_seen("id-1", "hash-abc")
        assert not dedup.is_duplicate("id-2", "hash-xyz")

    def test_ttl_eviction(self):
        """Items older than TTL are no longer considered duplicates."""
        store = InMemoryDedupStore(ttl_seconds=0.05, max_entries=1000)
        store.mark_seen("id-1", "hash-abc")
        assert store.is_duplicate("id-1", "hash-abc")

        time.sleep(0.1)  # exceed TTL

        # After TTL, the item should be treated as new
        assert not store.is_duplicate("id-1", "hash-abc")

    def test_max_entries_eviction(self):
        """Store never exceeds max_entries raw keys; oldest entries are evicted."""
        # max_entries=10 means 10 internal keys.  Each mark_seen() inserts 2 keys
        # (id: + hash:).  With 15 calls and batch-eviction of 10% we expect the
        # store to stay bounded well below 30 (15×2) raw keys.
        store = InMemoryDedupStore(ttl_seconds=3600.0, max_entries=10)
        for i in range(15):
            store.mark_seen(f"id-{i}", f"hash-{i}")
        # After 15 insertions the eviction should have fired at least once,
        # keeping the store materially below 30 keys (2 per mark_seen call).
        assert store.size() < 30
        # And the cap itself should have been respected (never more than
        # max_entries + one batch = 10 + 10*0.1*2 ≈ 12 at peak)
        assert store.size() <= 20

    def test_size_tracking(self, dedup):
        assert dedup.size() == 0
        dedup.mark_seen("id-1", "hash-1")
        # Two keys inserted (id: and hash:)
        assert dedup.size() == 2


# ── EntityLinker ──────────────────────────────────────────────────────────────

class TestEntityLinker:
    def test_apple_announced(self, linker):
        result = linker.link("Apple announced a new product today")
        assert "AAPL" in result

    def test_nvda_cashtag(self, linker):
        result = linker.link("$NVDA chip export restrictions tighten")
        assert "NVDA" in result

    def test_nvda_bare_ticker(self, linker):
        result = linker.link("NVDA is up 3% today")
        assert "NVDA" in result

    def test_jpmorgan_and_goldman(self, linker):
        result = linker.link("JPMorgan and Goldman Sachs report strong Q4 results")
        assert "JPM" in result
        assert "GS" in result

    def test_case_insensitive(self, linker):
        result = linker.link("apple inc is trading higher")
        assert "AAPL" in result

    def test_multiple_tickers(self, linker):
        result = linker.link("Microsoft and Amazon compete in cloud")
        assert "MSFT" in result
        assert "AMZN" in result

    def test_no_match_returns_empty(self, linker):
        result = linker.link("The weather is nice today")
        assert result == []

    def test_empty_text(self, linker):
        assert linker.link("") == []

    def test_deduplicates_same_ticker(self, linker):
        # "Apple" and "AAPL" both refer to the same ticker
        result = linker.link("Apple (AAPL) hit a new all-time high")
        assert result.count("AAPL") == 1

    def test_facebook_maps_to_meta(self, linker):
        result = linker.link("Facebook parent Meta posts record revenue")
        assert "META" in result

    def test_visa_ticker(self, linker):
        result = linker.link("Visa expands contactless payments globally")
        assert "V" in result

    def test_cashtag_with_dollar_sign(self, linker):
        result = linker.link("$JPM earnings call at 9 AM")
        assert "JPM" in result


# ── RawDocument ───────────────────────────────────────────────────────────────

class TestRawDocument:
    def test_construction_all_fields(self):
        doc = RawDocument(
            article_id="test-001",
            source="news",
            ticker_symbols=["AAPL", "MSFT"],
            title="Apple and Microsoft announce partnership",
            body="Full story here...",
            url="https://example.com/article",
            published_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            source_weight=0.85,
            raw_metadata={"domain": "reuters.com"},
        )
        assert doc.article_id == "test-001"
        assert doc.source == "news"
        assert "AAPL" in doc.ticker_symbols
        assert "MSFT" in doc.ticker_symbols
        assert doc.source_weight == 0.85
        assert doc.raw_metadata["domain"] == "reuters.com"

    def test_default_source_weight_news(self):
        doc = RawDocument(
            article_id="x",
            source="news",
            ticker_symbols=[],
            title="t",
            body="b",
            url="http://x.com",
            published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert doc.source_weight == SOURCE_WEIGHT_DEFAULTS["news"]

    def test_default_source_weight_social(self):
        doc = RawDocument(
            article_id="x",
            source="social",
            ticker_symbols=[],
            title="t",
            body="b",
            url="http://x.com",
            published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert doc.source_weight == SOURCE_WEIGHT_DEFAULTS["social"]

    def test_default_source_weight_filing(self):
        doc = RawDocument(
            article_id="x",
            source="filing",
            ticker_symbols=[],
            title="t",
            body="b",
            url="http://x.com",
            published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert doc.source_weight == SOURCE_WEIGHT_DEFAULTS["filing"]

    def test_ticker_symbols_uppercased(self):
        doc = RawDocument(
            article_id="x",
            source="news",
            ticker_symbols=["aapl", "msft", "AAPL"],  # lowercase + duplicate
            title="t",
            body="b",
            url="http://x.com",
            published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert doc.ticker_symbols == ["AAPL", "MSFT"]  # deduped, uppercased

    def test_raw_metadata_defaults_to_empty_dict(self):
        doc = RawDocument(
            article_id="x",
            source="news",
            ticker_symbols=[],
            title="t",
            body="b",
            url="http://x.com",
            published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        assert doc.raw_metadata == {}

    def test_source_weight_defaults_to_filing(self):
        assert SOURCE_WEIGHT_DEFAULTS["filing"] == 1.0

    def test_source_weight_defaults_to_social(self):
        assert SOURCE_WEIGHT_DEFAULTS["social"] == 0.40


# ── GDELTAdapter ──────────────────────────────────────────────────────────────

class TestGDELTAdapter:
    """Tests GDELTAdapter with a fully mocked httpx.AsyncClient."""

    def _make_gdelt_response(self, ticker: str) -> dict[str, Any]:
        return {
            "articles": [
                {
                    "url": f"https://reuters.com/{ticker.lower()}-story",
                    "title": f"{ticker} stock surges after earnings beat",
                    "seendate": "20240115T120000Z",
                    "domain": "reuters.com",
                    "socialimage": "",
                },
                {
                    "url": f"https://bloomberg.com/{ticker.lower()}-news",
                    "title": f"Analysts raise {ticker} price target",
                    "seendate": "20240115T110000Z",
                    "domain": "bloomberg.com",
                    "socialimage": "",
                },
            ]
        }

    @pytest.mark.asyncio
    async def test_fetch_returns_documents(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value=self._make_gdelt_response("AAPL")
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        dedup = InMemoryDedupStore()
        linker = EntityLinker()
        adapter = GDELTAdapter(
            dedup_store=dedup,
            entity_linker=linker,
            http_client=mock_client,
        )

        # Patch UNIVERSE so only AAPL is queried (avoids 20x mock calls)
        with patch("app.ingestion.gdelt_adapter.UNIVERSE", ["AAPL"]):
            docs = await adapter.fetch()

        assert len(docs) == 2
        assert all(d.source == "news" for d in docs)
        assert all("AAPL" in d.ticker_symbols for d in docs)
        assert docs[0].title == "AAPL stock surges after earnings beat"

    @pytest.mark.asyncio
    async def test_fetch_deduplicates_within_call(self):
        """Two articles with the same normalised title should produce one document."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={
                "articles": [
                    {
                        "url": "https://a.com/1",
                        "title": "Apple reports record earnings",
                        "seendate": "20240115T120000Z",
                        "domain": "a.com",
                        "socialimage": "",
                    },
                    {
                        "url": "https://b.com/2",
                        "title": "Apple Reports Record Earnings",  # same, different case
                        "seendate": "20240115T120100Z",
                        "domain": "b.com",
                        "socialimage": "",
                    },
                ]
            }
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = GDELTAdapter(http_client=mock_client)
        with patch("app.ingestion.gdelt_adapter.UNIVERSE", ["AAPL"]):
            docs = await adapter.fetch()

        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_fetch_handles_http_error_gracefully(self):
        """HTTP errors on one ticker should not abort the fetch cycle."""
        import httpx

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "429 Too Many Requests",
                request=MagicMock(),
                response=MagicMock(status_code=429),
            )
        )

        adapter = GDELTAdapter(http_client=mock_client)
        with patch("app.ingestion.gdelt_adapter.UNIVERSE", ["AAPL"]):
            docs = await adapter.fetch()

        assert docs == []

    @pytest.mark.asyncio
    async def test_fetch_handles_empty_articles(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"articles": []})

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = GDELTAdapter(http_client=mock_client)
        with patch("app.ingestion.gdelt_adapter.UNIVERSE", ["AAPL"]):
            docs = await adapter.fetch()

        assert docs == []


class TestGDELTDateParsing:
    def test_standard_format(self):
        dt = _parse_gdelt_date("20240115T123456Z")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 12
        assert dt.tzinfo is not None

    def test_without_trailing_z(self):
        dt = _parse_gdelt_date("20240115T123456")
        assert dt.year == 2024

    def test_invalid_falls_back_to_now(self):
        dt = _parse_gdelt_date("not-a-date")
        assert dt.tzinfo is not None
        # Should be recent (within last minute)
        delta = abs((datetime.now(tz=timezone.utc) - dt).total_seconds())
        assert delta < 60


# ── EDGARAdapter ──────────────────────────────────────────────────────────────

class TestExtractEventTypeHint:
    """Tests the item-code-to-event-type mapping in isolation."""

    def test_item_101_maps_to_ma(self):
        assert _extract_event_type_hint("Item 1.01 Entry into a Material Agreement") == "M&A"

    def test_item_103_maps_to_credit(self):
        assert _extract_event_type_hint("Item 1.03 Bankruptcy") == "Credit"

    def test_item_201_maps_to_ma(self):
        assert _extract_event_type_hint("Item 2.01 Completion of Acquisition") == "M&A"

    def test_item_402_maps_to_credit(self):
        assert _extract_event_type_hint("Item 4.02 Non-Reliance") == "Credit"

    def test_item_502_maps_to_leadership(self):
        assert _extract_event_type_hint("Item 5.02 Departure of Directors") == "Leadership"

    def test_item_701_maps_to_earnings(self):
        assert _extract_event_type_hint("Item 7.01 Regulation FD Disclosure") == "Earnings"

    def test_item_801_maps_to_product(self):
        assert _extract_event_type_hint("Item 8.01 Other Events") == "Product"

    def test_unknown_item_returns_unknown(self):
        assert _extract_event_type_hint("Item 9.99 Future Item") == "Unknown"

    def test_empty_string_returns_unknown(self):
        assert _extract_event_type_hint("") == "Unknown"

    def test_case_insensitive_item(self):
        assert _extract_event_type_hint("item 1.01 entry into agreement") == "M&A"

    def test_first_recognised_code_wins(self):
        # If multiple item codes appear, the first recognised one is returned.
        hint = _extract_event_type_hint("Item 5.02 and Item 7.01 disclosure")
        assert hint == "Leadership"


class TestEDGARAdapter:
    def _make_edgar_response(self, ticker: str) -> dict[str, Any]:
        return {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "0001234567-24-000001",
                        "_source": {
                            "accession_no": "0001234567-24-000001",
                            "entity_name": "Apple Inc.",
                            "form_type": "8-K",
                            "file_date": "2024-01-15",
                            "description": "Item 5.02 Departure of Directors or Certain Officers",
                            "cik": "320193",
                        },
                    }
                ],
            }
        }

    @pytest.mark.asyncio
    async def test_fetch_returns_filing_documents(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value=self._make_edgar_response("AAPL")
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = EDGARAdapter(http_client=mock_client)
        with patch("app.ingestion.edgar_adapter.UNIVERSE", ["AAPL"]):
            docs = await adapter.fetch()

        assert len(docs) == 1
        doc = docs[0]
        assert doc.source == "filing"
        assert "AAPL" in doc.ticker_symbols
        assert doc.raw_metadata["event_type_hint"] == "Leadership"
        assert doc.raw_metadata["form_type"] == "8-K"

    @pytest.mark.asyncio
    async def test_item_code_parsed_to_event_type_hint(self):
        """event_type_hint in raw_metadata reflects the 8-K item code."""
        test_cases = [
            ("Item 1.01 Entry into agreement", "M&A"),
            ("Item 7.01 Regulation FD", "Earnings"),
            ("Item 5.02 Leadership change", "Leadership"),
        ]
        for description, expected_hint in test_cases:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = MagicMock(
                return_value={
                    "hits": {
                        "total": {"value": 1},
                        "hits": [
                            {
                                "_id": f"acc-{expected_hint}",
                                "_source": {
                                    "accession_no": f"acc-{expected_hint}",
                                    "entity_name": "MSFT Inc.",
                                    "form_type": "8-K",
                                    "file_date": "2024-01-15",
                                    "description": description,
                                },
                            }
                        ],
                    }
                }
            )

            mock_client = AsyncMock(spec=["get", "aclose"])
            mock_client.get = AsyncMock(return_value=mock_response)

            adapter = EDGARAdapter(http_client=mock_client)
            with patch("app.ingestion.edgar_adapter.UNIVERSE", ["MSFT"]):
                docs = await adapter.fetch()

            assert len(docs) == 1
            assert docs[0].raw_metadata["event_type_hint"] == expected_hint, (
                f"Expected {expected_hint!r} for description {description!r}, "
                f"got {docs[0].raw_metadata['event_type_hint']!r}"
            )

    @pytest.mark.asyncio
    async def test_fetch_handles_empty_hits(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value={"hits": {"total": {"value": 0}, "hits": []}}
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = EDGARAdapter(http_client=mock_client)
        with patch("app.ingestion.edgar_adapter.UNIVERSE", ["AAPL"]):
            docs = await adapter.fetch()

        assert docs == []

    @pytest.mark.asyncio
    async def test_fetch_deduplicates_same_accession(self):
        """Same accession number queried twice should only produce one document."""
        single_hit = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "0001234567-24-000001",
                        "_source": {
                            "accession_no": "0001234567-24-000001",
                            "entity_name": "Apple Inc.",
                            "form_type": "8-K",
                            "file_date": "2024-01-15",
                            "description": "Item 1.01",
                        },
                    }
                ],
            }
        }
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value=single_hit)

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        shared_dedup = InMemoryDedupStore()
        adapter = EDGARAdapter(dedup_store=shared_dedup, http_client=mock_client)

        with patch("app.ingestion.edgar_adapter.UNIVERSE", ["AAPL"]):
            docs1 = await adapter.fetch()

        # Second fetch should return nothing (already seen)
        with patch("app.ingestion.edgar_adapter.UNIVERSE", ["AAPL"]):
            docs2 = await adapter.fetch()

        assert len(docs1) == 1
        assert len(docs2) == 0


# ── StockTwitsAdapter ─────────────────────────────────────────────────────────

class TestStockTwitsAdapter:
    def _make_stocktwits_response(self, ticker: str) -> dict[str, Any]:
        return {
            "symbol": {"symbol": ticker},
            "messages": [
                {
                    "id": 111111,
                    "body": f"${ticker} looking very bullish, love this stock!",
                    "created_at": "2024-01-15T12:00:00Z",
                    "user": {"id": 1, "username": "bull_trader"},
                    "entities": {"sentiment": {"basic": "Bullish"}},
                },
                {
                    "id": 222222,
                    "body": f"Taking profits on {ticker}, might come back down",
                    "created_at": "2024-01-15T11:30:00Z",
                    "user": {"id": 2, "username": "swing_trader"},
                    "entities": {"sentiment": {"basic": "Bearish"}},
                },
                {
                    "id": 333333,
                    "body": f"{ticker} neutral zone, waiting for breakout",
                    "created_at": "2024-01-15T11:00:00Z",
                    "user": {"id": 3, "username": "neutral_observer"},
                    "entities": {},  # No sentiment tag
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_fetch_returns_social_documents(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value=self._make_stocktwits_response("TSLA")
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = StockTwitsAdapter(http_client=mock_client)
        with patch("app.ingestion.stocktwits_adapter.UNIVERSE", ["TSLA"]):
            docs = await adapter.fetch()

        assert len(docs) == 3
        assert all(d.source == "social" for d in docs)
        assert all("TSLA" in d.ticker_symbols for d in docs)

    @pytest.mark.asyncio
    async def test_bullish_sentiment_hint_mapped(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value=self._make_stocktwits_response("TSLA")
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = StockTwitsAdapter(http_client=mock_client)
        with patch("app.ingestion.stocktwits_adapter.UNIVERSE", ["TSLA"]):
            docs = await adapter.fetch()

        # First message is Bullish
        bullish_doc = next(d for d in docs if "111111" in d.article_id)
        assert bullish_doc.raw_metadata["sentiment_hint"] == "positive"

    @pytest.mark.asyncio
    async def test_bearish_sentiment_hint_mapped(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value=self._make_stocktwits_response("TSLA")
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = StockTwitsAdapter(http_client=mock_client)
        with patch("app.ingestion.stocktwits_adapter.UNIVERSE", ["TSLA"]):
            docs = await adapter.fetch()

        bearish_doc = next(d for d in docs if "222222" in d.article_id)
        assert bearish_doc.raw_metadata["sentiment_hint"] == "negative"

    @pytest.mark.asyncio
    async def test_no_sentiment_tag_results_in_none(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value=self._make_stocktwits_response("TSLA")
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = StockTwitsAdapter(http_client=mock_client)
        with patch("app.ingestion.stocktwits_adapter.UNIVERSE", ["TSLA"]):
            docs = await adapter.fetch()

        neutral_doc = next(d for d in docs if "333333" in d.article_id)
        assert neutral_doc.raw_metadata["sentiment_hint"] is None

    @pytest.mark.asyncio
    async def test_deduplicates_by_message_id(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(
            return_value=self._make_stocktwits_response("TSLA")
        )

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        shared_dedup = InMemoryDedupStore()
        adapter = StockTwitsAdapter(dedup_store=shared_dedup, http_client=mock_client)

        with patch("app.ingestion.stocktwits_adapter.UNIVERSE", ["TSLA"]):
            docs1 = await adapter.fetch()
            docs2 = await adapter.fetch()

        assert len(docs1) == 3
        assert len(docs2) == 0  # All already seen

    @pytest.mark.asyncio
    async def test_rate_limit_response_returns_empty(self):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock(spec=["get", "aclose"])
        mock_client.get = AsyncMock(return_value=mock_response)

        adapter = StockTwitsAdapter(http_client=mock_client)
        with patch("app.ingestion.stocktwits_adapter.UNIVERSE", ["AAPL"]):
            with patch("app.ingestion.stocktwits_adapter._RATE_LIMIT_BACKOFF", 0.0):
                docs = await adapter.fetch()

        assert docs == []


# ── KaggleTweetReplayAdapter ──────────────────────────────────────────────────

class TestKaggleTweetReplayAdapter:
    def test_missing_file_generates_placeholder_docs(self, tmp_path):
        """When CSV is missing, adapter falls back to placeholder tweets."""
        missing_path = tmp_path / "does_not_exist.csv"
        adapter = KaggleTweetReplayAdapter(csv_path=missing_path)
        docs = asyncio.get_event_loop().run_until_complete(adapter.fetch())
        # Placeholder generates 4 tweets per ticker × 20 tickers = 80 docs
        # (some may be skipped if entity linking fails, but should have many)
        assert len(docs) > 0
        assert all(d.source == "social" for d in docs)

    def test_no_path_generates_placeholder_docs(self):
        """When no CSV path is configured, fall back to placeholders."""
        adapter = KaggleTweetReplayAdapter(csv_path=None)
        docs = asyncio.get_event_loop().run_until_complete(adapter.fetch())
        assert len(docs) > 0

    def test_reads_csv_file(self, tmp_path):
        """Adapter reads a valid CSV and returns documents."""
        csv_content = (
            "tweet_id,text,created_at,ticker\n"
            "1001,AAPL looking great today,2024-01-15T09:30:00Z,AAPL\n"
            "1002,Microsoft cloud revenue beats estimates,2024-01-15T09:31:00Z,MSFT\n"
            "1003,Tesla delivery numbers disappoint,2024-01-15T09:32:00Z,TSLA\n"
        )
        csv_file = tmp_path / "tweets.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        adapter = KaggleTweetReplayAdapter(csv_path=csv_file)
        docs = asyncio.get_event_loop().run_until_complete(adapter.fetch())

        assert len(docs) == 3
        tickers_found = {t for d in docs for t in d.ticker_symbols}
        assert "AAPL" in tickers_found
        assert "MSFT" in tickers_found
        assert "TSLA" in tickers_found

    def test_deduplicates_tweet_ids(self, tmp_path):
        """Duplicate tweet_ids in the CSV produce only one document each."""
        csv_content = (
            "tweet_id,text,created_at,ticker\n"
            "9001,AAPL to the moon!,2024-01-15T10:00:00Z,AAPL\n"
            "9001,AAPL to the moon!,2024-01-15T10:00:00Z,AAPL\n"  # duplicate
        )
        csv_file = tmp_path / "dedup_test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        adapter = KaggleTweetReplayAdapter(csv_path=csv_file)
        docs = asyncio.get_event_loop().run_until_complete(adapter.fetch())

        assert len(docs) == 1

    def test_handles_alternate_column_names(self, tmp_path):
        """Adapter detects 'content' for text and 'stock_name' for ticker."""
        csv_content = (
            "id,content,timestamp,stock_name\n"
            "5001,Nvidia GPU shortage continues,2024-01-15T11:00:00Z,NVDA\n"
        )
        csv_file = tmp_path / "alt_cols.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        adapter = KaggleTweetReplayAdapter(csv_path=csv_file)
        docs = asyncio.get_event_loop().run_until_complete(adapter.fetch())

        assert len(docs) == 1
        assert "NVDA" in docs[0].ticker_symbols

    def test_skips_rows_with_no_recognisable_ticker(self, tmp_path):
        """Rows that can't be linked to a known ticker are discarded."""
        csv_content = (
            "tweet_id,text,created_at,ticker\n"
            "8001,The weather is lovely today,2024-01-15T12:00:00Z,UNKNOWN_XYZ\n"
        )
        csv_file = tmp_path / "no_ticker.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        adapter = KaggleTweetReplayAdapter(csv_path=csv_file)
        docs = asyncio.get_event_loop().run_until_complete(adapter.fetch())

        # "UNKNOWN_XYZ" is not in the universe; entity linking won't match either
        assert len(docs) == 0

    def test_placeholder_tweets_are_social_documents(self):
        adapter = KaggleTweetReplayAdapter(csv_path=None)
        docs = asyncio.get_event_loop().run_until_complete(adapter.fetch())
        for doc in docs:
            assert doc.source == "social"
            assert len(doc.ticker_symbols) > 0

    def test_csv_timestamps_are_timezone_aware(self, tmp_path):
        csv_content = (
            "tweet_id,text,created_at,ticker\n"
            "7001,Apple earnings tomorrow,2024-01-15T08:00:00Z,AAPL\n"
        )
        csv_file = tmp_path / "tz_test.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        adapter = KaggleTweetReplayAdapter(csv_path=csv_file)
        docs = asyncio.get_event_loop().run_until_complete(adapter.fetch())

        assert len(docs) == 1
        assert docs[0].published_at.tzinfo is not None
