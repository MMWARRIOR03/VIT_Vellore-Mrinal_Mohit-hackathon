"""
Tests for DuckDBStore.

Uses an in-memory DuckDB database (path=":memory:") so no filesystem
side-effects are created and tests run fully isolated from each other.

Covers:
  - write_signal / get_signals round-trip
  - ticker filter matches the correct signal and excludes others
  - limit parameter is honoured
  - since (datetime) filter works
  - duplicate event_id insert is silently ignored (INSERT OR IGNORE)
  - write_article / get_articles_recent round-trip
  - ticker filter on articles
  - write_stress_run / get_stress_runs round-trip
  - write_rebalance_event / get_rebalance_events round-trip
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef
from app.storage.duckdb_store import DuckDBStore

# ── Helpers ───────────────────────────────────────────────────────────────────

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_signal(
    event_id: str = "evt-001",
    ticker: str = "AAPL",
    event_type: str = "Earnings",
    ts_published_offset_seconds: int = 300,
    sentiment: float = 0.5,
    impact: float = 6.0,
) -> RiskSignal:
    published = NOW - timedelta(seconds=ts_published_offset_seconds)
    return RiskSignal(
        event_id=event_id,
        ts_published=published,
        ts_processed=NOW,
        source="news",
        tickers=[TickerRef(ticker=ticker, exchange="NYSE", relevance=1.0)],
        sentiment=sentiment,
        sentiment_conf=0.80,
        event_type=event_type,
        event_probs={event_type: 0.85, "Macro": 0.15},
        impact=impact,
        impact_range=(impact - 1.0, impact + 1.0),
        confidence=0.80,
        novelty=0.50,
        diffusion=DiffusionInfo(sources=3, velocity=1.0, score=0.40),
        decay_halflife_hours=12.0,
        explanation="Test signal.",
        cluster_id=f"cluster-{event_id}",
        model_versions={"sentiment": "test-v1"},
    )


@pytest_asyncio.fixture
async def db():
    """In-memory DuckDB store, initialised and torn down per test."""
    store = DuckDBStore(path=Path(":memory:"))
    await store.initialise()
    yield store
    await store.close()


# ── Signals ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestSignalWriteRead:

    async def test_write_then_get_returns_signal(self, db: DuckDBStore):
        signal = _make_signal(event_id="s-001", ticker="AAPL")
        await db.write_signal(signal)
        results = await db.get_signals()
        assert len(results) == 1
        assert results[0].event_id == "s-001"

    async def test_round_trip_preserves_fields(self, db: DuckDBStore):
        original = _make_signal(event_id="s-rt", ticker="MSFT", sentiment=-0.3)
        await db.write_signal(original)
        results = await db.get_signals()
        restored = results[0]
        assert restored.event_id == original.event_id
        assert restored.sentiment == pytest.approx(original.sentiment)
        assert restored.tickers[0].ticker == "MSFT"
        assert restored.diffusion.sources == original.diffusion.sources

    async def test_duplicate_event_id_is_silently_ignored(self, db: DuckDBStore):
        s1 = _make_signal(event_id="s-dup")
        s2 = _make_signal(event_id="s-dup")   # same ID
        await db.write_signal(s1)
        await db.write_signal(s2)              # should not raise
        results = await db.get_signals()
        assert len(results) == 1

    async def test_ticker_filter_returns_matching_only(self, db: DuckDBStore):
        await db.write_signal(_make_signal(event_id="s-aapl", ticker="AAPL"))
        await db.write_signal(_make_signal(event_id="s-msft", ticker="MSFT"))
        await db.write_signal(_make_signal(event_id="s-nvda", ticker="NVDA"))

        aapl_results = await db.get_signals(ticker="AAPL")
        assert len(aapl_results) == 1
        assert aapl_results[0].event_id == "s-aapl"

    async def test_ticker_filter_case_insensitive(self, db: DuckDBStore):
        await db.write_signal(_make_signal(event_id="s-lower", ticker="AAPL"))
        results = await db.get_signals(ticker="aapl")
        assert len(results) == 1

    async def test_ticker_filter_no_match_returns_empty(self, db: DuckDBStore):
        await db.write_signal(_make_signal(event_id="s-001", ticker="AAPL"))
        results = await db.get_signals(ticker="TSLA")
        assert results == []

    async def test_limit_parameter_is_honoured(self, db: DuckDBStore):
        for i in range(10):
            await db.write_signal(_make_signal(event_id=f"s-{i:03d}", ticker="JPM"))
        results = await db.get_signals(limit=3)
        assert len(results) == 3

    async def test_since_filter_excludes_older_signals(self, db: DuckDBStore):
        old_signal = _make_signal(
            event_id="s-old", ticker="GS", ts_published_offset_seconds=3600
        )  # published 1 hour ago
        new_signal = _make_signal(
            event_id="s-new", ticker="GS", ts_published_offset_seconds=60
        )  # published 1 minute ago
        await db.write_signal(old_signal)
        await db.write_signal(new_signal)

        cutoff = NOW - timedelta(minutes=10)
        results = await db.get_signals(since=cutoff)
        ids = [r.event_id for r in results]
        assert "s-new" in ids
        assert "s-old" not in ids

    async def test_get_signals_returns_newest_first(self, db: DuckDBStore):
        for i in range(5):
            await db.write_signal(
                _make_signal(event_id=f"s-{i}", ticker="BA", ts_published_offset_seconds=i * 60)
            )
        results = await db.get_signals(limit=5)
        # Newest published (smallest offset) should come first
        offsets = [
            (NOW - r.ts_published).total_seconds() for r in results
        ]
        assert offsets == sorted(offsets)

    async def test_multi_ticker_signal_found_by_either_ticker(self, db: DuckDBStore):
        multi = RiskSignal(
            event_id="multi-001",
            ts_published=NOW - timedelta(seconds=60),
            ts_processed=NOW,
            source="news",
            tickers=[
                TickerRef(ticker="AAPL", relevance=1.0),
                TickerRef(ticker="MSFT", relevance=0.7),
            ],
            sentiment=0.2,
            sentiment_conf=0.8,
            event_type="M&A",
            event_probs={"M&A": 0.90, "Regulatory": 0.10},
            impact=7.0,
            impact_range=(6.0, 8.0),
            confidence=0.85,
            novelty=0.70,
            diffusion=DiffusionInfo(sources=5, velocity=2.0, score=0.60),
            decay_halflife_hours=48.0,
            explanation=None,
            cluster_id="cluster-ma",
            model_versions={},
        )
        await db.write_signal(multi)
        # Should be found by AAPL
        r1 = await db.get_signals(ticker="AAPL")
        assert len(r1) == 1
        # Should also be found by MSFT
        r2 = await db.get_signals(ticker="MSFT")
        assert len(r2) == 1


# ── Articles ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestArticleWriteRead:

    def _make_article(self, article_id: str, tickers: list[str], headline: str = "Test headline") -> dict:
        return {
            "article_id": article_id,
            "ts_published": NOW - timedelta(minutes=5),
            "source": "Reuters",
            "url": f"https://example.com/{article_id}",
            "headline": headline,
            "body": "Test article body.",
            "tickers": tickers,
        }

    async def test_write_then_get_returns_article(self, db: DuckDBStore):
        article = self._make_article("art-001", ["AAPL"], "Apple reports record earnings")
        await db.write_article(article)
        results = await db.get_articles_recent()
        assert len(results) == 1
        assert results[0]["article_id"] == "art-001"

    async def test_ticker_filter_on_articles(self, db: DuckDBStore):
        await db.write_article(self._make_article("art-aapl", ["AAPL"]))
        await db.write_article(self._make_article("art-msft", ["MSFT"]))

        results = await db.get_articles_recent(ticker="AAPL")
        assert len(results) == 1
        assert results[0]["article_id"] == "art-aapl"

    async def test_article_limit(self, db: DuckDBStore):
        for i in range(5):
            await db.write_article(self._make_article(f"art-{i}", ["JPM"]))
        results = await db.get_articles_recent(limit=2)
        assert len(results) == 2

    async def test_duplicate_article_id_ignored(self, db: DuckDBStore):
        art = self._make_article("art-dup", ["GS"])
        await db.write_article(art)
        await db.write_article(art)
        results = await db.get_articles_recent()
        assert len(results) == 1


# ── Stress runs ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestStressRunWriteRead:

    def _make_run(self, run_id: str, scenario: str = "Geopolitical") -> dict:
        return {
            "run_id": run_id,
            "ts_created": NOW,
            "scenario_name": scenario,
            "trigger_signal": None,
            "pnl_impact": -125_000.0,
            "var_95": 95_000.0,
            "cvar_95": 142_000.0,
            "results": {"equity_pnl": -80_000, "bond_pnl": -45_000},
            "meta": {"version": "1.0"},
        }

    async def test_write_then_get(self, db: DuckDBStore):
        await db.write_stress_run(self._make_run("run-001"))
        results = await db.get_stress_runs()
        assert len(results) == 1
        assert results[0]["run_id"] == "run-001"
        assert results[0]["pnl_impact"] == pytest.approx(-125_000.0)

    async def test_scenario_filter(self, db: DuckDBStore):
        await db.write_stress_run(self._make_run("run-geo", "Geopolitical"))
        await db.write_stress_run(self._make_run("run-cred", "Credit"))
        results = await db.get_stress_runs(scenario_name="Credit")
        assert len(results) == 1
        assert results[0]["scenario_name"] == "Credit"

    async def test_results_dict_preserved(self, db: DuckDBStore):
        run = self._make_run("run-dict")
        run["results"] = {"equity_pnl": -55_000, "swap_pnl": 12_000}
        await db.write_stress_run(run)
        results = await db.get_stress_runs()
        assert results[0]["results"]["equity_pnl"] == -55_000


# ── Rebalance events ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestRebalanceEventWriteRead:

    def _make_rebalance(self, rebalance_id: str) -> dict:
        return {
            "rebalance_id": rebalance_id,
            "ts_created": NOW,
            "trigger_signal": "evt-001",
            "delta": {"AAPL": -0.02, "JPM": +0.02},
            "reason": "High-impact earnings signal",
            "meta": {},
        }

    async def test_write_then_get(self, db: DuckDBStore):
        await db.write_rebalance_event(self._make_rebalance("reb-001"))
        results = await db.get_rebalance_events()
        assert len(results) == 1
        assert results[0]["rebalance_id"] == "reb-001"

    async def test_delta_dict_preserved(self, db: DuckDBStore):
        await db.write_rebalance_event(self._make_rebalance("reb-delta"))
        results = await db.get_rebalance_events()
        assert results[0]["delta"]["AAPL"] == pytest.approx(-0.02)
        assert results[0]["delta"]["JPM"] == pytest.approx(0.02)

    async def test_limit_honoured(self, db: DuckDBStore):
        for i in range(6):
            await db.write_rebalance_event(self._make_rebalance(f"reb-{i}"))
        results = await db.get_rebalance_events(limit=4)
        assert len(results) == 4
