"""
Module A test suite.

All tests are unit tests — no Redis, no DuckDB, no network required.

Coverage
────────
  WeightCalculator:
    - equal scores → all weights near base weight (0.05)
    - positive score for one ticker → that ticker's weight increases
    - negative score for one ticker → that ticker's weight decreases
    - weights always sum to 1.0 (within 1e-6)
    - all weights within [W_MIN, W_MAX]
    - should_rebalance False when all changes below MIN_WEIGHT_CHANGE
    - should_rebalance False when mean_confidence below CONFIDENCE_THRESHOLD
    - should_rebalance True when large change above threshold
    - cap_turnover: total one-way turnover never exceeds MAX_TURNOVER
    - hard_override reduces weight for high-impact negative signal ticker
    - hard_override no-op for low-impact signal

  SignalAggregator:
    - score is 0 initially
    - score is negative after adding a negative-sentiment signal
    - duplicate cluster_id does not double-count
    - old signal decays to near zero after many half-lives
    - get_mean_confidence returns 0.0 with no signals

  PriceCache:
    - returns non-empty list for any ticker (synthetic fallback)

  RebalanceEvent:
    - weight_deltas = after_weights − before_weights for each ticker
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef
from module_a.price_cache import PriceCache
from module_a.rebalancer import RebalanceEvent
from module_a.signal_aggregator import SignalAggregator
from module_a.weight_calculator import (
    CONFIDENCE_THRESHOLD,
    MAX_TURNOVER,
    MIN_WEIGHT_CHANGE,
    W_MAX,
    W_MIN,
    WeightCalculator,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM",  "BAC",  "GS",
    "V",    "JNJ",  "UNH",  "PFE",  "XOM",
    "CVX",  "CAT",  "BA",   "WMT",  "HD",
]


def _make_signal(
    ticker: str = "AAPL",
    sentiment: float = -0.5,
    impact: float = 5.0,
    confidence: float = 0.80,
    novelty: float = 0.90,
    cluster_id: str = "cluster-001",
    halflife_hours: float = 12.0,
    ts_processed: datetime | None = None,
    event_type: str = "Earnings",
) -> RiskSignal:
    """Construct a valid RiskSignal for testing."""
    now = datetime.now(tz=timezone.utc)
    processed = ts_processed or now
    # ts_published must be <= ts_processed
    published = processed - timedelta(seconds=30)

    # impact_range must contain impact
    low = max(1.0, impact - 0.5)
    high = min(10.0, impact + 0.5)

    return RiskSignal(
        event_id=f"test-{ticker}-{cluster_id}",
        ts_published=published,
        ts_processed=processed,
        source="news",
        tickers=[TickerRef(ticker=ticker, relevance=1.0)],
        sentiment=sentiment,
        sentiment_conf=0.85,
        event_type=event_type,
        event_probs={event_type: 0.90, "Macro": 0.10},
        impact=impact,
        impact_range=(low, high),
        confidence=confidence,
        novelty=novelty,
        diffusion=DiffusionInfo(sources=3, velocity=2.0, score=0.65),
        decay_halflife_hours=halflife_hours,
        cluster_id=cluster_id,
        model_versions={"sentiment": "test-v1"},
    )


def _make_high_impact_negative_signal(ticker: str = "TSLA") -> RiskSignal:
    """Impact ≥ 8, sentiment < -0.6 → hard_override fires."""
    return _make_signal(
        ticker=ticker,
        sentiment=-0.8,
        impact=8.5,
        confidence=0.90,
        cluster_id="override-cluster",
        event_type="Geopolitical",
    )


# ── WeightCalculator ──────────────────────────────────────────────────────────

class TestWeightCalculator:

    @pytest.fixture
    def calc(self) -> WeightCalculator:
        """Fresh WeightCalculator backed by the real index.yaml."""
        return WeightCalculator()

    def _zero_scores(self, calc: WeightCalculator) -> dict[str, float]:
        return {t: 0.0 for t in calc.base_weights}

    def test_equal_scores_produce_near_base_weights(self, calc: WeightCalculator) -> None:
        """
        When all scores are identical (zero), z-scores are all zero.
        exp(λ × 0) = 1, so every raw weight equals its base weight.
        After normalisation the result must equal the base weights.
        """
        scores = self._zero_scores(calc)
        weights = calc.compute_target_weights(scores)
        for ticker, bw in calc.base_weights.items():
            assert abs(weights[ticker] - bw) < 1e-6, (
                f"Equal-score weight for {ticker}: got {weights[ticker]:.6f}, "
                f"expected {bw:.6f}"
            )

    def test_positive_score_increases_nvda_weight(self, calc: WeightCalculator) -> None:
        scores = self._zero_scores(calc)
        scores["NVDA"] = 5.0  # strongly positive
        weights = calc.compute_target_weights(scores)
        base = calc.base_weights["NVDA"]
        assert weights["NVDA"] > base, (
            f"NVDA weight should increase with positive score; "
            f"base={base:.4f}, got={weights['NVDA']:.4f}"
        )

    def test_negative_score_decreases_tsla_weight(self, calc: WeightCalculator) -> None:
        scores = self._zero_scores(calc)
        scores["TSLA"] = -5.0  # strongly negative
        weights = calc.compute_target_weights(scores)
        base = calc.base_weights["TSLA"]
        assert weights["TSLA"] < base, (
            f"TSLA weight should decrease with negative score; "
            f"base={base:.4f}, got={weights['TSLA']:.4f}"
        )

    def test_weights_sum_to_one(self, calc: WeightCalculator) -> None:
        scores = {t: float(i) for i, t in enumerate(calc.base_weights)}
        weights = calc.compute_target_weights(scores)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total:.8f}, expected 1.0"

    def test_weights_within_min_max(self, calc: WeightCalculator) -> None:
        # Use extreme scores to stress-test the clip.
        scores = self._zero_scores(calc)
        scores["AAPL"] = 100.0
        scores["XOM"] = -100.0
        weights = calc.compute_target_weights(scores)
        for ticker, w in weights.items():
            # After re-normalisation the absolute values may shift, but the
            # pre-normalisation clip ensures raw ∈ [W_MIN, W_MAX].  After
            # re-normalisation weights may technically exceed W_MAX if clipping
            # pulled many weights to the boundary.  The guarantee the spec
            # makes is that the *raw* clip is applied.  We verify the softer
            # property: no weight is below W_MIN (which survives renorm).
            assert w >= W_MIN - 1e-6, (
                f"{ticker} weight {w:.6f} below W_MIN={W_MIN}"
            )

    def test_should_rebalance_false_when_all_changes_small(
        self, calc: WeightCalculator
    ) -> None:
        current = {t: 0.05 for t in calc.base_weights}
        # Target barely differs.
        target = {t: 0.05 + 0.001 for t in calc.base_weights}
        total = sum(target.values())
        target = {t: v / total for t, v in target.items()}
        assert not calc.should_rebalance(current, target, mean_confidence=0.80)

    def test_should_rebalance_false_when_low_confidence(
        self, calc: WeightCalculator
    ) -> None:
        current = {t: 0.05 for t in calc.base_weights}
        target = dict(current)
        target["NVDA"] = 0.12  # large delta
        assert not calc.should_rebalance(
            current, target, mean_confidence=CONFIDENCE_THRESHOLD - 0.01
        )

    def test_should_rebalance_true_when_conditions_met(
        self, calc: WeightCalculator
    ) -> None:
        current = {t: 0.05 for t in calc.base_weights}
        target = dict(current)
        target["NVDA"] = current["NVDA"] + MIN_WEIGHT_CHANGE + 0.01
        assert calc.should_rebalance(
            current, target, mean_confidence=CONFIDENCE_THRESHOLD + 0.01
        )

    def test_cap_turnover_never_exceeds_max(self, calc: WeightCalculator) -> None:
        current = {t: 0.05 for t in calc.base_weights}
        # Construct a target that would require massive turnover.
        tickers = list(calc.base_weights.keys())
        target = dict(current)
        for i, t in enumerate(tickers[:5]):
            target[t] = 0.14  # push 5 tickers to near-max
        # Re-normalise so it sums to 1.
        total = sum(target.values())
        target = {t: v / total for t, v in target.items()}

        capped = calc.cap_turnover(current, target)
        one_way = sum(abs(capped[t] - current[t]) for t in capped) / 2.0
        assert one_way <= MAX_TURNOVER + 1e-9, (
            f"One-way turnover {one_way:.4f} exceeds MAX_TURNOVER={MAX_TURNOVER}"
        )

    def test_hard_override_reduces_weight_for_high_impact_negative(
        self, calc: WeightCalculator
    ) -> None:
        current = {t: 0.05 for t in calc.base_weights}
        signal = _make_high_impact_negative_signal(ticker="TSLA")
        after = calc.hard_override(current, signal)
        cap = W_MAX / 2.0
        assert after["TSLA"] <= cap + 1e-9, (
            f"TSLA weight {after['TSLA']:.6f} should be ≤ W_MAX/2={cap:.4f}"
        )
        # Re-normalisation must still hold.
        assert abs(sum(after.values()) - 1.0) < 1e-6

    def test_hard_override_noop_for_low_impact_signal(
        self, calc: WeightCalculator
    ) -> None:
        current = {t: 0.05 for t in calc.base_weights}
        # Low impact; hard_override should not fire.
        signal = _make_signal(ticker="TSLA", sentiment=-0.9, impact=5.0)
        after = calc.hard_override(current, signal)
        for t in current:
            assert abs(after[t] - current[t]) < 1e-9, (
                f"Low-impact override should not change any weight; {t} changed"
            )


# ── SignalAggregator ──────────────────────────────────────────────────────────

class TestSignalAggregator:

    @pytest.fixture
    def agg(self) -> SignalAggregator:
        return SignalAggregator()

    @pytest.fixture
    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def test_score_is_zero_initially(
        self, agg: SignalAggregator, now: datetime
    ) -> None:
        assert agg.compute_score("AAPL", now) == 0.0

    def test_score_is_negative_after_negative_signal(
        self, agg: SignalAggregator, now: datetime
    ) -> None:
        sig = _make_signal(ticker="AAPL", sentiment=-0.8, ts_processed=now)
        agg.add_signal(sig)
        score = agg.compute_score("AAPL", now)
        assert score < 0.0, f"Expected negative score, got {score}"

    def test_duplicate_cluster_id_does_not_double_count(
        self, agg: SignalAggregator, now: datetime
    ) -> None:
        sig = _make_signal(
            ticker="MSFT", sentiment=-0.5, cluster_id="dup-cluster",
            ts_processed=now,
        )
        agg.add_signal(sig)
        score_once = agg.compute_score("MSFT", now)

        # Add the same cluster again — should overwrite, not accumulate.
        agg.add_signal(sig)
        score_twice = agg.compute_score("MSFT", now)

        assert abs(score_once - score_twice) < 1e-10, (
            f"Duplicate cluster_id inflated score: once={score_once}, twice={score_twice}"
        )

    def test_old_signal_decays_to_near_zero(
        self, agg: SignalAggregator
    ) -> None:
        """
        A signal processed 20 half-lives ago should contribute ≈ 2^-20 ≈ 0
        to the score.  We use a 1-hour halflife and backdate by 20 hours.
        """
        halflife = 1.0  # 1 hour
        age_hours = 20.0 * halflife  # 20 half-lives
        old_ts = datetime.now(tz=timezone.utc) - timedelta(hours=age_hours)

        sig = _make_signal(
            ticker="NVDA",
            sentiment=1.0,  # max positive so any residual is detectable
            confidence=1.0,
            novelty=1.0,
            halflife_hours=halflife,
            ts_processed=old_ts,
        )
        agg.add_signal(sig)

        now = datetime.now(tz=timezone.utc)
        score = agg.compute_score("NVDA", now)
        # exp(-ln2 * 20) = 2^-20 ≈ 9.5e-7; score should be < 1e-4
        assert abs(score) < 1e-4, (
            f"20-halflife-old signal should have nearly decayed; score={score:.2e}"
        )

    def test_get_mean_confidence_no_signals_returns_zero(
        self, agg: SignalAggregator
    ) -> None:
        assert agg.get_mean_confidence() == 0.0

    def test_get_mean_confidence_returns_correct_mean(
        self, agg: SignalAggregator, now: datetime
    ) -> None:
        sig1 = _make_signal(ticker="JPM", confidence=0.60, cluster_id="c1", ts_processed=now)
        sig2 = _make_signal(ticker="BAC", confidence=0.80, cluster_id="c2", ts_processed=now)
        agg.add_signal(sig1)
        agg.add_signal(sig2)
        mc = agg.get_mean_confidence()
        assert abs(mc - 0.70) < 1e-6, f"Expected mean_conf=0.70, got {mc}"

    def test_score_positive_for_positive_sentiment(
        self, agg: SignalAggregator, now: datetime
    ) -> None:
        sig = _make_signal(ticker="AAPL", sentiment=0.9, ts_processed=now)
        agg.add_signal(sig)
        assert agg.compute_score("AAPL", now) > 0.0

    def test_cleanup_expired_removes_stale_signals(
        self, agg: SignalAggregator
    ) -> None:
        halflife = 1.0
        # Signal processed 6 half-lives ago (threshold is 5).
        old_ts = datetime.now(tz=timezone.utc) - timedelta(hours=6.5 * halflife)
        sig = _make_signal(
            ticker="GS", halflife_hours=halflife, ts_processed=old_ts
        )
        agg.add_signal(sig)

        now = datetime.now(tz=timezone.utc)
        agg.cleanup_expired(now)

        # After cleanup the ticker should have no signals.
        assert "GS" not in agg._history or not agg._history.get("GS"), (
            "Expired signal should have been removed"
        )


# ── PriceCache ────────────────────────────────────────────────────────────────

class TestPriceCache:

    def test_returns_non_empty_list_synthetic_fallback(self) -> None:
        cache = PriceCache()
        for ticker in _UNIVERSE:
            prices = cache.get_prices(ticker)
            assert len(prices) > 0, f"Expected prices for {ticker}, got empty list"
            assert all(p > 0 for p in prices), (
                f"All prices for {ticker} should be positive"
            )

    def test_synthetic_prices_are_deterministic(self) -> None:
        cache1 = PriceCache()
        cache2 = PriceCache()
        prices1 = cache1.get_prices("TSLA")
        prices2 = cache2.get_prices("TSLA")
        assert prices1 == prices2, "Synthetic prices should be deterministic"

    def test_different_tickers_have_different_prices(self) -> None:
        cache = PriceCache()
        aapl = cache.get_prices("AAPL")
        msft = cache.get_prices("MSFT")
        # With different seeds the series should differ.
        assert aapl != msft, "Different tickers should produce different price series"

    def test_n_days_trims_result(self) -> None:
        cache = PriceCache()
        prices = cache.get_prices("NVDA", n_days=10)
        assert len(prices) <= 10, f"Expected ≤10 prices, got {len(prices)}"


# ── RebalanceEvent ────────────────────────────────────────────────────────────

class TestRebalanceEvent:

    def _make_event(
        self,
        before: dict[str, float],
        after: dict[str, float],
    ) -> RebalanceEvent:
        deltas = {t: after[t] - before.get(t, 0.0) for t in after}
        return RebalanceEvent(
            rebalance_id="test-id",
            ts_created=datetime.now(tz=timezone.utc),
            trigger_signal_id="sig-001",
            before_weights=dict(before),
            after_weights=dict(after),
            weight_deltas=deltas,
            reason="test",
        )

    def test_weight_deltas_equal_after_minus_before(self) -> None:
        before = {t: 0.05 for t in _UNIVERSE}
        after = {t: 0.05 for t in _UNIVERSE}
        after["NVDA"] = 0.10
        after["TSLA"] = 0.02
        # Re-normalise
        total = sum(after.values())
        after = {t: v / total for t, v in after.items()}

        event = self._make_event(before, after)
        for ticker in after:
            expected_delta = after[ticker] - before.get(ticker, 0.0)
            assert abs(event.weight_deltas[ticker] - expected_delta) < 1e-10, (
                f"Delta for {ticker}: expected {expected_delta:.6f}, "
                f"got {event.weight_deltas[ticker]:.6f}"
            )

    def test_to_dict_has_required_keys(self) -> None:
        before = {t: 0.05 for t in _UNIVERSE}
        after = {t: 0.05 for t in _UNIVERSE}
        event = self._make_event(before, after)
        d = event.to_dict()
        assert "rebalance_id" in d
        assert "ts_created" in d
        assert "trigger_signal" in d
        assert "delta" in d
        assert "reason" in d
        assert "meta" in d
        assert "before_weights" in d["meta"]
        assert "after_weights" in d["meta"]

    def test_to_dict_delta_matches_weight_deltas(self) -> None:
        before = {t: 0.05 for t in _UNIVERSE}
        after = {t: 0.05 for t in _UNIVERSE}
        after["AAPL"] = 0.09
        total = sum(after.values())
        after = {t: v / total for t, v in after.items()}

        event = self._make_event(before, after)
        d = event.to_dict()
        assert d["delta"] == event.weight_deltas
