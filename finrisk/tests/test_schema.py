"""
Tests for the RiskSignal schema contract.

Covers:
  - Successful construction with valid data
  - Field validator enforcement (bounds, UTC requirement, event_probs invariants)
  - Cross-field validators (impact within impact_range, ts ordering, event_type argmax)
  - JSON round-trip via model_dump_json / model_validate_json
  - Pydantic v2 model_validate (dict → model)
  - Immutability via frozen=True
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef

# ── Fixtures ──────────────────────────────────────────────────────────────────

NOW_UTC = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
PUBLISHED_UTC = datetime(2024, 6, 1, 11, 55, 0, tzinfo=timezone.utc)


def _valid_kwargs(**overrides) -> dict:
    """Return a dict of valid RiskSignal constructor kwargs, optionally overridden."""
    base = {
        "event_id": "test-event-001",
        "ts_published": PUBLISHED_UTC,
        "ts_processed": NOW_UTC,
        "source": "news",
        "tickers": [TickerRef(ticker="AAPL", exchange="NASDAQ", relevance=1.0)],
        "sentiment": -0.35,
        "sentiment_conf": 0.82,
        "event_type": "Earnings",
        "event_probs": {"Earnings": 0.75, "Macro": 0.15, "Regulatory": 0.10},
        "impact": 6.0,
        "impact_range": (5.0, 7.5),
        "confidence": 0.80,
        "novelty": 0.60,
        "diffusion": DiffusionInfo(sources=4, velocity=1.2, score=0.55),
        "decay_halflife_hours": 12.0,
        "explanation": "Test explanation.",
        "cluster_id": "cluster-abc123",
        "model_versions": {"sentiment": "v1", "classifier": "v2"},
    }
    base.update(overrides)
    return base


# ── Construction ──────────────────────────────────────────────────────────────

class TestRiskSignalConstruction:

    def test_valid_signal_builds_without_error(self):
        signal = RiskSignal(**_valid_kwargs())
        assert signal.event_id == "test-event-001"
        assert signal.event_type == "Earnings"

    def test_tickers_normalised_to_uppercase(self):
        signal = RiskSignal(**_valid_kwargs(
            tickers=[TickerRef(ticker="aapl", relevance=1.0)]
        ))
        assert signal.tickers[0].ticker == "AAPL"

    def test_optional_explanation_can_be_none(self):
        signal = RiskSignal(**_valid_kwargs(explanation=None))
        assert signal.explanation is None

    def test_optional_exchange_can_be_none(self):
        signal = RiskSignal(**_valid_kwargs(
            tickers=[TickerRef(ticker="TSLA")]
        ))
        assert signal.tickers[0].exchange is None

    def test_model_versions_defaults_to_empty_dict(self):
        kwargs = _valid_kwargs()
        del kwargs["model_versions"]
        signal = RiskSignal(**kwargs)
        assert signal.model_versions == {}

    def test_multiple_tickers_accepted(self):
        tickers = [
            TickerRef(ticker="AAPL", relevance=1.0),
            TickerRef(ticker="MSFT", relevance=0.7),
            TickerRef(ticker="NVDA", relevance=0.4),
        ]
        signal = RiskSignal(**_valid_kwargs(tickers=tickers))
        assert len(signal.tickers) == 3


# ── Sentiment bounds ──────────────────────────────────────────────────────────

class TestSentimentBounds:

    @pytest.mark.parametrize("sentiment", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_boundary_sentiments_accepted(self, sentiment: float):
        signal = RiskSignal(**_valid_kwargs(sentiment=sentiment))
        assert signal.sentiment == sentiment

    @pytest.mark.parametrize("sentiment", [-1.001, 1.001, -2.0, 2.0])
    def test_out_of_bounds_sentiment_rejected(self, sentiment: float):
        with pytest.raises(ValidationError) as exc_info:
            RiskSignal(**_valid_kwargs(sentiment=sentiment))
        errors = exc_info.value.errors()
        assert any("sentiment" in str(e["loc"]) for e in errors)

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_sentiment_conf_boundaries(self, conf: float):
        signal = RiskSignal(**_valid_kwargs(sentiment_conf=conf))
        assert signal.sentiment_conf == conf

    @pytest.mark.parametrize("conf", [-0.01, 1.01])
    def test_sentiment_conf_out_of_bounds_rejected(self, conf: float):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(sentiment_conf=conf))


# ── Impact bounds ─────────────────────────────────────────────────────────────

class TestImpactBounds:

    @pytest.mark.parametrize("impact,low,high", [
        (1.0, 1.0, 1.0),
        (5.5, 4.0, 7.0),
        (10.0, 10.0, 10.0),
    ])
    def test_valid_impact_values(self, impact, low, high):
        signal = RiskSignal(**_valid_kwargs(impact=impact, impact_range=(low, high)))
        assert signal.impact == impact

    def test_impact_below_1_rejected(self):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(impact=0.9, impact_range=(0.5, 1.5)))

    def test_impact_above_10_rejected(self):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(impact=10.1, impact_range=(9.0, 11.0)))

    def test_impact_outside_impact_range_rejected(self):
        """impact must lie within [impact_range.low, impact_range.high]."""
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(impact=8.0, impact_range=(3.0, 5.0)))

    def test_impact_range_inverted_rejected(self):
        """low > high should be rejected."""
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(impact=5.0, impact_range=(7.0, 4.0)))

    def test_impact_range_out_of_1_to_10_rejected(self):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(impact=5.0, impact_range=(0.5, 5.0)))


# ── Confidence and novelty bounds ─────────────────────────────────────────────

class TestConfidenceNoveltybounds:

    @pytest.mark.parametrize("field,value", [
        ("confidence", 0.0), ("confidence", 1.0),
        ("novelty", 0.0), ("novelty", 1.0),
    ])
    def test_boundaries_accepted(self, field, value):
        signal = RiskSignal(**_valid_kwargs(**{field: value}))
        assert getattr(signal, field) == value

    @pytest.mark.parametrize("field,value", [
        ("confidence", -0.01), ("confidence", 1.01),
        ("novelty", -0.01), ("novelty", 1.01),
    ])
    def test_out_of_bounds_rejected(self, field, value):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(**{field: value}))


# ── UTC timestamp enforcement ─────────────────────────────────────────────────

class TestTimestampValidation:

    def test_naive_datetime_rejected(self):
        """A tz-naive datetime must be rejected."""
        naive = datetime(2024, 1, 1, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(ts_published=naive))

    def test_iso_string_with_utc_accepted(self):
        """ISO string with Z suffix should be accepted."""
        signal = RiskSignal(**_valid_kwargs(
            ts_published="2024-06-01T11:55:00+00:00",
            ts_processed="2024-06-01T12:00:00+00:00",
        ))
        assert signal.ts_published.tzinfo is not None

    def test_ts_processed_before_ts_published_rejected(self):
        """Processing time before publication time must be rejected."""
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(
                ts_published=NOW_UTC,
                ts_processed=PUBLISHED_UTC,   # earlier than published
            ))


# ── event_probs invariants ────────────────────────────────────────────────────

class TestEventProbs:

    def test_event_probs_sum_over_one_rejected(self):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(
                event_type="Earnings",
                event_probs={"Earnings": 0.80, "Macro": 0.30},  # sum = 1.10
            ))

    def test_negative_probability_rejected(self):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(
                event_type="Earnings",
                event_probs={"Earnings": 0.80, "Macro": -0.05},
            ))

    def test_empty_event_probs_rejected(self):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(event_probs={}))

    def test_event_type_not_argmax_rejected(self):
        """event_type must be the highest-probability label."""
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(
                event_type="Macro",
                event_probs={"Earnings": 0.75, "Macro": 0.25},
                # Argmax is "Earnings" but event_type says "Macro"
            ))

    def test_valid_single_class_probs(self):
        signal = RiskSignal(**_valid_kwargs(
            event_type="Earnings",
            event_probs={"Earnings": 0.95},
        ))
        assert signal.event_probs["Earnings"] == pytest.approx(0.95)


# ── Source literal ────────────────────────────────────────────────────────────

class TestSourceLiteral:

    @pytest.mark.parametrize("source", ["news", "social", "filing"])
    def test_valid_sources(self, source):
        signal = RiskSignal(**_valid_kwargs(source=source))
        assert signal.source == source

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            RiskSignal(**_valid_kwargs(source="blog"))


# ── DiffusionInfo ─────────────────────────────────────────────────────────────

class TestDiffusionInfo:

    def test_sources_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            DiffusionInfo(sources=0, velocity=0.0, score=0.5)

    def test_velocity_non_negative(self):
        with pytest.raises(ValidationError):
            DiffusionInfo(sources=1, velocity=-1.0, score=0.5)

    def test_score_out_of_bounds_rejected(self):
        with pytest.raises(ValidationError):
            DiffusionInfo(sources=1, velocity=0.0, score=1.01)

    def test_valid_diffusion(self):
        d = DiffusionInfo(sources=10, velocity=3.5, score=0.72)
        assert d.sources == 10


# ── TickerRef ─────────────────────────────────────────────────────────────────

class TestTickerRef:

    def test_ticker_coerced_to_uppercase(self):
        t = TickerRef(ticker="aapl")
        assert t.ticker == "AAPL"

    def test_relevance_must_be_0_to_1(self):
        with pytest.raises(ValidationError):
            TickerRef(ticker="AAPL", relevance=1.5)

    def test_empty_ticker_rejected(self):
        with pytest.raises(ValidationError):
            TickerRef(ticker="")


# ── JSON round-trip ───────────────────────────────────────────────────────────

class TestJsonRoundTrip:

    def test_model_dump_json_then_model_validate_json(self):
        original = RiskSignal(**_valid_kwargs())
        json_str = original.model_dump_json()
        restored = RiskSignal.model_validate_json(json_str)
        assert restored.event_id == original.event_id
        assert restored.sentiment == original.sentiment
        assert restored.impact_range == original.impact_range
        assert restored.diffusion.score == original.diffusion.score

    def test_json_is_valid_json(self):
        signal = RiskSignal(**_valid_kwargs())
        json_str = signal.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["event_id"] == "test-event-001"
        assert isinstance(parsed["tickers"], list)
        assert parsed["diffusion"]["sources"] == 4

    def test_model_validate_from_dict(self):
        """Pydantic v2 model_validate should accept a plain dict."""
        data = {
            "event_id": "v2-validate-test",
            "ts_published": "2024-06-01T11:55:00+00:00",
            "ts_processed": "2024-06-01T12:00:00+00:00",
            "source": "filing",
            "tickers": [{"ticker": "JPM", "exchange": "NYSE", "relevance": 0.9}],
            "sentiment": 0.1,
            "sentiment_conf": 0.70,
            "event_type": "Regulatory",
            "event_probs": {"Regulatory": 0.80, "Credit": 0.20},
            "impact": 5.0,
            "impact_range": [4.0, 6.0],
            "confidence": 0.75,
            "novelty": 0.40,
            "diffusion": {"sources": 3, "velocity": 0.5, "score": 0.30},
            "decay_halflife_hours": 72.0,
            "explanation": None,
            "cluster_id": "cluster-xyz",
            "model_versions": {},
        }
        signal = RiskSignal.model_validate(data)
        assert signal.event_id == "v2-validate-test"
        assert signal.source == "filing"

    def test_round_trip_preserves_all_tickers(self):
        tickers = [
            TickerRef(ticker="GS", exchange="NYSE", relevance=1.0),
            TickerRef(ticker="JPM", exchange="NYSE", relevance=0.8),
        ]
        original = RiskSignal(**_valid_kwargs(tickers=tickers))
        restored = RiskSignal.model_validate_json(original.model_dump_json())
        assert len(restored.tickers) == 2
        assert restored.tickers[0].ticker == "GS"
        assert restored.tickers[1].ticker == "JPM"


# ── Immutability ──────────────────────────────────────────────────────────────

class TestImmutability:

    def test_frozen_model_raises_on_attribute_assignment(self):
        signal = RiskSignal(**_valid_kwargs())
        with pytest.raises(Exception):  # ValidationError or TypeError
            signal.sentiment = 0.0  # type: ignore[misc]

    def test_model_copy_with_update_creates_new_instance(self):
        original = RiskSignal(**_valid_kwargs())
        updated = original.model_copy(update={"novelty": 0.99})
        assert updated.novelty == pytest.approx(0.99)
        assert original.novelty != updated.novelty
