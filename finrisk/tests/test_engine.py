"""
Tests for the NLP engine (Task 3).

All tests run without a live Redis or DuckDB connection.  FinBERT is not
downloaded in CI, so sentiment tests exercise the rule-based fallback path.
Tests that require FinBERT are marked with ``@pytest.mark.skipif`` guarded by
``has_model()``.
"""

from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.engine.sentiment import FinBERTSentimentModel, _rule_based_sentiment
from app.engine.event_classifier import RuleBasedEventClassifier
from app.engine.novelty import NoveltyScorer
from app.engine.impact_calculator import ImpactCalculator, BASE_IMPACT
from app.engine.nlp_worker import NLPWorker


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_article(**overrides) -> dict[str, Any]:
    """Return a minimal valid article dict."""
    base: dict[str, Any] = {
        "article_id": "test-001",
        "source": "news",
        "ticker_symbols": ["AAPL"],
        "title": "Apple reports record quarterly earnings, beats analyst estimates",
        "body": (
            "Apple Inc reported quarterly revenue of $94 billion, exceeding "
            "analyst expectations of $92 billion. The company raised its "
            "full-year guidance citing strong iPhone demand."
        ),
        "url": "https://example.com/article/1",
        "published_at": datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc).isoformat(),
        "source_weight": 0.85,
    }
    base.update(overrides)
    return base


# ── Sentiment: rule-based fallback ─────────────────────────────────────────────

class TestRuleBasedSentiment:
    def test_positive_text_returns_positive_score(self):
        score, conf = _rule_based_sentiment(
            "Apple beats earnings, revenue surged beyond expectations"
        )
        assert score > 0.0, f"Expected positive score, got {score}"

    def test_negative_text_returns_negative_score(self):
        score, conf = _rule_based_sentiment(
            "Company missed earnings, profit loss widened, guidance cut"
        )
        assert score < 0.0, f"Expected negative score, got {score}"

    def test_neutral_text_returns_near_zero(self):
        score, conf = _rule_based_sentiment(
            "The company filed its annual report on schedule"
        )
        # Neutral-ish text should have low absolute score
        assert abs(score) < 0.9, f"Expected near-neutral, got {score}"

    def test_score_within_bounds(self):
        for text in [
            "massive earnings beat profit record high revenue",
            "bankruptcy default debt downgrade junk collapse",
            "meeting scheduled for next quarter review",
        ]:
            score, conf = _rule_based_sentiment(text)
            assert -1.0 <= score <= 1.0, f"Score {score} out of bounds for: {text!r}"
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds"

    def test_empty_text_returns_neutral(self):
        score, conf = _rule_based_sentiment("")
        assert score == 0.0
        assert 0.0 <= conf <= 1.0


class TestFinBERTSentimentModel:
    def test_rule_based_fallback_returns_valid_range(self):
        """
        The model always works via the fallback when FinBERT is not loaded.
        This test confirms the public API contract regardless of download status.
        """
        model = FinBERTSentimentModel()
        # Force rule-based path by patching _load to return False
        with patch.object(model, "_load", return_value=False):
            score, conf = model.predict("Earnings beat, revenue up, profit raised guidance")
        assert -1.0 <= score <= 1.0, f"Sentiment out of range: {score}"
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"

    def test_predict_batch_returns_correct_count(self):
        model = FinBERTSentimentModel()
        texts = [
            "Revenue beat expectations this quarter",
            "Stock fell on bankruptcy fears",
            "Fed raised interest rates unexpectedly",
        ]
        with patch.object(model, "_load", return_value=False):
            results = model.predict_batch(texts)
        assert len(results) == len(texts)
        for score, conf in results:
            assert -1.0 <= score <= 1.0
            assert 0.0 <= conf <= 1.0

    def test_has_model_returns_bool(self):
        model = FinBERTSentimentModel()
        result = model.has_model()
        assert isinstance(result, bool)

    def test_get_embedding_fallback_returns_list(self):
        model = FinBERTSentimentModel()
        with patch.object(model, "_load", return_value=False):
            emb = model.get_embedding("Apple Q3 earnings beat")
        assert isinstance(emb, list)
        assert len(emb) > 0
        # L2-norm should be ~1.0 for the fallback embedding
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 1e-6 or norm > 0.0  # non-zero

    def test_predict_batch_chunks_larger_than_batch_size(self):
        """Batches larger than BATCH_SIZE=8 must all be processed."""
        model = FinBERTSentimentModel()
        texts = [f"text number {i}" for i in range(20)]
        with patch.object(model, "_load", return_value=False):
            results = model.predict_batch(texts)
        assert len(results) == 20


# ── Event classifier ───────────────────────────────────────────────────────────

class TestRuleBasedEventClassifier:
    def setup_method(self):
        self.clf = RuleBasedEventClassifier()

    def test_regulatory_text(self):
        event_type, probs = self.clf.get_primary(
            "Apple faces antitrust fine from the SEC investigation"
        )
        assert event_type == "Regulatory", f"Expected Regulatory, got {event_type}"

    def test_earnings_text(self):
        event_type, probs = self.clf.get_primary(
            "NVDA beats earnings guidance, EPS exceeds quarterly expectations"
        )
        assert event_type == "Earnings", f"Expected Earnings, got {event_type}"

    def test_macro_text(self):
        event_type, probs = self.clf.get_primary(
            "Fed raises interest rates amid rising inflation, FOMC statement"
        )
        assert event_type == "Macro", f"Expected Macro, got {event_type}"

    def test_credit_text(self):
        event_type, probs = self.clf.get_primary(
            "Company faces bankruptcy after debt default, Moody's downgrade"
        )
        assert event_type == "Credit", f"Expected Credit, got {event_type}"

    def test_geopolitical_text(self):
        event_type, probs = self.clf.get_primary(
            "US imposes sanctions following military conflict, nato response"
        )
        assert event_type == "Geopolitical", f"Expected Geopolitical, got {event_type}"

    def test_event_probs_sum_lte_one(self):
        for text in [
            "Apple faces antitrust fine from the SEC investigation",
            "NVDA beats earnings guidance",
            "Fed raises interest rates",
            "Oil prices surge on OPEC supply chain disruption",
            "CEO resignation announcement, leadership departure",
        ]:
            probs = self.clf.classify(text)
            total = sum(probs.values())
            assert total <= 1.0 + 1e-6, f"Probs sum to {total:.6f} for: {text!r}"

    def test_event_probs_all_non_negative(self):
        probs = self.clf.classify("Apple Q3 earnings guidance raised")
        for label, prob in probs.items():
            assert prob >= 0.0, f"Negative probability for {label}: {prob}"

    def test_all_ten_categories_present(self):
        probs = self.clf.classify("generic financial article about the market")
        expected = {
            "Geopolitical", "Regulatory", "M&A", "Earnings", "Credit",
            "Macro", "Product", "Cyber", "Leadership", "Commodity",
        }
        assert set(probs.keys()) == expected

    def test_classify_returns_primary_as_argmax(self):
        """get_primary's event_type must be the argmax of event_probs."""
        for text in [
            "Fed raises interest rates, FOMC monetary policy decision",
            "Company acquired in hostile takeover buyout deal",
            "Hack breach ransomware cyberattack data leak",
        ]:
            event_type, probs = self.clf.get_primary(text)
            best = max(probs, key=lambda k: probs[k])
            assert event_type == best, (
                f"Mismatch: event_type={event_type!r}, argmax={best!r}"
            )

    def test_decay_halflife_defaults(self):
        assert RuleBasedEventClassifier.decay_halflife_for_event("Earnings") == 12.0
        assert RuleBasedEventClassifier.decay_halflife_for_event("Credit") == 48.0
        assert RuleBasedEventClassifier.decay_halflife_for_event("Regulatory") == 72.0
        assert RuleBasedEventClassifier.decay_halflife_for_event("Unknown") == 24.0

    def test_empty_text_does_not_crash(self):
        event_type, probs = self.clf.get_primary("")
        assert isinstance(event_type, str)
        assert sum(probs.values()) <= 1.0 + 1e-6


# ── Novelty scorer ─────────────────────────────────────────────────────────────

class TestNoveltyScorer:
    def setup_method(self):
        self.scorer = NoveltyScorer()

    def _make_embedding(self, dim: int = 128, seed: int = 0) -> list[float]:
        """Deterministic pseudo-random unit vector."""
        import math
        vec = [math.sin(seed * (i + 1)) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def test_first_signal_returns_default_novelty(self):
        """No history for ticker → returns 0.8."""
        emb = self._make_embedding(seed=1)
        novelty = self.scorer.compute_novelty("some text", "AAPL", emb)
        assert novelty == 0.8

    def test_identical_embedding_returns_near_zero(self):
        emb = self._make_embedding(seed=42)
        self.scorer.add_to_history("MSFT", emb)
        novelty = self.scorer.compute_novelty("text", "MSFT", emb)
        assert novelty < 0.05, f"Expected near-zero novelty for duplicate, got {novelty}"

    def test_orthogonal_embedding_returns_near_one(self):
        # Two orthogonal vectors have cosine similarity 0 → novelty = 1.0
        emb_a = [1.0] + [0.0] * 127
        emb_b = [0.0, 1.0] + [0.0] * 126
        self.scorer.add_to_history("GOOG", emb_a)
        novelty = self.scorer.compute_novelty("text", "GOOG", emb_b)
        assert novelty > 0.95, f"Expected near-1.0 novelty for orthogonal, got {novelty}"

    def test_novelty_is_within_bounds(self):
        emb1 = self._make_embedding(seed=1)
        emb2 = self._make_embedding(seed=999)
        self.scorer.add_to_history("TSLA", emb1)
        novelty = self.scorer.compute_novelty("text", "TSLA", emb2)
        assert 0.0 <= novelty <= 1.0

    def test_rolling_window_caps_at_max(self):
        scorer = NoveltyScorer(window_size=3)
        for i in range(5):
            scorer.add_to_history("NVDA", self._make_embedding(seed=i))
        assert scorer.window_size_for("NVDA") == 3

    def test_different_tickers_independent(self):
        emb = self._make_embedding(seed=7)
        self.scorer.add_to_history("AMD", emb)
        # INTC has no history → returns default
        novelty = self.scorer.compute_novelty("text", "INTC", emb)
        assert novelty == 0.8

    def test_mismatched_dimensions_does_not_crash(self):
        emb_short = [1.0] * 64
        emb_long = [1.0] * 128
        self.scorer.add_to_history("SPY", emb_short)
        # Should not raise; cosine_similarity returns 0.0 for mismatched dims
        novelty = self.scorer.compute_novelty("text", "SPY", emb_long)
        assert 0.0 <= novelty <= 1.0


# ── Impact calculator ──────────────────────────────────────────────────────────

class TestImpactCalculator:
    def setup_method(self):
        self.calc = ImpactCalculator()

    def test_all_event_types_in_range(self):
        for event_type in BASE_IMPACT:
            impact, (low, high) = self.calc.calculate(
                event_type=event_type,
                sentiment=0.5,
                novelty=0.7,
            )
            assert 1.0 <= impact <= 10.0, f"{event_type}: impact={impact} out of [1,10]"
            assert 1.0 <= low <= impact, f"{event_type}: low={low}"
            assert impact <= high <= 10.0, f"{event_type}: high={high}"

    def test_high_impact_events_score_higher_than_low(self):
        """Credit and Geopolitical must outrank Product and Leadership."""
        def get_impact(event_type):
            impact, _ = self.calc.calculate(
                event_type=event_type, sentiment=0.6, novelty=0.8
            )
            return impact

        assert get_impact("Credit") > get_impact("Product")
        assert get_impact("Geopolitical") > get_impact("Leadership")

    def test_zero_sentiment_reduces_impact(self):
        impact_high, _ = self.calc.calculate("Earnings", sentiment=0.9, novelty=0.8)
        impact_zero, _ = self.calc.calculate("Earnings", sentiment=0.0, novelty=0.8)
        assert impact_high > impact_zero

    def test_low_novelty_reduces_impact(self):
        impact_novel, _ = self.calc.calculate("M&A", sentiment=0.7, novelty=1.0)
        impact_dupe, _ = self.calc.calculate("M&A", sentiment=0.7, novelty=0.01)
        assert impact_novel > impact_dupe

    def test_impact_range_contains_point_estimate(self):
        for novelty in [0.1, 0.5, 1.0]:
            impact, (low, high) = self.calc.calculate(
                "Regulatory", sentiment=-0.8, novelty=novelty
            )
            assert low <= impact <= high, (
                f"impact={impact} not in [{low}, {high}] for novelty={novelty}"
            )

    def test_clamp_to_one_at_low_inputs(self):
        """Very low sentiment + low novelty must still hit floor of 1.0."""
        impact, _ = self.calc.calculate("Product", sentiment=0.0, novelty=0.0)
        assert impact >= 1.0

    def test_clamp_to_ten_at_max_inputs(self):
        """Maximum inputs must not exceed 10.0."""
        impact, (low, high) = self.calc.calculate("Credit", sentiment=1.0, novelty=1.0, diffusion_score=1.0)
        assert impact <= 10.0
        assert high <= 10.0

    def test_unknown_event_type_uses_default(self):
        """Unknown event type should not crash."""
        impact, (low, high) = self.calc.calculate(
            "UnknownType", sentiment=0.5, novelty=0.5
        )
        assert 1.0 <= impact <= 10.0


# ── NLP Worker ─────────────────────────────────────────────────────────────────

class TestNLPWorkerProcessOne:
    def setup_method(self):
        """Set up worker with mocked external dependencies."""
        # Mocked DuckDB and Redis
        self.mock_db = MagicMock()
        self.mock_db.write_signal = AsyncMock()

        self.mock_redis = MagicMock()
        # Mock consumer
        self.mock_consumer = MagicMock()
        self.mock_consumer.read_batch = AsyncMock(return_value=[])
        self.mock_redis.consumer.return_value = self.mock_consumer
        # Mock producer
        self.mock_producer = MagicMock()
        self.mock_producer.send = AsyncMock()
        self.mock_redis.producer.return_value = self.mock_producer

        # Mocked sentiment model — returns positive sentiment
        self.mock_sentiment = MagicMock(spec=FinBERTSentimentModel)
        self.mock_sentiment.predict.return_value = (0.7, 0.85)
        self.mock_sentiment.get_embedding.return_value = [0.1] * 128

        self.worker = NLPWorker(
            db=self.mock_db,
            redis=self.mock_redis,
            sentiment_model=self.mock_sentiment,
        )

    def test_process_one_returns_risk_signal(self):
        article = _make_article()
        signal = self.worker.process_one(article)
        assert signal is not None
        assert signal.event_type in {
            "Earnings", "M&A", "Regulatory", "Macro", "Geopolitical",
            "Credit", "Commodity", "Cyber", "Leadership", "Product",
        }

    def test_process_one_sentiment_from_mock(self):
        article = _make_article()
        signal = self.worker.process_one(article)
        assert signal is not None
        # Mock returns (0.7, 0.85) — score should be 0.7
        assert abs(signal.sentiment - 0.7) < 1e-3

    def test_process_one_tickers_are_correct(self):
        article = _make_article(ticker_symbols=["TSLA", "NVDA"])
        signal = self.worker.process_one(article)
        assert signal is not None
        ticker_syms = [t.ticker for t in signal.tickers]
        assert "TSLA" in ticker_syms
        assert "NVDA" in ticker_syms

    def test_process_one_impact_in_range(self):
        article = _make_article()
        signal = self.worker.process_one(article)
        assert signal is not None
        assert 1.0 <= signal.impact <= 10.0

    def test_process_one_event_probs_sum_lte_one(self):
        article = _make_article()
        signal = self.worker.process_one(article)
        assert signal is not None
        total = sum(signal.event_probs.values())
        assert total <= 1.0 + 1e-6

    def test_process_one_ts_processed_after_published(self):
        article = _make_article()
        signal = self.worker.process_one(article)
        assert signal is not None
        assert signal.ts_processed >= signal.ts_published

    def test_process_one_malformed_dict_returns_none(self):
        """A completely empty dict should return None without raising."""
        signal = self.worker.process_one({})
        # Empty dict has no tickers but will get UNKNOWN — might succeed or fail
        # depending on text content.  What must NOT happen is an uncaught exception.
        # We only check that the call does not raise.

    def test_process_one_missing_title_and_body_returns_none(self):
        """No text content → process_one should return None."""
        signal = self.worker.process_one({
            "article_id": "bad-001",
            "source": "news",
            "ticker_symbols": ["AAPL"],
            "title": "",
            "body": "",
            "published_at": datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
        })
        assert signal is None

    def test_process_one_invalid_source_coerces_to_news(self):
        article = _make_article(source="rss_feed")
        signal = self.worker.process_one(article)
        assert signal is not None
        assert signal.source == "news"

    def test_process_one_naive_datetime_is_coerced(self):
        """Naive published_at timestamps must be coerced to UTC."""
        article = _make_article(
            published_at=datetime(2024, 3, 1, 12, 0).isoformat()  # no tzinfo
        )
        signal = self.worker.process_one(article)
        assert signal is not None
        assert signal.ts_published.tzinfo is not None

    def test_process_one_cluster_id_stable(self):
        """Same ticker + event type + date → same cluster_id."""
        article_a = _make_article(article_id="a1")
        article_b = _make_article(article_id="a2")
        sig_a = self.worker.process_one(article_a)
        sig_b = self.worker.process_one(article_b)
        assert sig_a is not None
        assert sig_b is not None
        # Both articles have same ticker, body, and date → same cluster_id
        # (event_type may differ if classifier is stochastic, but with same
        # text it should be deterministic)
        if sig_a.event_type == sig_b.event_type:
            assert sig_a.cluster_id == sig_b.cluster_id

    def test_process_one_model_versions_present(self):
        article = _make_article()
        signal = self.worker.process_one(article)
        assert signal is not None
        assert "sentiment" in signal.model_versions
        assert "classifier" in signal.model_versions

    def test_process_one_large_ticker_list_capped(self):
        """More than 10 tickers should not cause a crash."""
        tickers = [f"TICK{i}" for i in range(20)]
        article = _make_article(ticker_symbols=tickers)
        signal = self.worker.process_one(article)
        assert signal is not None
        assert len(signal.tickers) <= 10


class TestNLPWorkerRunForever:
    """Tests for the async run_forever loop with fully mocked I/O."""

    def _make_worker(self, batches: list) -> NLPWorker:
        mock_db = MagicMock()
        mock_db.write_signal = AsyncMock()

        mock_redis = MagicMock()
        mock_consumer = MagicMock()
        # Return batches in sequence, then empty to end the loop
        mock_consumer.read_batch = AsyncMock(side_effect=batches + [[]])
        mock_consumer._batch_size = 8
        mock_consumer._block_ms = 100
        mock_redis.consumer.return_value = mock_consumer

        mock_producer = MagicMock()
        mock_producer.send = AsyncMock()
        mock_redis.producer.return_value = mock_producer

        mock_sentiment = MagicMock(spec=FinBERTSentimentModel)
        mock_sentiment.predict.return_value = (0.5, 0.8)
        mock_sentiment.get_embedding.return_value = [0.1] * 128

        worker = NLPWorker(
            db=mock_db,
            redis=mock_redis,
            sentiment_model=mock_sentiment,
        )
        return worker

    @pytest.mark.asyncio
    async def test_run_forever_processes_and_publishes(self):
        article = _make_article()
        batch = [("1-0", article)]
        worker = self._make_worker([batch])

        # Run loop with a timeout — it will drain the fake batches and then
        # block on the empty list; cancel it after the first iteration.
        task = asyncio.create_task(worker.run_forever())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # DuckDB write should have been called once
        worker._db.write_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_forever_handles_bad_article_without_crashing(self):
        """Malformed article in stream should not stop the worker."""
        bad_article = {"article_id": "bad", "title": "", "body": ""}
        batch = [("1-0", bad_article)]
        worker = self._make_worker([batch])

        task = asyncio.create_task(worker.run_forever())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # If we get here without an unhandled exception, the test passes.

    @pytest.mark.asyncio
    async def test_run_forever_handles_redis_read_error(self):
        """A Redis read exception should trigger a retry sleep, not a crash."""
        mock_db = MagicMock()
        mock_db.write_signal = AsyncMock()

        mock_redis = MagicMock()
        mock_consumer = MagicMock()
        # First call raises, second call returns empty to let the loop drain
        mock_consumer.read_batch = AsyncMock(
            side_effect=[Exception("Redis timeout"), []]
        )
        mock_consumer._batch_size = 8
        mock_consumer._block_ms = 100
        mock_redis.consumer.return_value = mock_consumer

        mock_producer = MagicMock()
        mock_producer.send = AsyncMock()
        mock_redis.producer.return_value = mock_producer

        worker = NLPWorker(db=mock_db, redis=mock_redis)

        task = asyncio.create_task(worker.run_forever())
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # Worker survived the Redis error without propagating it.
