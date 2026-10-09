"""
NLP worker — consumes raw.articles, produces processed.signals.

Processing pipeline for each article:
  1. Parse JSON dict back to a RawDocument-compatible structure.
  2. Run FinBERT sentiment on (title + first 512 chars of body).
  3. Extract a FinBERT embedding for novelty comparison.
  4. Run the event classifier on (title + body).
     Classifier selection is controlled by the EVENT_CLASSIFIER env var:
       "deberta" → DeBERTaEventClassifier (with rule-based fallback)
       default   → RuleBasedEventClassifier
  5. Compute novelty against per-ticker history.
  6. Record article arrival in DiffusionScorer; compute L1+L2 DiffusionInfo.
  7. Compute impact via ImpactCalculator.
  8. Build a RiskSignal.
  9. Persist to DuckDB.
  10. Publish to processed.signals.

Graceful degradation
────────────────────
If FinBERT fails or is not loaded, the rule-based sentiment scorer in
``sentiment.py`` is used transparently.  The worker never crashes on a single
bad article — errors are logged and the message is skipped.

Cluster ID
──────────
cluster_id = SHA256(ticker + "|" + event_type + "|" + date_str)[:16]
This groups same-event same-ticker signals within the same calendar day.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from app.engine.sentiment import FinBERTSentimentModel
from app.engine.event_classifier import RuleBasedEventClassifier
from app.engine.deberta_classifier import DeBERTaEventClassifier
from app.engine.novelty import NoveltyScorer
from app.engine.impact_calculator import ImpactCalculator
from app.engine.cluster_buffer import ClusterSentimentBuffer
from app.diffusion.diffusion_scorer import DiffusionScorer
from app.schema.risk_signal import RiskSignal, TickerRef, DiffusionInfo
from app.streams.redis_streams import (
    RedisStreamClient,
    STREAM_RAW_ARTICLES,
    STREAM_PROCESSED_SIGNALS,
)
from app.storage.duckdb_store import DuckDBStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_BATCH_SIZE = 8
_CONSUMER_BLOCK_MS = 2_000
_RETRY_SLEEP_SECONDS = 1.0


def _make_cluster_id(ticker: str, event_type: str, ts: datetime) -> str:
    """
    Stable intra-day cluster identifier for (ticker, event_type, date).

    Signals about the same underlying event on the same day for the same
    ticker will share this ID regardless of which article triggered them.
    """
    date_str = ts.strftime("%Y-%m-%d")
    raw = f"{ticker}|{event_type}|{date_str}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _build_input_text(article_dict: dict) -> str:
    """Combine title and first 512 chars of body for NLP input."""
    title = article_dict.get("title", "") or ""
    body = article_dict.get("body", "") or ""
    combined = f"{title}. {body[:512]}"
    return combined.strip(". ")


def _extract_tickers(article_dict: dict) -> list[str]:
    """Extract ticker list from article dict, normalising to uppercase."""
    raw = article_dict.get("ticker_symbols", []) or article_dict.get("tickers", [])
    if isinstance(raw, str):
        raw = [t.strip() for t in raw.split(",") if t.strip()]
    return [str(t).upper().strip() for t in raw if t]


def _parse_published_at(article_dict: dict) -> datetime:
    """
    Parse the published_at field from the article dict.

    Accepts datetime objects, ISO strings, and Unix timestamps.
    Falls back to utcnow() if parsing fails.
    """
    val = article_dict.get("published_at")
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val
    if isinstance(val, str):
        try:
            dt = datetime.fromisoformat(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            pass
    if isinstance(val, (int, float)):
        try:
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            pass
    logger.warning("Could not parse published_at=%r — using utcnow()", val)
    return datetime.now(tz=timezone.utc)


class NLPWorker:
    """
    Async consumer that transforms raw articles into RiskSignals.

    ``run_forever()`` is designed to be run as an asyncio task.  It never
    exits under normal operation; callers cancel it to stop it.

    Classifier selection
    ────────────────────
    Set the EVENT_CLASSIFIER environment variable (or settings.event_classifier)
    to "deberta" to use the fine-tuned DeBERTa model.  The rule-based classifier
    is always constructed and used as a fallback for DeBERTa.
    """

    MODEL_VERSION = "nlp-engine-v1"
    SENTIMENT_VERSION = "finbert-prosus-v1"
    CLASSIFIER_VERSION = "rule-based-v1"
    DEBERTA_CLASSIFIER_VERSION = "deberta-v3-base-finetuned-v1"

    def __init__(
        self,
        db: DuckDBStore,
        redis: RedisStreamClient,
        *,
        sentiment_model: FinBERTSentimentModel | None = None,
        event_classifier: RuleBasedEventClassifier | None = None,
        novelty_scorer: NoveltyScorer | None = None,
        impact_calculator: ImpactCalculator | None = None,
    ) -> None:
        self._db = db
        self._redis = redis
        # Allow injection for testing; construct defaults otherwise.
        self._sentiment = sentiment_model or FinBERTSentimentModel()
        self._classifier = event_classifier or RuleBasedEventClassifier()
        self._novelty = novelty_scorer or NoveltyScorer()
        self._impact = impact_calculator or ImpactCalculator()

        # ── DiffusionScorer ───────────────────────────────────────────────────
        self._diffusion_scorer = DiffusionScorer()

        # ── Cluster sentiment buffer ──────────────────────────────────────────
        self._cluster_buffer = ClusterSentimentBuffer(max_age_hours=2.0)
        self._process_count: int = 0

        # ── Active classifier selection ───────────────────────────────────────
        # Check EVENT_CLASSIFIER env var; fall back to settings if available.
        classifier_choice = os.environ.get("EVENT_CLASSIFIER", "").lower()
        if not classifier_choice:
            try:
                from app.config import get_settings
                settings = get_settings()
                classifier_choice = getattr(settings, "event_classifier", "").lower()
            except Exception:
                classifier_choice = ""

        if classifier_choice == "deberta":
            deberta_model_path: Path | None = None
            try:
                from app.config import get_settings
                settings = get_settings()
                raw_path = getattr(settings, "deberta_model_path", None)
                if raw_path:
                    deberta_model_path = Path(raw_path)
            except Exception:
                pass

            self._active_classifier = DeBERTaEventClassifier(
                model_path=deberta_model_path,
                fallback_classifier=self._classifier,
            )
            logger.info(
                "NLPWorker: using DeBERTaEventClassifier "
                "(loaded=%s, fallback=RuleBasedEventClassifier)",
                self._active_classifier.is_available(),
            )
        else:
            self._active_classifier = self._classifier
            logger.info("NLPWorker: using RuleBasedEventClassifier")

    def process_one(self, article_dict: dict) -> RiskSignal | None:
        """
        Process a single article dict and return a RiskSignal.

        Returns None if the article is malformed or processing fails.
        Intentionally synchronous so it can be called from tests without an
        event loop.
        """
        try:
            return self._process_article(article_dict)
        except Exception as exc:
            article_id = article_dict.get("article_id", "<unknown>")
            logger.error(
                "Failed to process article %s: %s", article_id, exc, exc_info=True
            )
            return None

    def _process_article(self, article_dict: dict) -> RiskSignal:
        """Core processing logic — raises on any failure."""
        # ── 1. Extract fields ─────────────────────────────────────────────────
        article_id: str = str(article_dict.get("article_id", uuid.uuid4()))
        source = article_dict.get("source", "news")
        if source not in ("news", "social", "filing"):
            source = "news"

        tickers = _extract_tickers(article_dict)
        if not tickers:
            # Use a generic ticker if none resolved — signals must have ≥ 1
            tickers = ["UNKNOWN"]

        ts_published = _parse_published_at(article_dict)
        ts_processed = datetime.now(tz=timezone.utc)

        # ── 2. Build NLP input text ───────────────────────────────────────────
        input_text = _build_input_text(article_dict)
        if not input_text:
            raise ValueError("Empty text — cannot process article without content")

        # ── 3. Sentiment ──────────────────────────────────────────────────────
        sentiment_score, sentiment_conf = self._sentiment.predict(input_text)

        # ── 4. Embedding for novelty ──────────────────────────────────────────
        embedding = self._sentiment.get_embedding(input_text)

        # ── 5. Event classification ───────────────────────────────────────────
        full_text = (
            f"{article_dict.get('title', '')} {article_dict.get('body', '')}"
        ).strip()
        event_type, event_probs = self._active_classifier.get_primary(
            full_text or input_text
        )

        # ── 6. Novelty (use the primary ticker for history lookups) ───────────
        primary_ticker = tickers[0]
        novelty = self._novelty.compute_novelty(input_text, primary_ticker, embedding)
        self._novelty.add_to_history(primary_ticker, embedding)

        # ── 7. Cluster ID (needed before diffusion record) ────────────────────
        cluster_id = _make_cluster_id(primary_ticker, event_type, ts_published)

        # ── 7b. Cluster buffer: add this article and get aggregate ────────────
        source_weight = float(article_dict.get("source_weight", 0.85))
        self._cluster_buffer.add(
            cluster_id, sentiment_score, sentiment_conf, source_weight, ts_published
        )
        agg_sentiment, agg_conf = self._cluster_buffer.get_aggregate(cluster_id)
        cluster_source_count = self._cluster_buffer.get_source_count(cluster_id)

        # Periodic cleanup (~every 100 articles)
        self._process_count += 1
        if self._process_count % 100 == 0:
            self._cluster_buffer.cleanup_old()

        # ── 8. Diffusion (L1 + L2 via DiffusionScorer) ───────────────────────
        self._diffusion_scorer.record_arrival(cluster_id, ts_published)
        diffusion = self._diffusion_scorer.compute(
            cluster_id, source_count=cluster_source_count
        )

        # ── 9. Impact ─────────────────────────────────────────────────────────
        impact, impact_range = self._impact.calculate(
            event_type=event_type,
            sentiment=agg_sentiment,
            novelty=novelty,
            diffusion_score=diffusion.score,
        )

        # ── 10. Decay half-life ───────────────────────────────────────────────
        decay_halflife = RuleBasedEventClassifier.decay_halflife_for_event(event_type)

        # ── 11. Confidence ────────────────────────────────────────────────────
        # Combine aggregated sentiment confidence, novelty, and source weight
        confidence = round(
            0.5 * agg_conf + 0.3 * novelty + 0.2 * source_weight, 4
        )
        confidence = max(0.0, min(1.0, confidence))

        # ── 12. Build TickerRefs ───────────────────────────────────────────────
        ticker_refs = [
            TickerRef(ticker=t, relevance=1.0 if i == 0 else 0.6)
            for i, t in enumerate(tickers[:10])  # cap at 10 tickers
        ]

        # ── 13. Explanation snippet ───────────────────────────────────────────
        title = article_dict.get("title", "")
        explanation = (
            f"[{event_type}] {title[:200]}" if title else None
        )

        # ── 14. Build RiskSignal ──────────────────────────────────────────────
        # Determine which classifier version tag to record
        if isinstance(self._active_classifier, DeBERTaEventClassifier):
            classifier_version = self.DEBERTA_CLASSIFIER_VERSION
        else:
            classifier_version = self.CLASSIFIER_VERSION

        signal = RiskSignal(
            event_id=str(uuid.uuid4()),
            ts_published=ts_published,
            ts_processed=ts_processed,
            source=source,
            tickers=ticker_refs,
            sentiment=round(float(agg_sentiment), 4),
            sentiment_conf=round(float(agg_conf), 4),
            event_type=event_type,
            event_probs=event_probs,
            impact=round(impact, 4),
            impact_range=(round(impact_range[0], 4), round(impact_range[1], 4)),
            confidence=confidence,
            novelty=round(float(novelty), 4),
            diffusion=diffusion,
            decay_halflife_hours=decay_halflife,
            explanation=explanation,
            cluster_id=cluster_id,
            model_versions={
                "sentiment": self.SENTIMENT_VERSION,
                "classifier": classifier_version,
                "engine": self.MODEL_VERSION,
            },
        )
        return signal

    async def run_forever(self) -> None:
        """
        Main consumer loop.

        Reads from the raw.articles stream in batches of 8, processes each
        article, persists to DuckDB, and publishes to processed.signals.
        Never exits under normal operation; cancel the task to stop it.
        """
        consumer = self._redis.consumer(
            STREAM_RAW_ARTICLES,
            last_id="$",
        )
        consumer._batch_size = _BATCH_SIZE
        consumer._block_ms = _CONSUMER_BLOCK_MS

        producer = self._redis.producer(STREAM_PROCESSED_SIGNALS)

        logger.info("NLPWorker started — consuming %s", STREAM_RAW_ARTICLES)

        while True:
            try:
                batch = await consumer.read_batch()
            except asyncio.CancelledError:
                logger.info("NLPWorker cancelled — shutting down")
                return
            except Exception as exc:
                logger.error("Error reading from stream: %s", exc)
                await asyncio.sleep(_RETRY_SLEEP_SECONDS)
                continue

            for msg_id, payload in batch:
                # The stream may carry the article as a nested "data" key
                # (already decoded by StreamConsumer) or as a flat dict.
                article_dict = payload

                signal = self.process_one(article_dict)
                if signal is None:
                    continue

                # Persist to DuckDB
                try:
                    await self._db.write_signal(signal)
                except Exception as exc:
                    logger.error(
                        "DuckDB write failed for event %s: %s",
                        signal.event_id, exc,
                    )

                # Publish to processed.signals
                try:
                    await producer.send(json.loads(signal.model_dump_json()))
                    logger.debug(
                        "Published signal %s (%s) → %s",
                        signal.event_id, signal.event_type, STREAM_PROCESSED_SIGNALS,
                    )
                except Exception as exc:
                    logger.error(
                        "Redis publish failed for event %s: %s",
                        signal.event_id, exc,
                    )
