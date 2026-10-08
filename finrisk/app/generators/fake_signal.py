"""
Synthetic RiskSignal generator for development and integration testing.

Produces statistically plausible but entirely fictional risk signals for the
20-stock index universe.  The generator is intentionally biased so that:

  - High-impact events (impact ≥ 7) are rarer than moderate ones.
  - Sentiment is correlated with event type (e.g. earnings beats → positive).
  - Multi-ticker events occur ~20 % of the time.
  - Cluster IDs reuse recently seen IDs ~30 % of the time to simulate signal
    deduplication in the NLP engine.

``run_forever`` publishes each signal to Redis (processed.signals) and persists
it to DuckDB.  It is safe to run as a background asyncio Task.
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef
from app.storage.duckdb_store import DuckDBStore
from app.streams.redis_streams import STREAM_PROCESSED_SIGNALS, RedisStreamClient

logger = logging.getLogger(__name__)

# ── Universe ──────────────────────────────────────────────────────────────────

INDEX_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "BAC", "GS",
    "V", "JNJ", "UNH", "PFE", "XOM",
    "CVX", "CAT", "BA", "WMT", "HD",
]

SOURCES: list[str] = ["news", "social", "filing"]
SOURCE_WEIGHTS: list[float] = [0.60, 0.25, 0.15]

EVENT_TYPES: list[str] = [
    "Earnings",
    "M&A",
    "Regulatory",
    "Macro",
    "Geopolitical",
    "Credit",
    "Commodity",
    "Cyber",
    "Leadership",
    "Product",
]

# Sentiment bias per event type: (mean, std)
_EVENT_SENTIMENT_BIAS: dict[str, tuple[float, float]] = {
    "Earnings":     (0.15, 0.45),
    "M&A":          (0.20, 0.35),
    "Regulatory":   (-0.20, 0.40),
    "Macro":        (-0.10, 0.35),
    "Geopolitical": (-0.30, 0.40),
    "Credit":       (-0.25, 0.35),
    "Commodity":    (-0.05, 0.40),
    "Cyber":        (-0.40, 0.30),
    "Leadership":   (0.05, 0.40),
    "Product":      (0.10, 0.45),
}

# Typical half-lives per event type (hours)
_EVENT_HALFLIFE: dict[str, float] = {
    "Earnings":     6.0,
    "M&A":          48.0,
    "Regulatory":   72.0,
    "Macro":        24.0,
    "Geopolitical": 36.0,
    "Credit":       18.0,
    "Commodity":    12.0,
    "Cyber":        8.0,
    "Leadership":   24.0,
    "Product":      10.0,
}

MODEL_VERSIONS: dict[str, str] = {
    "sentiment": "fake-generator-v1.0",
    "classifier": "fake-generator-v1.0",
}

EXPLANATION_TEMPLATES: list[str] = [
    "{ticker} {event_type} event with {sentiment_word} market sentiment "
    "and moderate diffusion across {sources} sources.",
    "Synthetic {event_type} signal for {ticker}: impact estimated at {impact:.1f}/10, "
    "cluster {cluster_id}.",
    "{ticker} flagged for {event_type} — confidence {confidence:.0%}, "
    "novelty {novelty:.0%}.",
]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sentiment_word(s: float) -> str:
    if s > 0.3:
        return "positive"
    if s < -0.3:
        return "negative"
    return "mixed"


def _make_event_probs(primary: str) -> dict[str, float]:
    """
    Build a probability distribution over event types where ``primary`` wins.
    The remaining probability is distributed among a random 2-3 other types.
    Always sums to exactly ≤ 1.0 by clamping the last entry.
    """
    primary_prob = round(random.uniform(0.50, 0.85), 4)
    others = [e for e in EVENT_TYPES if e != primary]
    n_others = random.randint(2, 4)
    other_types = random.sample(others, n_others)
    remainder = 1.0 - primary_prob
    shares = [random.random() for _ in other_types]
    total_shares = sum(shares)
    probs: dict[str, float] = {primary: primary_prob}
    running_sum = primary_prob
    for i, (t, s) in enumerate(zip(other_types, shares)):
        if i < len(other_types) - 1:
            v = round((s / total_shares) * remainder, 4)
            probs[t] = v
            running_sum += v
        else:
            # Last entry absorbs the exact remainder so sum never exceeds 1.0
            last_val = round(1.0 - running_sum, 4)
            probs[t] = max(0.0, last_val)
    return probs


class FakeSignalGenerator:
    """
    Emits synthetic RiskSignal objects.

    Parameters
    ──────────
    db      : DuckDBStore — for persisting each signal.
    redis   : RedisStreamClient — for publishing to processed.signals.
    seed    : Optional RNG seed for deterministic output (testing).
    """

    def __init__(
        self,
        db: DuckDBStore,
        redis: RedisStreamClient,
        seed: int | None = None,
    ) -> None:
        self._db = db
        self._redis = redis
        # Keep a small ring buffer of recent cluster_ids for re-use
        self._recent_clusters: list[str] = []
        self._max_cluster_history = 20
        if seed is not None:
            random.seed(seed)

    # ── Signal construction ───────────────────────────────────────────────────

    def _make_cluster_id(self) -> str:
        """Return an existing cluster ~30 % of the time to simulate deduplication."""
        if self._recent_clusters and random.random() < 0.30:
            return random.choice(self._recent_clusters)
        new_id = f"cluster-{uuid.uuid4().hex[:12]}"
        self._recent_clusters.append(new_id)
        if len(self._recent_clusters) > self._max_cluster_history:
            self._recent_clusters.pop(0)
        return new_id

    def _make_tickers(self) -> list[TickerRef]:
        """Pick 1 ticker 80 % of the time, 2-3 the other 20 %."""
        if random.random() < 0.80:
            count = 1
        else:
            count = random.randint(2, 3)
        chosen = random.sample(INDEX_TICKERS, min(count, len(INDEX_TICKERS)))
        refs = []
        # First ticker gets highest relevance; extras get decreasing relevance
        for i, ticker in enumerate(chosen):
            relevance = _clamp(1.0 - i * random.uniform(0.1, 0.3), 0.1, 1.0)
            refs.append(
                TickerRef(
                    ticker=ticker,
                    exchange="NASDAQ" if ticker in {"AAPL", "MSFT", "NVDA", "GOOGL",
                                                     "AMZN", "META", "TSLA"} else "NYSE",
                    relevance=round(relevance, 3),
                )
            )
        return refs

    def _build_signal(self) -> RiskSignal:
        now = datetime.now(tz=timezone.utc)
        published_offset = timedelta(seconds=random.uniform(1, 300))
        ts_published = now - published_offset
        ts_processed = now

        event_type = random.choice(EVENT_TYPES)
        mean_sent, std_sent = _EVENT_SENTIMENT_BIAS[event_type]
        sentiment = _clamp(random.gauss(mean_sent, std_sent), -1.0, 1.0)
        sentiment_conf = _clamp(random.gauss(0.75, 0.12), 0.40, 0.99)

        event_probs = _make_event_probs(event_type)

        # Impact: log-normal-ish; most events moderate (3-6), rare extremes
        raw_impact = random.betavariate(2.5, 2.5) * 9.0 + 1.0  # beta → [1, 10]
        impact = round(_clamp(raw_impact, 1.0, 10.0), 2)
        half_width = random.uniform(0.3, 1.5)
        impact_low = round(_clamp(impact - half_width, 1.0, impact), 2)
        impact_high = round(_clamp(impact + half_width, impact, 10.0), 2)

        confidence = _clamp(random.gauss(0.70, 0.15), 0.20, 0.99)
        novelty = _clamp(random.betavariate(2.0, 3.0), 0.01, 1.0)

        sources_count = max(1, int(random.lognormvariate(1.5, 0.8)))
        velocity = round(max(0.0, random.expovariate(1 / 3.0)), 2)
        diffusion_score = _clamp(
            (min(sources_count, 50) / 50) * 0.6 + (min(velocity, 10) / 10) * 0.4,
            0.0, 1.0,
        )
        diffusion = DiffusionInfo(
            sources=sources_count,
            velocity=velocity,
            score=round(diffusion_score, 4),
        )

        base_halflife = _EVENT_HALFLIFE[event_type]
        decay_halflife_hours = round(
            _clamp(random.gauss(base_halflife, base_halflife * 0.25), 0.5, 168.0), 2
        )

        tickers = self._make_tickers()
        cluster_id = self._make_cluster_id()
        source = random.choices(SOURCES, weights=SOURCE_WEIGHTS, k=1)[0]

        # Build explanation from a template
        tmpl = random.choice(EXPLANATION_TEMPLATES)
        explanation = tmpl.format(
            ticker=tickers[0].ticker,
            event_type=event_type,
            sentiment_word=_sentiment_word(sentiment),
            sources=sources_count,
            impact=impact,
            cluster_id=cluster_id,
            confidence=confidence,
            novelty=novelty,
        )

        return RiskSignal(
            event_id=str(uuid.uuid4()),
            ts_published=ts_published,
            ts_processed=ts_processed,
            source=source,  # type: ignore[arg-type]
            tickers=tickers,
            sentiment=round(sentiment, 4),
            sentiment_conf=round(sentiment_conf, 4),
            event_type=event_type,
            event_probs=event_probs,
            impact=impact,
            impact_range=(impact_low, impact_high),
            confidence=round(confidence, 4),
            novelty=round(novelty, 4),
            diffusion=diffusion,
            decay_halflife_hours=decay_halflife_hours,
            explanation=explanation,
            cluster_id=cluster_id,
            model_versions=MODEL_VERSIONS,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    async def generate(self) -> AsyncIterator[RiskSignal]:
        """
        Async generator that yields RiskSignal objects on demand.

        The caller controls the rate — this method never sleeps::

            async for signal in generator.generate():
                await do_something(signal)
                if enough:
                    break
        """
        while True:
            yield self._build_signal()

    async def emit_one(self) -> RiskSignal:
        """Build, persist, and publish a single RiskSignal.  Returns the signal."""
        signal = self._build_signal()

        # Persist to DuckDB
        try:
            await self._db.write_signal(signal)
        except Exception as exc:
            logger.error("DuckDB write failed for signal %s: %s", signal.event_id, exc)

        # Publish to Redis stream
        try:
            await self._redis.publish_json_str(
                STREAM_PROCESSED_SIGNALS,
                signal.model_dump_json(),
            )
        except Exception as exc:
            logger.error(
                "Redis publish failed for signal %s: %s", signal.event_id, exc
            )

        return signal

    async def run_forever(self, interval_seconds: float = 5.0) -> None:
        """
        Emit one signal every ``interval_seconds``, forever.

        Designed to run as an asyncio background Task.  Catches all exceptions
        internally so a bad signal never kills the generator loop.
        """
        logger.info(
            "FakeSignalGenerator starting (interval=%.1fs) …", interval_seconds
        )
        while True:
            try:
                signal = await self.emit_one()
                logger.debug(
                    "Emitted signal %s  type=%s  tickers=%s  impact=%.1f",
                    signal.event_id,
                    signal.event_type,
                    [t.ticker for t in signal.tickers],
                    signal.impact,
                )
            except asyncio.CancelledError:
                logger.info("FakeSignalGenerator cancelled — shutting down.")
                raise
            except Exception as exc:
                logger.exception("Unexpected error in FakeSignalGenerator: %s", exc)

            await asyncio.sleep(interval_seconds)


# ── Standalone entry point ────────────────────────────────────────────────────

async def _main() -> None:
    """Run the generator standalone (without the full API server)."""
    import os
    from pathlib import Path

    from app.config import get_settings

    settings = get_settings()
    db = DuckDBStore(settings.duckdb_path)
    await db.initialise()

    redis = RedisStreamClient(settings.redis_url)
    await redis.connect()

    gen = FakeSignalGenerator(db=db, redis=redis)
    try:
        await gen.run_forever(settings.fake_signal_interval_seconds)
    finally:
        await redis.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(_main())
