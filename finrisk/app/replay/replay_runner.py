"""
ReplayRunner — offline demo mode.

Reads pre-cached articles from data/sample/articles.jsonl and publishes them
to the raw.articles Redis stream at accelerated speed.  When the file doesn't
exist it generates 200 synthetic articles covering all event types and the
FinRisk-20 ticker universe, saves them for future runs, then starts replaying.

The loop restarts from the beginning after every pass (with a 30-second pause)
so a demo can run indefinitely without a live internet connection.

Replay timing
─────────────
Articles are published in the order they appear in the file.  The sleep
between consecutive articles is:

    sleep = (delta_ts_seconds) / speed_multiplier

where delta_ts is the difference in published_at between adjacent articles.
At speed_multiplier=10 a 1-hour dataset replays in 6 minutes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.streams.redis_streams import RedisStreamClient

logger = logging.getLogger(__name__)

# ── Ticker universe (FinRisk-20) ──────────────────────────────────────────────
_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "BRK-B", "JPM", "V",
    "JNJ", "WMT", "PG", "MA", "UNH",
    "HD", "BAC", "XOM", "DIS", "NFLX",
]

# ── Event types (must match RuleBasedEventClassifier labels) ──────────────────
_EVENT_TYPES = [
    "Earnings", "M&A", "Regulatory", "Product Launch",
    "Executive Change", "Macroeconomic", "Legal", "ESG",
    "Geopolitical", "Analyst Rating",
]

# ── Headline templates keyed by event type ────────────────────────────────────
_HEADLINES: dict[str, list[str]] = {
    "Earnings": [
        "{ticker} Q3 earnings beat consensus by 12%, raises FY guidance",
        "{ticker} reports quarterly revenue miss on slowing consumer demand",
        "{ticker} posts record EPS of $4.82, shares surge in after-hours",
        "{ticker} earnings disappoint: gross margin contracts 200 bps",
        "{ticker} raises annual revenue outlook following strong earnings",
    ],
    "M&A": [
        "{ticker} to acquire rival in $8.5B all-stock deal",
        "{ticker} exploring strategic alternatives including potential sale",
        "{ticker} completes divestiture of non-core business unit for $2.1B",
        "{ticker} merger talks with European partner collapse over valuation",
        "Private equity consortium bids for {ticker} at 30% premium",
    ],
    "Regulatory": [
        "EU antitrust regulators open formal probe into {ticker} practices",
        "{ticker} fined $450M by FTC over data privacy violations",
        "SEC launches investigation into {ticker} accounting disclosures",
        "{ticker} wins regulatory approval for new product line in 12 markets",
        "Congressional hearing targets {ticker} market dominance",
    ],
    "Product Launch": [
        "{ticker} unveils next-generation product suite at annual conference",
        "{ticker} launches AI-powered platform, analysts raise price targets",
        "{ticker} delays flagship product release citing supply chain issues",
        "{ticker} product recall affects 2.3M units, stock drops 5%",
        "{ticker} announces major software update with enterprise features",
    ],
    "Executive Change": [
        "{ticker} CEO announces unexpected resignation effective immediately",
        "{ticker} appoints new CFO with turnaround track record",
        "{ticker} board ousts chairman following activist investor pressure",
        "{ticker} names industry veteran as Chief Technology Officer",
        "{ticker} COO promoted to President amid restructuring effort",
    ],
    "Macroeconomic": [
        "Federal Reserve rate decision creates headwinds for {ticker}",
        "Rising inflation data weighs on {ticker} margin outlook",
        "Strong dollar pressures {ticker} international revenue guidance",
        "Credit tightening cycle expected to impact {ticker} lending business",
        "GDP growth surprise boosts outlook for cyclical names including {ticker}",
    ],
    "Legal": [
        "{ticker} faces class action lawsuit over misleading disclosures",
        "{ticker} patent dispute with competitor heads to federal court",
        "Judge dismisses antitrust case against {ticker}, shares rally",
        "{ticker} settles product liability claims for $310M",
        "{ticker} loses appeal, ordered to pay $1.2B in damages",
    ],
    "ESG": [
        "{ticker} commits to net-zero emissions by 2040 in new climate pledge",
        "ESG index drops {ticker} after governance concerns raised",
        "{ticker} sustainability report reveals water usage increased 18%",
        "Activist fund targets {ticker} board diversity, files shareholder proposal",
        "{ticker} raises minimum wage to $22/hr, boosts employer brand rating",
    ],
    "Geopolitical": [
        "Export restrictions on semiconductors hit {ticker} supply chain",
        "{ticker} suspends operations in conflict region, flags revenue risk",
        "Trade tariff escalation clouds {ticker} manufacturing cost outlook",
        "{ticker} diversifies supply chain away from geopolitical risk zones",
        "Sanctions compliance review delays {ticker} expansion into new market",
    ],
    "Analyst Rating": [
        "Goldman Sachs upgrades {ticker} to Buy, raises PT to $240",
        "Morgan Stanley cuts {ticker} to Underweight citing valuation",
        "JP Morgan initiates {ticker} coverage with Overweight and $195 PT",
        "Barclays downgrades {ticker} on slowing growth expectations",
        "UBS raises {ticker} price target by 15% following channel checks",
    ],
}

# ── Sentiment tendency per event type (+/neutral/-) ───────────────────────────
# Negative index = bearish template bias for impact calculation
_SENTIMENT_BIAS: dict[str, float] = {
    "Earnings": 0.0,           # mixed
    "M&A": 0.3,                # slightly positive (deal premium)
    "Regulatory": -0.5,        # negative
    "Product Launch": 0.3,
    "Executive Change": -0.2,
    "Macroeconomic": -0.1,
    "Legal": -0.5,
    "ESG": 0.1,
    "Geopolitical": -0.4,
    "Analyst Rating": 0.2,
}

# ── Impact tendency per event type ────────────────────────────────────────────
_IMPACT_BASE: dict[str, tuple[float, float]] = {
    "Earnings": (4.0, 9.0),
    "M&A": (5.0, 9.5),
    "Regulatory": (5.0, 8.5),
    "Product Launch": (3.0, 7.0),
    "Executive Change": (4.0, 8.0),
    "Macroeconomic": (3.0, 7.5),
    "Legal": (3.5, 8.0),
    "ESG": (2.0, 5.0),
    "Geopolitical": (4.0, 9.0),
    "Analyst Rating": (2.0, 6.0),
}


def _make_article(
    ticker: str,
    event_type: str,
    ts: datetime,
    template_idx: int | None = None,
) -> dict:
    """Build a synthetic article dict matching the ingestion pipeline contract."""
    templates = _HEADLINES[event_type]
    idx = template_idx if template_idx is not None else random.randrange(len(templates))
    headline = templates[idx % len(templates)].format(ticker=ticker)

    impact_low, impact_high = _IMPACT_BASE[event_type]
    impact_val = random.uniform(impact_low, impact_high)
    sentiment_bias = _SENTIMENT_BIAS[event_type]
    sentiment_noise = random.gauss(0, 0.3)
    sentiment_val = max(-1.0, min(1.0, sentiment_bias + sentiment_noise))

    return {
        "article_id": str(uuid.uuid4()),
        "title": headline,
        "body": (
            f"{headline}. "
            f"Analysts are closely watching {ticker} following this development. "
            f"The news comes amid broader market uncertainty and shifting sector dynamics. "
            f"Institutional investors have begun reassessing their positions in {ticker}."
        ),
        "source": random.choice(["news", "news", "news", "social", "filing"]),
        "ticker_symbols": [ticker],
        "published_at": ts.isoformat(),
        "url": f"https://example.com/news/{ticker.lower()}-{event_type.lower().replace(' ', '-')}-{ts.strftime('%Y%m%d%H%M%S')}",
        "source_weight": round(random.uniform(0.6, 1.0), 2),
        "source_count": random.randint(1, 5),
        # Metadata hints for test/replay only — not part of the NLP contract
        "_synthetic": True,
        "_event_type_hint": event_type,
        "_impact_hint": round(impact_val, 2),
        "_sentiment_hint": round(sentiment_val, 2),
    }


def generate_sample_articles(n: int = 200, hours: float = 2.0) -> list[dict]:
    """
    Generate ``n`` synthetic articles spread over the last ``hours`` hours.

    Distributes articles across all event types and tickers, ensuring:
    - Each event type appears at least once per ticker bucket
    - ~15% of articles have impact >= 7.0 to reliably trigger stress tests
    - Timestamps are strictly increasing (required for correct replay ordering)
    """
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(hours=hours)
    # Spread timestamps across the window with small random jitter
    step_seconds = (hours * 3600) / n
    articles = []

    for i in range(n):
        # Cycle through tickers and event types for even coverage
        ticker = _TICKERS[i % len(_TICKERS)]
        event_type = _EVENT_TYPES[i % len(_EVENT_TYPES)]

        # Approximately 15% of articles are "high impact" (use index 0 = positive/strong)
        # The remaining use random template selection
        high_impact = (i % 7 == 0)
        template_idx = 0 if high_impact else None

        jitter = random.uniform(-step_seconds * 0.4, step_seconds * 0.4)
        ts = start + timedelta(seconds=i * step_seconds + jitter)
        ts = ts.replace(microsecond=0)

        articles.append(_make_article(ticker, event_type, ts, template_idx))

    # Sort by published_at so replay is in chronological order
    articles.sort(key=lambda a: a["published_at"])
    return articles


class ReplayRunner:
    """
    Replays pre-cached sample articles into the raw.articles Redis stream.

    Parameters
    ──────────
    sample_dir        : directory to load/save articles.jsonl
    speed_multiplier  : replay speed (10.0 = 10× faster than real time)
    redis_client      : RedisStreamClient; must be connected before run_forever()
    """

    STREAM = "raw.articles"

    def __init__(
        self,
        sample_dir: Path,
        speed_multiplier: float = 10.0,
        redis_client: "RedisStreamClient | None" = None,
    ) -> None:
        self._sample_dir = Path(sample_dir)
        self._speed = max(0.01, speed_multiplier)
        self._redis = redis_client
        self._articles_path = self._sample_dir / "articles.jsonl"

    def set_redis(self, redis_client: "RedisStreamClient") -> None:
        """Attach the Redis client after construction (used in lifespan)."""
        self._redis = redis_client

    # ── Data loading / generation ─────────────────────────────────────────────

    def _load_or_generate(self) -> list[dict]:
        """Return articles from disk, or generate and persist them."""
        if self._articles_path.exists():
            try:
                articles = []
                with self._articles_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            articles.append(json.loads(line))
                if articles:
                    logger.info(
                        "ReplayRunner loaded %d articles from %s",
                        len(articles), self._articles_path,
                    )
                    return articles
            except Exception as exc:
                logger.warning("Failed to load %s: %s — regenerating", self._articles_path, exc)

        logger.info("Generating synthetic replay data (200 articles)…")
        articles = generate_sample_articles(n=200, hours=2.0)
        self._save(articles)
        return articles

    def _save(self, articles: list[dict]) -> None:
        """Persist articles to JSONL so subsequent runs skip generation."""
        self._sample_dir.mkdir(parents=True, exist_ok=True)
        with self._articles_path.open("w") as f:
            for article in articles:
                f.write(json.dumps(article, default=str) + "\n")
        logger.info("Saved %d synthetic articles to %s", len(articles), self._articles_path)

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """
        Publish articles to raw.articles indefinitely.

        Iterates through the article list in chronological order, sleeping
        proportionally to the inter-article time gap divided by speed_multiplier.
        After reaching the end, waits 30 seconds then restarts.
        """
        if self._redis is None:
            raise RuntimeError("ReplayRunner: Redis client not set.  Call set_redis() first.")

        articles = self._load_or_generate()
        logger.info(
            "ReplayRunner starting: %d articles, speed=%.1f×",
            len(articles), self._speed,
        )

        pass_count = 0
        while True:
            pass_count += 1
            logger.info("ReplayRunner: starting pass %d", pass_count)
            published = 0

            for i, article in enumerate(articles):
                try:
                    # Compute sleep from inter-article gap
                    if i > 0:
                        prev_ts = _parse_ts(articles[i - 1]["published_at"])
                        curr_ts = _parse_ts(article["published_at"])
                        gap_seconds = (curr_ts - prev_ts).total_seconds()
                        sleep_seconds = max(0.0, gap_seconds / self._speed)
                        # Cap individual sleep at 5s so the replay stays responsive
                        sleep_seconds = min(sleep_seconds, 5.0)
                        if sleep_seconds > 0.05:
                            await asyncio.sleep(sleep_seconds)

                    # Stamp the article with "now" so NLP timestamps are fresh
                    stamped = dict(article)
                    stamped["published_at"] = datetime.now(tz=timezone.utc).isoformat()

                    await self._redis.publish(self.STREAM, stamped)
                    published += 1

                    if published % 20 == 0:
                        logger.debug(
                            "ReplayRunner: published %d/%d articles",
                            published, len(articles),
                        )

                except asyncio.CancelledError:
                    logger.info("ReplayRunner cancelled after %d articles", published)
                    raise
                except Exception as exc:
                    logger.error("ReplayRunner publish error (article %d): %s", i, exc)
                    await asyncio.sleep(1.0)

            logger.info(
                "ReplayRunner pass %d complete: published %d articles. "
                "Restarting in 30s…",
                pass_count, published,
            )
            try:
                await asyncio.sleep(30.0)
            except asyncio.CancelledError:
                logger.info("ReplayRunner cancelled during inter-pass sleep")
                raise


def _parse_ts(ts_val: str | datetime) -> datetime:
    """Parse a published_at value to a UTC-aware datetime."""
    if isinstance(ts_val, datetime):
        if ts_val.tzinfo is None:
            return ts_val.replace(tzinfo=timezone.utc)
        return ts_val
    try:
        dt = datetime.fromisoformat(str(ts_val))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return datetime.now(tz=timezone.utc)
