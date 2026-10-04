"""
SignalAggregator — decay-weighted, volatility-adjusted score per ticker.

Design
──────
Signals are stored in a two-level dict keyed by (ticker, cluster_id) so that
multiple signals from the same underlying event are deduplicated.  Only the
most recent signal per cluster survives, which prevents a single viral story
from inflating scores by arriving from many sources.

Each signal contributes:

    contribution = sentiment × confidence × novelty × decay × relevance

where decay = exp(-ln(2) × age_hours / halflife).  The sign of sentiment
determines whether the contribution is bullish (+) or bearish (-).

Risk adjustment divides the raw score by realised annualised volatility,
so high-conviction signals on stable stocks get more weight than the same
signal on a volatile name.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Optional

from app.schema.risk_signal import RiskSignal

logger = logging.getLogger(__name__)

# Cap per ticker to avoid unbounded memory growth in long-running processes.
MAX_SIGNALS_PER_TICKER = 100

# Signal is considered expired when its age exceeds this many half-lives.
_EXPIRY_HALFLIFE_MULTIPLIER = 5.0

_LN2 = math.log(2.0)


class SignalAggregator:
    """
    Maintains a rolling window of RiskSignals per ticker, deduplicated by
    cluster_id, and computes decay-weighted risk-adjusted scores.
    """

    def __init__(self) -> None:
        # {ticker: {cluster_id: RiskSignal}}
        self._history: dict[str, dict[str, RiskSignal]] = {}

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_signal(self, signal: RiskSignal) -> None:
        """
        Register a signal for every ticker it references.

        If a signal with the same cluster_id already exists for a ticker,
        it is replaced with the newer one (deduplication).  If the per-ticker
        buffer would exceed MAX_SIGNALS_PER_TICKER, the oldest signal by
        ts_processed is evicted first.
        """
        for ticker_ref in signal.tickers:
            ticker = ticker_ref.ticker
            if ticker not in self._history:
                self._history[ticker] = {}

            bucket = self._history[ticker]
            bucket[signal.cluster_id] = signal

            # Enforce the per-ticker cap by evicting the oldest entry.
            if len(bucket) > MAX_SIGNALS_PER_TICKER:
                oldest_cluster = min(
                    bucket, key=lambda cid: bucket[cid].ts_processed
                )
                del bucket[oldest_cluster]
                logger.debug(
                    "Evicted oldest signal for ticker %s (cluster %s)",
                    ticker, oldest_cluster,
                )

    # ── Scoring ───────────────────────────────────────────────────────────────

    def compute_score(self, ticker: str, now: datetime) -> float:
        """
        Compute the raw decay-weighted sentiment score for a ticker.

        Returns 0.0 if no signals have been registered for the ticker.
        The score can be negative (net bearish signal stack).
        """
        bucket = self._history.get(ticker, {})
        if not bucket:
            return 0.0

        score = 0.0
        for sig in bucket.values():
            age_hours = (now - sig.ts_processed).total_seconds() / 3600.0
            # Exponential decay: half the value each decay_halflife_hours.
            decay = math.exp(-_LN2 * age_hours / sig.decay_halflife_hours)
            # Look up this ticker's relevance in the signal's ticker list.
            relevance = next(
                (tr.relevance for tr in sig.tickers if tr.ticker == ticker),
                1.0,
            )
            contribution = (
                sig.sentiment
                * sig.confidence
                * sig.novelty
                * decay
                * relevance
            )
            score += contribution

        return score

    @staticmethod
    def get_realized_vol(ticker: str, prices: list[float]) -> float:
        """
        Annualised realised volatility from a list of prices (newest-first).

        Uses daily log-returns.  Returns 0.20 (20%) when fewer than two
        prices are available, and is floored at 0.05 (5%) to prevent
        division-by-zero in the risk adjustment step.
        """
        if len(prices) < 2:
            return 0.20

        # prices is newest-first; compute return[i] = (p[i+1] - p[i]) / p[i]
        # iterating oldest→newest means p[i+1] is newer than p[i] in the
        # reversed list — but we just need the magnitude.
        daily_returns: list[float] = []
        for i in range(len(prices) - 1):
            p_new = prices[i]
            p_old = prices[i + 1]
            if p_old > 0:
                daily_returns.append((p_new - p_old) / p_old)

        if len(daily_returns) < 2:
            return 0.20

        vol = stdev(daily_returns) * math.sqrt(252)
        return max(vol, 0.05)

    def compute_risk_adjusted_score(
        self,
        ticker: str,
        now: datetime,
        prices: list[float],
    ) -> float:
        """
        Divide raw score by annualised vol to penalise noisy, high-vol tickers.
        A small epsilon prevents division by zero.
        """
        score = self.compute_score(ticker, now)
        vol = self.get_realized_vol(ticker, prices)
        return score / (vol + 1e-6)

    def get_all_scores(
        self,
        tickers: list[str],
        now: datetime,
        prices_cache: dict[str, list[float]],
    ) -> dict[str, float]:
        """
        Return a risk-adjusted score for every ticker in the list.

        Tickers with no signals get a score of 0.0, which means
        WeightCalculator will assign them their base weight when z-scores are
        computed (z = 0 → raw = base_weight × exp(0) = base_weight).
        """
        return {
            t: self.compute_risk_adjusted_score(t, now, prices_cache.get(t, []))
            for t in tickers
        }

    # ── Confidence ────────────────────────────────────────────────────────────

    def get_mean_confidence(self, ticker: Optional[str] = None) -> float:
        """
        Mean confidence across all stored signals.

        If ``ticker`` is provided, scoped to that ticker's signals only.
        Returns 0.0 when there are no matching signals.
        """
        if ticker is not None:
            signals = list(self._history.get(ticker, {}).values())
        else:
            signals = [
                sig
                for bucket in self._history.values()
                for sig in bucket.values()
            ]

        if not signals:
            return 0.0

        return mean(sig.confidence for sig in signals)

    # ── Maintenance ───────────────────────────────────────────────────────────

    def cleanup_expired(self, now: datetime) -> None:
        """
        Remove signals whose age exceeds 5× their declared half-life.

        This prevents stale signals from accumulating in long-running
        processes where the stream may have periods of high activity
        followed by silence.
        """
        for ticker in list(self._history.keys()):
            bucket = self._history[ticker]
            expired = [
                cid
                for cid, sig in bucket.items()
                if (now - sig.ts_processed).total_seconds() / 3600.0
                > _EXPIRY_HALFLIFE_MULTIPLIER * sig.decay_halflife_hours
            ]
            for cid in expired:
                del bucket[cid]
                logger.debug("Expired signal cluster %s for ticker %s", cid, ticker)

            # Clean up empty ticker entries to avoid dict bloat.
            if not bucket:
                del self._history[ticker]
