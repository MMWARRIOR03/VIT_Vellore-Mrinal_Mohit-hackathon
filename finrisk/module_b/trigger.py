"""
Stress-test trigger gate with per-(ticker, event_type) cooldown.

Trigger conditions (all must be met)
──────────────────────────────────────
  1. signal.impact >= 7.0
  2. signal.confidence >= 0.55
  3. No cooldown active for any ticker × event_type pair in the signal

Cooldown logic
──────────────
After a trigger fires for (ticker, event_type), that pair is locked for 30
minutes.  Subsequent signals for the same pair within the window are silently
dropped.  This prevents a burst of near-identical signals (e.g. 10 syndicated
news articles about the same event) from firing 10 stress runs during a demo.

The cooldown key is f"{ticker}:{event_type}".  For multi-ticker signals, ALL
tickers are checked — the trigger only fires if none of them are on cooldown.
When it does fire, all tickers in the signal are marked.

Thread safety: designed for single-threaded asyncio.  The cooldown registry
is a plain dict; no locking is required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from app.schema.risk_signal import RiskSignal

logger = logging.getLogger(__name__)

# Default thresholds (can be overridden via constructor for testing)
_DEFAULT_IMPACT_THRESHOLD: float = 7.0
_DEFAULT_CONFIDENCE_THRESHOLD: float = 0.55
_DEFAULT_COOLDOWN_MINUTES: int = 30


class StressTrigger:
    """
    Gate that decides whether a RiskSignal should fire a stress test.

    Parameters
    ──────────
    impact_threshold     : minimum impact score (default 7.0)
    confidence_threshold : minimum confidence (default 0.55)
    cooldown_minutes     : cooldown window in minutes (default 30)
    """

    def __init__(
        self,
        impact_threshold: float = _DEFAULT_IMPACT_THRESHOLD,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        cooldown_minutes: int = _DEFAULT_COOLDOWN_MINUTES,
    ) -> None:
        self._impact_threshold = impact_threshold
        self._confidence_threshold = confidence_threshold
        self._cooldown_delta = timedelta(minutes=cooldown_minutes)
        # {f"{ticker}:{event_type}": datetime_of_last_trigger}
        self._cooldown_registry: dict[str, datetime] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def consume_signals(self, signal: RiskSignal) -> bool:
        """
        Evaluate whether ``signal`` should trigger a stress test.

        Returns True if all conditions pass and the trigger fires.
        Marks all tickers on cooldown if it does fire.

        Decision log is emitted at DEBUG level to aid diagnostics.
        """
        # 1. Impact gate
        if signal.impact < self._impact_threshold:
            logger.debug(
                "Trigger skipped: impact %.2f < threshold %.2f  (event_id=%s)",
                signal.impact, self._impact_threshold, signal.event_id,
            )
            return False

        # 2. Confidence gate
        if signal.confidence < self._confidence_threshold:
            logger.debug(
                "Trigger skipped: confidence %.3f < threshold %.3f  (event_id=%s)",
                signal.confidence, self._confidence_threshold, signal.event_id,
            )
            return False

        # 3. Cooldown gate — check all tickers in the signal
        for ticker_ref in signal.tickers:
            if self.is_on_cooldown(ticker_ref.ticker, signal.event_type):
                logger.debug(
                    "Trigger skipped: %s:%s on cooldown  (event_id=%s)",
                    ticker_ref.ticker, signal.event_type, signal.event_id,
                )
                return False

        # All gates passed — fire and mark cooldown
        for ticker_ref in signal.tickers:
            self.mark_triggered(ticker_ref.ticker, signal.event_type)

        logger.info(
            "Stress trigger FIRED: event_id=%s  event_type=%s  impact=%.2f  "
            "confidence=%.3f  tickers=%s",
            signal.event_id,
            signal.event_type,
            signal.impact,
            signal.confidence,
            [t.ticker for t in signal.tickers],
        )
        return True

    def is_on_cooldown(self, ticker: str, event_type: str) -> bool:
        """
        Return True if the (ticker, event_type) pair was triggered within the
        cooldown window.
        """
        key = self._make_key(ticker, event_type)
        last = self._cooldown_registry.get(key)
        if last is None:
            return False
        now = datetime.now(tz=timezone.utc)
        return (now - last) < self._cooldown_delta

    def mark_triggered(self, ticker: str, event_type: str) -> None:
        """Record that (ticker, event_type) has just triggered."""
        key = self._make_key(ticker, event_type)
        self._cooldown_registry[key] = datetime.now(tz=timezone.utc)

    def reset_cooldown(self, ticker: str, event_type: str) -> None:
        """Remove the cooldown entry for (ticker, event_type).  Mainly for testing."""
        key = self._make_key(ticker, event_type)
        self._cooldown_registry.pop(key, None)

    def reset_all_cooldowns(self) -> None:
        """Clear the entire cooldown registry.  Mainly for testing."""
        self._cooldown_registry.clear()

    def cooldown_remaining(self, ticker: str, event_type: str) -> float:
        """
        Return seconds remaining on the cooldown for (ticker, event_type).
        Returns 0.0 if not on cooldown.
        """
        key = self._make_key(ticker, event_type)
        last = self._cooldown_registry.get(key)
        if last is None:
            return 0.0
        elapsed = (datetime.now(tz=timezone.utc) - last).total_seconds()
        remaining = self._cooldown_delta.total_seconds() - elapsed
        return max(0.0, remaining)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(ticker: str, event_type: str) -> str:
        return f"{ticker.upper()}:{event_type}"

    @property
    def active_cooldowns(self) -> dict[str, float]:
        """
        Return a snapshot of all active cooldowns as {key: seconds_remaining}.
        Expired entries are excluded.
        """
        now = datetime.now(tz=timezone.utc)
        result: dict[str, float] = {}
        for key, last in self._cooldown_registry.items():
            elapsed = (now - last).total_seconds()
            remaining = self._cooldown_delta.total_seconds() - elapsed
            if remaining > 0:
                result[key] = round(remaining, 1)
        return result
