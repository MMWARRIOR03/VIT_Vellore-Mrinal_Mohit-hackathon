"""
IndexRebalancer — orchestrates signal aggregation, weight computation,
and rebalance event emission for the FinRisk-20 tactical index.

Flow per incoming signal
────────────────────────
  on_signal(signal)
    │
    ├── SignalAggregator.add_signal            (update history)
    ├── SignalAggregator.cleanup_expired       (prune stale entries)
    ├── PriceCache.get_prices × 20             (vol estimation inputs)
    ├── SignalAggregator.get_all_scores        (decay-weighted, vol-adjusted)
    ├── WeightCalculator.compute_target_weights (z-score → exp tilt → clip)
    ├── WeightCalculator.hard_override         (extreme negative event gate)
    ├── WeightCalculator.should_rebalance      (confidence + magnitude gate)
    ├── WeightCalculator.cap_turnover          (turnover budget guard)
    └── RebalanceEvent (persist + publish)

The rebalancer is stateful (current_weights is updated in-place) and is
designed for a single-coroutine async environment.  It is not thread-safe.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.schema.risk_signal import RiskSignal
from module_a.price_cache import PriceCache
from module_a.signal_aggregator import SignalAggregator
from module_a.weight_calculator import WeightCalculator

logger = logging.getLogger(__name__)


@dataclass
class RebalanceEvent:
    """
    Immutable record of a single index rebalance.

    ``weight_deltas`` is after_weights − before_weights for each ticker.
    The to_dict() shape matches DuckDBStore.write_rebalance_event's contract.
    """

    rebalance_id: str
    ts_created: datetime
    trigger_signal_id: str
    before_weights: dict[str, float]
    after_weights: dict[str, float]
    weight_deltas: dict[str, float]
    reason: str

    def to_dict(self) -> dict:
        """Serialise to the shape expected by DuckDBStore.write_rebalance_event."""
        return {
            "rebalance_id": self.rebalance_id,
            "ts_created": self.ts_created,
            "trigger_signal": self.trigger_signal_id,
            "delta": self.weight_deltas,
            "reason": self.reason,
            "meta": {
                "before_weights": self.before_weights,
                "after_weights": self.after_weights,
            },
        }

    async def persist(self, db) -> None:  # db: DuckDBStore
        """Write this event to DuckDB."""
        await db.write_rebalance_event(self.to_dict())


class IndexRebalancer:
    """
    Signal-driven rebalancer for the FinRisk-20 index.

    Parameters
    ──────────
    config_path : path to configs/index.yaml.  None → auto-detect.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self._aggregator = SignalAggregator()
        self._calc = WeightCalculator(config_path)
        self._price_cache = PriceCache()

        # Initialise to equal weight over the full universe.
        self._current_weights: dict[str, float] = {
            t: w for t, w in self._calc.base_weights.items()
        }
        self._history: list[RebalanceEvent] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def on_signal(self, signal: RiskSignal) -> Optional[RebalanceEvent]:
        """
        Process an incoming RiskSignal and optionally rebalance.

        Returns a RebalanceEvent when a rebalance is triggered, None otherwise.
        The caller is responsible for persisting and publishing the event.
        """
        now = datetime.now(tz=timezone.utc)

        # 1. Update the signal history.
        self._aggregator.add_signal(signal)

        # 2. Prune expired entries.
        self._aggregator.cleanup_expired(now)

        # 3. Gather price data for vol estimation.
        all_tickers = list(self._current_weights.keys())
        prices = {t: self._price_cache.get_prices(t) for t in all_tickers}

        # 4. Compute risk-adjusted scores for all tickers.
        scores = self._aggregator.get_all_scores(all_tickers, now, prices)

        # 5. Overall confidence gate.
        mean_conf = self._aggregator.get_mean_confidence()

        # 6. Compute target weights from the score distribution.
        target = self._calc.compute_target_weights(scores)

        # 7. Apply hard override for extreme negative events before gating.
        target = self._calc.hard_override(target, signal)

        # 8. Gate: only proceed if magnitude and confidence thresholds are met.
        if not self._calc.should_rebalance(self._current_weights, target, mean_conf):
            logger.debug(
                "Rebalance gated: signal=%s mean_conf=%.3f",
                signal.event_id, mean_conf,
            )
            return None

        # 9. Cap turnover to stay within the MAX_TURNOVER budget.
        target = self._calc.cap_turnover(self._current_weights, target)

        # 10. Compute weight deltas.
        deltas = {
            t: target[t] - self._current_weights.get(t, 0.0)
            for t in target
        }

        # 11. Build the event record.
        primary_ticker = signal.tickers[0].ticker if signal.tickers else "UNKNOWN"
        event = RebalanceEvent(
            rebalance_id=str(uuid.uuid4()),
            ts_created=now,
            trigger_signal_id=signal.event_id,
            before_weights=dict(self._current_weights),
            after_weights=target,
            weight_deltas=deltas,
            reason=(
                f"Signal: {primary_ticker} {signal.event_type} "
                f"impact={signal.impact:.1f}"
            ),
        )

        # 12. Commit the new weights.
        self._current_weights.update(target)

        # 13. Keep a rolling 100-event history.
        self._history.append(event)
        if len(self._history) > 100:
            self._history.pop(0)

        logger.info(
            "Rebalanced: id=%s trigger=%s max_delta=%.4f",
            event.rebalance_id,
            signal.event_id,
            max(abs(d) for d in deltas.values()),
        )
        return event

    def get_current_weights(self) -> dict[str, float]:
        """Return a copy of the current weight vector."""
        return dict(self._current_weights)

    def get_weight_history(self) -> list[dict]:
        """Return the last ≤100 rebalance events as dicts."""
        return [e.to_dict() for e in self._history[-100:]]

    @property
    def price_cache(self) -> PriceCache:
        """Expose the price cache so the runner can trigger a refresh."""
        return self._price_cache
