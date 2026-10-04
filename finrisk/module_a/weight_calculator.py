"""
WeightCalculator — signal scores → constrained index weights.

Algorithm
─────────
1. Z-score the risk-adjusted scores across all tickers.
2. Exponential tilt: raw[t] = base_weight[t] × exp(λ × z[t]).
   λ controls how aggressively high-scoring tickers are overweighted.
3. Normalise to sum-to-1.
4. Clip each weight to [W_MIN, W_MAX] and re-normalise.
5. Apply hard override for extreme negative events (impact ≥ 8, sentiment < -0.6).
6. Gate on mean_confidence before accepting any rebalance.
7. Cap total one-way turnover to MAX_TURNOVER.

The z-score step means the tilt is always relative: a ticker with zero signal
gets exactly its base weight when all other tickers also have zero signal.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import yaml

from app.schema.risk_signal import RiskSignal

logger = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────

# Controls the aggressiveness of the exponential tilt.
LAMBDA: float = 0.5

# Per-ticker weight floor and ceiling (pre- and post-normalisation).
W_MIN: float = 0.02
W_MAX: float = 0.15

# Maximum one-way portfolio turnover per rebalance event.
MAX_TURNOVER: float = 0.20

# Minimum weight change (max delta across all tickers) to trigger a rebalance.
MIN_WEIGHT_CHANGE: float = 0.02

# Minimum mean signal confidence required to allow a rebalance.
CONFIDENCE_THRESHOLD: float = 0.65

# Fallback 20-stock universe if configs/index.yaml is not found.
_FALLBACK_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM",  "BAC",  "GS",
    "V",    "JNJ",  "UNH",  "PFE",  "XOM",
    "CVX",  "CAT",  "BA",   "WMT",  "HD",
]


def _load_base_weights(config_path: Optional[Path]) -> dict[str, float]:
    """
    Parse configs/index.yaml and return {ticker: base_weight}.
    Falls back to equal-weight over _FALLBACK_UNIVERSE on any error.
    """
    path = config_path
    if path is None:
        # Resolve relative to this file's location: module_a/ → finrisk/
        path = Path(__file__).parent.parent / "configs" / "index.yaml"

    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
        constituents = data["constituents"]
        weights = {c["ticker"]: float(c["base_weight"]) for c in constituents}
        logger.info("Loaded %d base weights from %s", len(weights), path)
        return weights
    except Exception as exc:
        logger.warning(
            "Could not load index.yaml (%s); using equal-weight fallback.", exc
        )
        return {t: 0.05 for t in _FALLBACK_UNIVERSE}


class WeightCalculator:
    """
    Converts a dict of risk-adjusted scores into index-weight targets.

    Parameters
    ──────────
    config_path : path to configs/index.yaml.  None → auto-detect.
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.base_weights: dict[str, float] = _load_base_weights(config_path)

    # ── Core weight computation ───────────────────────────────────────────────

    def compute_target_weights(self, scores: dict[str, float]) -> dict[str, float]:
        """
        Map risk-adjusted scores to constrained portfolio weights.

        Tickers absent from ``scores`` are treated as having a score of 0.0.
        The output always sums to 1.0 (within floating-point precision).
        """
        tickers = list(self.base_weights.keys())
        values = [scores.get(t, 0.0) for t in tickers]

        # Z-score (cross-sectional normalisation).
        mu = mean(values)
        sigma = stdev(values) if len(values) > 1 else 0.0
        sigma = max(sigma, 1e-6)  # avoid division by zero when all scores equal
        z = {t: (scores.get(t, 0.0) - mu) / sigma for t in tickers}

        # Exponential tilt from base weights.
        raw = {t: self.base_weights[t] * math.exp(LAMBDA * z[t]) for t in tickers}

        # Normalise to sum-to-1.
        total = sum(raw.values())
        normalised = {t: raw[t] / total for t in tickers}

        # Clip to [W_MIN, W_MAX].
        clipped = {t: max(W_MIN, min(W_MAX, normalised[t])) for t in tickers}

        # Re-normalise after clipping (clipping breaks the sum-to-1 invariant).
        total2 = sum(clipped.values())
        return {t: clipped[t] / total2 for t in tickers}

    # ── Rebalance gate ────────────────────────────────────────────────────────

    def should_rebalance(
        self,
        current: dict[str, float],
        target: dict[str, float],
        mean_confidence: float,
    ) -> bool:
        """
        Return True only when both conditions hold:
        - At least one ticker would move more than MIN_WEIGHT_CHANGE.
        - Mean signal confidence exceeds CONFIDENCE_THRESHOLD.

        Both conditions must be satisfied; either alone is not enough.
        """
        if mean_confidence <= CONFIDENCE_THRESHOLD:
            return False
        max_delta = max(
            abs(target.get(t, 0.0) - current.get(t, 0.0))
            for t in target
        )
        return max_delta > MIN_WEIGHT_CHANGE

    # ── Turnover cap ──────────────────────────────────────────────────────────

    def cap_turnover(
        self,
        current: dict[str, float],
        target: dict[str, float],
    ) -> dict[str, float]:
        """
        Scale back moves linearly so that one-way turnover ≤ MAX_TURNOVER.

        One-way turnover = sum of positive weight changes (= sum of absolute
        changes / 2, since the portfolio is fully invested).  If turnover is
        already within budget the target is returned unchanged.
        """
        one_way = (
            sum(abs(target.get(t, 0.0) - current.get(t, 0.0)) for t in target) / 2.0
        )
        if one_way <= MAX_TURNOVER:
            return target

        scale = MAX_TURNOVER / one_way
        adjusted = {
            t: current.get(t, 0.0) + scale * (target.get(t, 0.0) - current.get(t, 0.0))
            for t in target
        }
        # Re-normalise to fix any floating-point drift.
        total = sum(adjusted.values())
        return {t: adjusted[t] / total for t in target}

    # ── Hard override ─────────────────────────────────────────────────────────

    def hard_override(
        self,
        target: dict[str, float],
        signal: RiskSignal,
    ) -> dict[str, float]:
        """
        Force-reduce affected tickers when a signal is extremely negative.

        Triggered when impact ≥ 8.0 AND sentiment < -0.6.  Affected tickers
        (those mentioned in signal.tickers) are capped at W_MAX / 2, then
        the portfolio is re-normalised.  This is a one-way floor: the override
        only reduces weights, never increases them.
        """
        if signal.impact >= 8.0 and signal.sentiment < -0.6:
            result = dict(target)
            cap = W_MAX / 2.0
            for tr in signal.tickers:
                if tr.ticker in result:
                    result[tr.ticker] = min(result[tr.ticker], cap)
            total = sum(result.values())
            return {t: v / total for t, v in result.items()}
        return target
