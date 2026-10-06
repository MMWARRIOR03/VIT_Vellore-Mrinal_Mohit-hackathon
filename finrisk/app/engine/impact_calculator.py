"""
impact_calculator.py — Translates NLP signals into a 1–10 risk impact score.

Two modes controlled by the MODEL_IMPACT env var:

  MODEL_IMPACT=rules    (default) Deterministic formula using event-type base
                        scores, sentiment magnitude, novelty, and diffusion.

  MODEL_IMPACT=xgboost  Loads the pre-trained XGBoost regressor from
                        data/models/impact_xgb.pkl (built by
                        scripts/train_impact_model.py).  Falls back to the
                        rule-based formula if the file is not present.

The XGBoostImpactModel lives in app/engine/xgboost_model.py.  ImpactCalculator
is the only caller of that class in production; tests may use it directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_XGB_MODEL_PATH = Path("data/models/impact_xgb.pkl")
_XGB_FEATURES_PATH = Path("data/models/impact_xgb_features.json")

# ── Base impact scores per event type ─────────────────────────────────────────

BASE_IMPACT: dict[str, float] = {
    "Earnings": 5.0,
    "M&A": 5.5,
    "Regulatory": 6.0,
    "Macro": 6.0,
    "Geopolitical": 7.0,
    "Credit": 8.0,
    "Commodity": 5.5,
    "Cyber": 6.5,
    "Leadership": 4.5,
    "Product": 4.0,
}

_DEFAULT_BASE = 5.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ── ImpactCalculator ──────────────────────────────────────────────────────────

class ImpactCalculator:
    """
    Computes a risk impact score (1–10) and a ±1.5 credible interval.

    Parameters
    ──────────
    model_path  Path to the XGBoost pkl file.  Defaults to the standard
                project location.  Useful for tests that supply a mock model.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        from app.engine.xgboost_model import XGBoostImpactModel

        self._xgb: XGBoostImpactModel | None = None
        self._use_xgb: bool = False

        effective_path = Path(model_path) if model_path is not None else _XGB_MODEL_PATH

        if os.environ.get("MODEL_IMPACT", "").lower() == "xgboost":
            xgb = XGBoostImpactModel()
            if xgb.load(model_path=effective_path, features_path=_XGB_FEATURES_PATH):
                self._xgb = xgb
                self._use_xgb = True
                logger.info("ImpactCalculator: using XGBoost model from %s", effective_path)
            else:
                logger.info(
                    "MODEL_IMPACT=xgboost set but model not available — "
                    "falling back to rule-based calculator"
                )

    # ── Public interface ──────────────────────────────────────────────────────

    def calculate(
        self,
        event_type: str,
        sentiment: float,
        novelty: float,
        diffusion_score: float = 0.5,
        article_dict: dict[str, Any] | None = None,
        source_count: int = 1,
    ) -> tuple[float, tuple[float, float]]:
        """
        Calculate impact score and credible interval.

        Parameters
        ──────────
        event_type      One of the 10 canonical event type labels.
        sentiment       Signed sentiment in [-1, 1] (from FinBERT or rule-based).
        novelty         Novelty score in [0, 1] (from NoveltyScorer).
        diffusion_score Composite diffusion in [0, 1] (default 0.5).
        article_dict    Optional raw article dict.  Required for XGBoost path
                        (supplies published_at, ticker_symbols, etc.).
        source_count    Number of distinct sources reporting this event.

        Returns
        ───────
        (impact, impact_range):
          impact        float in [1.0, 10.0]
          impact_range  (low, high) each in [1.0, 10.0], low ≤ impact ≤ high
        """
        if self._use_xgb and self._xgb is not None:
            return self._calculate_xgb(
                event_type, sentiment, novelty, diffusion_score,
                article_dict or {}, source_count,
            )
        return self._calculate_rule_based(event_type, sentiment, novelty, diffusion_score)

    # ── Rule-based path ───────────────────────────────────────────────────────

    def _calculate_rule_based(
        self,
        event_type: str,
        sentiment: float,
        novelty: float,
        diffusion_score: float,
    ) -> tuple[float, tuple[float, float]]:
        base = BASE_IMPACT.get(event_type, _DEFAULT_BASE)

        # Novelty floor: even duplicate signals carry some impact.
        effective_novelty = max(novelty, 0.3)

        impact = (
            base
            * (0.5 + 0.5 * abs(sentiment))
            * effective_novelty
            * (0.8 + 0.2 * diffusion_score)
        )
        impact = _clamp(impact, 1.0, 10.0)

        low  = _clamp(impact - 1.5, 1.0, impact)
        high = _clamp(impact + 1.5, impact, 10.0)
        return impact, (low, high)

    # ── XGBoost path ──────────────────────────────────────────────────────────

    def _calculate_xgb(
        self,
        event_type: str,
        sentiment: float,
        novelty: float,
        diffusion_score: float,
        article_dict: dict[str, Any],
        source_count: int,
    ) -> tuple[float, tuple[float, float]]:
        """XGBoost inference path — only reached when the model is loaded."""
        assert self._xgb is not None  # guarded by caller

        features = self._xgb.get_feature_vector(
            article_dict=article_dict,
            sentiment=sentiment,
            event_type=event_type,
            source_count=source_count,
        )

        # For XGBoost, the raw score is already market-calibrated.
        # Apply only a mild diffusion modifier (fresh multi-source events → slight boost)
        # but do NOT multiply by novelty — the model already learned impact from real moves.
        diffusion_mod = 0.9 + 0.1 * diffusion_score  # range [0.9, 1.0]

        raw_impact = self._xgb.predict(features)
        impact = _clamp(raw_impact * diffusion_mod, 1.0, 10.0)

        low  = _clamp(impact - 1.5, 1.0, impact)
        high = _clamp(impact + 1.5, impact, 10.0)
        return impact, (low, high)
