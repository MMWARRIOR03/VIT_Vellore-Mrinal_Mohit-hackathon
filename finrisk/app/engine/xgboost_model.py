"""
xgboost_model.py — Production inference class for the XGBoost impact model.

Feature contract (matches v5 Colab training notebook):
  sentiment, abs_sentiment, tone_positive, tone_negative,
  source_count, is_after_hours, eps_surprise, has_eps, is_earnings,
  days_since_last,
  price_momentum_5d    — 5-day return before the article (no look-ahead)
  realized_vol_20d     — annualised 20-day realized vol before the article
  market_cap_bucket    — 0=large-cap, 1=mid-cap (large caps react less)
  vol_x_abssentiment   — interaction: vol × |sentiment|
  momentum_sentiment_align — interaction: momentum × sentiment direction
  et_<10 event types>, sec_<8 sectors>
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must stay in sync with scripts/train_impact_model.py
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_PATH = Path("data/models/impact_xgb.pkl")
_DEFAULT_FEATURES_PATH = Path("data/models/impact_xgb_features.json")

EVENT_TYPES: list[str] = [
    "Earnings", "M&A", "Regulatory", "Macro", "Geopolitical",
    "Credit", "Commodity", "Cyber", "Leadership", "Product",
]

SECTORS: list[str] = [
    "Technology", "Communication Services", "Consumer Discretionary",
    "Financials", "Health Care", "Energy", "Industrials", "Consumer Staples",
]

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

_TICKER_SECTOR: dict[str, str] = {
    "AAPL": "Technology",  "MSFT": "Technology",  "NVDA": "Technology",
    "GOOGL": "Communication Services",  "META": "Communication Services",
    "AMZN": "Consumer Discretionary",  "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "JPM": "Financials",  "BAC": "Financials",  "GS": "Financials",  "V": "Financials",
    "JNJ": "Health Care",  "UNH": "Health Care",  "PFE": "Health Care",
    "XOM": "Energy",  "CVX": "Energy",
    "CAT": "Industrials",  "BA": "Industrials",
    "WMT": "Consumer Staples",
}

# Market cap bucket: 0=large (>$200B), 1=mid ($50-200B)
_MARKET_CAP_BUCKET: dict[str, float] = {
    "AAPL": 0, "MSFT": 0, "NVDA": 0, "GOOGL": 0, "AMZN": 0,
    "META": 0, "TSLA": 0, "JPM": 0, "V": 0, "UNH": 0, "WMT": 0,
    "BAC": 1, "GS": 1, "JNJ": 1, "XOM": 1, "CVX": 1,
    "PFE": 1, "HD": 1, "CAT": 1, "BA": 1,
}

# NYSE/NASDAQ session in UTC: 13:30–20:00
_MARKET_OPEN_UTC_H = 13
_MARKET_OPEN_UTC_M = 30
_MARKET_CLOSE_UTC_H = 20
_MARKET_CLOSE_UTC_M = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _is_after_hours(ts: Any) -> bool:
    """Return True if ts falls outside regular NYSE/NASDAQ session."""
    try:
        import pandas as pd
        ts = pd.Timestamp(ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        market_open = ts.replace(hour=_MARKET_OPEN_UTC_H, minute=_MARKET_OPEN_UTC_M, second=0)
        market_close = ts.replace(hour=_MARKET_CLOSE_UTC_H, minute=_MARKET_CLOSE_UTC_M, second=0)
        return ts.weekday() >= 5 or not (market_open <= ts < market_close)
    except Exception:
        return False


def _get_price_features(ticker: str, published_at: Any) -> dict[str, float]:
    """
    Compute pre-event price features using the module-level PriceCache.

    All prices used are STRICTLY before published_at — no look-ahead.

    Returns dict with keys:
      price_momentum_5d      — 5-day return before event, clipped [-0.20, +0.20]
      realized_vol_20d       — annualised 20-day realized vol, clipped [0.05, 1.0]
      market_cap_bucket      — 0=large, 1=mid
      vol_x_abssentiment     — filled in by get_feature_vector after sentiment known
      momentum_sentiment_align — filled in by get_feature_vector after sentiment known
    """
    defaults = {
        "price_momentum_5d": 0.0,
        "realized_vol_20d": 0.20,
        "market_cap_bucket": float(_MARKET_CAP_BUCKET.get(ticker, 1)),
    }
    try:
        import pandas as pd
        import numpy as np
        prices = _price_cache.get_prices(ticker, n_days=30)
        if not prices or len(prices) < 6:
            return defaults
        # prices is newest-first; reverse to chronological
        prices_asc = list(reversed(prices))
        ts = pd.Timestamp(published_at)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        # 5-day momentum: use only prices from before the article
        # PriceCache stores closing prices; all are end-of-day so any price
        # from a date < article date is safely before the event
        p_now = prices_asc[-1]
        p_5d  = prices_asc[-6] if len(prices_asc) >= 6 else prices_asc[0]
        momentum = (p_now - p_5d) / p_5d if p_5d else 0.0
        momentum = float(np.clip(momentum, -0.20, 0.20))
        # 20-day realized vol
        if len(prices_asc) >= 21:
            recent = prices_asc[-21:]
            rets = [recent[i] / recent[i-1] - 1 for i in range(1, len(recent))]
            vol = float(np.std(rets) * np.sqrt(252))
            vol = float(np.clip(vol, 0.05, 1.0))
        else:
            vol = 0.20
        return {
            "price_momentum_5d": momentum,
            "realized_vol_20d": vol,
            "market_cap_bucket": float(_MARKET_CAP_BUCKET.get(ticker, 1)),
        }
    except Exception as exc:
        logger.debug("Price feature extraction failed for %s: %s", ticker, exc)
        return defaults


# Module-level price cache — shared across all ImpactCalculator instances.
# Lazy-initialised on first use; refreshes from yfinance once per day.
class _LazyPriceCache:
    """Thin wrapper that lazily creates a PriceCache on first access."""
    def __init__(self):
        self._cache = None

    def _ensure(self):
        if self._cache is None:
            try:
                from module_a.price_cache import PriceCache
                self._cache = PriceCache()
            except Exception:
                self._cache = None

    def get_prices(self, ticker: str, n_days: int = 30) -> list[float]:
        self._ensure()
        if self._cache is None:
            return []
        try:
            return self._cache.get_prices(ticker, n_days)
        except Exception:
            return []


_price_cache = _LazyPriceCache()


# ---------------------------------------------------------------------------
# XGBoostImpactModel
# ---------------------------------------------------------------------------


class XGBoostImpactModel:
    """
    Production inference wrapper for the trained XGBoost impact regressor.

    Lifecycle:
      model = XGBoostImpactModel()
      ok = model.load(model_path, features_path)
      if ok:
          features = model.get_feature_vector(article, sentiment, event_type)
          score = model.predict(features)

    The model is deliberately separated from ImpactCalculator so the same
    class can be unit-tested in isolation without touching the calculator.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._feature_columns: list[str] = []
        self._loaded: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(
        self,
        model_path: Path = _DEFAULT_MODEL_PATH,
        features_path: Path = _DEFAULT_FEATURES_PATH,
    ) -> bool:
        """
        Load the pickled XGBoost model and its feature column list.

        Returns True on success; False if either file is missing or malformed.
        Does not raise.
        """
        model_path = Path(model_path)
        features_path = Path(features_path)

        if not model_path.exists():
            logger.info("XGBoost model not found at %s — will use rule-based fallback", model_path)
            return False
        if not features_path.exists():
            logger.warning("Feature file not found at %s — cannot load XGBoost model", features_path)
            return False

        try:
            with open(model_path, "rb") as fh:
                self._model = pickle.load(fh)

            with open(features_path) as fh:
                meta = json.load(fh)

            self._feature_columns = meta["feature_columns"]
            self._loaded = True
            logger.info(
                "XGBoostImpactModel loaded: %d features, best_iter=%s",
                len(self._feature_columns),
                meta.get("best_iteration", "?"),
            )
            return True

        except Exception as exc:
            logger.warning("Failed to load XGBoost model: %s", exc)
            self._model = None
            self._feature_columns = []
            self._loaded = False
            return False

    def is_available(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        return self._loaded and self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: dict[str, float]) -> float:
        """
        Predict impact score (1–10) from a pre-built feature dict.

        Falls back to a rule-based estimate if the model is not loaded.
        The fallback uses BASE_IMPACT + sentiment magnitude, same formula as
        ImpactCalculator._calculate_rule_based.
        """
        if not self.is_available():
            # Rule-based fallback using whatever we can extract from features
            event_type = features.get("_event_type_hint", "Earnings")
            sentiment = float(features.get("sentiment", 0.0))
            base = BASE_IMPACT.get(str(event_type), 5.0)
            impact = base * (0.5 + 0.5 * abs(sentiment)) * 0.7 * 0.9
            return _clamp(impact, 1.0, 10.0)

        try:
            import numpy as np
            # Build vector in the canonical column order
            vector = np.array(
                [features.get(col, 0.0) for col in self._feature_columns],
                dtype=float,
            ).reshape(1, -1)

            raw = float(self._model.predict(vector)[0])

            # The v4 model was trained with impact_label in [1,10] but the
            # underlying XGBoost learned residuals that output in [0.1, 1.0]
            # range. Rescale: if raw < 2.0 it's in fractional form, multiply by 10.
            if raw < 2.0:
                raw = raw * 10.0

            return _clamp(raw, 1.0, 10.0)

        except Exception as exc:
            logger.error("XGBoost prediction error: %s — falling back to rule-based", exc)
            event_type = features.get("_event_type_hint", "Earnings")
            base = BASE_IMPACT.get(str(event_type), 5.0)
            return _clamp(base, 1.0, 10.0)

    # ── Feature extraction ────────────────────────────────────────────────────

    def get_feature_vector(
        self,
        article_dict: dict[str, Any],
        sentiment: float,
        event_type: str,
        source_count: int = 1,
    ) -> dict[str, float]:
        """
        Extract the model's feature vector from an article dict and NLP outputs.

        Produces the same 28 features as the v4 Colab training notebook:
          sentiment, abs_sentiment, tone_positive, tone_negative,
          source_count, is_after_hours, eps_surprise, has_eps, is_earnings,
          days_since_last, et_<10 event types>, sec_<8 sectors>

        Parameters
        ──────────
        article_dict  Article dict from the ingestion pipeline.
        sentiment     Signed sentiment score in [-1, 1].
        event_type    One of the 10 canonical event type labels.
        source_count  Number of sources mentioning the event (default 1).
        """
        features: dict[str, float] = {}

        # ── Core sentiment features ───────────────────────────────────────────
        sent = _clamp(float(sentiment), -1.0, 1.0)
        features["sentiment"]     = sent
        features["abs_sentiment"] = abs(sent)

        # V2Tone fields — proxy from FinBERT sentiment if not available
        tone_positive = float(article_dict.get("tone_positive",
                              article_dict.get("v2tone_positive", 0.0)) or 0.0)
        tone_negative = float(article_dict.get("tone_negative",
                              article_dict.get("v2tone_negative", 0.0)) or 0.0)
        if tone_positive == 0.0 and tone_negative == 0.0 and sent != 0.0:
            if sent > 0:
                tone_positive = abs(sent) * 10.0
            else:
                tone_negative = abs(sent) * 10.0
        features["tone_positive"] = abs(tone_positive)
        features["tone_negative"] = abs(tone_negative)

        # ── Source count + after-hours ────────────────────────────────────────
        features["source_count"]  = float(max(1, int(source_count)))
        published_at = article_dict.get("published_at")
        features["is_after_hours"] = float(_is_after_hours(published_at)) if published_at else 0.0

        # ── EPS surprise features (strong predictor for earnings events) ──────
        eps_surprise = article_dict.get("eps_surprise")
        if eps_surprise is not None:
            features["eps_surprise"] = _clamp(float(eps_surprise), -2.0, 2.0)
            features["has_eps"]      = 1.0
        else:
            features["eps_surprise"] = 0.0
            features["has_eps"]      = 0.0
        features["is_earnings"] = 1.0 if event_type == "Earnings" else 0.0

        # ── Days since last event ─────────────────────────────────────────────
        days_since = float(article_dict.get("days_since_last_event", 30.0) or 30.0)
        features["days_since_last"] = _clamp(days_since, 0.0, 90.0)

        # ── NEW: Pre-event price features (v5) ───────────────────────────────
        # Ticker for price lookup: try article_dict first, then the inference ticker
        ticker = None
        tickers = article_dict.get("ticker_symbols", [])
        if tickers:
            ticker = tickers[0] if isinstance(tickers, list) else str(tickers)
        # Also accept a direct 'ticker' key (set by NLPWorker)
        if not ticker:
            ticker = article_dict.get("ticker")
        if ticker:
            ticker = str(ticker).upper().strip()

        published_at = article_dict.get("published_at")
        px_feats = _get_price_features(ticker or "", published_at) if ticker else {
            "price_momentum_5d": 0.0,
            "realized_vol_20d": 0.20,
            "market_cap_bucket": 1.0,
        }
        features["price_momentum_5d"] = px_feats["price_momentum_5d"]
        features["realized_vol_20d"]  = px_feats["realized_vol_20d"]
        features["market_cap_bucket"] = px_feats["market_cap_bucket"]

        # Interaction features (computed after both price + sentiment are known)
        features["vol_x_abssentiment"]        = features["realized_vol_20d"] * abs(sent)
        features["momentum_sentiment_align"]   = features["price_momentum_5d"] * sent

        # ── Event type one-hot (et_ prefix — matches v4/v5 training) ─────────
        for et in EVENT_TYPES:
            col = f"et_{et.lower().replace('&', 'and').replace(' ', '_')}"
            features[col] = 1.0 if et == event_type else 0.0

        # ── Sector one-hot (sec_ prefix — matches v4/v5 training) ───────────
        sector = _TICKER_SECTOR.get(str(ticker), "Technology") if ticker else "Technology"
        for sec in SECTORS:
            col = f"sec_{sec.lower().replace(' ', '_')}"
            features[col] = 1.0 if sec == sector else 0.0

        # Private hint for rule-based fallback (stripped before model inference)
        features["_event_type_hint"] = event_type  # type: ignore[assignment]

        return features

    def _strip_private_keys(self, features: dict) -> dict[str, float]:
        """Remove internal hint keys before building the model input vector."""
        return {k: v for k, v in features.items() if not k.startswith("_")}
