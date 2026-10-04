"""
train_impact_model.py — Train the XGBoost impact scoring model.

Usage:
    python scripts/train_impact_model.py

Pipeline:
  1. Load labeled_events.parquet produced by build_training_labels.py.
  2. Engineer features (same set used at inference in xgboost_model.py).
  3. Time-based train / validation / test split — NO random shuffling.
  4. Train an XGBoost regressor targeting the continuous impact_label.
  5. Evaluate: MAE, RMSE, Spearman ρ vs rule-based baseline.
  6. Save model to data/models/impact_xgb.pkl and feature names to
     data/models/impact_xgb_features.json.

Feature contract
────────────────
The features computed here must exactly match those computed by
XGBoostImpactModel.get_feature_vector() in app/engine/xgboost_model.py.
If you add/remove features here, update that method too.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "training"
_MODELS_DIR = _PROJECT_ROOT / "data" / "models"

_LABELED_PATH = _DATA_DIR / "labeled_events.parquet"
_MODEL_OUT = _MODELS_DIR / "impact_xgb.pkl"
_FEATURES_OUT = _MODELS_DIR / "impact_xgb_features.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must stay in sync with xgboost_model.py
# ---------------------------------------------------------------------------

EVENT_TYPES: list[str] = [
    "Earnings", "M&A", "Regulatory", "Macro", "Geopolitical",
    "Credit", "Commodity", "Cyber", "Leadership", "Product",
]

SECTORS: list[str] = [
    "Technology", "Communication Services", "Consumer Discretionary",
    "Financials", "Health Care", "Energy", "Industrials", "Consumer Staples",
]

# Rule-based base scores — used to compute baseline predictions
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

# NYSE/NASDAQ regular session: 09:30–16:00 ET  =  13:30–20:00 UTC
_MARKET_OPEN_UTC_H = 13
_MARKET_OPEN_UTC_M = 30
_MARKET_CLOSE_UTC_H = 20
_MARKET_CLOSE_UTC_M = 0


# ---------------------------------------------------------------------------
# Event type inference from V2Themes
# ---------------------------------------------------------------------------


def _infer_event_type(v2themes: str) -> str:
    """
    Infer the primary event type from GDELT V2Themes string.

    Uses keyword matching on the semicolon-separated GDELT themes.
    Returns one of the 10 canonical event type labels.
    """
    text = v2themes.lower()

    if any(k in text for k in ["earn", "revenue", "eps", "quarterly", "guidance", "profit"]):
        return "Earnings"
    if any(k in text for k in ["merger", "acquisition", "takeover", "buyout", "divest"]):
        return "M&A"
    if any(k in text for k in ["regulation", "sec_", "antitrust", "penalty", "fine", "lawsuit"]):
        return "Regulatory"
    if any(k in text for k in ["federal_reserve", "interest_rate", "inflation", "gdp", "unemployment"]):
        return "Macro"
    if any(k in text for k in ["sanction", "war", "tariff", "military", "geopolit", "conflict"]):
        return "Geopolitical"
    if any(k in text for k in ["credit", "debt", "downgrade", "bankrupt", "bond", "default"]):
        return "Credit"
    if any(k in text for k in ["oil", "gas", "commodity", "opec", "energy_price", "crude"]):
        return "Commodity"
    if any(k in text for k in ["cyber", "hack", "breach", "ransomware", "malware"]):
        return "Cyber"
    if any(k in text for k in ["ceo", "cfo", "resign", "appoint", "executive", "departure"]):
        return "Leadership"
    if any(k in text for k in ["product", "launch", "recall", "patent", "drug", "approval"]):
        return "Product"

    return "Earnings"  # default — Earnings is the most common


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _is_after_hours(ts: pd.Timestamp) -> bool:
    """Return True if ts falls outside NYSE/NASDAQ regular session (UTC)."""
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    market_open = ts.replace(hour=_MARKET_OPEN_UTC_H, minute=_MARKET_OPEN_UTC_M, second=0)
    market_close = ts.replace(hour=_MARKET_CLOSE_UTC_H, minute=_MARKET_CLOSE_UTC_M, second=0)
    is_weekend = ts.weekday() >= 5
    return is_weekend or not (market_open <= ts < market_close)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all training features from the labeled events DataFrame.

    Features produced (must match XGBoostImpactModel.get_feature_vector):
      sentiment          float  normalized GDELT tone [-1, +1]
      abs_sentiment      float  |sentiment|
      tone_positive      float  GDELT V2Tone positive score (raw)
      tone_negative      float  GDELT V2Tone negative score (raw)
      source_count       int    (always 1 per GDELT row; bumped in diffusion)
      is_after_hours     float  0/1 — article published outside regular session
      days_since_last    float  days since previous news on same ticker
      event_type_*       float  one-hot columns for 10 event types
      sector_*           float  one-hot columns for 8 sectors
    """
    out = pd.DataFrame(index=df.index)

    # Sentiment from GDELT V2Tone: field is -10 to +10; normalise to [-1, +1]
    out["sentiment"] = (df["tone_overall"] / 10.0).clip(-1.0, 1.0)
    out["abs_sentiment"] = out["sentiment"].abs()
    out["tone_positive"] = df["tone_positive"].fillna(0.0)
    out["tone_negative"] = df["tone_negative"].fillna(0.0).abs()  # positive magnitude

    # Source count — GDELT rows are individual articles; 1 per row
    out["source_count"] = 1.0

    # After-hours flag
    out["is_after_hours"] = df["article_ts"].apply(
        lambda ts: float(_is_after_hours(ts)) if pd.notna(ts) else 0.0
    )

    # Days since last event per ticker
    df_sorted = df.sort_values(["ticker", "article_ts"]).copy()
    df_sorted["prev_ts"] = df_sorted.groupby("ticker")["article_ts"].shift(1)
    df_sorted["days_since_last"] = (
        (df_sorted["article_ts"] - df_sorted["prev_ts"])
        .dt.total_seconds()
        .div(86400)
        .fillna(30.0)  # default 30 days if no prior event
        .clip(0.0, 90.0)
    )
    # Align back to original index order
    out["days_since_last"] = df_sorted["days_since_last"].reindex(df.index).fillna(30.0)

    # Event type one-hot
    event_types = df["V2Themes"].apply(_infer_event_type)
    for et in EVENT_TYPES:
        col = f"event_type_{et.lower().replace('&', 'and').replace(' ', '_')}"
        out[col] = (event_types == et).astype(float)

    # Sector one-hot
    sectors = df["ticker"].map(_TICKER_SECTOR).fillna("Technology")
    for sec in SECTORS:
        col = f"sector_{sec.lower().replace(' ', '_')}"
        out[col] = (sectors == sec).astype(float)

    return out


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------


def _rule_based_impact(row: pd.Series, event_type: str) -> float:
    """Replicate the rule-based formula from ImpactCalculator._calculate_rule_based."""
    base = BASE_IMPACT.get(event_type, 5.0)
    sentiment = float(row.get("sentiment", 0.0))
    effective_novelty = 0.7  # assumed average novelty for baseline
    diffusion_score = 0.5    # assumed average diffusion for baseline
    impact = base * (0.5 + 0.5 * abs(sentiment)) * effective_novelty * (0.8 + 0.2 * diffusion_score)
    return float(np.clip(impact, 1.0, 10.0))


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def train() -> None:
    if not _LABELED_PATH.exists():
        raise FileNotFoundError(
            f"Labeled data not found at {_LABELED_PATH}. "
            "Run scripts/build_training_labels.py first."
        )

    log.info("Loading labeled events from %s", _LABELED_PATH)
    df = pd.read_parquet(_LABELED_PATH)
    log.info("Loaded %d rows", len(df))

    # Ensure chronological order — required for time-based split
    df = df.sort_values("article_ts").reset_index(drop=True)

    # ── Feature engineering ───────────────────────────────────────────────────
    log.info("Engineering features …")
    X = build_features(df)
    y = df["impact_label"].values.astype(float)

    feature_columns = list(X.columns)
    log.info("Feature set (%d features): %s", len(feature_columns), feature_columns)

    # ── Time-based split — NO random shuffling ────────────────────────────────
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end].values, y[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end].values, y[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:].values, y[val_end:]

    log.info(
        "Split: train=%d  val=%d  test=%d  (%.0f/%.0f/%.0f%%)",
        len(X_train), len(X_val), len(X_test),
        100 * len(X_train) / n, 100 * len(X_val) / n, 100 * len(X_test) / n,
    )

    date_train_end = df.iloc[train_end]["article_ts"]
    date_val_end   = df.iloc[val_end]["article_ts"]
    log.info("Train covers:      up to %s", date_train_end)
    log.info("Validation covers: %s → %s", date_train_end, date_val_end)
    log.info("Test covers:       %s → %s", date_val_end, df["article_ts"].max())

    # ── XGBoost training ──────────────────────────────────────────────────────
    log.info("Training XGBoost regressor …")
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=20,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = model.best_iteration
    log.info("Best iteration: %d (early stopping on validation RMSE)", best_iter)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred_test = np.clip(model.predict(X_test), 1.0, 10.0)

    # Rule-based baseline predictions on test set
    test_df = df.iloc[val_end:].copy()
    test_df_feat = X.iloc[val_end:].copy()
    test_df["_sentiment"] = test_df_feat["sentiment"].values
    event_types_test = test_df["V2Themes"].apply(_infer_event_type)

    baseline_preds = np.array([
        _rule_based_impact(row, event_types_test.iloc[i])
        for i, (_, row) in enumerate(test_df.iterrows())
    ])

    xgb_mae  = mean_absolute_error(y_test, y_pred_test)
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    xgb_rho,  _ = spearmanr(y_test, y_pred_test)

    base_mae  = mean_absolute_error(y_test, baseline_preds)
    base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))
    base_rho, _ = spearmanr(y_test, baseline_preds)

    # ── Validation metrics ────────────────────────────────────────────────────
    y_pred_val = np.clip(model.predict(X_val), 1.0, 10.0)
    val_mae  = mean_absolute_error(y_val, y_pred_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))

    log.info("\n%s", "=" * 60)
    log.info("  Evaluation Results (test set: %d samples)", len(y_test))
    log.info("%s", "=" * 60)
    log.info("  %-30s  %8s  %8s  %8s", "Model", "MAE", "RMSE", "Spearman ρ")
    log.info("  %s", "-" * 57)
    log.info("  %-30s  %8.4f  %8.4f  %+8.4f", "Rule-based baseline", base_mae, base_rmse, base_rho)
    log.info("  %-30s  %8.4f  %8.4f  %+8.4f", "XGBoost (test)",      xgb_mae,  xgb_rmse,  xgb_rho)
    log.info("  %-30s  %8.4f  %8.4f", "XGBoost (validation)", val_mae, val_rmse)
    log.info("%s", "=" * 60)

    improvement_mae = (base_mae - xgb_mae) / base_mae * 100 if base_mae > 0 else 0.0
    log.info("  MAE improvement over rule-based: %+.1f%%", improvement_mae)
    log.info("%s\n", "=" * 60)

    # ── Feature importance ────────────────────────────────────────────────────
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)
    log.info("Top 10 feature importances:")
    for feat, imp in feat_imp[:10]:
        bar = "█" * int(imp * 50)
        log.info("  %-40s  %.4f  %s", feat, imp, bar)

    # ── Save artifacts ────────────────────────────────────────────────────────
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(_MODEL_OUT, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("Model saved to %s", _MODEL_OUT)

    features_meta = {
        "feature_columns": feature_columns,
        "event_types": EVENT_TYPES,
        "sectors": SECTORS,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "best_iteration": int(best_iter),
        "test_mae": float(xgb_mae),
        "test_rmse": float(xgb_rmse),
        "test_spearman_rho": float(xgb_rho),
        "baseline_mae": float(base_mae),
        "baseline_spearman_rho": float(base_rho),
    }
    _FEATURES_OUT.write_text(json.dumps(features_meta, indent=2))
    log.info("Feature metadata saved to %s", _FEATURES_OUT)

    log.info("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    train()
