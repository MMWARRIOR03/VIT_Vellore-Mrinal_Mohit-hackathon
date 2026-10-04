"""
build_training_labels.py — Compute abnormal-return labels for the training set.

Usage:
    python scripts/build_training_labels.py

For each GDELT news event (article) at timestamp t linked to ticker X:
  1. Find the next tradeable price AFTER t (no look-ahead).
  2. Compute the 1-hour and 1-day abnormal return vs SPY.
  3. Map the maximum |abnormal return| to an integer impact label 1–10
     using empirical percentile bins from the training set.
  4. Assert that every label's return observation timestamp strictly follows
     the article timestamp — any violation raises immediately.

Output files:
  data/training/labeled_events.parquet  — full labeled dataset
  data/training/impact_percentiles.json — percentile boundaries for inference
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "training"
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

_GDELT_PATH = _DATA_DIR / "gdelt_gkg_filtered.parquet"
_LABELED_OUT = _DATA_DIR / "labeled_events.parquet"
_PERCENTILES_OUT = _DATA_DIR / "impact_percentiles.json"

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
# Constants
# ---------------------------------------------------------------------------

# 20-ticker universe (SPY is the market benchmark — not labeled)
_UNIVERSE_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "BAC", "GS", "V", "JNJ", "UNH", "PFE",
    "XOM", "CVX", "CAT", "BA", "WMT", "HD",
]

# Percentile bin edges → impact label mapping
# The label boundaries below define which percentile of |abnormal return|
# maps to which 1-10 label bucket.
_PERCENTILE_BINS: list[tuple[float, float, tuple[int, int]]] = [
    (0.0,  10.0, (1, 2)),   # negligible
    (10.0, 30.0, (3, 4)),   # low
    (30.0, 60.0, (4, 6)),   # moderate
    (60.0, 80.0, (6, 8)),   # high
    (80.0, 100.0, (8, 10)), # extreme
]

# How to look up tickers from GDELT rows: check V2Themes + SourceCommonName
_TICKER_ALIAS_CACHE: dict[str, list[str]] | None = None


# ---------------------------------------------------------------------------
# Ticker alias loading
# ---------------------------------------------------------------------------


def _get_ticker_aliases() -> dict[str, list[str]]:
    """Return {ticker: [alias1, alias2, ...]} from configs/ticker_map.yaml."""
    global _TICKER_ALIAS_CACHE
    if _TICKER_ALIAS_CACHE is not None:
        return _TICKER_ALIAS_CACHE

    try:
        import yaml
        path = _CONFIGS_DIR / "ticker_map.yaml"
        with path.open() as fh:
            raw = yaml.safe_load(fh)
        result: dict[str, list[str]] = {}
        for ticker, info in raw.get("tickers", {}).items():
            aliases = [info["company"].lower()] + [a.lower() for a in info.get("aliases", [])]
            result[ticker] = aliases
        _TICKER_ALIAS_CACHE = result
        return result
    except Exception as exc:
        log.warning("Could not load ticker_map.yaml: %s — using ticker symbols only", exc)
        _TICKER_ALIAS_CACHE = {t: [t.lower()] for t in _UNIVERSE_TICKERS}
        return _TICKER_ALIAS_CACHE


def _identify_tickers(v2themes: str, source: str) -> list[str]:
    """Return all tickers mentioned in the GDELT row."""
    text = (v2themes + " " + source).lower()
    aliases = _get_ticker_aliases()
    found: list[str] = []
    for ticker in _UNIVERSE_TICKERS:
        for alias in aliases.get(ticker, [ticker.lower()]):
            if alias and alias in text:
                found.append(ticker)
                break
    return found


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def _load_hourly_prices(ticker: str) -> pd.Series | None:
    """Load 1-hour close prices for ticker, indexed by UTC datetime."""
    path = _DATA_DIR / f"prices_1h_{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df["Close"].sort_index()


def _load_daily_prices(ticker: str) -> pd.Series | None:
    """Load daily close prices for ticker, indexed by UTC datetime."""
    path = _DATA_DIR / f"prices_daily_{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df["Close"].sort_index()


def _next_price_after(prices: pd.Series, ts: pd.Timestamp) -> tuple[float, pd.Timestamp] | None:
    """
    Return (price, price_ts) for the first bar whose timestamp is strictly
    after `ts`.  Returns None if no such bar exists.

    This is the core no-look-ahead guarantee: we never use prices at or
    before the article timestamp.
    """
    future = prices[prices.index > ts]
    if future.empty:
        return None
    price_ts = future.index[0]
    return float(future.iloc[0]), price_ts


def _price_at_or_after(prices: pd.Series, ts: pd.Timestamp, window_td: pd.Timedelta) -> float | None:
    """
    Return the first close price in (ts, ts + window_td].
    Used to look up the end-of-window price without look-ahead.
    """
    window = prices[(prices.index > ts) & (prices.index <= ts + window_td)]
    if window.empty:
        return None
    return float(window.iloc[-1])


# ---------------------------------------------------------------------------
# Abnormal return computation
# ---------------------------------------------------------------------------


def _compute_abnormal_returns(
    article_ts: pd.Timestamp,
    ticker: str,
    hourly_prices: dict[str, pd.Series],
    daily_prices: dict[str, pd.Series],
    spy_hourly: pd.Series,
    spy_daily: pd.Series,
) -> dict | None:
    """
    Compute 1h and 1d abnormal returns for a single (article, ticker) pair.

    Returns a dict with keys:
      entry_ts, entry_price, ar_1h, ar_1d, abs_ar_max, return_ts
    or None if price data is insufficient.

    CRITICAL: every return observation is strictly after article_ts.
    """
    stock_h = hourly_prices.get(ticker)
    stock_d = daily_prices.get(ticker)
    if stock_h is None and stock_d is None:
        return None

    # Entry point: first available bar after article_ts
    entry = None
    if stock_h is not None:
        entry = _next_price_after(stock_h, article_ts)
    if entry is None and stock_d is not None:
        entry = _next_price_after(stock_d, article_ts)
    if entry is None:
        return None

    entry_price, entry_ts = entry

    # ── 1-hour abnormal return ────────────────────────────────────────────────
    ar_1h: float | None = None
    if stock_h is not None:
        end_price_1h = _price_at_or_after(stock_h, article_ts, pd.Timedelta(hours=1))
        spy_end_1h = _price_at_or_after(spy_hourly, article_ts, pd.Timedelta(hours=1))
        spy_start_1h = _next_price_after(spy_hourly, article_ts)

        if end_price_1h is not None and spy_end_1h is not None and spy_start_1h is not None:
            spy_start_price, spy_start_ts = spy_start_1h
            # Ensure no look-ahead
            assert spy_start_ts > article_ts, f"SPY look-ahead: {spy_start_ts} <= {article_ts}"

            stock_ret_1h = (end_price_1h - entry_price) / entry_price if entry_price != 0 else 0.0
            spy_ret_1h = (spy_end_1h - spy_start_price) / spy_start_price if spy_start_price != 0 else 0.0
            ar_1h = stock_ret_1h - spy_ret_1h

    # ── 1-day abnormal return ─────────────────────────────────────────────────
    ar_1d: float | None = None
    prices_for_day = stock_d if stock_d is not None else stock_h
    spy_for_day = spy_daily if spy_daily is not None else spy_hourly

    end_price_1d = _price_at_or_after(prices_for_day, article_ts, pd.Timedelta(days=1))
    spy_end_1d = _price_at_or_after(spy_for_day, article_ts, pd.Timedelta(days=1))
    spy_start_1d = _next_price_after(spy_for_day, article_ts)

    if end_price_1d is not None and spy_end_1d is not None and spy_start_1d is not None:
        spy_start_price_d, spy_start_ts_d = spy_start_1d
        assert spy_start_ts_d > article_ts, f"SPY daily look-ahead: {spy_start_ts_d} <= {article_ts}"

        stock_ret_1d = (end_price_1d - entry_price) / entry_price if entry_price != 0 else 0.0
        spy_ret_1d = (spy_end_1d - spy_start_price_d) / spy_start_price_d if spy_start_price_d != 0 else 0.0
        ar_1d = stock_ret_1d - spy_ret_1d

    if ar_1h is None and ar_1d is None:
        return None

    # Max |abnormal return| for a robust label
    abs_candidates = [abs(ar_1h) if ar_1h is not None else 0.0,
                      abs(ar_1d) if ar_1d is not None else 0.0]
    abs_ar_max = max(abs_candidates)

    return {
        "entry_ts": entry_ts,
        "entry_price": entry_price,
        "ar_1h": ar_1h,
        "ar_1d": ar_1d,
        "abs_ar_max": abs_ar_max,
        "return_ts": entry_ts,  # timestamp of the return observation
    }


# ---------------------------------------------------------------------------
# Impact label mapping
# ---------------------------------------------------------------------------


def _build_impact_mapper(abs_ar_values: np.ndarray) -> tuple[list[float], callable]:
    """
    Compute percentile boundaries from the training distribution of |AR|.

    Returns (boundaries, map_fn) where:
      boundaries: list of percentile threshold values (saved for inference)
      map_fn: callable(abs_ar) -> float in [1.0, 10.0]
    """
    percentile_thresholds: list[float] = []
    for pct, _, _ in _PERCENTILE_BINS[1:]:  # skip 0th percentile start
        percentile_thresholds.append(float(np.percentile(abs_ar_values, pct)))

    def _map(abs_ar: float) -> float:
        for i, (_, end_pct, (lo, hi)) in enumerate(_PERCENTILE_BINS):
            threshold = percentile_thresholds[i] if i < len(percentile_thresholds) else float("inf")
            if abs_ar < threshold:
                # Interpolate within the band
                start_threshold = percentile_thresholds[i - 1] if i > 0 else 0.0
                band_width = threshold - start_threshold
                if band_width > 0:
                    frac = (abs_ar - start_threshold) / band_width
                else:
                    frac = 0.5
                return float(lo) + frac * (hi - lo)
        # Beyond the last threshold: extreme range
        lo, hi = _PERCENTILE_BINS[-1][2]
        last_threshold = percentile_thresholds[-1]
        # Cap smoothly at 10
        scale = min((abs_ar - last_threshold) / (last_threshold + 1e-9), 1.0)
        return float(lo) + scale * (hi - lo)

    return percentile_thresholds, _map


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_labels() -> pd.DataFrame:
    """Load GDELT + prices, compute labels, save outputs."""

    if not _GDELT_PATH.exists():
        raise FileNotFoundError(
            f"GDELT data not found at {_GDELT_PATH}. "
            "Run scripts/fetch_training_data.py first."
        )

    log.info("Loading GDELT data from %s", _GDELT_PATH)
    gdelt_df = pd.read_parquet(_GDELT_PATH)
    log.info("GDELT rows: %d", len(gdelt_df))

    # ── Load all price series into memory ────────────────────────────────────
    log.info("Loading price data …")
    hourly_prices: dict[str, pd.Series] = {}
    daily_prices: dict[str, pd.Series] = {}
    missing_tickers: list[str] = []

    for ticker in _UNIVERSE_TICKERS + ["SPY"]:
        h = _load_hourly_prices(ticker)
        d = _load_daily_prices(ticker)
        if h is not None:
            hourly_prices[ticker] = h
        if d is not None:
            daily_prices[ticker] = d
        if h is None and d is None:
            missing_tickers.append(ticker)

    if missing_tickers:
        log.warning("No price data found for: %s", missing_tickers)

    spy_hourly = hourly_prices.get("SPY", pd.Series(dtype=float))
    spy_daily = daily_prices.get("SPY", pd.Series(dtype=float))

    if spy_hourly.empty and spy_daily.empty:
        raise RuntimeError("SPY price data is required for abnormal return calculation.")

    # ── First pass: compute raw |AR| for all events ──────────────────────────
    log.info("Computing abnormal returns …")
    records: list[dict] = []

    for _, row in tqdm(gdelt_df.iterrows(), total=len(gdelt_df), desc="Events", unit="event"):
        article_ts = row["article_ts"]
        if pd.isna(article_ts):
            continue

        tickers = _identify_tickers(
            str(row.get("V2Themes", "")),
            str(row.get("SourceCommonName", "")),
        )

        for ticker in tickers:
            try:
                ar_data = _compute_abnormal_returns(
                    article_ts=article_ts,
                    ticker=ticker,
                    hourly_prices=hourly_prices,
                    daily_prices=daily_prices,
                    spy_hourly=spy_hourly,
                    spy_daily=spy_daily,
                )
            except AssertionError as e:
                log.error("Look-ahead bias detected: %s — skipping row", e)
                continue

            if ar_data is None:
                continue

            # ── CRITICAL: assert no look-ahead ─────────────────────────────
            assert ar_data["return_ts"] > article_ts, (
                f"Look-ahead bias: return_ts={ar_data['return_ts']} "
                f"<= article_ts={article_ts} for {ticker}"
            )

            # Parse V2Tone: comma-separated, first field is overall tone
            v2tone_raw = str(row.get("V2Tone", "0,0,0"))
            tone_parts = v2tone_raw.split(",")
            try:
                tone_overall = float(tone_parts[0]) if tone_parts else 0.0
                tone_positive = float(tone_parts[1]) if len(tone_parts) > 1 else 0.0
                tone_negative = float(tone_parts[2]) if len(tone_parts) > 2 else 0.0
            except (ValueError, IndexError):
                tone_overall = tone_positive = tone_negative = 0.0

            records.append({
                "GKGRECORDID": row.get("GKGRECORDID", ""),
                "article_ts": article_ts,
                "ticker": ticker,
                "SourceCommonName": row.get("SourceCommonName", ""),
                "DocumentIdentifier": row.get("DocumentIdentifier", ""),
                "V2Themes": row.get("V2Themes", ""),
                "V2Tone_raw": v2tone_raw,
                "tone_overall": tone_overall,
                "tone_positive": tone_positive,
                "tone_negative": tone_negative,
                "entry_ts": ar_data["entry_ts"],
                "entry_price": ar_data["entry_price"],
                "ar_1h": ar_data["ar_1h"],
                "ar_1d": ar_data["ar_1d"],
                "abs_ar_max": ar_data["abs_ar_max"],
            })

    if not records:
        log.error("No labeled events produced. Check that price data covers the GDELT time range.")
        raise RuntimeError("No labeled events — cannot continue.")

    df = pd.DataFrame(records)
    log.info("Produced %d event-ticker pairs", len(df))

    # ── Build impact labels from empirical distribution ───────────────────────
    log.info("Computing impact labels from percentile bins …")
    abs_ar_arr = df["abs_ar_max"].values.astype(float)

    # Filter extreme outliers before computing percentiles (winsorise at 99.5%)
    p995 = float(np.percentile(abs_ar_arr, 99.5))
    abs_ar_clipped = np.clip(abs_ar_arr, 0.0, p995)

    percentile_thresholds, impact_map = _build_impact_mapper(abs_ar_clipped)
    df["impact_label"] = df["abs_ar_max"].apply(impact_map).clip(1.0, 10.0)

    # Round to 1 decimal for readability while keeping continuous target
    df["impact_label"] = df["impact_label"].round(1)

    # ── Save percentile boundaries for inference ──────────────────────────────
    pct_data = {
        "percentile_thresholds": percentile_thresholds,
        "percentile_levels": [p for p, _, _ in _PERCENTILE_BINS[1:]],
        "bin_definitions": [
            {"pct_range": [lo, hi], "label_range": list(label)}
            for lo, hi, label in _PERCENTILE_BINS
        ],
        "p99_5_winsor": float(p995),
    }
    _PERCENTILES_OUT.write_text(json.dumps(pct_data, indent=2))
    log.info("Percentile boundaries saved to %s", _PERCENTILES_OUT)

    # ── Save labeled dataset ──────────────────────────────────────────────────
    df = df.sort_values("article_ts").reset_index(drop=True)
    df.to_parquet(_LABELED_OUT, index=False)
    log.info("Labeled events saved to %s", _LABELED_OUT)

    # ── Statistics ────────────────────────────────────────────────────────────
    log.info("\n=== Label Statistics ===")
    log.info("Total events:      %d", len(df))
    log.info("Date range:        %s → %s", df["article_ts"].min(), df["article_ts"].max())
    log.info("\nPer-ticker counts:")
    ticker_counts = df["ticker"].value_counts().sort_index()
    for ticker, count in ticker_counts.items():
        log.info("  %-6s  %d", ticker, count)

    log.info("\nImpact label distribution (binned):")
    bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10.1]
    bin_labels = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-9", "9-10"]
    counts, _ = np.histogram(df["impact_label"].values, bins=bins)
    for label, count in zip(bin_labels, counts):
        bar = "█" * min(count // max(1, len(df) // 40), 40)
        log.info("  %s  %6d  %s", label, count, bar)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    build_labels()
