"""
fetch_training_data.py — Download and cache GDELT GKG + yfinance price data.

Usage:
    python scripts/fetch_training_data.py [--days 730]

What it does:
  1. Downloads GDELT 2.0 GKG 15-minute bulk files for the requested lookback window,
     filters rows mentioning any company in our 20-ticker universe, and saves the
     result to data/training/gdelt_gkg_filtered.parquet.

  2. Downloads yfinance OHLCV data for all 20 tickers + SPY:
       - Daily bars for the full lookback window (up to 2 years)
       - Hourly bars for the last 60 days (yfinance limit)
     and saves each series to data/training/prices_{interval}_{ticker}.parquet.

  3. Writes a manifest to data/training/manifest.json recording what was fetched
     and when.

No look-ahead risk here: this script is offline data preparation; temporal
integrity is enforced in build_training_labels.py.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "training"
_CONFIGS_DIR = _PROJECT_ROOT / "configs"

_GDELT_OUT = _DATA_DIR / "gdelt_gkg_filtered.parquet"
_MANIFEST_OUT = _DATA_DIR / "manifest.json"

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

# 20-ticker universe + benchmark
_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "JPM", "BAC", "GS", "V", "JNJ", "UNH", "PFE",
    "XOM", "CVX", "CAT", "BA", "WMT", "HD",
    "SPY",  # market benchmark
]

# Company name fragments used to filter GDELT rows.
# Drawn from ticker_map.yaml aliases (lower-cased for substring match).
_COMPANY_FILTERS: list[str] = [
    "apple", "microsoft", "nvidia", "alphabet", "google", "amazon",
    "meta", "facebook", "tesla", "jpmorgan", "jp morgan", "bank of america",
    "goldman sachs", "visa", "johnson", "unitedhealth", "pfizer",
    "exxon", "chevron", "caterpillar", "boeing", "walmart", "home depot",
]

# GDELT 2.0 GKG columns we need (the full file has ~60 columns)
_GKG_COLS = {
    0: "GKGRECORDID",
    1: "DATE",
    3: "SourceCommonName",
    4: "DocumentIdentifier",
    7: "V2Themes",
    15: "V2Tone",
}

_GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
_GDELT_MASTER_URL = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
_RETRY_WAIT = 5   # seconds between retries
_MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_ticker_map() -> dict[str, str]:
    """Return {alias_lower: ticker} from configs/ticker_map.yaml."""
    try:
        import yaml
    except ImportError:
        log.warning("PyYAML not available — using built-in company filter list")
        return {}

    path = _CONFIGS_DIR / "ticker_map.yaml"
    if not path.exists():
        return {}

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    alias_to_ticker: dict[str, str] = {}
    for ticker, info in raw.get("tickers", {}).items():
        alias_to_ticker[info["company"].lower()] = ticker
        for alias in info.get("aliases", []):
            alias_to_ticker[alias.lower()] = ticker
    return alias_to_ticker


def _row_matches_universe(themes: str, source: str, alias_map: dict[str, str]) -> bool:
    """Return True if the GKG row is about a company in our universe."""
    combined = (themes + " " + source).lower()
    if alias_map:
        return any(alias in combined for alias in alias_map)
    return any(frag in combined for frag in _COMPANY_FILTERS)


def _retry_get(url: str, stream: bool = False, timeout: int = 30) -> requests.Response | None:
    """GET with retry/backoff; returns None on final failure."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, stream=stream, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None  # file genuinely missing — don't retry
            log.warning("HTTP %d for %s (attempt %d/%d)", resp.status_code, url, attempt, _MAX_RETRIES)
        except requests.RequestException as exc:
            log.warning("Request error for %s: %s (attempt %d/%d)", url, exc, attempt, _MAX_RETRIES)
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_WAIT * attempt)
    return None


def _parse_gkg_chunk(raw_bytes: bytes, alias_map: dict[str, str]) -> list[dict]:
    """
    Parse a single GKG CSV (tab-separated, no header) and return matching rows.

    GDELT GKG v2 fields are tab-separated.  We only extract the columns listed
    in _GKG_COLS; everything else is discarded immediately to keep memory low.
    """
    rows: list[dict] = []
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            # Guard against malformed rows
            max_col = max(_GKG_COLS.keys())
            if len(fields) <= max_col:
                continue

            themes = fields[7] if len(fields) > 7 else ""
            source = fields[3] if len(fields) > 3 else ""

            if not _row_matches_universe(themes, source, alias_map):
                continue

            rows.append({
                "GKGRECORDID": fields[0],
                "DATE": fields[1],
                "SourceCommonName": fields[3] if len(fields) > 3 else "",
                "DocumentIdentifier": fields[4] if len(fields) > 4 else "",
                "V2Themes": fields[7] if len(fields) > 7 else "",
                "V2Tone": fields[15] if len(fields) > 15 else "",
            })
    except Exception as exc:
        log.debug("Chunk parse error: %s", exc)
    return rows


def _gdelt_filename_for_dt(dt: datetime) -> str:
    """Return the GKG filename for the 15-min bucket containing dt."""
    # Round down to nearest 15 minutes
    minute = (dt.minute // 15) * 15
    snapped = dt.replace(minute=minute, second=0, microsecond=0)
    return snapped.strftime("%Y%m%d%H%M%S") + ".gkg.csv.zip"


def _build_gdelt_urls(days: int) -> list[str]:
    """
    Build a list of GDELT GKG file URLs for the last `days` days.

    Attempts to use the GDELT master file list first (faster lookup);
    falls back to constructing URLs directly from timestamps.
    """
    end_dt = datetime.now(tz=timezone.utc)
    start_dt = end_dt - timedelta(days=days)

    # Try master file list
    log.info("Fetching GDELT master file list …")
    resp = _retry_get(_GDELT_MASTER_URL, timeout=60)
    if resp is not None:
        urls = []
        for line in resp.text.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                url = parts[2].strip()
                if url.endswith(".gkg.csv.zip"):
                    # Extract timestamp from filename
                    fname = url.split("/")[-1]
                    ts_str = fname.replace(".gkg.csv.zip", "")
                    try:
                        file_dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                        if start_dt <= file_dt <= end_dt:
                            urls.append(url)
                    except ValueError:
                        continue
        if urls:
            log.info("Master list: found %d GKG files in range", len(urls))
            return sorted(urls)

    # Fallback: construct URLs every 15 minutes
    log.info("Constructing GKG URLs from timestamps …")
    urls = []
    current = start_dt
    while current <= end_dt:
        fname = _gdelt_filename_for_dt(current)
        urls.append(f"{_GDELT_BASE_URL}/{fname}")
        current += timedelta(minutes=15)
    return urls


# ---------------------------------------------------------------------------
# GDELT fetch
# ---------------------------------------------------------------------------


def fetch_gdelt(days: int) -> pd.DataFrame:
    """
    Download and filter GDELT GKG files for the last `days` days.

    Returns a DataFrame with columns:
        GKGRECORDID, DATE, SourceCommonName, DocumentIdentifier,
        V2Themes, V2Tone, article_ts (datetime, UTC)
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    alias_map = _load_ticker_map()

    urls = _build_gdelt_urls(days)
    log.info("Will attempt %d GDELT GKG files", len(urls))

    all_rows: list[dict] = []
    skipped = 0

    for url in tqdm(urls, desc="GDELT GKG files", unit="file"):
        resp = _retry_get(url, stream=True, timeout=60)
        if resp is None:
            skipped += 1
            continue

        try:
            zip_bytes = resp.content
            with gzip.GzipFile(fileobj=io.BytesIO(zip_bytes)) as gz:
                csv_bytes = gz.read()
            rows = _parse_gkg_chunk(csv_bytes, alias_map)
            all_rows.extend(rows)
        except Exception as exc:
            log.debug("Failed to decompress %s: %s", url, exc)
            skipped += 1

    log.info("GDELT: %d matching rows fetched, %d files skipped", len(all_rows), skipped)

    if not all_rows:
        log.warning("No GDELT rows found — returning empty DataFrame")
        return pd.DataFrame(columns=["GKGRECORDID", "DATE", "SourceCommonName",
                                      "DocumentIdentifier", "V2Themes", "V2Tone", "article_ts"])

    df = pd.DataFrame(all_rows)

    # Parse DATE field: GDELT uses YYYYMMDDHHmmss format
    def _parse_gdelt_date(s: str) -> pd.Timestamp | pd.NaT:
        try:
            return pd.Timestamp(datetime.strptime(str(s), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc))
        except (ValueError, TypeError):
            return pd.NaT

    df["article_ts"] = df["DATE"].apply(_parse_gdelt_date)
    df = df.dropna(subset=["article_ts"])
    df = df.sort_values("article_ts").reset_index(drop=True)

    log.info("Saving %d rows to %s", len(df), _GDELT_OUT)
    df.to_parquet(_GDELT_OUT, index=False)
    return df


# ---------------------------------------------------------------------------
# Price fetch
# ---------------------------------------------------------------------------


def fetch_prices(days: int) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Download OHLCV data for all tickers and save per-ticker parquet files.

    Returns {ticker: {"daily": df, "1h": df}}.
    """
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    price_data: dict[str, dict[str, pd.DataFrame]] = {}

    end_dt = datetime.now(tz=timezone.utc)
    daily_start = (end_dt - timedelta(days=days)).strftime("%Y-%m-%d")
    hourly_start = (end_dt - timedelta(days=60)).strftime("%Y-%m-%d")  # yfinance 60d limit for 1h

    for ticker in tqdm(_TICKERS, desc="Downloading prices", unit="ticker"):
        price_data[ticker] = {}
        t = yf.Ticker(ticker)

        # ── Daily bars ────────────────────────────────────────────────────────
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                daily = t.history(start=daily_start, interval="1d", auto_adjust=True)
                if not daily.empty:
                    daily = daily.tz_convert("UTC") if daily.index.tz is not None else daily.tz_localize("UTC")
                    daily.index.name = "datetime"
                    daily_path = _DATA_DIR / f"prices_daily_{ticker}.parquet"
                    daily.to_parquet(daily_path)
                    price_data[ticker]["daily"] = daily
                    log.debug("Daily %s: %d rows", ticker, len(daily))
                break
            except Exception as exc:
                log.warning("Daily download failed for %s (attempt %d): %s", ticker, attempt, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_WAIT * attempt)

        # ── Hourly bars ───────────────────────────────────────────────────────
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                hourly = t.history(start=hourly_start, interval="1h", auto_adjust=True)
                if not hourly.empty:
                    hourly = hourly.tz_convert("UTC") if hourly.index.tz is not None else hourly.tz_localize("UTC")
                    hourly.index.name = "datetime"
                    hourly_path = _DATA_DIR / f"prices_1h_{ticker}.parquet"
                    hourly.to_parquet(hourly_path)
                    price_data[ticker]["1h"] = hourly
                    log.debug("Hourly %s: %d rows", ticker, len(hourly))
                break
            except Exception as exc:
                log.warning("Hourly download failed for %s (attempt %d): %s", ticker, attempt, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_WAIT * attempt)

        # Polite delay between tickers
        time.sleep(0.5)

    return price_data


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(days: int, gdelt_rows: int, tickers_fetched: list[str]) -> None:
    manifest = {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "lookback_days": days,
        "gdelt_rows_filtered": gdelt_rows,
        "tickers": tickers_fetched,
        "files": {
            "gdelt": str(_GDELT_OUT.relative_to(_PROJECT_ROOT)),
            "prices": [
                f"data/training/prices_daily_{t}.parquet" for t in tickers_fetched
            ] + [
                f"data/training/prices_1h_{t}.parquet" for t in tickers_fetched
            ],
        },
    }
    _MANIFEST_OUT.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to %s", _MANIFEST_OUT)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GDELT + price training data")
    parser.add_argument(
        "--days",
        type=int,
        default=730,
        help="Lookback window in days (default: 730 = ~2 years)",
    )
    args = parser.parse_args()

    log.info("=== fetch_training_data.py  [lookback: %d days] ===", args.days)

    # 1. GDELT
    gdelt_df = fetch_gdelt(args.days)

    # 2. Prices
    price_data = fetch_prices(args.days)
    tickers_ok = [t for t, d in price_data.items() if d]

    # 3. Manifest
    write_manifest(args.days, len(gdelt_df), tickers_ok)

    log.info("Done. GDELT rows: %d | Tickers with data: %d", len(gdelt_df), len(tickers_ok))


if __name__ == "__main__":
    main()
