"""
download_prices_for_colab.py

Downloads daily + hourly price data and earnings dates for all 20 tickers + SPY,
then zips everything into a single file: data/training/prices_for_colab.zip

Run locally (yfinance works on your Mac, not in Colab):
    cd finrisk
    .venv/bin/python scripts/download_prices_for_colab.py

Then upload prices_for_colab.zip to Colab (drag-and-drop into the file browser).
"""
from __future__ import annotations

import json
import pickle
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA",
    "JPM","BAC","GS","V","JNJ","UNH","PFE",
    "XOM","CVX","CAT","BA","WMT","HD","SPY",
]
LOOKBACK_DAYS = 730
OUT_DIR = Path("data/training/prices_colab")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ZIP_OUT = Path("data/training/prices_for_colab.zip")

daily_start  = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
hourly_start = (datetime.now() - timedelta(days=59)).strftime("%Y-%m-%d")

print(f"Downloading daily prices from {daily_start}...")
df_daily = yf.download(TICKERS, start=daily_start, interval="1d",
                       auto_adjust=True, progress=True, threads=True)
close_d = df_daily["Close"]
saved = 0
for ticker in TICKERS:
    if ticker in close_d.columns:
        s = close_d[ticker].dropna()
        if not s.empty:
            s.index = s.index.tz_convert("UTC") if s.index.tz else s.index.tz_localize("UTC")
            s.index.name = "datetime"
            s.to_frame("Close").to_parquet(OUT_DIR / f"prices_daily_{ticker}.parquet")
            saved += 1
print(f"✓ Daily saved: {saved}/{len(TICKERS)}")

print(f"\nDownloading hourly prices from {hourly_start}...")
time.sleep(2)
df_hourly = yf.download(TICKERS, start=hourly_start, interval="1h",
                        auto_adjust=True, progress=True, threads=True)
close_h = df_hourly["Close"]
saved_h = 0
for ticker in TICKERS:
    if ticker in close_h.columns:
        s = close_h[ticker].dropna()
        if not s.empty:
            s.index = s.index.tz_convert("UTC") if s.index.tz else s.index.tz_localize("UTC")
            s.index.name = "datetime"
            s.to_frame("Close").to_parquet(OUT_DIR / f"prices_1h_{ticker}.parquet")
            saved_h += 1
print(f"✓ Hourly saved: {saved_h}/{len(TICKERS)}")

print("\nDownloading earnings dates...")
earnings = {}
UNIVERSE = [t for t in TICKERS if t != "SPY"]
for ticker in UNIVERSE:
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            ed.index = ed.index.tz_convert("UTC")
            # Convert to dict for JSON serialisation
            earnings[ticker] = {
                str(ts.isoformat()): {
                    "Reported EPS": row.get("Reported EPS"),
                    "EPS Estimate": row.get("EPS Estimate"),
                }
                for ts, row in ed.iterrows()
            }
            print(f"  {ticker}: {len(ed)} quarters")
    except Exception as e:
        print(f"  {ticker}: failed ({e})")
    time.sleep(0.5)

with open(OUT_DIR / "earnings_dates.json", "w") as f:
    json.dump(earnings, f)
print(f"✓ Earnings saved: {len(earnings)} tickers")

# Zip everything
print(f"\nZipping to {ZIP_OUT}...")
with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
    for file in OUT_DIR.iterdir():
        zf.write(file, file.name)

size_mb = ZIP_OUT.stat().st_size / 1e6
print(f"✓ Done: {ZIP_OUT} ({size_mb:.1f} MB)")
print(f"\nNow upload prices_for_colab.zip to Colab (drag into the file browser on the left).")
