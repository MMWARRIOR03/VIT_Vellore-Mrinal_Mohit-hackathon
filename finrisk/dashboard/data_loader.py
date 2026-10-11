"""
DataLoader — fetches data for the Streamlit dashboard.

Strategy: calls the FastAPI endpoints (not DuckDB directly) so there is
no file-locking conflict when the API server holds the DuckDB connection.

Falls back gracefully when the API is not reachable (empty data).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
_TIMEOUT = 5  # seconds


def _get(path: str, params: dict | None = None) -> list | dict | None:
    """GET from the API, return parsed JSON or None on any error."""
    try:
        resp = requests.get(f"{_API_BASE}{path}", params=params or {}, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        logger.debug("API %s returned %d", path, resp.status_code)
        return None
    except Exception as exc:
        logger.debug("API call failed %s: %s", path, exc)
        return None


class DataLoader:
    """Fetch-based data loader that calls the FastAPI server."""

    def __init__(self, api_base: str | None = None) -> None:
        global _API_BASE
        if api_base:
            _API_BASE = api_base

    # ── Signals ──────────────────────────────────────────────────────────────

    def get_recent_signals(self, limit: int = 50, ticker: str | None = None) -> list[dict]:
        params: dict = {"limit": limit}
        if ticker and ticker != "All":
            params["ticker"] = ticker.upper()
        result = _get("/signals", params)
        if isinstance(result, list):
            return result
        # Keep dashboard charts useful when the API is restarting or unavailable.
        # This mirrors the existing read-only fallback used for stress runs.
        return _read_signals(limit, ticker)

    # ── Articles ─────────────────────────────────────────────────────────────

    def get_recent_articles(self, limit: int = 50, ticker: str | None = None) -> list[dict]:
        params: dict = {"limit": limit}
        if ticker and ticker != "All":
            params["ticker"] = ticker.upper()
        result = _get("/articles/recent", params)
        if isinstance(result, list):
            return result
        return []

    # ── Stress runs — read from DuckDB with retry ─────────────────────────────

    def get_stress_runs(self, limit: int = 20) -> list[dict]:
        """Stress runs are stored in DuckDB — try a short-lived connection."""
        return _read_stress_runs(limit)

    # ── Rebalance events ──────────────────────────────────────────────────────

    def get_rebalance_events(self, limit: int = 100) -> list[dict]:
        return _read_rebalance_events(limit)

    # ── Summary stats ─────────────────────────────────────────────────────────

    def get_summary_stats(self) -> dict:
        signals = self.get_recent_signals(limit=200)
        if not signals:
            return {
                "total_signals_today": 0,
                "mean_impact": 0.0,
                "active_tickers": 0,
                "latest_ts": None,
            }

        today = datetime.now(timezone.utc).date().isoformat()
        today_count = 0
        impacts: list[float] = []
        tickers: set[str] = set()
        latest_ts = None

        for sig in signals:
            ts = sig.get("ts_processed") or sig.get("ts_published", "")
            if ts and str(ts)[:10] == today:
                today_count += 1
            if ts and (latest_ts is None or str(ts) > str(latest_ts)):
                latest_ts = ts
            impacts.append(float(sig.get("impact", 0.0)))
            for t in sig.get("tickers", []):
                if isinstance(t, dict):
                    tickers.add(t.get("ticker", ""))
                elif isinstance(t, str):
                    tickers.add(t)

        return {
            "total_signals_today": today_count,
            "mean_impact": round(sum(impacts) / len(impacts), 2) if impacts else 0.0,
            "active_tickers": len(tickers - {""}),
            "latest_ts": str(latest_ts)[:19].replace("T", " ") if latest_ts else None,
        }

    # ── Ticker stats ──────────────────────────────────────────────────────────

    def get_ticker_stats(self) -> pd.DataFrame:
        signals = self.get_recent_signals(limit=500)
        if not signals:
            return pd.DataFrame(columns=["ticker", "mean_sentiment", "mean_impact", "signal_count"])

        records: list[dict] = []
        for sig in signals:
            sentiment = float(sig.get("sentiment", 0.0))
            impact = float(sig.get("impact", 1.0))
            for t in sig.get("tickers", []):
                ticker = t.get("ticker", "") if isinstance(t, dict) else str(t)
                if ticker:
                    records.append({"ticker": ticker, "sentiment": sentiment, "impact": impact})

        if not records:
            return pd.DataFrame(columns=["ticker", "mean_sentiment", "mean_impact", "signal_count"])

        df = pd.DataFrame(records)
        grouped = (
            df.groupby("ticker")
            .agg(mean_sentiment=("sentiment", "mean"), mean_impact=("impact", "mean"), signal_count=("ticker", "count"))
            .reset_index()
        )
        grouped["mean_sentiment"] = grouped["mean_sentiment"].round(3)
        grouped["mean_impact"] = grouped["mean_impact"].round(2)
        return grouped.sort_values("signal_count", ascending=False)

    # ── Weight history ────────────────────────────────────────────────────────

    def get_weight_history(self) -> list[dict]:
        events = self.get_rebalance_events(limit=200)
        result: list[dict] = []
        for ev in events:
            meta = ev.get("meta", {})
            after = meta.get("after_weights", {})
            if after:
                result.append({"ts_created": ev["ts_created"], "weights": after})
        return result

    # ── Current weights ───────────────────────────────────────────────────────

    def get_current_weights(self) -> dict[str, float]:
        _UNIVERSE = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
            "META", "TSLA", "JPM", "BAC", "GS",
            "V", "JNJ", "UNH", "PFE", "XOM",
            "CVX", "CAT", "BA", "WMT", "HD",
        ]
        eq_weight = round(1.0 / len(_UNIVERSE), 6)
        equal_weights: dict[str, float] = {t: eq_weight for t in _UNIVERSE}

        events = self.get_rebalance_events(limit=1)
        if not events:
            return equal_weights
        meta = events[0].get("meta", {})
        after = meta.get("after_weights", {})
        return after if after else equal_weights

    # ── All tickers ────────────────────────────────────────────────────────────

    def get_all_tickers(self) -> list[str]:
        stats = self.get_ticker_stats()
        if stats.empty:
            return []
        return sorted(stats["ticker"].tolist())


# ── DuckDB fallback for stress runs and rebalance events ─────────────────────
# These are written by Module B/A runners, not via the API.
# We try a short-lived DuckDB connection with a retry to get them.

def _try_duckdb_query(sql: str, params: list | None = None) -> list[tuple] | None:
    """Try to open DuckDB read-only, return rows or None if locked."""
    import duckdb
    db_path = os.getenv("DUCKDB_PATH", "./data/finrisk.duckdb")
    # Resolve relative to finrisk/ root
    p = Path(db_path)
    if not p.is_absolute():
        candidates = [p, Path(__file__).resolve().parents[1] / p]
        for c in candidates:
            if c.exists():
                db_path = str(c)
                break
    try:
        conn = duckdb.connect(db_path, read_only=True)
        rows = conn.execute(sql, params or []).fetchall()
        conn.close()
        return rows
    except Exception as exc:
        logger.debug("DuckDB query failed: %s", exc)
        return None


def _read_stress_runs(limit: int) -> list[dict]:
    rows = _try_duckdb_query(
        "SELECT run_id, ts_created, scenario_name, trigger_signal, pnl_impact, var_95, cvar_95, results_json, meta_json FROM stress_runs ORDER BY ts_created DESC LIMIT ?",
        [limit],
    )
    if rows is None:
        return []
    results = []
    for row in rows:
        run_id, ts, s_name, trigger, pnl, var95, cvar95, res_json, meta_json = row
        results.append({
            "run_id": run_id, "ts_created": ts, "scenario_name": s_name,
            "trigger_signal": trigger, "pnl_impact": pnl, "var_95": var95, "cvar_95": cvar95,
            "results": json.loads(res_json) if res_json else {},
            "meta": json.loads(meta_json) if meta_json else {},
        })
    return results


def _read_signals(limit: int, ticker: str | None = None) -> list[dict]:
    """Read persisted signal payloads when the API endpoint is unavailable."""
    if ticker and ticker != "All":
        symbol = ticker.upper()
        rows = _try_duckdb_query(
            """SELECT payload FROM signals
               WHERE ticker_csv = ? OR ticker_csv LIKE ? OR ticker_csv LIKE ? OR ticker_csv LIKE ?
               ORDER BY ts_published DESC LIMIT ?""",
            [symbol, f"{symbol},%", f"%,{symbol}", f"%,{symbol},%", limit],
        )
    else:
        rows = _try_duckdb_query(
            "SELECT payload FROM signals ORDER BY ts_published DESC LIMIT ?", [limit]
        )
    if rows is None:
        return []
    signals: list[dict] = []
    for (payload,) in rows:
        try:
            signals.append(json.loads(payload))
        except (TypeError, json.JSONDecodeError) as exc:
            logger.warning("Skipping malformed persisted signal: %s", exc)
    return signals


def _read_rebalance_events(limit: int) -> list[dict]:
    rows = _try_duckdb_query(
        "SELECT rebalance_id, ts_created, trigger_signal, delta_json, reason, meta_json FROM rebalance_events ORDER BY ts_created DESC LIMIT ?",
        [limit],
    )
    if rows is None:
        return []
    results = []
    for row in rows:
        r_id, ts, trigger, delta_json, reason, meta_json = row
        results.append({
            "rebalance_id": r_id, "ts_created": ts, "trigger_signal": trigger,
            "delta": json.loads(delta_json) if delta_json else {},
            "reason": reason or "",
            "meta": json.loads(meta_json) if meta_json else {},
        })
    return results
