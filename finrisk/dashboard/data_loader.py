"""
DataLoader — read-only DuckDB access layer for the Streamlit dashboard.

All methods use short-lived connections (connect → query → close) and
read_only=True to avoid contention with the FastAPI write path.

Graceful degradation: every method returns [] or an empty DataFrame when
the database is missing, a table doesn't exist yet, or a query fails.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def _db_path() -> str:
    """Resolve DuckDB path from env or fallback default."""
    raw = os.getenv("DUCKDB_PATH", "./data/finrisk.duckdb")
    # Make path relative to finrisk/ root when running from dashboard/
    p = Path(raw)
    if not p.is_absolute() and not p.exists():
        # Try one level up (running from dashboard/)
        candidate = Path(__file__).resolve().parents[1] / p
        if candidate.exists():
            return str(candidate)
    return str(p)


class DataLoader:
    """
    Read-only data access for the dashboard.

    Parameters
    ──────────
    db_path : override the DuckDB path (defaults to DUCKDB_PATH env var).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._path = db_path or _db_path()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Open a read-only connection; raises if db doesn't exist."""
        return duckdb.connect(self._path, read_only=True)

    # ── Signals ───────────────────────────────────────────────────────────────

    def get_recent_signals(
        self,
        limit: int = 50,
        ticker: str | None = None,
    ) -> list[dict]:
        """
        Return last ``limit`` signals as plain dicts parsed from the payload column.

        Optionally filtered by ticker substring match.
        Returns [] on any error.
        """
        try:
            conn = self._connect()
            params: list = []
            clauses: list[str] = []

            if ticker is not None and ticker != "All":
                t = ticker.upper()
                clauses.append(
                    "(ticker_csv = ? OR ticker_csv LIKE ? "
                    "OR ticker_csv LIKE ? OR ticker_csv LIKE ?)"
                )
                params.extend([t, f"{t},%", f"%,{t}", f"%,{t},%"])

            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"""
                SELECT payload
                FROM signals
                {where}
                ORDER BY ts_published DESC
                LIMIT ?
            """
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            conn.close()

            results: list[dict] = []
            for (payload,) in rows:
                try:
                    d = json.loads(payload)
                    results.append(d)
                except Exception:
                    pass
            return results

        except Exception as exc:
            logger.debug("get_recent_signals failed: %s", exc)
            return []

    # ── Articles ──────────────────────────────────────────────────────────────

    def get_recent_articles(
        self,
        limit: int = 50,
        ticker: str | None = None,
    ) -> list[dict]:
        """Return last ``limit`` articles as dicts. Returns [] on error."""
        try:
            conn = self._connect()
            params: list = []
            clauses: list[str] = []

            if ticker is not None and ticker != "All":
                t = ticker.upper()
                clauses.append(
                    "(ticker_csv = ? OR ticker_csv LIKE ? "
                    "OR ticker_csv LIKE ? OR ticker_csv LIKE ?)"
                )
                params.extend([t, f"{t},%", f"%,{t}", f"%,{t},%"])

            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            sql = f"""
                SELECT raw_json
                FROM articles
                {where}
                ORDER BY ts_published DESC
                LIMIT ?
            """
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            conn.close()
            return [json.loads(r[0]) for r in rows]

        except Exception as exc:
            logger.debug("get_recent_articles failed: %s", exc)
            return []

    # ── Stress runs ───────────────────────────────────────────────────────────

    def get_stress_runs(self, limit: int = 20) -> list[dict]:
        """
        Return last ``limit`` stress runs.

        Each dict contains: run_id, ts_created, scenario_name, pnl_impact,
        var_95, cvar_95, results (parsed dict), meta (parsed dict).
        """
        try:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT run_id, ts_created, scenario_name, trigger_signal,
                       pnl_impact, var_95, cvar_95, results_json, meta_json
                FROM stress_runs
                ORDER BY ts_created DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
            conn.close()

            results: list[dict] = []
            for row in rows:
                (run_id, ts_created, scenario_name, trigger,
                 pnl, var95, cvar95, results_json, meta_json) = row
                results.append(
                    {
                        "run_id": run_id,
                        "ts_created": ts_created,
                        "scenario_name": scenario_name,
                        "trigger_signal": trigger,
                        "pnl_impact": pnl,
                        "var_95": var95,
                        "cvar_95": cvar95,
                        "results": json.loads(results_json) if results_json else {},
                        "meta": json.loads(meta_json) if meta_json else {},
                    }
                )
            return results

        except Exception as exc:
            logger.debug("get_stress_runs failed: %s", exc)
            return []

    # ── Rebalance events ──────────────────────────────────────────────────────

    def get_rebalance_events(self, limit: int = 100) -> list[dict]:
        """
        Return last ``limit`` rebalance events.

        Each dict: rebalance_id, ts_created, trigger_signal, delta (parsed dict),
        reason, meta (parsed dict containing before_weights / after_weights).
        """
        try:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT rebalance_id, ts_created, trigger_signal,
                       delta_json, reason, meta_json
                FROM rebalance_events
                ORDER BY ts_created DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
            conn.close()

            results: list[dict] = []
            for row in rows:
                (r_id, ts_created, trigger, delta_json, reason, meta_json) = row
                results.append(
                    {
                        "rebalance_id": r_id,
                        "ts_created": ts_created,
                        "trigger_signal": trigger,
                        "delta": json.loads(delta_json) if delta_json else {},
                        "reason": reason or "",
                        "meta": json.loads(meta_json) if meta_json else {},
                    }
                )
            return results

        except Exception as exc:
            logger.debug("get_rebalance_events failed: %s", exc)
            return []

    # ── Summary stats ─────────────────────────────────────────────────────────

    def get_summary_stats(self) -> dict:
        """
        Returns:
            total_signals_today : int
            mean_impact         : float (last 50 signals, 0 if none)
            active_tickers      : int  (distinct tickers in last 50)
            latest_ts           : str | None
        """
        try:
            conn = self._connect()

            # today UTC
            today = datetime.now(timezone.utc).date().isoformat()

            total_today = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE CAST(ts_published AS DATE) = ?",
                [today],
            ).fetchone()[0]

            stats = conn.execute(
                """
                SELECT AVG(impact), MAX(ts_published)
                FROM (
                    SELECT impact, ts_published
                    FROM signals
                    ORDER BY ts_published DESC
                    LIMIT 50
                ) sub
                """
            ).fetchone()

            mean_impact = float(stats[0]) if stats[0] is not None else 0.0
            latest_ts = str(stats[1]) if stats[1] is not None else None

            # Distinct tickers from last 50 signals
            rows = conn.execute(
                "SELECT ticker_csv FROM signals ORDER BY ts_published DESC LIMIT 50"
            ).fetchall()
            conn.close()

            tickers: set[str] = set()
            for (csv_str,) in rows:
                for t in csv_str.split(","):
                    t = t.strip()
                    if t:
                        tickers.add(t)

            return {
                "total_signals_today": int(total_today),
                "mean_impact": round(mean_impact, 2),
                "active_tickers": len(tickers),
                "latest_ts": latest_ts,
            }

        except Exception as exc:
            logger.debug("get_summary_stats failed: %s", exc)
            return {
                "total_signals_today": 0,
                "mean_impact": 0.0,
                "active_tickers": 0,
                "latest_ts": None,
            }

    # ── Ticker stats ──────────────────────────────────────────────────────────

    def get_ticker_stats(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
            ticker, mean_sentiment, mean_impact, signal_count

        Computed by exploding ticker_csv and grouping by ticker.
        Returns empty DataFrame on error.
        """
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT ticker_csv, sentiment, impact FROM signals"
            ).fetchall()
            conn.close()

            if not rows:
                return pd.DataFrame(
                    columns=["ticker", "mean_sentiment", "mean_impact", "signal_count"]
                )

            records: list[dict] = []
            for (csv_str, sentiment, impact) in rows:
                for t in csv_str.split(","):
                    t = t.strip()
                    if t:
                        records.append(
                            {"ticker": t, "sentiment": sentiment, "impact": impact}
                        )

            if not records:
                return pd.DataFrame(
                    columns=["ticker", "mean_sentiment", "mean_impact", "signal_count"]
                )

            df = pd.DataFrame(records)
            grouped = (
                df.groupby("ticker")
                .agg(
                    mean_sentiment=("sentiment", "mean"),
                    mean_impact=("impact", "mean"),
                    signal_count=("ticker", "count"),
                )
                .reset_index()
            )
            grouped["mean_sentiment"] = grouped["mean_sentiment"].round(3)
            grouped["mean_impact"] = grouped["mean_impact"].round(2)
            return grouped.sort_values("signal_count", ascending=False)

        except Exception as exc:
            logger.debug("get_ticker_stats failed: %s", exc)
            return pd.DataFrame(
                columns=["ticker", "mean_sentiment", "mean_impact", "signal_count"]
            )

    # ── Weight history ────────────────────────────────────────────────────────

    def get_weight_history(self) -> list[dict]:
        """
        Return list of dicts, one per rebalance event, each containing:
            ts_created : str
            weights    : dict[str, float]  (after_weights from meta_json)
        """
        try:
            conn = self._connect()
            rows = conn.execute(
                """
                SELECT ts_created, meta_json
                FROM rebalance_events
                ORDER BY ts_created ASC
                LIMIT 200
                """
            ).fetchall()
            conn.close()

            results: list[dict] = []
            for (ts, meta_json) in rows:
                try:
                    meta = json.loads(meta_json) if meta_json else {}
                    after = meta.get("after_weights", {})
                    if after:
                        results.append({"ts_created": ts, "weights": after})
                except Exception:
                    pass
            return results

        except Exception as exc:
            logger.debug("get_weight_history failed: %s", exc)
            return []

    # ── Current weights ───────────────────────────────────────────────────────

    def get_current_weights(self) -> dict[str, float]:
        """
        Return after_weights from the most recent rebalance event.
        Returns equal-weight dict over the 20 FinRisk tickers if no events exist.
        """
        _UNIVERSE = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "JPM", "BAC", "WFC", "GS", "MS",
            "JNJ", "PFE", "UNH", "ABBV", "MRK",
            "XOM", "CVX", "COP", "SLB", "EOG",
        ]
        eq_weight = round(1.0 / len(_UNIVERSE), 6)
        equal_weights: dict[str, float] = {t: eq_weight for t in _UNIVERSE}

        try:
            conn = self._connect()
            row = conn.execute(
                """
                SELECT meta_json
                FROM rebalance_events
                ORDER BY ts_created DESC
                LIMIT 1
                """
            ).fetchone()
            conn.close()

            if row is None:
                return equal_weights

            meta = json.loads(row[0]) if row[0] else {}
            after = meta.get("after_weights", {})
            return after if after else equal_weights

        except Exception as exc:
            logger.debug("get_current_weights failed: %s", exc)
            return equal_weights

    # ── All known tickers ─────────────────────────────────────────────────────

    def get_all_tickers(self) -> list[str]:
        """Return sorted list of all distinct tickers seen in the signals table."""
        try:
            conn = self._connect()
            rows = conn.execute("SELECT DISTINCT ticker_csv FROM signals").fetchall()
            conn.close()
            tickers: set[str] = set()
            for (csv_str,) in rows:
                for t in csv_str.split(","):
                    t = t.strip()
                    if t:
                        tickers.add(t)
            return sorted(tickers)
        except Exception:
            return []
