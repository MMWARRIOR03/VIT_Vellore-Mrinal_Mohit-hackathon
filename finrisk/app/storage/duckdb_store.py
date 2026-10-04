"""
DuckDB persistence layer.

Threading model
───────────────
DuckDB's in-process connection is not safe for concurrent writes from multiple
coroutines.  All write paths acquire ``_write_lock`` (an asyncio.Lock) before
touching the database.  Reads do NOT acquire the lock — DuckDB supports
multiple concurrent readers — but they do run in a thread-pool executor to
avoid blocking the event loop on I/O.

Schema notes
────────────
- ``signals``          : one row per RiskSignal.  Full JSON payload is stored
                         in ``payload`` so no schema migration is needed when
                         RiskSignal gains new fields.  Indexed columns
                         (ticker_csv, ts_published, event_type, cluster_id)
                         support the common query patterns without JSON parsing.
- ``articles``         : raw article metadata from the ingestion layer.
- ``clusters``         : aggregate metadata per cluster_id.
- ``rebalance_events`` : output events from Module A.
- ``stress_runs``      : output records from Module B.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from app.schema.risk_signal import RiskSignal

logger = logging.getLogger(__name__)

# ── DDL statements ────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS signals (
    event_id      TEXT        PRIMARY KEY,
    ts_published  TIMESTAMPTZ NOT NULL,
    ts_processed  TIMESTAMPTZ NOT NULL,
    source        TEXT        NOT NULL,
    event_type    TEXT        NOT NULL,
    ticker_csv    TEXT        NOT NULL,   -- comma-separated ticker symbols for LIKE queries
    cluster_id    TEXT        NOT NULL,
    sentiment     DOUBLE      NOT NULL,
    impact        DOUBLE      NOT NULL,
    confidence    DOUBLE      NOT NULL,
    novelty       DOUBLE      NOT NULL,
    payload       TEXT        NOT NULL    -- full RiskSignal JSON
);

CREATE INDEX IF NOT EXISTS idx_signals_ts       ON signals (ts_published);
CREATE INDEX IF NOT EXISTS idx_signals_ticker   ON signals (ticker_csv);
CREATE INDEX IF NOT EXISTS idx_signals_cluster  ON signals (cluster_id);
CREATE INDEX IF NOT EXISTS idx_signals_type     ON signals (event_type);

CREATE TABLE IF NOT EXISTS articles (
    article_id    TEXT        PRIMARY KEY,
    ts_published  TIMESTAMPTZ NOT NULL,
    source        TEXT        NOT NULL,
    url           TEXT,
    headline      TEXT        NOT NULL,
    body          TEXT,
    ticker_csv    TEXT        NOT NULL,
    raw_json      TEXT        NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_ts      ON articles (ts_published);
CREATE INDEX IF NOT EXISTS idx_articles_ticker  ON articles (ticker_csv);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id     TEXT        PRIMARY KEY,
    first_seen     TIMESTAMPTZ NOT NULL,
    last_seen      TIMESTAMPTZ NOT NULL,
    signal_count   INTEGER     NOT NULL DEFAULT 0,
    dominant_type  TEXT,
    tickers_csv    TEXT,
    meta_json      TEXT
);

CREATE TABLE IF NOT EXISTS rebalance_events (
    rebalance_id   TEXT        PRIMARY KEY,
    ts_created     TEXT        NOT NULL,   -- ISO 8601 UTC string
    trigger_signal TEXT,                   -- soft reference to signals.event_id (no FK)
    delta_json     TEXT        NOT NULL,   -- {ticker: weight_delta}
    reason         TEXT,
    meta_json      TEXT
);

CREATE INDEX IF NOT EXISTS idx_rebalance_ts     ON rebalance_events (ts_created);

CREATE TABLE IF NOT EXISTS stress_runs (
    run_id         TEXT        PRIMARY KEY,
    ts_created     TEXT        NOT NULL,   -- ISO 8601 UTC string; avoids pytz dependency
    scenario_name  TEXT        NOT NULL,
    trigger_signal TEXT,                   -- soft reference to signals.event_id (no FK)
    pnl_impact     DOUBLE,
    var_95         DOUBLE,
    cvar_95        DOUBLE,
    results_json   TEXT        NOT NULL,
    meta_json      TEXT
);

CREATE INDEX IF NOT EXISTS idx_stress_ts        ON stress_runs (ts_created);
"""


# ── Store class ───────────────────────────────────────────────────────────────

class DuckDBStore:
    """Async-friendly DuckDB wrapper with a single-writer lock."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._write_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialise(self) -> None:
        """Open the DuckDB connection and apply DDL (idempotent)."""
        self._loop = asyncio.get_running_loop()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self._path))
        # DDL is safe to run without the write lock at startup
        self._conn.executemany  # ensure connection is live
        await self._run_in_executor(self._conn.execute, _DDL)
        logger.info("DuckDB initialised at %s", self._path)

    async def close(self) -> None:
        if self._conn is not None:
            await self._run_in_executor(self._conn.close)
            self._conn = None
            logger.info("DuckDB connection closed.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _run_in_executor(self, fn, *args):  # noqa: ANN001, ANN002
        """Run a synchronous DuckDB call in the default thread-pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args)

    def _assert_open(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDBStore is not open.  Call initialise() first.")
        return self._conn

    # ── Signals ───────────────────────────────────────────────────────────────

    async def write_signal(self, signal: RiskSignal) -> None:
        """Persist a RiskSignal.  Silently ignores duplicate event_id."""
        conn = self._assert_open()
        ticker_csv = ",".join(t.ticker for t in signal.tickers)
        payload = signal.model_dump_json()

        def _insert():
            conn.execute(
                """
                INSERT OR IGNORE INTO signals
                    (event_id, ts_published, ts_processed, source, event_type,
                     ticker_csv, cluster_id, sentiment, impact, confidence,
                     novelty, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    signal.event_id,
                    signal.ts_published,
                    signal.ts_processed,
                    signal.source,
                    signal.event_type,
                    ticker_csv,
                    signal.cluster_id,
                    signal.sentiment,
                    signal.impact,
                    signal.confidence,
                    signal.novelty,
                    payload,
                ],
            )

        async with self._write_lock:
            await self._run_in_executor(_insert)

    async def get_signals(
        self,
        *,
        ticker: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[RiskSignal]:
        """
        Fetch signals with optional filters.

        ``ticker`` matches if the ticker appears anywhere in ticker_csv.
        ``since`` filters by ts_published (inclusive).
        Results are ordered newest-first.
        """
        conn = self._assert_open()

        clauses: list[str] = []
        params: list[Any] = []

        if ticker is not None:
            # Use LIKE to match the CSV column: ',AAPL,' or 'AAPL,' or ',AAPL'
            clauses.append(
                "(ticker_csv = ? OR ticker_csv LIKE ? OR ticker_csv LIKE ? OR ticker_csv LIKE ?)"
            )
            t = ticker.upper()
            params.extend([t, f"{t},%", f"%,{t}", f"%,{t},%"])

        if since is not None:
            clauses.append("ts_published >= ?")
            params.append(since)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT payload FROM signals
            {where}
            ORDER BY ts_published DESC
            LIMIT ?
        """
        params.append(limit)

        def _query():
            return conn.execute(sql, params).fetchall()

        rows = await self._run_in_executor(_query)
        signals: list[RiskSignal] = []
        for (payload,) in rows:
            try:
                signals.append(RiskSignal.model_validate_json(payload))
            except Exception as exc:
                logger.warning("Failed to deserialise signal payload: %s", exc)
        return signals

    # ── Articles ──────────────────────────────────────────────────────────────

    async def write_article(self, article: dict[str, Any]) -> None:
        """
        Persist a raw article dict.

        Expected keys: article_id, ts_published (datetime), source, url,
        headline, body, tickers (list[str]).
        """
        conn = self._assert_open()
        ticker_csv = ",".join(str(t) for t in article.get("tickers", []))
        raw_json = json.dumps(article, default=str)

        def _insert():
            conn.execute(
                """
                INSERT OR IGNORE INTO articles
                    (article_id, ts_published, source, url, headline, body,
                     ticker_csv, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    article["article_id"],
                    article["ts_published"],
                    article.get("source", "unknown"),
                    article.get("url"),
                    article.get("headline", ""),
                    article.get("body"),
                    ticker_csv,
                    raw_json,
                ],
            )

        async with self._write_lock:
            await self._run_in_executor(_insert)

    async def get_articles_recent(
        self,
        *,
        ticker: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return the most recent articles, optionally filtered by ticker."""
        conn = self._assert_open()

        clauses: list[str] = []
        params: list[Any] = []

        if ticker is not None:
            t = ticker.upper()
            clauses.append(
                "(ticker_csv = ? OR ticker_csv LIKE ? OR ticker_csv LIKE ? OR ticker_csv LIKE ?)"
            )
            params.extend([t, f"{t},%", f"%,{t}", f"%,{t},%"])

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT raw_json FROM articles
            {where}
            ORDER BY ts_published DESC
            LIMIT ?
        """
        params.append(limit)

        def _query():
            return conn.execute(sql, params).fetchall()

        rows = await self._run_in_executor(_query)
        return [json.loads(row[0]) for row in rows]

    # ── Stress runs ───────────────────────────────────────────────────────────

    async def write_stress_run(self, run: dict[str, Any]) -> None:
        """
        Persist a stress test result from Module B.

        Expected keys: run_id, ts_created, scenario_name, trigger_signal (str|None),
        pnl_impact, var_95, cvar_95, results (dict), meta (dict|None).
        """
        conn = self._assert_open()
        results_json = json.dumps(run.get("results", {}), default=str)
        meta_json = json.dumps(run.get("meta", {}), default=str)

        ts_str = run["ts_created"].isoformat() if hasattr(run["ts_created"], "isoformat") else str(run["ts_created"])

        def _insert():
            conn.execute(
                """
                INSERT OR IGNORE INTO stress_runs
                    (run_id, ts_created, scenario_name, trigger_signal,
                     pnl_impact, var_95, cvar_95, results_json, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run["run_id"],
                    ts_str,
                    run["scenario_name"],
                    run.get("trigger_signal"),
                    run.get("pnl_impact"),
                    run.get("var_95"),
                    run.get("cvar_95"),
                    results_json,
                    meta_json,
                ],
            )

        async with self._write_lock:
            await self._run_in_executor(_insert)

    async def get_stress_runs(
        self,
        *,
        scenario_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return recent stress runs, optionally filtered by scenario name."""
        conn = self._assert_open()
        params: list[Any] = []
        where = ""
        if scenario_name is not None:
            where = "WHERE scenario_name = ?"
            params.append(scenario_name)

        sql = f"""
            SELECT run_id, ts_created, scenario_name, trigger_signal,
                   pnl_impact, var_95, cvar_95, results_json, meta_json
            FROM stress_runs
            {where}
            ORDER BY ts_created DESC
            LIMIT ?
        """
        params.append(limit)

        def _query():
            return conn.execute(sql, params).fetchall()

        rows = await self._run_in_executor(_query)
        results = []
        for row in rows:
            (run_id, ts_created, s_name, trigger, pnl, var95, cvar95,
             results_json, meta_json) = row
            results.append(
                {
                    "run_id": run_id,
                    "ts_created": ts_created,
                    "scenario_name": s_name,
                    "trigger_signal": trigger,
                    "pnl_impact": pnl,
                    "var_95": var95,
                    "cvar_95": cvar95,
                    "results": json.loads(results_json),
                    "meta": json.loads(meta_json),
                }
            )
        return results

    # ── Rebalance events ──────────────────────────────────────────────────────

    async def write_rebalance_event(self, event: dict[str, Any]) -> None:
        """
        Persist a portfolio rebalance event from Module A.

        Expected keys: rebalance_id, ts_created, trigger_signal (str|None),
        delta (dict[str, float]), reason (str|None), meta (dict|None).
        """
        conn = self._assert_open()
        delta_json = json.dumps(event.get("delta", {}), default=str)
        meta_json = json.dumps(event.get("meta", {}), default=str)

        ts_str = event["ts_created"].isoformat() if hasattr(event["ts_created"], "isoformat") else str(event["ts_created"])

        def _insert():
            conn.execute(
                """
                INSERT OR IGNORE INTO rebalance_events
                    (rebalance_id, ts_created, trigger_signal, delta_json,
                     reason, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    event["rebalance_id"],
                    ts_str,
                    event.get("trigger_signal"),
                    delta_json,
                    event.get("reason"),
                    meta_json,
                ],
            )

        async with self._write_lock:
            await self._run_in_executor(_insert)

    async def get_rebalance_events(
        self,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return recent rebalance events newest-first."""
        conn = self._assert_open()

        def _query():
            return conn.execute(
                """
                SELECT rebalance_id, ts_created, trigger_signal, delta_json,
                       reason, meta_json
                FROM rebalance_events
                ORDER BY ts_created DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()

        rows = await self._run_in_executor(_query)
        results = []
        for row in rows:
            (r_id, ts_created, trigger, delta_json, reason, meta_json) = row
            results.append(
                {
                    "rebalance_id": r_id,
                    "ts_created": ts_created,
                    "trigger_signal": trigger,
                    "delta": json.loads(delta_json),
                    "reason": reason,
                    "meta": json.loads(meta_json),
                }
            )
        return results
