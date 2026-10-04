"""
StressEngine — orchestrates the full stress-test pipeline.

Pipeline for each incoming signal
───────────────────────────────────
  1. StressTrigger.consume_signals     → gate: impact ≥ 7, confidence ≥ 0.55, no cooldown
  2. ScenarioEngine.get_scenario       → fetch base shock vector for event_type
  3. ScenarioEngine.apply_impact_scaling → scale magnitudes by impact-derived multiplier
  4. ScenarioEngine.apply_sentiment_sign → flip equity sign for Earnings/Product events
  5. ScenarioEngine.get_pd_lgd_multipliers → PD/LGD stress factors for loans
  6. Inject PD/LGD multipliers into shocks
  7. ScenarioEngine.flatten_shocks     → merge into a single flat dict
  8. PortfolioValuator.compute_total_pnl → full first-order P&L calculation
  9. Package into StressResult
 10. persist_result → DuckDB (async)
 11. publish to signals.stress Redis stream (async)

The portfolio is built once at engine construction and shared across all runs.
This is intentional: the portfolio is the fixed book under test, and we want
every stress run to operate on the same positions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.schema.risk_signal import RiskSignal
from app.storage.duckdb_store import DuckDBStore
from app.streams.redis_streams import STREAM_SIGNALS_STRESS, RedisStreamClient
from module_b.portfolio import SyntheticPortfolio
from module_b.scenarios import ScenarioEngine
from module_b.trigger import StressTrigger
from module_b.valuation import PnLResult, PortfolioValuator

logger = logging.getLogger(__name__)

MODEL_VERSION: str = "module-b-v1.0"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StressResult:
    """
    Complete output of a single stress-test run.

    All monetary values are in USD.  ``breakdown_by_asset_class`` and
    ``breakdown_by_sector`` are delta P&L dictionaries (negative = loss).
    ``shocks_applied`` is the fully-resolved flat shock dict after scaling
    and sentiment correction, for auditability.
    """

    run_id: str
    ts_created: datetime
    trigger_signal_id: str
    scenario_name: str
    event_type: str
    impact: float
    pre_nav: float
    post_nav: float
    delta_nav: float
    delta_pct: float
    breakdown_by_asset_class: dict[str, float]
    breakdown_by_sector: dict[str, float]
    shocks_applied: dict[str, Any]
    model_version: str = MODEL_VERSION
    # Top 10 biggest loss positions for the dashboard
    top_losers: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (safe for JSON encoding)."""
        d = {
            "run_id": self.run_id,
            "ts_created": self.ts_created.isoformat(),
            "trigger_signal_id": self.trigger_signal_id,
            "scenario_name": self.scenario_name,
            "event_type": self.event_type,
            "impact": self.impact,
            "pre_nav": self.pre_nav,
            "post_nav": self.post_nav,
            "delta_nav": self.delta_nav,
            "delta_pct": self.delta_pct,
            "breakdown_by_asset_class": self.breakdown_by_asset_class,
            "breakdown_by_sector": self.breakdown_by_sector,
            "shocks_applied": self.shocks_applied,
            "model_version": self.model_version,
            "top_losers": self.top_losers,
        }
        return d


# ── Engine ────────────────────────────────────────────────────────────────────

class StressEngine:
    """
    Orchestrates portfolio stress testing.

    Parameters
    ──────────
    portfolio      : SyntheticPortfolio (created once, shared across runs)
    valuator       : PortfolioValuator
    scenario_engine: ScenarioEngine
    trigger        : StressTrigger
    db             : DuckDBStore | None — if None, persistence is skipped
    redis          : RedisStreamClient | None — if None, publishing is skipped
    """

    def __init__(
        self,
        portfolio: SyntheticPortfolio | None = None,
        valuator: PortfolioValuator | None = None,
        scenario_engine: ScenarioEngine | None = None,
        trigger: StressTrigger | None = None,
        db: DuckDBStore | None = None,
        redis: RedisStreamClient | None = None,
    ) -> None:
        self.portfolio = portfolio or SyntheticPortfolio()
        self.valuator = valuator or PortfolioValuator()
        self.scenario_engine = scenario_engine or ScenarioEngine()
        self.trigger = trigger or StressTrigger()
        self._db = db
        self._redis = redis

        logger.info(
            "StressEngine initialised: %d positions, NAV=$%.0f",
            self.portfolio.position_count()["total"],
            self.portfolio.compute_nav(),
        )

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def run_stress_test(self, signal: RiskSignal) -> StressResult | None:
        """
        Run the full stress-test pipeline for a signal.

        Returns a StressResult if the trigger fired, None otherwise.
        This method is synchronous (pure computation); I/O happens in
        persist_result and on_signal.
        """
        # Step 1: Gate
        if not self.trigger.consume_signals(signal):
            return None

        # Step 2: Base scenario
        scenario = self.scenario_engine.get_scenario(signal.event_type)
        scenario_name = signal.event_type

        # Step 3: Scale shocks by impact
        scenario = self.scenario_engine.apply_impact_scaling(scenario, signal.impact)

        # Step 4: Sentiment sign correction (Earnings / Product)
        scenario = self.scenario_engine.apply_sentiment_sign(
            scenario, signal.sentiment, signal.event_type
        )

        # Step 5: PD/LGD multipliers for loans
        pd_mult, lgd_mult = self.scenario_engine.get_pd_lgd_multipliers(
            signal.event_type, signal.impact
        )

        # Step 6: Inject PD/LGD into shock dict
        flat_shocks = self.scenario_engine.flatten_shocks(scenario)
        flat_shocks["pd_multiplier"] = pd_mult
        flat_shocks["lgd_multiplier"] = lgd_mult

        # Step 7–8: Full valuation
        pnl: PnLResult = self.valuator.compute_total_pnl(self.portfolio, flat_shocks)

        # Step 9: Package result
        result = StressResult(
            run_id=str(uuid.uuid4()),
            ts_created=datetime.now(tz=timezone.utc),
            trigger_signal_id=signal.event_id,
            scenario_name=scenario_name,
            event_type=signal.event_type,
            impact=signal.impact,
            pre_nav=pnl.pre_nav,
            post_nav=pnl.post_nav,
            delta_nav=pnl.delta_nav,
            delta_pct=pnl.delta_pct,
            breakdown_by_asset_class=pnl.breakdown_by_asset_class,
            breakdown_by_sector=pnl.breakdown_by_sector,
            shocks_applied=flat_shocks,
            top_losers=pnl.top_losers,
        )

        logger.info(
            "StressResult: run_id=%s  scenario=%s  ΔNAV=$%,.0f (%.2f%%)  "
            "tickers=%s",
            result.run_id,
            result.scenario_name,
            result.delta_nav,
            result.delta_pct,
            [t.ticker for t in signal.tickers],
        )
        return result

    # ── Async I/O ─────────────────────────────────────────────────────────────

    async def persist_result(self, result: StressResult) -> None:
        """
        Persist a StressResult to DuckDB stress_runs table.

        The DuckDBStore.write_stress_run contract expects:
            run_id, ts_created, scenario_name, trigger_signal (str|None),
            pnl_impact, var_95, cvar_95, results (dict), meta (dict|None)

        var_95 and cvar_95 are left as None in v1 (Monte Carlo is out of scope
        for this deterministic module; they will be filled by a future MC layer).
        """
        if self._db is None:
            logger.debug("No DuckDB configured — skipping persistence for run %s", result.run_id)
            return

        run_dict: dict[str, Any] = {
            "run_id": result.run_id,
            "ts_created": result.ts_created,
            "scenario_name": result.scenario_name,
            "trigger_signal": result.trigger_signal_id,
            "pnl_impact": result.delta_nav,
            "var_95": None,
            "cvar_95": None,
            "results": {
                "pre_nav": result.pre_nav,
                "post_nav": result.post_nav,
                "delta_nav": result.delta_nav,
                "delta_pct": result.delta_pct,
                "breakdown_by_asset_class": result.breakdown_by_asset_class,
                "breakdown_by_sector": result.breakdown_by_sector,
                "top_losers": result.top_losers,
            },
            "meta": {
                "event_type": result.event_type,
                "impact": result.impact,
                "shocks_applied": result.shocks_applied,
                "model_version": result.model_version,
            },
        }

        try:
            await self._db.write_stress_run(run_dict)
            logger.debug("Persisted stress run %s to DuckDB", result.run_id)
        except Exception as exc:
            logger.error("Failed to persist stress run %s: %s", result.run_id, exc)

    async def on_signal(self, signal: RiskSignal) -> StressResult | None:
        """
        High-level entry point called when a signal arrives.

        1. Run the stress test (synchronous computation)
        2. If triggered, persist to DuckDB
        3. Publish to signals.stress Redis stream
        4. Return the result (or None if not triggered)
        """
        result = self.run_stress_test(signal)
        if result is None:
            return None

        # Persist and publish concurrently
        persist_task = asyncio.create_task(self.persist_result(result))

        if self._redis is not None:
            publish_task = asyncio.create_task(
                self._publish_result(result, signal)
            )
            await asyncio.gather(persist_task, publish_task, return_exceptions=True)
        else:
            await persist_task

        return result

    async def _publish_result(
        self,
        result: StressResult,
        original_signal: RiskSignal,
    ) -> None:
        """Publish the stress result to the signals.stress Redis stream."""
        if self._redis is None:
            return

        payload: dict[str, Any] = {
            "run_id": result.run_id,
            "ts_created": result.ts_created.isoformat(),
            "trigger_signal_id": result.trigger_signal_id,
            "scenario_name": result.scenario_name,
            "event_type": result.event_type,
            "impact": result.impact,
            "delta_nav": result.delta_nav,
            "delta_pct": result.delta_pct,
            "pre_nav": result.pre_nav,
            "post_nav": result.post_nav,
            "breakdown_by_asset_class": result.breakdown_by_asset_class,
            "breakdown_by_sector": result.breakdown_by_sector,
            "tickers": [t.ticker for t in original_signal.tickers],
            "model_version": result.model_version,
        }

        try:
            await self._redis.publish(STREAM_SIGNALS_STRESS, payload)
            logger.debug("Published stress run %s to %s", result.run_id, STREAM_SIGNALS_STRESS)
        except Exception as exc:
            logger.error(
                "Failed to publish stress run %s to Redis: %s", result.run_id, exc
            )
