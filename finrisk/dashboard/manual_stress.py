"""
ManualStressTester — synchronous wrapper used by the dashboard's
"Run Stress Test" button.

Creates a synthetic RiskSignal, bypasses the trigger gate (which
requires impact ≥ 7 and cooldown checks), runs the deterministic
stress pipeline + reverse stress synchronously, and returns both results.

No DB or Redis writes are performed — this is a pure in-process calculation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def run_manual_stress(
    event_type: str,
    impact: float,
    sentiment: float = -0.5,
) -> tuple:
    """
    Run a manual stress test synchronously.

    Parameters
    ──────────
    event_type  : scenario name (must be present in configs/scenarios.yaml)
    impact      : impact score 1–10 (gate is bypassed, so any value works)
    sentiment   : signal sentiment (-1 to +1); affects equity shock direction

    Returns
    ───────
    (StressResult | None, ReverseStressResult | None)

    Both are None only if an unexpected import or construction error occurs.
    """
    try:
        from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef
        from module_b.monte_carlo import MonteCarloEngine
        from module_b.portfolio import SyntheticPortfolio
        from module_b.reverse_stress import ReverseStressTester
        from module_b.scenarios import ScenarioEngine
        from module_b.stress_engine import StressEngine, StressResult
        from module_b.valuation import PortfolioValuator
    except ImportError as exc:
        logger.error("Manual stress import failed: %s", exc)
        return None, None

    # ── Build a synthetic signal that always passes the trigger gate ──────────
    now = datetime.now(tz=timezone.utc)
    impact_clamped = max(1.0, min(10.0, float(impact)))
    half_width = min(0.5, (impact_clamped - 1.0) / 2, (10.0 - impact_clamped) / 2)
    half_width = max(0.05, half_width)

    # Normalise event_probs: event_type gets all the probability
    event_probs: dict[str, float] = {event_type: 1.0}

    try:
        signal = RiskSignal(
            event_id=f"manual-{uuid.uuid4()}",
            ts_published=now,
            ts_processed=now,
            source="news",
            tickers=[TickerRef(ticker="AAPL", relevance=1.0)],
            sentiment=float(max(-1.0, min(1.0, sentiment))),
            sentiment_conf=0.9,
            event_type=event_type,
            event_probs=event_probs,
            impact=impact_clamped,
            impact_range=(
                max(1.0, impact_clamped - half_width),
                min(10.0, impact_clamped + half_width),
            ),
            confidence=0.9,
            novelty=1.0,
            diffusion=DiffusionInfo(sources=1, velocity=0.0, score=0.5),
            decay_halflife_hours=24.0,
            explanation=f"Manual stress test: {event_type} at impact {impact_clamped:.1f}",
            cluster_id=f"manual-cluster-{event_type.lower().replace(' ', '-')}",
        )
    except Exception as exc:
        logger.error("Manual stress signal construction failed: %s", exc)
        return None, None

    # ── Build engine components (no DB, no Redis) ─────────────────────────────
    try:
        portfolio = SyntheticPortfolio()
        valuator = PortfolioValuator()
        scenario_engine = ScenarioEngine()
        mc_engine = MonteCarloEngine(n_paths=5_000)  # reduced for dashboard speed
    except Exception as exc:
        logger.error("Manual stress engine construction failed: %s", exc)
        return None, None

    # ── Run deterministic pipeline (bypass trigger gate) ─────────────────────
    try:
        # Step 2-8: replicate StressEngine.run_stress_test without the gate
        scenario = scenario_engine.get_scenario(event_type)
        scenario = scenario_engine.apply_impact_scaling(scenario, impact_clamped)
        scenario = scenario_engine.apply_sentiment_sign(scenario, sentiment, event_type)
        pd_mult, lgd_mult = scenario_engine.get_pd_lgd_multipliers(event_type, impact_clamped)
        flat_shocks = scenario_engine.flatten_shocks(scenario)
        flat_shocks["pd_multiplier"] = pd_mult
        flat_shocks["lgd_multiplier"] = lgd_mult

        pnl = valuator.compute_total_pnl(portfolio, flat_shocks)

        import uuid as _uuid
        stress_result = StressResult(
            run_id=f"manual-{_uuid.uuid4()}",
            ts_created=now,
            trigger_signal_id=signal.event_id,
            scenario_name=event_type,
            event_type=event_type,
            impact=impact_clamped,
            pre_nav=pnl.pre_nav,
            post_nav=pnl.post_nav,
            delta_nav=pnl.delta_nav,
            delta_pct=pnl.delta_pct,
            breakdown_by_asset_class=pnl.breakdown_by_asset_class,
            breakdown_by_sector=pnl.breakdown_by_sector,
            shocks_applied=flat_shocks,
            top_losers=pnl.top_losers,
        )
    except Exception as exc:
        logger.error("Manual stress deterministic run failed: %s", exc)
        return None, None

    # ── Run Monte Carlo (synchronous, reduced paths) ──────────────────────────
    try:
        mc_result = mc_engine.run(
            portfolio,
            flat_shocks,
            impact_clamped,
            signal.impact_range,
            valuator,
        )
        stress_result.mc_var_95 = mc_result.var_95
        stress_result.mc_es_95 = mc_result.es_95
        stress_result.mc_expected_loss = mc_result.expected_loss
        stress_result.mc_worst_loss = mc_result.worst_loss
    except Exception as exc:
        logger.warning("Manual stress MC run failed: %s — continuing with det result", exc)

    # ── Reverse stress ────────────────────────────────────────────────────────
    reverse_result = None
    try:
        reverse_tester = ReverseStressTester()
        reverse_result = reverse_tester.find_breach_threshold(
            portfolio=portfolio,
            valuator=valuator,
            scenario_engine=scenario_engine,
            mc_engine=mc_engine,
            event_type=event_type,
            loss_limit_pct=0.10,
        )
    except Exception as exc:
        logger.warning("Manual reverse stress failed: %s", exc)

    logger.info(
        "Manual stress [%s] impact=%.1f: ΔNAV=$%.0f (%.2f%%)",
        event_type,
        impact_clamped,
        stress_result.delta_nav,
        stress_result.delta_pct,
    )

    return stress_result, reverse_result
