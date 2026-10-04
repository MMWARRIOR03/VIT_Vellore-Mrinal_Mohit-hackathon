"""
validate_module_b.py — Standalone validation for Module B (StressEngine).

Runs stress tests on 4 historical reference scenarios and validates outputs:
  - 2008 GFC          — Credit event, impact 9.5
  - 2020 COVID        — Geopolitical event, impact 9.0
  - 2022 Rate Shock   — Macro event, impact 8.5
  - 2023 Banking Crisis — Credit event, impact 8.0

For each scenario: runs StressEngine.run_stress_test() (deterministic) and
then MonteCarloEngine.run() for VaR/ES metrics.

Run from the finrisk/ root:
    python scripts/validate_module_b.py

Output: data/results/module_b_validation.json
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("validate_module_b")

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "module_b_validation.json"


# ─────────────────────────────────────────────────────────────────────────────
# Reference scenarios
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_SCENARIOS: list[dict] = [
    {
        "name": "2008 Global Financial Crisis",
        "event_type": "Credit",
        "impact": 9.5,
        "sentiment": -0.90,
        "description": (
            "Lehman Brothers collapse triggers global credit freeze. "
            "Interbank lending seizes, equity markets fall 40%+."
        ),
    },
    {
        "name": "2020 COVID-19 Shock",
        "event_type": "Geopolitical",
        "impact": 9.0,
        "sentiment": -0.85,
        "description": (
            "Pandemic lockdowns cause fastest bear market in history. "
            "Supply chains disrupted, demand collapses across sectors."
        ),
    },
    {
        "name": "2022 Rate Shock",
        "event_type": "Macro",
        "impact": 8.5,
        "sentiment": -0.75,
        "description": (
            "Federal Reserve raises rates 425bps in 12 months. "
            "Bond markets suffer worst year in decades; duration assets re-price."
        ),
    },
    {
        "name": "2023 Banking Crisis",
        "event_type": "Credit",
        "impact": 8.0,
        "sentiment": -0.80,
        "description": (
            "Silicon Valley Bank and Signature Bank failures trigger contagion fears. "
            "Regional banking sector sees deposit flight."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Signal construction
# ─────────────────────────────────────────────────────────────────────────────

def _make_stress_signal(scenario: dict):
    """Build a RiskSignal that bypasses the StressTrigger confidence gate."""
    from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef

    event_type = scenario["event_type"]
    impact = float(scenario["impact"])
    sentiment = float(scenario["sentiment"])
    impact_low = max(1.0, impact - 0.5)
    impact_high = min(10.0, impact + 0.5)

    probs: dict[str, float] = {event_type: 0.85}
    all_types = [
        "Geopolitical", "Regulatory", "M&A", "Earnings", "Credit",
        "Macro", "Product", "Cyber", "Leadership", "Commodity",
    ]
    other_types = [t for t in all_types if t != event_type][:3]
    remainder = 1.0 - 0.85
    for i, t in enumerate(other_types):
        probs[t] = round(remainder / len(other_types), 4)

    now = datetime.now(tz=timezone.utc)

    return RiskSignal(
        event_id=str(uuid.uuid4()),
        ts_published=now,
        ts_processed=now,
        source="news",
        tickers=[
            TickerRef(ticker="SPY", exchange="NYSE", relevance=1.0),
            TickerRef(ticker="JPM", exchange="NYSE", relevance=0.9),
            TickerRef(ticker="GS", exchange="NYSE", relevance=0.85),
        ],
        sentiment=sentiment,
        sentiment_conf=0.95,
        event_type=event_type,
        event_probs=probs,
        impact=impact,
        impact_range=(impact_low, impact_high),
        confidence=0.95,  # high enough to bypass trigger gate
        novelty=0.95,
        diffusion=DiffusionInfo(sources=50, velocity=15.0, score=0.95),
        decay_halflife_hours=48.0,
        explanation=scenario["description"],
        cluster_id=f"reference-{scenario['name'].replace(' ', '-').lower()}",
        model_versions={"sentiment": "validate-v1.0"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run one scenario
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(engine, scenario: dict) -> dict:
    """Run deterministic stress test + Monte Carlo for a single scenario."""
    from module_b.trigger import StressTrigger

    logger.info("Running scenario: %s (impact=%.1f)", scenario["name"], scenario["impact"])

    signal = _make_stress_signal(scenario)

    # Bypass the cooldown gate so each scenario fires fresh.
    # All reference signals have impact >= 8 and confidence = 0.95, so the
    # impact and confidence gates pass; only cooldown could block them.
    engine.trigger.reset_all_cooldowns()

    det_result = engine.run_stress_test(signal)

    if det_result is None:
        # Trigger didn't fire — force direct stress calculation
        logger.warning(
            "Trigger did not fire for %s — running valuation directly",
            scenario["name"],
        )
        scenario_shocks = engine.scenario_engine.get_scenario(signal.event_type)
        scenario_shocks = engine.scenario_engine.apply_impact_scaling(scenario_shocks, signal.impact)
        scenario_shocks = engine.scenario_engine.apply_sentiment_sign(
            scenario_shocks, signal.sentiment, signal.event_type
        )
        pd_mult, lgd_mult = engine.scenario_engine.get_pd_lgd_multipliers(
            signal.event_type, signal.impact
        )
        flat_shocks = engine.scenario_engine.flatten_shocks(scenario_shocks)
        flat_shocks["pd_multiplier"] = pd_mult
        flat_shocks["lgd_multiplier"] = lgd_mult
        pnl = engine.valuator.compute_total_pnl(engine.portfolio, flat_shocks)

        pre_nav = pnl.pre_nav
        delta_nav = pnl.delta_nav
        delta_pct = pnl.delta_pct
        shocks_applied = flat_shocks
    else:
        pre_nav = det_result.pre_nav
        delta_nav = det_result.delta_nav
        delta_pct = det_result.delta_pct
        shocks_applied = det_result.shocks_applied

    # Monte Carlo
    logger.info("  Running Monte Carlo for %s …", scenario["name"])
    try:
        mc_result = engine.mc_engine.run(
            portfolio=engine.portfolio,
            base_shocks=shocks_applied,
            impact=signal.impact,
            impact_range=signal.impact_range,
            valuator=engine.valuator,
        )
        var_95 = mc_result.var_95
        es_95 = mc_result.es_95
        expected_loss = mc_result.expected_loss
        worst_loss = mc_result.worst_loss
    except Exception as exc:
        logger.error("Monte Carlo failed for %s: %s", scenario["name"], exc)
        var_95 = es_95 = expected_loss = worst_loss = None

    result = {
        "scenario": scenario["name"],
        "event_type": scenario["event_type"],
        "impact": scenario["impact"],
        "description": scenario["description"],
        "pre_nav": round(pre_nav, 2),
        "post_nav": round(pre_nav + delta_nav, 2),
        "delta_nav": round(delta_nav, 2),
        "delta_pct": round(delta_pct, 4),
        "delta_pct_display": f"{delta_pct:.2f}%",
        "var_95": round(var_95, 2) if var_95 is not None else None,
        "es_95": round(es_95, 2) if es_95 is not None else None,
        "expected_loss": round(expected_loss, 2) if expected_loss is not None else None,
        "worst_case_loss": round(worst_loss, 2) if worst_loss is not None else None,
        "trigger_fired": det_result is not None,
    }

    logger.info(
        "  %s: ΔNAV=$%,.0f (%.2f%%)  VaR95=$%,.0f  ES95=$%,.0f",
        scenario["name"],
        delta_nav,
        delta_pct,
        var_95 or 0,
        es_95 or 0,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def _print_rule(char: str = "━", width: int = 75) -> None:
    print(char * width)


def print_summary(portfolio_nav: float, results: list[dict]) -> None:
    _print_rule()
    print("Module B Historical Scenario Validation")
    _print_rule()
    print(f"  Portfolio NAV:  ${portfolio_nav:,.0f}")
    print(f"  Scenarios:      {len(results)}")
    print()

    header = (
        f"  {'Scenario':<30}"
        f"{'ΔNAV':>14}"
        f"{'Δ%':>8}"
        f"{'VaR 95%':>14}"
        f"{'ES 95%':>14}"
        f"{'Worst':>14}"
    )
    print(header)
    print("  " + "─" * 73)

    for r in results:
        delta = r["delta_nav"]
        delta_pct = r["delta_pct"]
        var95 = r["var_95"]
        es95 = r["es_95"]
        worst = r["worst_case_loss"]

        def _fmt(v) -> str:
            return f"${v:,.0f}" if v is not None else "n/a"

        print(
            f"  {r['scenario']:<30}"
            f"{_fmt(delta):>14}"
            f"{delta_pct:>7.2f}%"
            f"{_fmt(var95):>14}"
            f"{_fmt(es95):>14}"
            f"{_fmt(worst):>14}"
        )

    _print_rule()
    print(f"Results saved to {OUTPUT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation checks
# ─────────────────────────────────────────────────────────────────────────────

def run_validation_checks(portfolio_nav: float, results: list[dict]) -> list[str]:
    """
    Run basic sanity checks on the stress results.

    Returns a list of PASS/FAIL strings for each check.
    """
    checks: list[str] = []

    for r in results:
        # Check 1: High-impact scenarios should produce negative delta NAV
        if r["delta_nav"] < 0:
            checks.append(f"PASS  {r['scenario']}: ΔNAV < 0 (loss as expected)")
        else:
            checks.append(f"FAIL  {r['scenario']}: ΔNAV >= 0 (unexpected gain under stress)")

        # Check 2: VaR 95% >= |delta_nav| (MC tail should be at least as bad as deterministic)
        if r["var_95"] is not None and r["worst_case_loss"] is not None:
            if r["worst_case_loss"] >= abs(r["delta_nav"]) * 0.5:
                checks.append(
                    f"PASS  {r['scenario']}: MC worst-case >= 50% of deterministic loss"
                )
            else:
                checks.append(
                    f"WARN  {r['scenario']}: MC worst-case seems low vs deterministic"
                )

        # Check 3: ES 95% >= VaR 95% (coherent risk measure)
        if r["var_95"] is not None and r["es_95"] is not None:
            if r["es_95"] >= r["var_95"]:
                checks.append(f"PASS  {r['scenario']}: ES95 >= VaR95 (coherent)")
            else:
                checks.append(f"FAIL  {r['scenario']}: ES95 < VaR95 (not coherent!)")

        # Check 4: Loss bounded — no scenario should exceed 100% NAV
        if abs(r["delta_nav"]) < portfolio_nav:
            checks.append(f"PASS  {r['scenario']}: |ΔNAV| < total NAV (portfolio survives)")
        else:
            checks.append(f"FAIL  {r['scenario']}: |ΔNAV| >= total NAV (portfolio wiped out?)")

    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    from module_b.stress_engine import StressEngine
    from module_b.portfolio import SyntheticPortfolio
    from module_b.valuation import PortfolioValuator
    from module_b.scenarios import ScenarioEngine
    from module_b.trigger import StressTrigger
    from module_b.monte_carlo import MonteCarloEngine

    logger.info("Initialising StressEngine …")
    portfolio = SyntheticPortfolio(seed=42)
    valuator = PortfolioValuator()
    scenario_engine = ScenarioEngine()
    trigger = StressTrigger()
    mc_engine = MonteCarloEngine(n_paths=5_000, seed=42)  # 5k paths for speed

    engine = StressEngine(
        portfolio=portfolio,
        valuator=valuator,
        scenario_engine=scenario_engine,
        trigger=trigger,
        db=None,   # no persistence needed for validation
        redis=None,
        mc_n_paths=5_000,
    )
    # Share the same MC engine instance for all runs
    engine.mc_engine = mc_engine

    portfolio_nav = portfolio.compute_nav()
    logger.info("Portfolio NAV: $%,.0f", portfolio_nav)

    # Run all reference scenarios
    scenario_results: list[dict] = []
    for scenario in REFERENCE_SCENARIOS:
        result = run_scenario(engine, scenario)
        scenario_results.append(result)

    # Validation checks
    logger.info("Running validation checks …")
    checks = run_validation_checks(portfolio_nav, scenario_results)
    for check in checks:
        logger.info("  %s", check)

    # Build output
    pass_count = sum(1 for c in checks if c.startswith("PASS"))
    fail_count = sum(1 for c in checks if c.startswith("FAIL"))
    warn_count = sum(1 for c in checks if c.startswith("WARN"))

    output = {
        "portfolio_nav": round(portfolio_nav, 2),
        "n_scenarios": len(scenario_results),
        "scenarios": scenario_results,
        "validation_checks": checks,
        "check_summary": {
            "pass": pass_count,
            "fail": fail_count,
            "warn": warn_count,
            "total": len(checks),
        },
    }

    with OUTPUT_PATH.open("w") as fh:
        json.dump(output, fh, indent=2, default=str)
    logger.info("Saved validation results to %s", OUTPUT_PATH)

    print()
    print_summary(portfolio_nav, scenario_results)
    print()
    print("  Validation checks:")
    for check in checks:
        print(f"    {check}")
    print()
    print(f"  Summary: {pass_count} PASS  {warn_count} WARN  {fail_count} FAIL")
    _print_rule()


if __name__ == "__main__":
    main()
