"""
Module B — Wholesale Banking Portfolio Stress Tester
=====================================================

This module implements deterministic factor-shock stress testing for a
synthetic ~300-position wholesale banking portfolio.

Public API
──────────
    from module_b import StressEngine, ModuleBRunner, SyntheticPortfolio

    # Standalone: generate the portfolio and inspect it
    portfolio = SyntheticPortfolio()
    print(f"NAV = ${portfolio.compute_nav():,.0f}")
    print(portfolio.to_dataframe().head())

    # Full engine: wired to DuckDB + Redis
    engine = StressEngine()
    result = engine.run_stress_test(signal)

Component layout
────────────────
  portfolio.py     — SyntheticPortfolio: deterministic position generator
  valuation.py     — PortfolioValuator: first-order factor-shock pricing
  scenarios.py     — ScenarioEngine: shock vector library + scaling logic
  trigger.py       — StressTrigger: gate + 30-min cooldown
  stress_engine.py — StressEngine: orchestrates all components
  runner.py        — ModuleBRunner: async consumer loop (Redis → DuckDB)
"""

from module_b.portfolio import SyntheticPortfolio
from module_b.scenarios import ScenarioEngine
from module_b.stress_engine import StressEngine, StressResult
from module_b.trigger import StressTrigger
from module_b.valuation import PnLResult, PortfolioValuator

__all__ = [
    "SyntheticPortfolio",
    "ScenarioEngine",
    "StressEngine",
    "StressResult",
    "StressTrigger",
    "PnLResult",
    "PortfolioValuator",
]
