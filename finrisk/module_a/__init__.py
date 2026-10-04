"""
Module A — Tactical Index Rebalancer
=====================================

Signal-driven rebalancer for the FinRisk-20 synthetic equity index.
Consumes processed.signals from Redis, computes risk-adjusted scores,
and emits weight-delta events to DuckDB and the signals.rebalance stream.

Public API
──────────
    from module_a import IndexRebalancer, ModuleARunner, SignalAggregator, WeightCalculator

    # Standalone: inspect weights without running the loop
    rebalancer = IndexRebalancer()
    event = await rebalancer.on_signal(signal)

    # Full runner: wired to DuckDB + Redis
    runner = ModuleARunner(db=db, redis=redis)
    await runner.start()
    await runner.run_forever()

Component layout
────────────────
  signal_aggregator.py  — decay-weighted score aggregation per ticker
  weight_calculator.py  — z-score → exponential tilt → constrained weights
  price_cache.py        — yfinance price cache with synthetic fallback
  rebalancer.py         — IndexRebalancer: orchestrates all components
  runner.py             — ModuleARunner: async consumer loop (Redis → DuckDB)
"""

from module_a.rebalancer import IndexRebalancer, RebalanceEvent
from module_a.runner import ModuleARunner
from module_a.signal_aggregator import SignalAggregator
from module_a.weight_calculator import WeightCalculator

__all__ = [
    "IndexRebalancer",
    "RebalanceEvent",
    "ModuleARunner",
    "SignalAggregator",
    "WeightCalculator",
]
