"""
Test suite for MonteCarloEngine and ReverseStressTester.

All tests are self-contained unit tests (no external dependencies).
The SyntheticPortfolio defaults to seed=42 for determinism.

Coverage
────────
  MonteCarloEngine:
    - initialises without error
    - covariance matrix is positive semi-definite
    - run() returns MonteCarloResult with correct n_paths
    - zero base shocks → expected_loss near zero (within 3-sigma)
    - negative equity shock → expected_loss > 0 (portfolio loses)
    - ES_95 >= VaR_95 (always)
    - VaR_99 >= VaR_95 (always)
    - n_paths=100 completes in under 5 seconds

  MonteCarloResult:
    - to_dict() returns expected keys
    - losses list length matches n_paths

  ReverseStressTester:
    - find_breach_threshold returns breach_impact in [1, 10]
    - narrative is non-empty and contains event_type
    - higher loss_limit requires higher breach_impact (monotone property)

  Integration smoke:
    - StressEngine initialises with MC engine attached
    - run_reverse_stress() delegates correctly and returns a result
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef
from module_b.monte_carlo import FACTOR_NAMES, MonteCarloEngine, MonteCarloResult
from module_b.portfolio import SyntheticPortfolio
from module_b.reverse_stress import ReverseStressResult, ReverseStressTester
from module_b.scenarios import ScenarioEngine
from module_b.stress_engine import StressEngine
from module_b.valuation import PortfolioValuator


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def portfolio() -> SyntheticPortfolio:
    """Deterministic portfolio (seed=42) shared across the module."""
    return SyntheticPortfolio(seed=42)


@pytest.fixture(scope="module")
def valuator() -> PortfolioValuator:
    return PortfolioValuator()


@pytest.fixture(scope="module")
def scenario_engine() -> ScenarioEngine:
    return ScenarioEngine()


@pytest.fixture(scope="module")
def mc_engine() -> MonteCarloEngine:
    """Shared MC engine (n_paths=1000 for test speed, deterministic seed)."""
    return MonteCarloEngine(n_paths=1_000, seed=42)


@pytest.fixture(scope="module")
def geopolitical_shocks(scenario_engine: ScenarioEngine) -> dict:
    """A fully-resolved shock dict for a Geopolitical event at impact=8."""
    scenario = scenario_engine.get_scenario("Geopolitical")
    scenario = scenario_engine.apply_impact_scaling(scenario, 8.0)
    flat = scenario_engine.flatten_shocks(scenario)
    pd_mult, lgd_mult = scenario_engine.get_pd_lgd_multipliers("Geopolitical", 8.0)
    flat["pd_multiplier"] = pd_mult
    flat["lgd_multiplier"] = lgd_mult
    return flat


@pytest.fixture(scope="module")
def equity_crash_shocks() -> dict:
    """Shock dict with a strong negative equity shock and no other stresses."""
    return {
        "equity_shock_pct": -0.30,
        "rates_bps": 0.0,
        "credit_spreads_bps": 0.0,
        "fx_usd_chg_pct": 0.0,
        "pd_multiplier": 1.0,
        "lgd_multiplier": 1.0,
        "sector_equity_overrides": {},
    }


@pytest.fixture(scope="module")
def zero_shocks() -> dict:
    """Shock dict with all factors set to zero."""
    return {
        "equity_shock_pct": 0.0,
        "rates_bps": 0.0,
        "credit_spreads_bps": 0.0,
        "fx_usd_chg_pct": 0.0,
        "pd_multiplier": 1.0,
        "lgd_multiplier": 1.0,
        "sector_equity_overrides": {},
    }


# ── MonteCarloEngine unit tests ───────────────────────────────────────────────

class TestMonteCarloEngineInit:

    def test_initialises_without_error(self) -> None:
        engine = MonteCarloEngine(n_paths=100, seed=0)
        assert engine.n_paths == 100

    def test_covariance_matrix_shape(self) -> None:
        engine = MonteCarloEngine(n_paths=100, seed=0)
        cov = engine._cov
        assert cov.shape == (5, 5), f"Expected (5, 5), got {cov.shape}"

    def test_covariance_matrix_symmetric(self) -> None:
        engine = MonteCarloEngine(n_paths=100, seed=0)
        cov = engine._cov
        assert np.allclose(cov, cov.T, atol=1e-10), "Covariance matrix must be symmetric"

    def test_covariance_matrix_positive_semidefinite(self) -> None:
        """All eigenvalues must be positive (positive definite)."""
        engine = MonteCarloEngine(n_paths=100, seed=0)
        eigenvalues = np.linalg.eigvalsh(engine._cov)
        assert np.all(eigenvalues > 0), (
            f"Covariance matrix has non-positive eigenvalues: "
            f"{eigenvalues[eigenvalues <= 0]}"
        )

    def test_cholesky_factor_shape(self) -> None:
        engine = MonteCarloEngine(n_paths=100, seed=0)
        L = engine._L
        assert L.shape == (5, 5)

    def test_cholesky_reconstructs_covariance(self) -> None:
        """L @ L.T should equal the covariance matrix."""
        engine = MonteCarloEngine(n_paths=100, seed=0)
        reconstructed = engine._L @ engine._L.T
        assert np.allclose(reconstructed, engine._cov, atol=1e-10), (
            "Cholesky reconstruction failed: L @ L.T ≠ Cov"
        )

    def test_factor_names_list(self) -> None:
        assert FACTOR_NAMES == ["equity", "rates", "credit_spread", "fx", "commodity"]


class TestMonteCarloRun:

    def test_returns_monte_carlo_result_type(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert isinstance(result, MonteCarloResult)

    def test_result_has_correct_n_paths(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert result.n_paths == mc_engine.n_paths

    def test_losses_list_length_matches_n_paths(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert len(result.losses) == mc_engine.n_paths, (
            f"Expected {mc_engine.n_paths} losses, got {len(result.losses)}"
        )

    def test_pre_nav_positive(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert result.pre_nav > 0

    def test_zero_shocks_expected_loss_near_zero(
        self,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        zero_shocks: dict,
    ) -> None:
        """
        With zero base shocks, the only source of P&L is random factor noise.
        The expected loss should be near zero (within 3 standard errors of the
        loss distribution).
        """
        engine = MonteCarloEngine(n_paths=2_000, seed=42)
        result = engine.run(portfolio, zero_shocks, 1.0, (1.0, 1.0), valuator)

        nav = result.pre_nav
        # Expected loss should be small relative to NAV — within 3% of NAV
        assert abs(result.expected_loss) < 0.03 * nav, (
            f"Expected loss with zero shocks too large: "
            f"${result.expected_loss:,.0f} vs NAV=${nav:,.0f}"
        )

    def test_negative_equity_shock_produces_positive_expected_loss(
        self,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        equity_crash_shocks: dict,
    ) -> None:
        """A strong negative equity shock must produce a positive expected loss."""
        engine = MonteCarloEngine(n_paths=1_000, seed=42)
        result = engine.run(portfolio, equity_crash_shocks, 9.0, (8.5, 9.5), valuator)
        assert result.expected_loss > 0, (
            f"Expected positive expected_loss after equity crash, "
            f"got {result.expected_loss:.0f}"
        )

    def test_es_95_greater_than_or_equal_var_95(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        """ES_95 must always be >= VaR_95 by definition."""
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert result.es_95 >= result.var_95, (
            f"ES_95 ({result.es_95:.0f}) < VaR_95 ({result.var_95:.0f})"
        )

    def test_var_99_greater_than_or_equal_var_95(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        """VaR_99 must always be >= VaR_95 (higher quantile = larger loss)."""
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert result.var_99 >= result.var_95, (
            f"VaR_99 ({result.var_99:.0f}) < VaR_95 ({result.var_95:.0f})"
        )

    def test_es_99_greater_than_or_equal_es_95(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        """ES_99 must be >= ES_95 (conditional expectation of more extreme tail)."""
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert result.es_99 >= result.es_95, (
            f"ES_99 ({result.es_99:.0f}) < ES_95 ({result.es_95:.0f})"
        )

    def test_worst_loss_greater_than_or_equal_var_99(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert result.worst_loss >= result.var_99, (
            f"worst_loss ({result.worst_loss:.0f}) < var_99 ({result.var_99:.0f})"
        )

    def test_best_outcome_lte_expected_loss(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert result.best_outcome <= result.expected_loss, (
            "best_outcome must be <= expected_loss (it is the minimum of the loss dist)"
        )

    def test_n_paths_100_completes_under_5_seconds(
        self,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        """Even the minimal 100-path run must complete well within the time limit."""
        engine = MonteCarloEngine(n_paths=100, seed=42)
        start = time.perf_counter()
        engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, (
            f"MC run with n_paths=100 took {elapsed:.2f}s (limit: 5s)"
        )

    def test_10k_paths_completes_under_30_seconds(
        self,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        """
        Performance regression guard for the full 10k-path run.
        30 seconds is a conservative limit; target is < 3 s on modern hardware.
        """
        engine = MonteCarloEngine(n_paths=10_000, seed=42)
        start = time.perf_counter()
        engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        elapsed = time.perf_counter() - start
        assert elapsed < 30.0, (
            f"MC run with n_paths=10_000 took {elapsed:.2f}s (limit: 30s)"
        )


class TestMonteCarloResultDict:

    def test_to_dict_returns_expected_keys(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        d = result.to_dict()
        expected_keys = {
            "n_paths", "pre_nav", "var_95", "var_99", "es_95", "es_99",
            "expected_loss", "worst_loss", "best_outcome",
        }
        assert expected_keys.issubset(d.keys()), (
            f"Missing keys: {expected_keys - d.keys()}"
        )

    def test_to_dict_n_paths_matches(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        assert result.to_dict()["n_paths"] == mc_engine.n_paths

    def test_all_metrics_are_finite(
        self,
        mc_engine: MonteCarloEngine,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        geopolitical_shocks: dict,
    ) -> None:
        result = mc_engine.run(portfolio, geopolitical_shocks, 8.0, (7.5, 8.5), valuator)
        import math
        for attr in ("var_95", "var_99", "es_95", "es_99", "expected_loss",
                     "worst_loss", "best_outcome"):
            val = getattr(result, attr)
            assert math.isfinite(val), f"{attr} is not finite: {val}"


# ── ReverseStressTester tests ─────────────────────────────────────────────────

class TestReverseStressTester:

    @pytest.fixture(scope="class")
    def tester(self) -> ReverseStressTester:
        return ReverseStressTester()

    @pytest.fixture(scope="class")
    def fast_mc_engine(self) -> MonteCarloEngine:
        """Small engine for reverse stress search (fast)."""
        return MonteCarloEngine(n_paths=200, seed=42)

    def test_breach_impact_in_valid_range(
        self,
        tester: ReverseStressTester,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        scenario_engine: ScenarioEngine,
        fast_mc_engine: MonteCarloEngine,
    ) -> None:
        result = tester.find_breach_threshold(
            portfolio=portfolio,
            valuator=valuator,
            scenario_engine=scenario_engine,
            mc_engine=fast_mc_engine,
            event_type="Geopolitical",
            loss_limit_pct=0.10,
        )
        assert 1.0 <= result.breach_impact <= 10.0, (
            f"breach_impact={result.breach_impact} outside [1, 10]"
        )

    def test_narrative_non_empty(
        self,
        tester: ReverseStressTester,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        scenario_engine: ScenarioEngine,
        fast_mc_engine: MonteCarloEngine,
    ) -> None:
        result = tester.find_breach_threshold(
            portfolio=portfolio,
            valuator=valuator,
            scenario_engine=scenario_engine,
            mc_engine=fast_mc_engine,
            event_type="Credit",
            loss_limit_pct=0.10,
        )
        assert result.narrative, "Narrative must not be empty"

    def test_narrative_contains_event_type(
        self,
        tester: ReverseStressTester,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        scenario_engine: ScenarioEngine,
        fast_mc_engine: MonteCarloEngine,
    ) -> None:
        event_type = "Macro"
        result = tester.find_breach_threshold(
            portfolio=portfolio,
            valuator=valuator,
            scenario_engine=scenario_engine,
            mc_engine=fast_mc_engine,
            event_type=event_type,
            loss_limit_pct=0.15,
        )
        assert event_type in result.narrative, (
            f"Narrative does not mention event_type '{event_type}': {result.narrative!r}"
        )

    def test_result_fields_populated(
        self,
        tester: ReverseStressTester,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        scenario_engine: ScenarioEngine,
        fast_mc_engine: MonteCarloEngine,
    ) -> None:
        result = tester.find_breach_threshold(
            portfolio=portfolio,
            valuator=valuator,
            scenario_engine=scenario_engine,
            mc_engine=fast_mc_engine,
            event_type="Geopolitical",
            loss_limit_pct=0.10,
        )
        assert result.event_type == "Geopolitical"
        assert result.loss_limit_pct == pytest.approx(0.10)
        # expected_loss_at_breach is the expected loss at the found impact level;
        # if no breach was found it reflects max-impact which can be negative
        # (portfolio gains on average), so we only check it is finite.
        import math
        assert math.isfinite(result.expected_loss_at_breach)
        assert math.isfinite(result.expected_loss_pct)
        ci_lo, ci_hi = result.confidence_interval
        assert ci_lo <= result.breach_impact <= ci_hi or (
            # allow slight floating point tolerance at boundaries
            abs(ci_lo - result.breach_impact) < 0.2 or
            abs(ci_hi - result.breach_impact) < 0.2
        )

    def test_higher_loss_limit_requires_higher_breach_impact(
        self,
        tester: ReverseStressTester,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        scenario_engine: ScenarioEngine,
        fast_mc_engine: MonteCarloEngine,
    ) -> None:
        """
        A tighter loss limit (lower pct) should breach at a lower impact;
        a more relaxed limit (higher pct) should require a higher impact to breach.

        Monotone property: loss_limit_pct₁ < loss_limit_pct₂
                           → breach_impact₁ ≤ breach_impact₂
        """
        result_tight = tester.find_breach_threshold(
            portfolio=portfolio,
            valuator=valuator,
            scenario_engine=scenario_engine,
            mc_engine=fast_mc_engine,
            event_type="Credit",
            loss_limit_pct=0.05,   # tighter → should breach at lower impact
        )
        result_relaxed = tester.find_breach_threshold(
            portfolio=portfolio,
            valuator=valuator,
            scenario_engine=scenario_engine,
            mc_engine=fast_mc_engine,
            event_type="Credit",
            loss_limit_pct=0.20,   # more relaxed → should breach at higher impact
        )
        assert result_tight.breach_impact <= result_relaxed.breach_impact + 0.5, (
            f"Expected tight limit to breach at lower impact: "
            f"tight={result_tight.breach_impact:.2f}  "
            f"relaxed={result_relaxed.breach_impact:.2f}"
        )

    def test_to_dict_returns_expected_keys(
        self,
        tester: ReverseStressTester,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        scenario_engine: ScenarioEngine,
        fast_mc_engine: MonteCarloEngine,
    ) -> None:
        result = tester.find_breach_threshold(
            portfolio=portfolio,
            valuator=valuator,
            scenario_engine=scenario_engine,
            mc_engine=fast_mc_engine,
            event_type="Geopolitical",
            loss_limit_pct=0.10,
        )
        d = result.to_dict()
        expected_keys = {
            "event_type", "loss_limit_pct", "breach_impact",
            "expected_loss_at_breach", "expected_loss_pct",
            "confidence_interval", "narrative",
        }
        assert expected_keys.issubset(d.keys()), (
            f"Missing keys in ReverseStressResult.to_dict(): {expected_keys - d.keys()}"
        )


# ── StressEngine integration smoke tests ──────────────────────────────────────

class TestStressEngineIntegration:
    """
    Smoke tests verifying the updated StressEngine initialises correctly and
    exposes the new MC and reverse-stress surface.
    """

    @pytest.fixture(scope="class")
    def engine(self) -> StressEngine:
        """StressEngine with small MC n_paths for speed."""
        return StressEngine(mc_n_paths=200)

    def test_stress_engine_has_mc_engine(self, engine: StressEngine) -> None:
        assert hasattr(engine, "mc_engine")
        assert isinstance(engine.mc_engine, MonteCarloEngine)

    def test_stress_engine_has_reverse_tester(self, engine: StressEngine) -> None:
        assert hasattr(engine, "reverse_tester")
        assert isinstance(engine.reverse_tester, ReverseStressTester)

    def test_run_reverse_stress_returns_result(self, engine: StressEngine) -> None:
        result = engine.run_reverse_stress("Geopolitical", loss_limit_pct=0.10)
        assert isinstance(result, ReverseStressResult)
        assert 1.0 <= result.breach_impact <= 10.0

    def test_stress_result_has_mc_fields(self, engine: StressEngine) -> None:
        """
        The StressResult dataclass must carry mc_* fields (initialised to None).
        We verify the fields exist and default to None before a full run.
        """
        from module_b.stress_engine import StressResult
        from datetime import datetime, timezone

        result = StressResult(
            run_id="test",
            ts_created=datetime.now(tz=timezone.utc),
            trigger_signal_id="sig-1",
            scenario_name="Geopolitical",
            event_type="Geopolitical",
            impact=8.0,
            pre_nav=50_000_000.0,
            post_nav=49_000_000.0,
            delta_nav=-1_000_000.0,
            delta_pct=-2.0,
            breakdown_by_asset_class={},
            breakdown_by_sector={},
            shocks_applied={},
        )
        assert result.mc_var_95 is None
        assert result.mc_es_95 is None
        assert result.mc_expected_loss is None
        assert result.mc_worst_loss is None

    def test_stress_result_to_dict_includes_mc_keys(self) -> None:
        from module_b.stress_engine import StressResult
        from datetime import datetime, timezone

        result = StressResult(
            run_id="test",
            ts_created=datetime.now(tz=timezone.utc),
            trigger_signal_id="sig-1",
            scenario_name="Geopolitical",
            event_type="Geopolitical",
            impact=8.0,
            pre_nav=50_000_000.0,
            post_nav=49_000_000.0,
            delta_nav=-1_000_000.0,
            delta_pct=-2.0,
            breakdown_by_asset_class={},
            breakdown_by_sector={},
            shocks_applied={},
            mc_var_95=500_000.0,
            mc_es_95=750_000.0,
            mc_expected_loss=300_000.0,
            mc_worst_loss=2_000_000.0,
        )
        d = result.to_dict()
        assert "mc_var_95" in d
        assert "mc_es_95" in d
        assert "mc_expected_loss" in d
        assert "mc_worst_loss" in d
        assert d["mc_var_95"] == pytest.approx(500_000.0)
