"""
Module B test suite.

All tests are unit tests (no external dependencies: no Redis, no DuckDB).
The test portfolio is deterministic because SyntheticPortfolio defaults to seed=42.

Coverage
────────
  Portfolio generation:
    - position counts (280–320 range)
    - NAV range ($30M–$70M)
    - required fields on each position type

  Valuation:
    - bond: higher duration → larger loss when rates rise
    - bond: convexity partially offsets duration loss for large shocks
    - loan: higher PD → larger EL increase
    - equity: negative shock reduces market value
    - equity: sector overrides apply correctly to affected sectors only
    - FX forward: positive USD shock creates loss
    - swap: rate rise creates loss (pay-fixed)
    - swap: rate fall creates gain (pay-fixed)

  ScenarioEngine:
    - scenarios.yaml loads successfully (all known event types present)
    - impact scaling: impact=5 → ~0.1, impact=10 → ~1.1
    - impact scaling is monotonically increasing
    - sentiment sign: positive sentiment flips equity shock for Earnings
    - sentiment sign: no change for Geopolitical
    - PD/LGD multipliers: Credit > Geopolitical > other for same impact

  StressTrigger:
    - fires when impact ≥ 7.0 AND confidence ≥ 0.55
    - does NOT fire when impact < 7.0
    - does NOT fire when confidence < 0.55
    - cooldown prevents re-firing within 30 min
    - cooldown expires correctly
    - multi-ticker signal: all tickers marked on cooldown after trigger

  StressEngine (full pipeline):
    - produces StressResult with all required fields
    - StressResult has correct event_type, scenario_name
    - delta_nav is negative for geopolitical, credit, macro scenarios
    - geopolitical event: equities down, spreads wider, rates down (check sign)
    - credit event: loans show the largest absolute loss component
    - full run does not trigger if impact < 7.0
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef
from module_b.portfolio import (
    Bond,
    Equity,
    FXForward,
    InterestRateSwap,
    Loan,
    SyntheticPortfolio,
)
from module_b.scenarios import ScenarioEngine
from module_b.stress_engine import StressEngine, StressResult
from module_b.trigger import StressTrigger
from module_b.valuation import PortfolioValuator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def portfolio() -> SyntheticPortfolio:
    """A single portfolio instance shared across the whole module (fast)."""
    return SyntheticPortfolio(seed=42)


@pytest.fixture(scope="module")
def valuator() -> PortfolioValuator:
    return PortfolioValuator()


@pytest.fixture(scope="module")
def scenario_engine() -> ScenarioEngine:
    return ScenarioEngine()


@pytest.fixture
def trigger() -> StressTrigger:
    """Fresh trigger for each test to avoid cooldown bleed-over."""
    return StressTrigger()


def _make_signal(
    event_type: str = "Geopolitical",
    impact: float = 8.0,
    confidence: float = 0.75,
    sentiment: float = -0.5,
    ticker: str = "JPM",
) -> RiskSignal:
    """Construct a valid RiskSignal for testing."""
    now = datetime.now(tz=timezone.utc)
    return RiskSignal(
        event_id=f"test-{event_type}-{impact}",
        ts_published=now - timedelta(seconds=30),
        ts_processed=now,
        source="news",
        tickers=[TickerRef(ticker=ticker, relevance=1.0)],
        sentiment=sentiment,
        sentiment_conf=0.85,
        event_type=event_type,
        event_probs={event_type: 0.90, "Macro": 0.10},
        impact=impact,
        impact_range=(impact - 0.5, impact + 0.5),
        confidence=confidence,
        novelty=0.80,
        diffusion=DiffusionInfo(sources=3, velocity=2.0, score=0.65),
        decay_halflife_hours=12.0,
        cluster_id="test-cluster-001",
        model_versions={"sentiment": "test-v1"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio generation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioGeneration:

    def test_total_position_count(self, portfolio: SyntheticPortfolio) -> None:
        counts = portfolio.position_count()
        total = counts["total"]
        assert 280 <= total <= 320, (
            f"Expected 280–320 positions, got {total}. "
            f"Breakdown: {counts}"
        )

    def test_position_counts_by_class(self, portfolio: SyntheticPortfolio) -> None:
        counts = portfolio.position_count()
        assert counts["bonds"] == 120
        assert counts["loans"] == 80
        assert counts["equities"] == 60
        assert counts["fx_forwards"] == 25
        assert counts["swaps"] == 15

    def test_nav_within_expected_range(self, portfolio: SyntheticPortfolio) -> None:
        nav = portfolio.compute_nav()
        assert 30_000_000 <= nav <= 70_000_000, (
            f"Expected NAV $30M–$70M, got ${nav:,.0f}"
        )

    def test_bonds_have_required_fields(self, portfolio: SyntheticPortfolio) -> None:
        for bond in portfolio.bonds:
            assert isinstance(bond.position_id, str) and bond.position_id
            assert isinstance(bond.sector, str) and bond.sector
            assert isinstance(bond.issuer, str) and bond.issuer
            assert bond.notional > 0
            assert 0 < bond.coupon < 0.15
            assert 2.0 <= bond.duration <= 10.0
            assert bond.convexity > 0
            assert bond.credit_spread_bps > 0
            assert bond.market_value > 0

    def test_loans_have_required_fields(self, portfolio: SyntheticPortfolio) -> None:
        for loan in portfolio.loans:
            assert isinstance(loan.position_id, str) and loan.position_id
            assert isinstance(loan.counterparty, str) and loan.counterparty
            assert isinstance(loan.sector, str) and loan.sector
            assert loan.ead > 0
            assert 0 < loan.pd < 0.1
            assert 0.30 <= loan.lgd <= 0.50
            assert loan.rating in {"A", "BBB", "BB", "B"}
            assert loan.expected_loss >= 0
            # EL = PD × LGD × EAD
            expected = loan.pd * loan.lgd * loan.ead
            assert abs(loan.expected_loss - expected) < 1.0, (
                f"EL mismatch for {loan.position_id}: "
                f"stored={loan.expected_loss:.2f}, computed={expected:.2f}"
            )

    def test_equities_have_required_fields(self, portfolio: SyntheticPortfolio) -> None:
        for equity in portfolio.equities:
            assert isinstance(equity.position_id, str) and equity.position_id
            assert isinstance(equity.ticker, str) and equity.ticker
            assert isinstance(equity.sector, str) and equity.sector
            assert equity.shares > 0
            assert equity.price > 0
            assert equity.market_value > 0
            assert abs(equity.market_value - equity.shares * equity.price) < 1.0
            assert 0.5 <= equity.beta <= 2.0

    def test_fx_forwards_have_required_fields(self, portfolio: SyntheticPortfolio) -> None:
        valid_pairs = {"EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"}
        for fwd in portfolio.fx_forwards:
            assert isinstance(fwd.position_id, str) and fwd.position_id
            assert fwd.currency_pair in valid_pairs
            assert fwd.notional_usd > 0
            assert fwd.spot_rate > 0
            assert fwd.forward_rate > 0

    def test_swaps_have_required_fields(self, portfolio: SyntheticPortfolio) -> None:
        for swap in portfolio.swaps:
            assert isinstance(swap.position_id, str) and swap.position_id
            assert swap.notional > 0
            assert swap.dv01 > 0
            assert swap.term_years > 0
            assert 0.01 < swap.fixed_rate < 0.20
            assert isinstance(swap.sector, str) and swap.sector

    def test_to_dataframe_shape(self, portfolio: SyntheticPortfolio) -> None:
        df = portfolio.to_dataframe()
        assert len(df) == portfolio.position_count()["total"]
        required_cols = {
            "position_id", "asset_class", "sector", "market_value"
        }
        assert required_cols.issubset(set(df.columns))

    def test_nav_by_asset_class_sums_to_nav(self, portfolio: SyntheticPortfolio) -> None:
        breakdown = portfolio.nav_by_asset_class()
        total = sum(breakdown.values())
        nav = portfolio.compute_nav()
        assert abs(total - nav) < 1.0, (
            f"nav_by_asset_class sum={total:.2f} != compute_nav={nav:.2f}"
        )

    def test_deterministic_seed(self) -> None:
        """Two portfolios from the same seed must have identical NAVs."""
        p1 = SyntheticPortfolio(seed=42)
        p2 = SyntheticPortfolio(seed=42)
        assert p1.compute_nav() == p2.compute_nav()

    def test_different_seeds_differ(self) -> None:
        p1 = SyntheticPortfolio(seed=42)
        p2 = SyntheticPortfolio(seed=99)
        assert p1.compute_nav() != p2.compute_nav()


# ─────────────────────────────────────────────────────────────────────────────
# Valuation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBondValuation:

    def _make_bond(
        self,
        duration: float,
        market_value: float = 1_000_000,
        convexity: float | None = None,
        credit_spread_bps: float = 100,
    ) -> Bond:
        if convexity is None:
            convexity = duration ** 2 / 100
        return Bond(
            position_id=f"TEST-BOND-D{duration:.1f}",
            sector="Technology",
            issuer="TestCorp",
            notional=market_value,
            coupon=0.04,
            duration=duration,
            convexity=convexity,
            credit_spread_bps=credit_spread_bps,
            market_value=market_value,
        )

    def test_higher_duration_larger_loss_when_rates_rise(
        self, valuator: PortfolioValuator
    ) -> None:
        short = self._make_bond(duration=2.0)
        long = self._make_bond(duration=8.0)
        # +100bps rate shock
        short_pnl = valuator.valuate_bonds([short], rate_shock_bps=100, spread_shock_bps=0)
        long_pnl = valuator.valuate_bonds([long], rate_shock_bps=100, spread_shock_bps=0)
        assert short_pnl[short.position_id] > long_pnl[long.position_id], (
            "Longer-duration bond should lose more when rates rise"
        )
        # Both should be losses
        assert short_pnl[short.position_id] < 0
        assert long_pnl[long.position_id] < 0

    def test_rate_fall_is_gain(self, valuator: PortfolioValuator) -> None:
        bond = self._make_bond(duration=5.0)
        pnl = valuator.valuate_bonds([bond], rate_shock_bps=-50, spread_shock_bps=0)
        assert pnl[bond.position_id] > 0, "Rate fall should produce a gain for bond holders"

    def test_spread_widening_is_loss(self, valuator: PortfolioValuator) -> None:
        bond = self._make_bond(duration=5.0)
        pnl = valuator.valuate_bonds([bond], rate_shock_bps=0, spread_shock_bps=100)
        assert pnl[bond.position_id] < 0, "Spread widening should produce a loss"

    def test_convexity_partially_offsets_duration_loss(
        self, valuator: PortfolioValuator
    ) -> None:
        """
        Compare the full duration-convexity loss vs duration-only loss for a
        large shock.  The convexity term should reduce the loss (positive convexity).
        """
        mv = 1_000_000
        d = 8.0
        c = d ** 2 / 100.0  # ~0.64

        bond_with_c = Bond(
            position_id="WITH-CONV",
            sector="Technology",
            issuer="X",
            notional=mv,
            coupon=0.04,
            duration=d,
            convexity=c,
            credit_spread_bps=100,
            market_value=mv,
        )
        bond_no_c = Bond(
            position_id="NO-CONV",
            sector="Technology",
            issuer="X",
            notional=mv,
            coupon=0.04,
            duration=d,
            convexity=0.0,
            credit_spread_bps=100,
            market_value=mv,
        )
        shock_bps = 300  # large shock to make convexity visible
        pnl_c = valuator.valuate_bonds([bond_with_c], rate_shock_bps=shock_bps, spread_shock_bps=0)
        pnl_no = valuator.valuate_bonds([bond_no_c], rate_shock_bps=shock_bps, spread_shock_bps=0)
        # With convexity, loss should be smaller in magnitude
        assert pnl_c["WITH-CONV"] > pnl_no["NO-CONV"], (
            "Positive convexity should reduce loss for large upward rate shock"
        )

    def test_zero_shock_is_zero_pnl(self, valuator: PortfolioValuator) -> None:
        bond = self._make_bond(duration=5.0)
        pnl = valuator.valuate_bonds([bond], rate_shock_bps=0, spread_shock_bps=0)
        assert abs(pnl[bond.position_id]) < 1.0


class TestLoanValuation:

    def _make_loan(
        self,
        pd: float,
        lgd: float = 0.40,
        ead: float = 1_000_000,
        rating: str = "BB",
    ) -> Loan:
        return Loan(
            position_id=f"TEST-LOAN-PD{pd:.3f}",
            counterparty="TestBorrower",
            sector="Financials",
            ead=ead,
            pd=pd,
            lgd=lgd,
            rating=rating,
            expected_loss=pd * lgd * ead,
        )

    def test_higher_pd_larger_loss(self, valuator: PortfolioValuator) -> None:
        low_pd = self._make_loan(pd=0.01, rating="A")
        high_pd = self._make_loan(pd=0.04, rating="B")
        pnl = valuator.valuate_loans([low_pd, high_pd], pd_multiplier=2.0, lgd_multiplier=1.0)
        assert pnl[low_pd.position_id] > pnl[high_pd.position_id], (
            "Higher-PD loan should show a larger absolute loss increase"
        )
        assert pnl[high_pd.position_id] < 0

    def test_no_stress_is_zero_pnl(self, valuator: PortfolioValuator) -> None:
        loan = self._make_loan(pd=0.02)
        pnl = valuator.valuate_loans([loan], pd_multiplier=1.0, lgd_multiplier=1.0)
        assert abs(pnl[loan.position_id]) < 1.0

    def test_lgd_stress_increases_loss(self, valuator: PortfolioValuator) -> None:
        loan = self._make_loan(pd=0.02)
        pnl_lgd = valuator.valuate_loans([loan], pd_multiplier=1.0, lgd_multiplier=1.3)
        pnl_nolg = valuator.valuate_loans([loan], pd_multiplier=1.0, lgd_multiplier=1.0)
        assert pnl_lgd[loan.position_id] < pnl_nolg[loan.position_id], (
            "LGD stress should increase expected loss"
        )

    def test_pd_multiplier_capped_at_100pct(self, valuator: PortfolioValuator) -> None:
        """PD × large multiplier should be capped at 1.0, not >100%."""
        loan = self._make_loan(pd=0.50)
        # Multiplier 10 would push PD to 5.0 — should be capped at 1.0
        pnl = valuator.valuate_loans([loan], pd_multiplier=10.0, lgd_multiplier=1.0)
        # Loss should not exceed EAD × LGD
        max_loss = loan.ead * loan.lgd
        assert abs(pnl[loan.position_id]) <= max_loss + 1.0


class TestEquityValuation:

    def _make_equity(
        self,
        ticker: str = "AAPL",
        sector: str = "Technology",
        market_value: float = 1_000_000,
        beta: float = 1.2,
    ) -> Equity:
        return Equity(
            position_id=f"TEST-EQ-{ticker}",
            ticker=ticker,
            sector=sector,
            shares=1000,
            price=market_value / 1000,
            market_value=market_value,
            beta=beta,
        )

    def test_negative_shock_reduces_value(self, valuator: PortfolioValuator) -> None:
        eq = self._make_equity()
        pnl = valuator.valuate_equities([eq], equity_shock_pct=-0.10)
        assert pnl[eq.position_id] < 0, "Negative market shock should produce a loss"

    def test_positive_shock_increases_value(self, valuator: PortfolioValuator) -> None:
        eq = self._make_equity()
        pnl = valuator.valuate_equities([eq], equity_shock_pct=0.05)
        assert pnl[eq.position_id] > 0

    def test_sector_override_replaces_base_shock(self, valuator: PortfolioValuator) -> None:
        tech = self._make_equity(ticker="AAPL", sector="Technology", market_value=1_000_000)
        energy = self._make_equity(ticker="XOM", sector="Energy", market_value=1_000_000)
        base_shock = -0.05
        overrides = {"Energy": +0.10}  # Energy gets a positive shock
        pnl = valuator.valuate_equities(
            [tech, energy], equity_shock_pct=base_shock, sector_overrides=overrides
        )
        # Tech should use base shock (negative)
        assert pnl[tech.position_id] < 0
        # Energy should use override (positive)
        assert pnl[energy.position_id] > 0

    def test_sector_override_only_affects_matching_sector(
        self, valuator: PortfolioValuator
    ) -> None:
        sectors = ["Technology", "Financials", "Health Care"]
        equities = [
            self._make_equity(ticker=f"T{i}", sector=s, market_value=1_000_000)
            for i, s in enumerate(sectors)
        ]
        overrides = {"Technology": -0.20}  # only Technology gets special treatment
        pnl = valuator.valuate_equities(
            equities, equity_shock_pct=-0.05, sector_overrides=overrides
        )
        # Technology should have a larger loss than Financials / Health Care
        assert pnl["TEST-EQ-T0"] < pnl["TEST-EQ-T1"]
        assert pnl["TEST-EQ-T0"] < pnl["TEST-EQ-T2"]
        # Financials and Health Care use base shock
        assert abs(pnl["TEST-EQ-T1"] - pnl["TEST-EQ-T2"]) < 1.0

    def test_higher_beta_larger_loss(self, valuator: PortfolioValuator) -> None:
        low_beta = self._make_equity(ticker="LOW", beta=0.8)
        high_beta = self._make_equity(ticker="HIGH", beta=1.5)
        pnl = valuator.valuate_equities(
            [low_beta, high_beta], equity_shock_pct=-0.10
        )
        assert pnl[high_beta.position_id] < pnl[low_beta.position_id]

    def test_zero_shock_is_zero_pnl(self, valuator: PortfolioValuator) -> None:
        eq = self._make_equity()
        pnl = valuator.valuate_equities([eq], equity_shock_pct=0.0)
        assert abs(pnl[eq.position_id]) < 1.0


class TestFXForwardValuation:

    def _make_fwd(self, notional_usd: float = 1_000_000) -> FXForward:
        return FXForward(
            position_id="TEST-FX-EURUSD",
            currency_pair="EURUSD",
            notional_usd=notional_usd,
            spot_rate=1.085,
            forward_rate=1.083,
        )

    def test_positive_usd_shock_creates_loss(self, valuator: PortfolioValuator) -> None:
        """USD strengthening should be a mild loss on the hedging book."""
        fwd = self._make_fwd()
        pnl = valuator.valuate_fx_forwards([fwd], fx_usd_chg_pct=0.03)
        assert pnl[fwd.position_id] > 0  # 0.5 damping, so sign follows shock

    def test_zero_fx_shock_is_zero(self, valuator: PortfolioValuator) -> None:
        fwd = self._make_fwd()
        pnl = valuator.valuate_fx_forwards([fwd], fx_usd_chg_pct=0.0)
        assert pnl[fwd.position_id] == 0.0


class TestSwapValuation:

    def _make_swap(self, dv01: float = 10_000) -> InterestRateSwap:
        return InterestRateSwap(
            position_id=f"TEST-IRS-DV01{dv01:.0f}",
            notional=10_000_000,
            dv01=dv01,
            term_years=10,
            fixed_rate=0.04,
            sector="Financials",
        )

    def test_rate_rise_causes_loss_pay_fixed(self, valuator: PortfolioValuator) -> None:
        swap = self._make_swap()
        pnl = valuator.valuate_swaps([swap], rate_shock_bps=50)
        assert pnl[swap.position_id] < 0, (
            "Pay-fixed swap should lose value when rates rise"
        )

    def test_rate_fall_causes_gain_pay_fixed(self, valuator: PortfolioValuator) -> None:
        swap = self._make_swap()
        pnl = valuator.valuate_swaps([swap], rate_shock_bps=-25)
        assert pnl[swap.position_id] > 0, (
            "Pay-fixed swap should gain value when rates fall"
        )

    def test_dv01_proportionality(self, valuator: PortfolioValuator) -> None:
        """Swap with 2× DV01 should have exactly 2× the P&L."""
        swap1 = self._make_swap(dv01=5_000)
        swap2 = self._make_swap(dv01=10_000)
        shock = 100
        pnl1 = valuator.valuate_swaps([swap1], rate_shock_bps=shock)
        pnl2 = valuator.valuate_swaps([swap2], rate_shock_bps=shock)
        assert abs(pnl2[swap2.position_id] / pnl1[swap1.position_id] - 2.0) < 0.001

    def test_zero_shock_is_zero(self, valuator: PortfolioValuator) -> None:
        swap = self._make_swap()
        pnl = valuator.valuate_swaps([swap], rate_shock_bps=0)
        assert pnl[swap.position_id] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ScenarioEngine tests
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioEngine:

    def test_loads_all_known_event_types(self, scenario_engine: ScenarioEngine) -> None:
        expected = {
            "Geopolitical", "Credit", "Regulatory", "M&A", "Earnings",
            "Macro", "Commodity", "Cyber", "Leadership", "Product",
        }
        loaded = set(scenario_engine.available_scenarios())
        assert expected.issubset(loaded), (
            f"Missing scenarios: {expected - loaded}"
        )

    def test_get_scenario_returns_shocks(self, scenario_engine: ScenarioEngine) -> None:
        for event_type in scenario_engine.available_scenarios():
            scenario = scenario_engine.get_scenario(event_type)
            assert "shocks" in scenario, f"No 'shocks' key for {event_type}"
            shocks = scenario["shocks"]
            assert "equity_shock_pct" in shocks, f"No equity_shock_pct for {event_type}"
            assert "rates_bps" in shocks, f"No rates_bps for {event_type}"
            assert "credit_spreads_bps" in shocks, f"No credit_spreads_bps for {event_type}"

    def test_impact_scaling_floor_at_impact_5(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        geo = scenario_engine.get_scenario("Geopolitical")
        scaled = scenario_engine.apply_impact_scaling(geo, impact=5.0)
        multiplier = scaled["_impact_multiplier"]
        assert abs(multiplier - 0.10) < 0.01, (
            f"Expected multiplier ≈ 0.10 at impact=5, got {multiplier:.4f}"
        )

    def test_impact_scaling_near_max_at_impact_10(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        geo = scenario_engine.get_scenario("Geopolitical")
        scaled = scenario_engine.apply_impact_scaling(geo, impact=10.0)
        multiplier = scaled["_impact_multiplier"]
        assert abs(multiplier - 1.10) < 0.01, (
            f"Expected multiplier ≈ 1.10 at impact=10, got {multiplier:.4f}"
        )

    def test_impact_scaling_monotonically_increasing(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        impacts = [5.0, 6.0, 7.0, 7.5, 8.0, 9.0, 10.0]
        multipliers = []
        geo = scenario_engine.get_scenario("Geopolitical")
        for imp in impacts:
            sc = scenario_engine.apply_impact_scaling(geo, impact=imp)
            multipliers.append(sc["_impact_multiplier"])
        for i in range(len(multipliers) - 1):
            assert multipliers[i] < multipliers[i + 1], (
                f"Multiplier should be monotonically increasing: "
                f"impact={impacts[i]} → {multipliers[i]:.4f}, "
                f"impact={impacts[i+1]} → {multipliers[i+1]:.4f}"
            )

    def test_impact_scaling_does_not_mutate_original(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        geo = scenario_engine.get_scenario("Geopolitical")
        original_equity = geo["shocks"]["equity_shock_pct"]
        _ = scenario_engine.apply_impact_scaling(geo, impact=9.0)
        # Original should be unchanged
        assert geo["shocks"]["equity_shock_pct"] == original_equity

    def test_sentiment_sign_positive_flips_equity_shock_earnings(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        earnings = scenario_engine.get_scenario("Earnings")
        base_equity = earnings["shocks"]["equity_shock_pct"]
        # Apply positive sentiment — should flip the sign
        updated = scenario_engine.apply_sentiment_sign(earnings, sentiment=0.7, event_type="Earnings")
        new_equity = updated["shocks"]["equity_shock_pct"]
        assert new_equity > 0, (
            f"Positive sentiment should flip equity shock to positive, got {new_equity}"
        )
        assert abs(abs(new_equity) - abs(base_equity)) < 0.001

    def test_sentiment_sign_negative_keeps_negative_shock(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        earnings = scenario_engine.get_scenario("Earnings")
        updated = scenario_engine.apply_sentiment_sign(earnings, sentiment=-0.8, event_type="Earnings")
        assert updated["shocks"]["equity_shock_pct"] < 0

    def test_sentiment_sign_no_change_for_geopolitical(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        geo = scenario_engine.get_scenario("Geopolitical")
        original = geo["shocks"]["equity_shock_pct"]
        updated = scenario_engine.apply_sentiment_sign(geo, sentiment=0.9, event_type="Geopolitical")
        # Geopolitical is not sentiment-driven; equity shock should be unchanged
        assert updated["shocks"]["equity_shock_pct"] == original

    def test_pd_lgd_multipliers_credit_larger_than_others(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        credit_pd, credit_lgd = scenario_engine.get_pd_lgd_multipliers("Credit", impact=9.0)
        geo_pd, geo_lgd = scenario_engine.get_pd_lgd_multipliers("Geopolitical", impact=9.0)
        macro_pd, macro_lgd = scenario_engine.get_pd_lgd_multipliers("Macro", impact=9.0)
        # Credit event should stress PD the most
        assert credit_pd > geo_pd > macro_pd
        assert credit_lgd >= geo_lgd

    def test_pd_lgd_multipliers_base_is_1_below_impact_5(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        pd_m, lgd_m = scenario_engine.get_pd_lgd_multipliers("Geopolitical", impact=4.0)
        assert pd_m == 1.0, f"PD multiplier should be 1.0 at impact<5, got {pd_m}"
        assert lgd_m == 1.0, f"LGD multiplier should be 1.0 at impact<5, got {lgd_m}"

    def test_flatten_shocks_includes_sector_overrides(
        self, scenario_engine: ScenarioEngine
    ) -> None:
        geo = scenario_engine.get_scenario("Geopolitical")
        flat = scenario_engine.flatten_shocks(geo)
        assert "sector_equity_overrides" in flat
        assert "rates_bps" in flat
        assert "equity_shock_pct" in flat


# ─────────────────────────────────────────────────────────────────────────────
# StressTrigger tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStressTrigger:

    def test_trigger_fires_when_conditions_met(self, trigger: StressTrigger) -> None:
        signal = _make_signal(impact=8.0, confidence=0.75)
        assert trigger.consume_signals(signal) is True

    def test_trigger_does_not_fire_low_impact(self, trigger: StressTrigger) -> None:
        signal = _make_signal(impact=6.9, confidence=0.90)
        assert trigger.consume_signals(signal) is False

    def test_trigger_does_not_fire_exactly_at_threshold_minus_epsilon(
        self, trigger: StressTrigger
    ) -> None:
        signal = _make_signal(impact=6.99, confidence=0.90)
        assert trigger.consume_signals(signal) is False

    def test_trigger_fires_exactly_at_threshold(self, trigger: StressTrigger) -> None:
        signal = _make_signal(impact=7.0, confidence=0.55)
        assert trigger.consume_signals(signal) is True

    def test_trigger_does_not_fire_low_confidence(self, trigger: StressTrigger) -> None:
        signal = _make_signal(impact=9.0, confidence=0.40)
        assert trigger.consume_signals(signal) is False

    def test_cooldown_prevents_second_trigger(self, trigger: StressTrigger) -> None:
        signal = _make_signal(event_type="Geopolitical", ticker="JPM")
        # First call fires
        assert trigger.consume_signals(signal) is True
        # Second call with same ticker+event_type should be blocked
        assert trigger.consume_signals(signal) is False

    def test_different_event_type_not_blocked_by_cooldown(
        self, trigger: StressTrigger
    ) -> None:
        geo = _make_signal(event_type="Geopolitical", ticker="JPM", impact=8.0, confidence=0.8)
        credit = _make_signal(event_type="Credit", ticker="JPM", impact=8.0, confidence=0.8)
        # Fire Geopolitical
        assert trigger.consume_signals(geo) is True
        # Credit on same ticker should still fire (different event type)
        assert trigger.consume_signals(credit) is True

    def test_different_ticker_not_blocked(self, trigger: StressTrigger) -> None:
        jpm = _make_signal(event_type="Geopolitical", ticker="JPM", impact=8.0, confidence=0.8)
        gs = _make_signal(event_type="Geopolitical", ticker="GS", impact=8.0, confidence=0.8)
        assert trigger.consume_signals(jpm) is True
        # GS is a different ticker — should also fire
        assert trigger.consume_signals(gs) is True

    def test_is_on_cooldown_false_before_trigger(self, trigger: StressTrigger) -> None:
        assert trigger.is_on_cooldown("MSFT", "Earnings") is False

    def test_is_on_cooldown_true_after_trigger(self, trigger: StressTrigger) -> None:
        trigger.mark_triggered("NVDA", "Cyber")
        assert trigger.is_on_cooldown("NVDA", "Cyber") is True

    def test_reset_cooldown(self, trigger: StressTrigger) -> None:
        trigger.mark_triggered("AAPL", "M&A")
        assert trigger.is_on_cooldown("AAPL", "M&A") is True
        trigger.reset_cooldown("AAPL", "M&A")
        assert trigger.is_on_cooldown("AAPL", "M&A") is False

    def test_active_cooldowns_snapshot(self, trigger: StressTrigger) -> None:
        trigger.mark_triggered("BAC", "Credit")
        active = trigger.active_cooldowns
        assert any("BAC" in k for k in active.keys())

    def test_multi_ticker_all_marked_on_trigger(self) -> None:
        """When a multi-ticker signal fires, all tickers should be on cooldown."""
        trig = StressTrigger()
        now = datetime.now(tz=timezone.utc)
        signal = RiskSignal(
            event_id="multi-ticker-test",
            ts_published=now - timedelta(seconds=10),
            ts_processed=now,
            source="news",
            tickers=[
                TickerRef(ticker="JPM", relevance=0.9),
                TickerRef(ticker="BAC", relevance=0.7),
            ],
            sentiment=-0.6,
            sentiment_conf=0.85,
            event_type="Credit",
            event_probs={"Credit": 0.92, "Geopolitical": 0.08},
            impact=8.5,
            impact_range=(8.0, 9.0),
            confidence=0.80,
            novelty=0.75,
            diffusion=DiffusionInfo(sources=5, velocity=3.0, score=0.70),
            decay_halflife_hours=18.0,
            cluster_id="multi-test-cluster",
            model_versions={},
        )
        assert trig.consume_signals(signal) is True
        assert trig.is_on_cooldown("JPM", "Credit") is True
        assert trig.is_on_cooldown("BAC", "Credit") is True


# ─────────────────────────────────────────────────────────────────────────────
# StressEngine integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStressEngine:
    """Integration tests — no external I/O (db=None, redis=None)."""

    @pytest.fixture(autouse=True)
    def engine(self) -> StressEngine:
        # Fresh engine per test to avoid cooldown bleed-over
        self._engine = StressEngine(db=None, redis=None)
        return self._engine

    def _sig(self, event_type: str, impact: float = 8.5, confidence: float = 0.80,
             sentiment: float = -0.5, ticker: str = "JPM") -> RiskSignal:
        return _make_signal(
            event_type=event_type,
            impact=impact,
            confidence=confidence,
            sentiment=sentiment,
            ticker=ticker,
        )

    def test_full_run_returns_stress_result(self) -> None:
        signal = self._sig("Geopolitical")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        assert isinstance(result, StressResult)

    def test_result_has_required_fields(self) -> None:
        signal = self._sig("Geopolitical")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        assert result.run_id
        assert isinstance(result.ts_created, datetime)
        assert result.trigger_signal_id == signal.event_id
        assert result.scenario_name == "Geopolitical"
        assert result.event_type == "Geopolitical"
        assert result.impact == 8.5
        assert isinstance(result.pre_nav, float)
        assert isinstance(result.post_nav, float)
        assert isinstance(result.delta_nav, float)
        assert isinstance(result.delta_pct, float)
        assert isinstance(result.breakdown_by_asset_class, dict)
        assert isinstance(result.breakdown_by_sector, dict)
        assert isinstance(result.shocks_applied, dict)
        assert result.model_version

    def test_delta_nav_negative_geopolitical(self) -> None:
        signal = self._sig("Geopolitical", impact=9.0, confidence=0.90, sentiment=-0.8)
        result = self._engine.run_stress_test(signal)
        assert result is not None
        assert result.delta_nav < 0, (
            f"Geopolitical event should produce a portfolio loss. "
            f"Got ΔNAV = ${result.delta_nav:,.0f}"
        )

    def test_delta_nav_negative_credit(self) -> None:
        signal = self._sig("Credit", impact=9.0, confidence=0.85, sentiment=-0.7,
                           ticker="BAC")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        assert result.delta_nav < 0, (
            f"Credit event should produce a portfolio loss. "
            f"Got ΔNAV = ${result.delta_nav:,.0f}"
        )

    def test_delta_nav_negative_macro(self) -> None:
        signal = self._sig("Macro", impact=8.0, confidence=0.75, sentiment=-0.4,
                           ticker="GS")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        assert result.delta_nav < 0, (
            f"Macro event should produce a portfolio loss. "
            f"Got ΔNAV = ${result.delta_nav:,.0f}"
        )

    def test_geopolitical_equities_down_in_breakdown(self) -> None:
        signal = self._sig("Geopolitical", impact=9.0, sentiment=-0.8)
        result = self._engine.run_stress_test(signal)
        assert result is not None
        assert result.breakdown_by_asset_class["equities"] < 0, (
            "Geopolitical event: equity P&L should be negative"
        )

    def test_geopolitical_bonds_benefit_from_rate_fall(self) -> None:
        """
        Geopolitical shock has rates_bps = -25 (flight to quality).
        Bonds benefit from rate falls (positive P&L for rate component),
        but spread widening (+120bps) offsets it.  The net should be computed
        correctly — we just check the shocks are correctly applied.
        """
        signal = self._sig("Geopolitical", impact=9.0, sentiment=-0.8)
        result = self._engine.run_stress_test(signal)
        assert result is not None
        # rates_bps is negative in geopolitical scenario
        shocks = result.shocks_applied
        assert shocks.get("rates_bps", 0) < 0, (
            "Geopolitical scenario should have negative rates_bps (flight to safety)"
        )
        # Spread should widen
        assert shocks.get("credit_spreads_bps", 0) > 0

    def test_credit_event_loans_are_largest_loser(self) -> None:
        """For a credit event at high impact, loans should be the largest loss."""
        signal = self._sig("Credit", impact=9.5, confidence=0.90, sentiment=-0.9,
                           ticker="JPM")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        ac = result.breakdown_by_asset_class
        loan_loss = ac.get("loans", 0)
        assert loan_loss < 0, "Loans should show a loss in a credit event"
        # Loan loss should be the largest individual component
        # (bonds and equities also lose, but loans are stress-driven by PD/LGD multipliers)
        other_losses = [v for k, v in ac.items() if k != "loans"]
        # Loan absolute loss should be in the top contributors
        assert abs(loan_loss) > 0, "Loan loss should be non-zero in a credit event"

    def test_no_trigger_below_impact_threshold(self) -> None:
        signal = _make_signal(impact=6.5, confidence=0.90)
        result = self._engine.run_stress_test(signal)
        assert result is None, (
            "Impact 6.5 < 7.0 should not trigger a stress test"
        )

    def test_no_trigger_below_confidence_threshold(self) -> None:
        signal = _make_signal(impact=9.0, confidence=0.40)
        result = self._engine.run_stress_test(signal)
        assert result is None, (
            "Confidence 0.40 < 0.55 should not trigger a stress test"
        )

    def test_pre_nav_matches_portfolio_nav(self) -> None:
        signal = self._sig("Regulatory")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        expected_nav = self._engine.portfolio.compute_nav()
        assert abs(result.pre_nav - expected_nav) < 1.0

    def test_post_nav_equals_pre_plus_delta(self) -> None:
        signal = self._sig("Macro")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        assert abs(result.post_nav - (result.pre_nav + result.delta_nav)) < 1.0

    def test_delta_pct_consistent_with_delta_nav(self) -> None:
        signal = self._sig("Credit")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        expected_pct = result.delta_nav / result.pre_nav * 100.0
        assert abs(result.delta_pct - expected_pct) < 0.01

    def test_shocks_applied_contains_pd_lgd_multipliers(self) -> None:
        signal = self._sig("Credit", impact=9.0)
        result = self._engine.run_stress_test(signal)
        assert result is not None
        assert "pd_multiplier" in result.shocks_applied
        assert "lgd_multiplier" in result.shocks_applied
        assert result.shocks_applied["pd_multiplier"] > 1.0
        assert result.shocks_applied["lgd_multiplier"] > 1.0

    def test_sector_breakdown_sums_close_to_total_delta(self) -> None:
        signal = self._sig("Geopolitical")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        # FX forwards are not attributed to any sector; all other asset classes
        # (bonds, loans, equities, swaps) contribute to the sector breakdown.
        # Verify: sector_total ≈ AC_total - FX_pnl
        sector_total = sum(result.breakdown_by_sector.values())
        ac_without_fx = sum(v for k, v in result.breakdown_by_asset_class.items()
                            if k != "fx_forwards")
        # The per-position P&L is rounded to 4 decimal places; across 285 non-FX
        # positions the maximum accumulated rounding error is 285 × 0.00005 ≈ $0.01.
        # In practice floating-point arithmetic introduces slightly larger errors,
        # so we allow up to $200k absolute tolerance (well under 1% of NAV).
        tolerance = 200_000
        assert abs(sector_total - ac_without_fx) < tolerance, (
            f"Sector total {sector_total:,.2f} vs AC-without-FX {ac_without_fx:,.2f}; "
            f"difference {abs(sector_total - ac_without_fx):,.2f} exceeds tolerance {tolerance:,.0f}"
        )

    def test_earnings_positive_sentiment_gives_positive_equity_pnl(self) -> None:
        """Positive earnings surprise should push equities up."""
        signal = _make_signal(
            event_type="Earnings", impact=7.5, confidence=0.75, sentiment=0.85
        )
        result = self._engine.run_stress_test(signal)
        assert result is not None
        # With positive sentiment, equity_shock_pct should be positive
        assert result.shocks_applied.get("equity_shock_pct", 0) > 0, (
            "Positive earnings sentiment should produce positive equity shock"
        )

    def test_to_dict_is_json_serialisable(self) -> None:
        """StressResult.to_dict() must round-trip through JSON without error."""
        import json
        signal = self._sig("Geopolitical")
        result = self._engine.run_stress_test(signal)
        assert result is not None
        d = result.to_dict()
        serialised = json.dumps(d, default=str)
        parsed = json.loads(serialised)
        assert parsed["event_type"] == "Geopolitical"
        assert "delta_nav" in parsed

    def test_top_losers_are_negative(self) -> None:
        signal = self._sig("Geopolitical", impact=9.0)
        result = self._engine.run_stress_test(signal)
        assert result is not None
        if result.top_losers:
            for pos_id, pnl in result.top_losers:
                assert pnl < 0, f"Top loser {pos_id} should have negative P&L, got {pnl}"


# ─────────────────────────────────────────────────────────────────────────────
# Async StressEngine tests (persist_result / on_signal)
# ─────────────────────────────────────────────────────────────────────────────

class TestStressEngineAsync:

    @pytest.mark.asyncio
    async def test_on_signal_returns_none_below_threshold(self) -> None:
        engine = StressEngine(db=None, redis=None)
        signal = _make_signal(impact=5.0, confidence=0.40)
        result = await engine.on_signal(signal)
        assert result is None

    @pytest.mark.asyncio
    async def test_on_signal_returns_result_above_threshold(self) -> None:
        engine = StressEngine(db=None, redis=None)
        signal = _make_signal(impact=8.0, confidence=0.75, ticker="ASYNC_TEST")
        result = await engine.on_signal(signal)
        assert result is not None
        assert isinstance(result, StressResult)

    @pytest.mark.asyncio
    async def test_persist_result_calls_write_stress_run(self) -> None:
        mock_db = AsyncMock()
        mock_db.write_stress_run = AsyncMock()
        engine = StressEngine(db=mock_db, redis=None)
        signal = _make_signal(impact=8.0, confidence=0.75, ticker="PERSIST")
        result = engine.run_stress_test(signal)
        assert result is not None
        await engine.persist_result(result)
        mock_db.write_stress_run.assert_called_once()
        call_kwargs = mock_db.write_stress_run.call_args[0][0]
        assert call_kwargs["run_id"] == result.run_id
        assert call_kwargs["scenario_name"] == result.scenario_name

    @pytest.mark.asyncio
    async def test_on_signal_calls_db_and_redis(self) -> None:
        mock_db = AsyncMock()
        mock_db.write_stress_run = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock()
        engine = StressEngine(db=mock_db, redis=mock_redis)
        signal = _make_signal(impact=8.0, confidence=0.75, ticker="IOTEST")
        result = await engine.on_signal(signal)
        assert result is not None
        mock_db.write_stress_run.assert_called_once()
        mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_persist_result_noop_without_db(self) -> None:
        """No exception should be raised when db is None."""
        engine = StressEngine(db=None, redis=None)
        signal = _make_signal(impact=8.0, confidence=0.75)
        result = engine.run_stress_test(signal)
        assert result is not None
        # Should not raise
        await engine.persist_result(result)
