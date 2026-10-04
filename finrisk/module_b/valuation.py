"""
First-order factor-shock portfolio valuator.

Pricing methodology
───────────────────
All valuation is delta-approximate: we compute the linear (and, for bonds,
second-order convexity) sensitivity of each position to a factor shock and sum
the dollar P&L.  Full revaluation would require a yield curve, credit term
structure, and FX vol surface; for an event-driven stress tester the factor
approximation is standard (it matches Basel FRTB's SA sensitivity approach).

Bond pricing
    ΔP = P × [-D × Δr + 0.5 × C × Δr²] + P × [-D_spread × Δs]
    where:
        D        = modified duration
        C        = convexity (years²)
        Δr       = rate shock in decimal (bps / 10000)
        D_spread = spread duration ≈ D × 0.85
        Δs       = spread shock in decimal

Loan loss
    ΔEL = (PD_stressed × LGD_stressed - PD_base × LGD_base) × EAD
    loss is reported as a negative P&L (increase in expected loss = portfolio loss)

Equity pricing
    ΔV = β × equity_shock × market_value
    sector_overrides replace the base equity_shock for positions in that sector

FX forward pricing
    ΔV ≈ notional_usd × fx_usd_chg_pct × 0.5
    the 0.5 factor reflects that only half the notional of a hedging book is
    long-USD on net (the portfolio.yaml notes near-zero net delta by construction)

Swap pricing
    ΔV = -DV01 × rate_shock_bps
    negative because pay-fixed swaps lose when rates rise
"""

from __future__ import annotations

from dataclasses import dataclass, field

from module_b.portfolio import (
    Bond,
    Equity,
    FXForward,
    InterestRateSwap,
    Loan,
    SyntheticPortfolio,
)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class PnLResult:
    """
    Full portfolio P&L breakdown after a scenario shock.

    Attributes
    ──────────
    pre_nav                 : portfolio NAV before the shock (USD)
    post_nav                : portfolio NAV after the shock (USD)
    delta_nav               : post_nav - pre_nav  (negative = loss)
    delta_pct               : delta_nav / pre_nav × 100
    breakdown_by_asset_class: {asset_class: delta_pnl_usd}
    breakdown_by_sector     : {sector: delta_pnl_usd}
    position_pnl            : {position_id: delta_pnl_usd} — full drill-down
    """

    pre_nav: float
    post_nav: float
    delta_nav: float
    delta_pct: float
    breakdown_by_asset_class: dict[str, float]
    breakdown_by_sector: dict[str, float]
    position_pnl: dict[str, float] = field(default_factory=dict)

    @property
    def top_losers(self) -> list[tuple[str, float]]:
        """Return top 10 positions by absolute loss (most negative first)."""
        return sorted(
            self.position_pnl.items(),
            key=lambda kv: kv[1],
        )[:10]


# ── Valuator ──────────────────────────────────────────────────────────────────

class PortfolioValuator:
    """
    Computes P&L for each asset class under a given set of factor shocks.

    All methods are pure functions of their inputs — no internal state is
    mutated.  This makes the valuator safe for concurrent use.
    """

    # Spread duration = fraction of modified duration used for credit-spread sens.
    # Standard market convention for IG bonds is ~0.85-0.90; we use 0.85.
    SPREAD_DURATION_FACTOR: float = 0.85

    # ── Bond valuation ────────────────────────────────────────────────────────

    def valuate_bonds(
        self,
        bonds: list[Bond],
        rate_shock_bps: float,
        spread_shock_bps: float,
    ) -> dict[str, float]:
        """
        Compute dollar P&L for each bond under rate and spread shocks.

        Returns
        ───────
        dict mapping position_id → delta_pnl (negative = loss)

        The second-order convexity term is included to avoid over-estimating
        losses for large rate shocks (bonds have positive convexity; the
        convexity term partially offsets the duration loss).
        """
        delta_rate = rate_shock_bps / 10_000.0   # convert bps → decimal
        delta_spread = spread_shock_bps / 10_000.0

        result: dict[str, float] = {}
        for bond in bonds:
            d = bond.duration
            c = bond.convexity
            mv = bond.market_value
            d_spread = d * self.SPREAD_DURATION_FACTOR

            # Rate component (duration-convexity approximation)
            pnl_rate = mv * (-d * delta_rate + 0.5 * c * delta_rate ** 2)
            # Spread component (credit spread widening → bond price falls)
            pnl_spread = mv * (-d_spread * delta_spread)

            result[bond.position_id] = round(pnl_rate + pnl_spread, 4)
        return result

    # ── Loan valuation ────────────────────────────────────────────────────────

    def valuate_loans(
        self,
        loans: list[Loan],
        pd_multiplier: float,
        lgd_multiplier: float,
    ) -> dict[str, float]:
        """
        Compute dollar P&L for each loan under stressed PD and LGD.

        The delta is the change in expected loss:
            ΔEL = (PD_stressed × LGD_stressed - PD_base × LGD_base) × EAD

        Since an increase in EL is a loss to the portfolio, the returned P&L
        is negative when PD or LGD increases (multipliers > 1).

        Parameters
        ──────────
        pd_multiplier  : factor applied to base PD (e.g. 1.5 → PD × 1.5)
        lgd_multiplier : factor applied to base LGD (e.g. 1.1 → LGD × 1.1)
        """
        result: dict[str, float] = {}
        for loan in loans:
            pd_stressed = min(loan.pd * pd_multiplier, 1.0)
            lgd_stressed = min(loan.lgd * lgd_multiplier, 1.0)
            el_stressed = pd_stressed * lgd_stressed * loan.ead
            el_base = loan.pd * loan.lgd * loan.ead
            delta_el = el_stressed - el_base  # positive = more loss
            result[loan.position_id] = round(-delta_el, 4)  # negative P&L
        return result

    # ── Equity valuation ──────────────────────────────────────────────────────

    def valuate_equities(
        self,
        equities: list[Equity],
        equity_shock_pct: float,
        sector_overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Compute dollar P&L for each equity position under a market-wide shock
        and optional per-sector overrides.

        The effective shock for a position in sector S is:
            shock = sector_overrides[S]  if S in sector_overrides
            shock = equity_shock_pct     otherwise

        Dollar P&L = beta × effective_shock × market_value

        Parameters
        ──────────
        equity_shock_pct : market-wide shock (e.g. -0.10 = −10 %)
        sector_overrides : map of sector → override shock (replaces base shock)
        """
        overrides = sector_overrides or {}
        result: dict[str, float] = {}
        for equity in equities:
            shock = overrides.get(equity.sector, equity_shock_pct)
            pnl = equity.beta * shock * equity.market_value
            result[equity.position_id] = round(pnl, 4)
        return result

    # ── FX forward valuation ──────────────────────────────────────────────────

    def valuate_fx_forwards(
        self,
        fx_forwards: list[FXForward],
        fx_usd_chg_pct: float,
    ) -> dict[str, float]:
        """
        Compute dollar P&L for each FX forward under a USD strength shock.

        The approximation ΔV = notional_usd × fx_usd_chg_pct × 0.5 reflects
        that the hedging book has near-zero net USD delta by construction
        (matching the portfolio.yaml spec); the 0.5 dampening factor models
        the net partial exposure that remains after netting long and short legs.

        A positive fx_usd_chg_pct (USD strengthens) creates a small loss on
        forward contracts that were entered to sell USD.
        """
        result: dict[str, float] = {}
        for fwd in fx_forwards:
            pnl = fwd.notional_usd * fx_usd_chg_pct * 0.5
            result[fwd.position_id] = round(pnl, 4)
        return result

    # ── Swap valuation ────────────────────────────────────────────────────────

    def valuate_swaps(
        self,
        swaps: list[InterestRateSwap],
        rate_shock_bps: float,
    ) -> dict[str, float]:
        """
        Compute dollar P&L for each pay-fixed IRS under a rate shock.

        ΔV = -DV01 × rate_shock_bps

        Pay-fixed swaps have negative DV01 exposure: when rates rise, the
        fixed-rate leg becomes more expensive relative to floating receipts,
        so the swap loses market value.  When rates fall (negative shock),
        the swap gains.
        """
        result: dict[str, float] = {}
        for swap in swaps:
            pnl = -swap.dv01 * rate_shock_bps
            result[swap.position_id] = round(pnl, 4)
        return result

    # ── Full portfolio valuation ──────────────────────────────────────────────

    def compute_total_pnl(
        self,
        portfolio: SyntheticPortfolio,
        shocks: dict,
    ) -> PnLResult:
        """
        Apply all factor shocks to the full portfolio and return a PnLResult.

        Parameters
        ──────────
        portfolio : SyntheticPortfolio
        shocks    : dict with keys from scenarios.yaml:
            rate_shock_bps    — int/float bps
            spread_shock_bps  — int/float bps
            equity_shock_pct  — float (e.g. -0.12)
            fx_usd_chg_pct    — float
            pd_multiplier     — float (default 1.0)
            lgd_multiplier    — float (default 1.0)
            sector_equity_overrides — dict[str, float] (optional)

        Returns
        ───────
        PnLResult with full breakdown
        """
        rate_shock = float(shocks.get("rates_bps", 0.0))
        spread_shock = float(shocks.get("credit_spreads_bps", 0.0))
        equity_shock = float(shocks.get("equity_shock_pct", 0.0))
        fx_shock = float(shocks.get("fx_usd_chg_pct", 0.0))
        pd_mult = float(shocks.get("pd_multiplier", 1.0))
        lgd_mult = float(shocks.get("lgd_multiplier", 1.0))
        sector_overrides: dict[str, float] = shocks.get("sector_equity_overrides", {})

        # Per-position P&L
        bond_pnl = self.valuate_bonds(portfolio.bonds, rate_shock, spread_shock)
        loan_pnl = self.valuate_loans(portfolio.loans, pd_mult, lgd_mult)
        equity_pnl = self.valuate_equities(
            portfolio.equities, equity_shock, sector_overrides
        )
        fx_pnl = self.valuate_fx_forwards(portfolio.fx_forwards, fx_shock)
        swap_pnl = self.valuate_swaps(portfolio.swaps, rate_shock)

        # Aggregate position P&L into one flat dict
        position_pnl: dict[str, float] = {}
        position_pnl.update(bond_pnl)
        position_pnl.update(loan_pnl)
        position_pnl.update(equity_pnl)
        position_pnl.update(fx_pnl)
        position_pnl.update(swap_pnl)

        # Asset class breakdown
        ac_breakdown = {
            "bonds":       round(sum(bond_pnl.values()), 2),
            "loans":       round(sum(loan_pnl.values()), 2),
            "equities":    round(sum(equity_pnl.values()), 2),
            "fx_forwards": round(sum(fx_pnl.values()), 2),
            "swaps":       round(sum(swap_pnl.values()), 2),
        }

        # Sector breakdown: aggregate over bonds + loans + equities + swaps
        sector_breakdown: dict[str, float] = {}
        for bond in portfolio.bonds:
            sector_breakdown[bond.sector] = (
                sector_breakdown.get(bond.sector, 0.0) + bond_pnl[bond.position_id]
            )
        for loan in portfolio.loans:
            sector_breakdown[loan.sector] = (
                sector_breakdown.get(loan.sector, 0.0) + loan_pnl[loan.position_id]
            )
        for equity in portfolio.equities:
            sector_breakdown[equity.sector] = (
                sector_breakdown.get(equity.sector, 0.0) + equity_pnl[equity.position_id]
            )
        for swap in portfolio.swaps:
            sector_breakdown[swap.sector] = (
                sector_breakdown.get(swap.sector, 0.0) + swap_pnl[swap.position_id]
            )
        sector_breakdown = {k: round(v, 2) for k, v in sector_breakdown.items()}

        # NAV aggregation
        pre_nav = portfolio.compute_nav()
        total_delta = sum(position_pnl.values())
        post_nav = pre_nav + total_delta
        delta_pct = (total_delta / pre_nav * 100.0) if pre_nav != 0 else 0.0

        return PnLResult(
            pre_nav=round(pre_nav, 2),
            post_nav=round(post_nav, 2),
            delta_nav=round(total_delta, 2),
            delta_pct=round(delta_pct, 4),
            breakdown_by_asset_class=ac_breakdown,
            breakdown_by_sector=sector_breakdown,
            position_pnl={k: round(v, 4) for k, v in position_pnl.items()},
        )
