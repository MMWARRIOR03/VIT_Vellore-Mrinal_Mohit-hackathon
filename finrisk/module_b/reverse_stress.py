"""
Reverse stress testing for Module B.

Reverse stress asks the inverse of the standard question: instead of
"how much do we lose in scenario X?", it asks "what severity of event X
is required to breach our loss limit?".

Algorithm
─────────
Binary search on impact in [1.0, 10.0].  For each candidate impact value,
we run a reduced Monte Carlo (n_paths=500 for speed) and check whether the
expected loss exceeds the loss_limit_pct * NAV.  The bisection terminates
when the hi-lo interval is narrower than the requested precision.

The result reports:
  - breach_impact:            the minimum impact at which the limit is breached
  - expected_loss_at_breach:  mean MC loss at that impact
  - expected_loss_pct:        that loss as a fraction of NAV
  - confidence_interval:      90 % CI on breach_impact (±half-precision width)
  - narrative:                plain-English summary
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from module_b.monte_carlo import MonteCarloEngine
from module_b.portfolio import SyntheticPortfolio
from module_b.scenarios import ScenarioEngine
from module_b.valuation import PortfolioValuator

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ReverseStressResult:
    """
    Output of a reverse stress test.

    Attributes
    ──────────
    event_type              : event category tested
    loss_limit_pct          : user-supplied loss limit as a fraction of NAV
    breach_impact           : minimum impact score (1–10) that breaches the limit
    expected_loss_at_breach : mean MC expected loss at breach_impact (USD)
    expected_loss_pct       : expected_loss_at_breach / NAV
    confidence_interval     : 90 % CI on breach_impact as (lo, hi)
    narrative               : plain-English description of the result
    """

    event_type: str
    loss_limit_pct: float
    breach_impact: float
    expected_loss_at_breach: float
    expected_loss_pct: float
    confidence_interval: tuple[float, float]
    narrative: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "event_type": self.event_type,
            "loss_limit_pct": round(self.loss_limit_pct, 4),
            "breach_impact": round(self.breach_impact, 2),
            "expected_loss_at_breach": round(self.expected_loss_at_breach, 2),
            "expected_loss_pct": round(self.expected_loss_pct, 4),
            "confidence_interval": (
                round(self.confidence_interval[0], 2),
                round(self.confidence_interval[1], 2),
            ),
            "narrative": self.narrative,
        }


# ── Reverse stress tester ─────────────────────────────────────────────────────

class ReverseStressTester:
    """
    Finds the minimum event impact that breaches a portfolio loss limit.

    Uses a binary search over impact in [1.0, 10.0], evaluating the expected
    loss from a reduced Monte Carlo run at each candidate impact value.
    """

    # n_paths used for each MC eval inside the binary search (speed trade-off)
    _SEARCH_N_PATHS: int = 500

    def find_breach_threshold(
        self,
        portfolio: SyntheticPortfolio,
        valuator: PortfolioValuator,
        scenario_engine: ScenarioEngine,
        mc_engine: MonteCarloEngine,
        event_type: str,
        loss_limit_pct: float = 0.10,
        precision: float = 0.1,
    ) -> ReverseStressResult:
        """
        Binary-search for the minimum impact that causes expected_loss to
        exceed loss_limit_pct × portfolio.compute_nav().

        Parameters
        ──────────
        portfolio       : SyntheticPortfolio
        valuator        : PortfolioValuator
        scenario_engine : ScenarioEngine (fetches base shocks for event_type)
        mc_engine       : MonteCarloEngine (its seed / covariance are reused;
                          n_paths is overridden to _SEARCH_N_PATHS for speed)
        event_type      : event category to reverse-stress
        loss_limit_pct  : loss threshold as fraction of NAV (default 0.10 = 10 %)
        precision       : binary-search convergence criterion (default 0.1 impact units)

        Returns
        ───────
        ReverseStressResult.  If no impact in [1, 10] breaches the limit,
        breach_impact is returned as 10.0 with a note in the narrative.
        If even impact=1.0 breaches the limit, breach_impact is 1.0.
        """
        nav = portfolio.compute_nav()
        loss_limit_usd = loss_limit_pct * nav

        # Build a fast MC engine with reduced n_paths for the search
        fast_mc = MonteCarloEngine(
            n_paths=self._SEARCH_N_PATHS,
            seed=mc_engine._rng.integers(0, 2**32),  # independent seed
        )

        def _expected_loss_at(impact_val: float) -> float:
            """Run fast MC for a given impact and return expected loss."""
            # Build scenario shocks for this impact level
            scenario = scenario_engine.get_scenario(event_type)
            scenario = scenario_engine.apply_impact_scaling(scenario, impact_val)
            # Apply neutral sentiment (conservative — no sign flip)
            scenario = scenario_engine.apply_sentiment_sign(scenario, sentiment=-0.5, event_type=event_type)
            pd_mult, lgd_mult = scenario_engine.get_pd_lgd_multipliers(event_type, impact_val)
            flat_shocks = scenario_engine.flatten_shocks(scenario)
            flat_shocks["pd_multiplier"] = pd_mult
            flat_shocks["lgd_multiplier"] = lgd_mult

            # Use a symmetric impact_range centred on the candidate impact value
            half_width = min(precision * 2, 0.5)
            impact_range = (
                max(1.0, impact_val - half_width),
                min(10.0, impact_val + half_width),
            )

            mc_result = fast_mc.run(
                portfolio, flat_shocks, impact_val, impact_range, valuator
            )
            return mc_result.expected_loss

        # ── Binary search ──────────────────────────────────────────────────────
        lo, hi = 1.0, 10.0
        breach_found = False

        # First check boundary conditions
        loss_at_hi = _expected_loss_at(hi)
        if loss_at_hi <= loss_limit_usd:
            # Even at max impact the limit is not breached
            breach_impact = 10.0
            expected_loss = loss_at_hi
            breach_found = False
        else:
            loss_at_lo = _expected_loss_at(lo)
            if loss_at_lo >= loss_limit_usd:
                # Already breached at minimum impact
                breach_impact = 1.0
                expected_loss = loss_at_lo
                breach_found = True
            else:
                # Binary search: find the crossing point
                breach_found = True
                iterations = 0
                while (hi - lo) > precision:
                    mid = (lo + hi) / 2.0
                    loss_mid = _expected_loss_at(mid)
                    if loss_mid >= loss_limit_usd:
                        hi = mid
                    else:
                        lo = mid
                    iterations += 1
                    if iterations > 50:  # safety guard
                        break
                breach_impact = (lo + hi) / 2.0
                expected_loss = _expected_loss_at(breach_impact)

        expected_loss_pct = expected_loss / nav if nav > 0 else 0.0

        # ── 90 % CI: ±precision/2 around the found breach point ───────────────
        ci_half = precision / 2.0
        ci = (
            max(1.0, breach_impact - ci_half),
            min(10.0, breach_impact + ci_half),
        )

        # ── Narrative ──────────────────────────────────────────────────────────
        if not breach_found:
            narrative = (
                f"A {event_type} event does not breach your "
                f"{loss_limit_pct:.0%} loss limit (${loss_limit_usd:,.0f}) "
                f"at any impact up to 10/10. "
                f"Maximum expected loss at impact=10 is "
                f"${expected_loss:,.0f} ({expected_loss_pct:.1%} of NAV)."
            )
        elif breach_impact <= 1.0:
            narrative = (
                f"A {event_type} event breaches your {loss_limit_pct:.0%} loss "
                f"limit (${loss_limit_usd:,.0f}) even at minimum impact (1/10). "
                f"Expected loss at breach: ${expected_loss:,.0f} "
                f"({expected_loss_pct:.1%} of NAV). "
                f"Consider reducing exposure to {event_type}-sensitive positions."
            )
        else:
            narrative = (
                f"A {event_type} event with impact \u2265 {breach_impact:.1f}/10 "
                f"would breach your {loss_limit_pct:.0%} loss limit "
                f"(${loss_limit_usd:,.0f}). "
                f"Expected loss at breach: ${expected_loss:,.0f} "
                f"({expected_loss_pct:.1%} of NAV)."
            )

        logger.info(
            "Reverse stress [%s]: breach_impact=%.2f  E[loss]=$%.0f (%.1f%% NAV)  limit=%.1f%%",
            event_type,
            breach_impact,
            expected_loss,
            expected_loss_pct * 100,
            loss_limit_pct * 100,
        )

        return ReverseStressResult(
            event_type=event_type,
            loss_limit_pct=loss_limit_pct,
            breach_impact=breach_impact,
            expected_loss_at_breach=expected_loss,
            expected_loss_pct=expected_loss_pct,
            confidence_interval=ci,
            narrative=narrative,
        )
