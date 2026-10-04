"""
Scenario engine — loads shock vectors from configs/scenarios.yaml and applies
impact scaling, sentiment-sign correction, and PD/LGD multiplier logic.

Scaling formula
───────────────
For a signal with impact score I (1–10):

    raw_scale = clip((I - 5) / 5, 0, 1)
    multiplier = raw_scale² + 0.1          ← 0.1 floor so something always fires
    capped multiplier ≤ 1.1               ← slight super-linear at impact=10

At impact 5:  raw_scale=0 → multiplier=0.1  (10 % of full shocks)
At impact 7:  raw_scale=0.4 → multiplier=0.26  (26 %)
At impact 8:  raw_scale=0.6 → multiplier=0.46  (46 %)
At impact 9:  raw_scale=0.8 → multiplier=0.74  (74 %)
At impact 10: raw_scale=1.0 → multiplier=1.1   (110 % for extreme events)

Sentiment sign override
───────────────────────
For Earnings and Product events, the scenario YAML stores a conservative
negative base for equity_shock_pct.  At runtime, if the signal's sentiment
is positive, the sign of equity_shock_pct is flipped to positive.
This is documented in scenarios.yaml ("sign overridden by sentiment").

PD/LGD multipliers
──────────────────
Two reference stress levels anchor the calibration; intermediate impacts
interpolate linearly:
  Credit event,     impact 9: PD × 2.5, LGD × 1.3
  Geopolitical,     impact 8: PD × 1.8, LGD × 1.15
  All other types,  impact 7: PD × 1.3, LGD × 1.05  (mild credit deterioration)
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Scenario YAML lives two levels up from this file: finrisk/configs/scenarios.yaml
_DEFAULT_SCENARIOS_PATH = Path(__file__).resolve().parents[1] / "configs" / "scenarios.yaml"


class ScenarioEngine:
    """
    Loads and applies stress scenarios.

    Parameters
    ──────────
    scenarios_path : pathlib.Path | str | None
        Path to scenarios.yaml.  Defaults to finrisk/configs/scenarios.yaml.
    """

    # Event types for which equity_shock_pct sign follows sentiment
    _SENTIMENT_DRIVEN_EVENTS: frozenset[str] = frozenset({"Earnings", "Product"})

    # Shock keys whose magnitudes should be scaled (sign preserved, magnitude scaled)
    _MAGNITUDE_KEYS: tuple[str, ...] = (
        "rates_bps",
        "credit_spreads_bps",
        "equity_shock_pct",
        "fx_usd_chg_pct",
        "commodity_chg_pct",
        "vol_shock_bps",
    )

    def __init__(self, scenarios_path: Path | str | None = None) -> None:
        path = Path(scenarios_path) if scenarios_path else _DEFAULT_SCENARIOS_PATH
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        self._scenarios: dict[str, Any] = raw.get("scenarios", {})
        logger.info(
            "ScenarioEngine loaded %d scenarios from %s",
            len(self._scenarios),
            path,
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def get_scenario(self, event_type: str) -> dict:
        """
        Return a deep copy of the raw shock vector for ``event_type``.

        Falls back to the 'Geopolitical' scenario if the event type has no
        dedicated entry — a conservative catch-all for unknown events.

        The returned dict includes:
            shocks: {rates_bps, credit_spreads_bps, equity_shock_pct,
                     fx_usd_chg_pct, commodity_chg_pct, vol_shock_bps}
            sector_equity_overrides: {sector: float}  (may be absent)
            description: str
        """
        if event_type in self._scenarios:
            return copy.deepcopy(self._scenarios[event_type])
        logger.warning(
            "No scenario defined for event_type '%s'; falling back to Geopolitical.",
            event_type,
        )
        return copy.deepcopy(self._scenarios.get("Geopolitical", {}))

    def available_scenarios(self) -> list[str]:
        """Return the list of scenario names loaded from YAML."""
        return list(self._scenarios.keys())

    def apply_impact_scaling(self, scenario: dict, impact: float) -> dict:
        """
        Scale all shock magnitudes by a non-linear multiplier derived from impact.

        Mutates a copy of ``scenario`` (caller's copy is unaffected).

        Scaling formula:
            raw_scale  = clip((impact - 5) / 5, 0, 1)
            multiplier = raw_scale² + 0.1       (floor 0.1, cap 1.1)

        The floor of 0.1 ensures a minimum shock is always applied — this
        prevents a trivially zero result for low-impact events that somehow
        cleared the trigger threshold.
        """
        impact = float(impact)
        raw_scale = max(0.0, min(1.0, (impact - 5.0) / 5.0))
        multiplier = raw_scale ** 2 + 0.1
        multiplier = min(multiplier, 1.1)  # slight super-linear cap at impact=10

        scaled = copy.deepcopy(scenario)
        shocks = scaled.get("shocks", {})

        for key in self._MAGNITUDE_KEYS:
            if key in shocks and shocks[key] != 0:
                shocks[key] = shocks[key] * multiplier

        # Scale sector equity overrides by the same multiplier
        sec_overrides = scaled.get("sector_equity_overrides", {})
        for sec in list(sec_overrides.keys()):
            sec_overrides[sec] = sec_overrides[sec] * multiplier

        scaled["shocks"] = shocks
        scaled["sector_equity_overrides"] = sec_overrides
        scaled["_impact_multiplier"] = round(multiplier, 4)
        return scaled

    def apply_sentiment_sign(
        self,
        scenario: dict,
        sentiment: float,
        event_type: str,
    ) -> dict:
        """
        For sentiment-driven event types, override the sign of equity_shock_pct.

        For Earnings and Product events, the base scenario stores a conservative
        negative equity shock.  If the signal's sentiment is positive, the shock
        is flipped to reflect an upside surprise.

        Parameters
        ──────────
        scenario   : scenario dict (will be deep-copied before modification)
        sentiment  : signal sentiment in [-1, +1]
        event_type : signal event_type string

        Returns
        ───────
        Updated scenario dict (deep copy; caller's original is unchanged).
        """
        if event_type not in self._SENTIMENT_DRIVEN_EVENTS:
            return scenario

        updated = copy.deepcopy(scenario)
        shocks = updated.get("shocks", {})

        if "equity_shock_pct" in shocks:
            base_magnitude = abs(shocks["equity_shock_pct"])
            # Positive sentiment → positive equity shock; negative → negative
            sign = 1.0 if sentiment > 0 else -1.0
            shocks["equity_shock_pct"] = sign * base_magnitude

        updated["shocks"] = shocks
        return updated

    def get_pd_lgd_multipliers(
        self,
        event_type: str,
        impact: float,
    ) -> tuple[float, float]:
        """
        Return (pd_multiplier, lgd_multiplier) for loan stress.

        Anchors:
            Credit event,  impact 9  → PD × 2.5,  LGD × 1.30
            Geopolitical,  impact 8  → PD × 1.8,  LGD × 1.15
            Other events,  impact 7  → PD × 1.3,  LGD × 1.05

        For impacts below 5: multipliers = 1.0 (no stress).
        Interpolation is linear between the anchor and 1.0 (no stress at impact=5).
        """
        impact = float(impact)

        if event_type == "Credit":
            # Anchor: impact=9 → (2.5, 1.30); impact≤5 → (1.0, 1.0)
            pd_mult = self._interpolate(impact, anchor_impact=9.0, anchor_val=2.5)
            lgd_mult = self._interpolate(impact, anchor_impact=9.0, anchor_val=1.30)
        elif event_type == "Geopolitical":
            pd_mult = self._interpolate(impact, anchor_impact=8.0, anchor_val=1.8)
            lgd_mult = self._interpolate(impact, anchor_impact=8.0, anchor_val=1.15)
        else:
            # Generic credit stress for all other event types
            pd_mult = self._interpolate(impact, anchor_impact=7.0, anchor_val=1.3)
            lgd_mult = self._interpolate(impact, anchor_impact=7.0, anchor_val=1.05)

        return round(pd_mult, 4), round(lgd_mult, 4)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _interpolate(
        impact: float,
        anchor_impact: float,
        anchor_val: float,
        base_impact: float = 5.0,
        base_val: float = 1.0,
    ) -> float:
        """
        Linear interpolation between (base_impact, base_val) and (anchor_impact, anchor_val).

        For impacts above anchor_impact, the value is linearly extrapolated
        (capped so PD never exceeds 1.0 at the call site).
        For impacts at or below base_impact, returns base_val (no stress).
        """
        if impact <= base_impact:
            return base_val
        t = (impact - base_impact) / max(anchor_impact - base_impact, 1e-9)
        return base_val + t * (anchor_val - base_val)

    def flatten_shocks(self, scenario: dict) -> dict:
        """
        Flatten a scenario dict into a single-level shocks dict suitable for
        PortfolioValuator.compute_total_pnl.

        Merges 'shocks' and 'sector_equity_overrides' into one flat dict with
        a 'sector_equity_overrides' key.
        """
        flat = dict(scenario.get("shocks", {}))
        flat["sector_equity_overrides"] = scenario.get("sector_equity_overrides", {})
        return flat
