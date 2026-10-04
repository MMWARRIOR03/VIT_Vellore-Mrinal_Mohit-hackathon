"""
Monte Carlo stress engine for Module B.

Design overview
───────────────
The engine simulates n_paths joint risk-factor shocks using a Student-t(df=5)
distribution (fat tails) correlated via a Cholesky factorisation of the
5×5 factor covariance matrix.  Instead of calling compute_total_pnl once per
path (too slow at 10k paths), we pre-compute a dollar sensitivity for each
factor via a perturbation approach and then evaluate the linear factor model
for all paths in vectorised numpy operations.

Factor ordering (index 0–4 throughout):
    0: equity
    1: rates
    2: credit_spread
    3: fx
    4: commodity

Speed guarantee: 10k paths in < 3 s on a modern laptop.

Monte Carlo P&L sign convention
────────────────────────────────
``losses`` stores positive numbers for losses (positive loss = bad).
A path where the portfolio gains value appears as a negative entry in the
losses array, which shows up as a negative ``best_outcome``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from module_b.portfolio import SyntheticPortfolio
from module_b.valuation import PortfolioValuator

logger = logging.getLogger(__name__)

# ── Factor universe ────────────────────────────────────────────────────────────

FACTOR_NAMES: list[str] = ["equity", "rates", "credit_spread", "fx", "commodity"]

# Annualised factor volatilities (decimal)
_FACTOR_VOLS: np.ndarray = np.array([
    0.20,   # equity
    0.015,  # rates (decimal move, not bps)
    0.25,   # credit_spread
    0.08,   # fx
    0.22,   # commodity
], dtype=np.float64)

# Factor index lookup for quick access
_F_IDX: dict[str, int] = {name: i for i, name in enumerate(FACTOR_NAMES)}


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    """
    Output of a single Monte Carlo stress run.

    All monetary values are in USD.  ``losses`` is a list of per-path losses
    (positive = loss, negative = gain) stored as a plain Python list for JSON
    serialisability.

    Attributes
    ──────────
    n_paths         : number of simulation paths
    pre_nav         : portfolio NAV before any shock (USD)
    losses          : per-path loss in USD (positive = portfolio lost money)
    var_95          : 95th-percentile loss (Value at Risk)
    var_99          : 99th-percentile loss
    es_95           : Expected Shortfall beyond 95th percentile (CVaR)
    es_99           : Expected Shortfall beyond 99th percentile
    expected_loss   : mean loss across all paths
    worst_loss      : maximum loss (worst-case path)
    best_outcome    : minimum loss (may be negative, i.e. a gain)
    """

    n_paths: int
    pre_nav: float
    losses: list[float]  # positive = loss (JSON-serialisable)
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    expected_loss: float
    worst_loss: float
    best_outcome: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "n_paths": self.n_paths,
            "pre_nav": self.pre_nav,
            # losses array omitted from summary dict — use separately if needed
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "es_95": round(self.es_95, 2),
            "es_99": round(self.es_99, 2),
            "expected_loss": round(self.expected_loss, 2),
            "worst_loss": round(self.worst_loss, 2),
            "best_outcome": round(self.best_outcome, 2),
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class MonteCarloEngine:
    """
    Vectorised Monte Carlo stress engine over a 5-factor correlated model.

    Parameters
    ──────────
    n_paths : number of simulation paths (default 10_000)
    seed    : random seed for reproducibility (default 42)
    """

    def __init__(self, n_paths: int = 10_000, seed: int = 42) -> None:
        self.n_paths = n_paths
        self._rng = np.random.default_rng(seed)
        self._cov = self._build_covariance()
        self._L = self._cholesky(self._cov)

        logger.info(
            "MonteCarloEngine initialised: n_paths=%d  factors=%s",
            n_paths,
            FACTOR_NAMES,
        )

    # ── Covariance construction ───────────────────────────────────────────────

    def _build_covariance(self) -> np.ndarray:
        """
        Build the 5×5 factor covariance matrix.

        Starts from a correlation matrix with the prescribed off-diagonal
        entries, then scales to a covariance matrix via
        Cov = diag(vols) @ Corr @ diag(vols).
        """
        # ── Correlation matrix (symmetric) ────────────────────────────────────
        # Index order: equity(0), rates(1), credit_spread(2), fx(3), commodity(4)
        corr = np.eye(5, dtype=np.float64)

        # equity vs others
        corr[0, 1] = corr[1, 0] = -0.25   # equity-rates
        corr[0, 2] = corr[2, 0] = -0.70   # equity-credit_spread
        corr[0, 3] = corr[3, 0] = -0.15   # equity-fx
        corr[0, 4] = corr[4, 0] = +0.30   # equity-commodity

        # rates vs others
        corr[1, 2] = corr[2, 1] = +0.20   # rates-credit_spread
        corr[1, 3] = corr[3, 1] = +0.35   # rates-fx
        corr[1, 4] = corr[4, 1] = -0.10   # rates-commodity

        # credit_spread vs others
        corr[2, 3] = corr[3, 2] = -0.20   # credit_spread-fx
        corr[2, 4] = corr[4, 2] = -0.30   # credit_spread-commodity

        # fx vs others
        corr[3, 4] = corr[4, 3] = -0.40   # fx-commodity

        # Covariance = diag(vols) @ corr @ diag(vols)
        D = np.diag(_FACTOR_VOLS)
        cov = D @ corr @ D
        return cov

    def _cholesky(self, cov: np.ndarray) -> np.ndarray:
        """
        Attempt a Cholesky decomposition.  If the matrix is not positive
        definite (can happen with extreme correlation inputs), regularise by
        adding a small diagonal perturbation and retry.
        """
        jitter = 0.0
        for _ in range(10):
            try:
                L = np.linalg.cholesky(cov + jitter * np.eye(len(cov)))
                return L
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10, 1e-6)
        raise RuntimeError(
            "Could not obtain a Cholesky decomposition for the covariance matrix "
            "after 10 regularisation attempts."
        )

    # ── Path sampling helpers ─────────────────────────────────────────────────

    def _sample_impacts(
        self,
        impact: float,
        impact_range: tuple[float, float],
    ) -> np.ndarray:
        """
        Draw per-path impact samples from Uniform(impact_range[0], impact_range[1]).

        Returns shape (n_paths,).  This propagates NLP model uncertainty about
        the event impact into each simulation path.
        """
        lo, hi = float(impact_range[0]), float(impact_range[1])
        # Guard: ensure lo <= hi (degenerate range is fine → constant)
        if lo >= hi:
            return np.full(self.n_paths, impact, dtype=np.float64)
        return self._rng.uniform(lo, hi, size=self.n_paths)

    # ── Factor sensitivity (one-time perturbation) ────────────────────────────

    def _compute_factor_sensitivities(
        self,
        portfolio: SyntheticPortfolio,
        base_shocks: dict,
        valuator: PortfolioValuator,
    ) -> dict[str, float]:
        """
        Estimate the portfolio's dollar sensitivity to each risk factor by
        perturbing the base shock for that factor by 1 % of its magnitude
        and computing the resulting P&L change.

        The perturbation size is chosen to be small enough to stay in the
        linear regime but large enough to avoid floating-point noise.

        Returns
        ───────
        dict mapping factor name → dollar P&L per unit of base shock
        (i.e. if base_shock doubles, so does the P&L contribution).
        """
        pre_nav = portfolio.compute_nav()
        sensitivities: dict[str, float] = {}

        # Mapping: factor name → (shock_key, perturbation_multiplier)
        # We perturb each factor individually from a zero-shock baseline
        # so sensitivities are orthogonal.
        factor_shock_keys: dict[str, str] = {
            "equity":        "equity_shock_pct",
            "rates":         "rates_bps",
            "credit_spread": "credit_spreads_bps",
            "fx":            "fx_usd_chg_pct",
            "commodity":     "equity_shock_pct",   # commodity proxied via equity channel
        }

        # Reference P&L with zero shocks (should be ~0 but captures numerical baseline)
        zero_shocks: dict = {
            "rates_bps": 0.0,
            "credit_spreads_bps": 0.0,
            "equity_shock_pct": 0.0,
            "fx_usd_chg_pct": 0.0,
            "pd_multiplier": base_shocks.get("pd_multiplier", 1.0),
            "lgd_multiplier": base_shocks.get("lgd_multiplier", 1.0),
            "sector_equity_overrides": {},
        }
        pnl_zero = valuator.compute_total_pnl(portfolio, zero_shocks).delta_nav

        # Perturbation magnitudes that give a measurable but linear P&L
        _perturbations: dict[str, float] = {
            "equity":        -0.01,    # −1 % equity shock
            "rates":          50.0,    # +50 bps
            "credit_spread":  50.0,    # +50 bps
            "fx":              0.01,   # +1 % USD change
            "commodity":      -0.01,   # −1 % (proxied via equity channel)
        }

        for factor in FACTOR_NAMES:
            perturb_val = _perturbations[factor]
            shock_key = factor_shock_keys[factor]
            perturb_shocks = dict(zero_shocks)
            perturb_shocks[shock_key] = perturb_val

            pnl_perturb = valuator.compute_total_pnl(portfolio, perturb_shocks).delta_nav
            delta_pnl = pnl_perturb - pnl_zero

            # Sensitivity = $ P&L per unit of perturbation
            if abs(perturb_val) > 1e-12:
                sensitivities[factor] = delta_pnl / perturb_val
            else:
                sensitivities[factor] = 0.0

        logger.debug("Factor sensitivities: %s", sensitivities)
        return sensitivities

    # ── Metrics ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_metrics(losses: np.ndarray) -> dict[str, float]:
        """
        Compute VaR and Expected Shortfall from the loss distribution.

        losses : (n_paths,) array where positive = portfolio loss.
        """
        sorted_losses = np.sort(losses)
        n = len(sorted_losses)

        idx_95 = int(np.ceil(0.95 * n)) - 1
        idx_99 = int(np.ceil(0.99 * n)) - 1

        var_95 = float(sorted_losses[idx_95])
        var_99 = float(sorted_losses[idx_99])

        # ES = mean of losses strictly beyond the VaR threshold
        tail_95 = sorted_losses[idx_95:]
        tail_99 = sorted_losses[idx_99:]

        es_95 = float(np.mean(tail_95)) if len(tail_95) > 0 else var_95
        es_99 = float(np.mean(tail_99)) if len(tail_99) > 0 else var_99

        return {
            "var_95": var_95,
            "var_99": var_99,
            "es_95": es_95,
            "es_99": es_99,
            "expected_loss": float(np.mean(losses)),
            "worst_loss": float(np.max(losses)),
            "best_outcome": float(np.min(losses)),
        }

    # ── Main simulation ───────────────────────────────────────────────────────

    def run(
        self,
        portfolio: SyntheticPortfolio,
        base_shocks: dict,
        impact: float,
        impact_range: tuple[float, float],
        valuator: PortfolioValuator,
    ) -> MonteCarloResult:
        """
        Run the vectorised Monte Carlo simulation.

        Algorithm
        ─────────
        1. Compute factor sensitivities (one-time, 5 valuator calls).
        2. Draw Z ~ Student-t(df=5), shape (n_paths, 5) — fat tails.
        3. Correlate draws: corr_Z = Z @ L.T
        4. Sample per-path impact from Uniform(impact_range) → (n_paths,)
        5. Compute shock_scale = impact_sampled / 10.0 (normalise to ~[0.1, 1.0])
        6. For each factor, per-path shock = base_component + noise_component
           where base_component = base_shock_mag[f] * shock_scale
                 noise_component = corr_Z[:, f] * vol_scale_f
        7. Portfolio loss per path = sum_f sensitivity[f] * path_shock[f]
           Loss sign: positive = portfolio lost money (shock in loss-inducing direction)

        Parameters
        ──────────
        portfolio    : SyntheticPortfolio
        base_shocks  : flat shock dict from ScenarioEngine.flatten_shocks + PD/LGD
        impact       : point-estimate impact score (1–10)
        impact_range : (lo, hi) credible interval for impact
        valuator     : PortfolioValuator
        """
        pre_nav = portfolio.compute_nav()
        mean_impact = float(impact)

        # ── Step 1: Factor sensitivities ──────────────────────────────────────
        sensitivities = self._compute_factor_sensitivities(portfolio, base_shocks, valuator)

        # ── Step 2: Draw Student-t shocks ─────────────────────────────────────
        # Student-t(df=5) has fatter tails than Gaussian; consistent with
        # empirical risk factor distributions.
        # scipy is an optional dep; fall back to Gaussian if unavailable.
        try:
            from scipy.stats import t as student_t  # type: ignore[import]
            Z = student_t.rvs(df=5, size=(self.n_paths, 5), random_state=self._rng)
        except ImportError:  # pragma: no cover
            logger.warning(
                "scipy not available; falling back to standard normal draws. "
                "Fat-tail properties will not be captured."
            )
            Z = self._rng.standard_normal(size=(self.n_paths, 5))

        Z = np.asarray(Z, dtype=np.float64)

        # ── Step 3: Correlate via Cholesky ────────────────────────────────────
        # corr_Z[i, f] is the correlated shock for path i, factor f
        corr_Z = Z @ self._L.T   # shape (n_paths, 5)

        # ── Step 4: Per-path impact sampling ──────────────────────────────────
        impact_sampled = self._sample_impacts(impact, impact_range)   # (n_paths,)

        # ── Step 5: Shock scale per path ──────────────────────────────────────
        shock_scale = impact_sampled / 10.0   # normalise impact [1,10] → [0.1, 1.0]

        # Stressed vol scale: proportional to mean impact severity
        vol_scale_global = 0.3 * (mean_impact / 10.0)

        # ── Step 6: Build base shock magnitudes for each factor ───────────────
        # These are the magnitudes of base shocks that each factor sensitivity
        # was calibrated against (see _perturbations in _compute_factor_sensitivities).
        # We need to convert base_shocks into factor-native units.
        base_shock_mags: dict[str, float] = {
            "equity":        float(base_shocks.get("equity_shock_pct", 0.0)),
            "rates":         float(base_shocks.get("rates_bps", 0.0)),
            "credit_spread": float(base_shocks.get("credit_spreads_bps", 0.0)),
            "fx":            float(base_shocks.get("fx_usd_chg_pct", 0.0)),
            # Commodity proxied via equity channel; use commodity_chg_pct if present
            "commodity":     float(base_shocks.get("commodity_chg_pct",
                                   base_shocks.get("equity_shock_pct", 0.0))),
        }

        # ── Step 7: Vectorised loss computation ───────────────────────────────
        # path_shock[i, f] = base_shock[f] * shock_scale[i] + corr_Z[i, f] * vol_scale[f]
        # Each factor column of corr_Z is already in correlation-unit space;
        # we scale it by the factor's vol (already embedded in L via the covariance)
        # and the stressed vol multiplier.

        # Build (n_paths, 5) array of per-path shocks in factor-native units
        path_shocks = np.empty((self.n_paths, 5), dtype=np.float64)
        for f_idx, factor in enumerate(FACTOR_NAMES):
            base_mag = base_shock_mags[factor]
            # Base component: scenario shock scaled by per-path impact weight
            base_component = base_mag * shock_scale
            # Noise component: correlated random shock scaled by stressed vol
            # corr_Z[:, f_idx] is in vol-scaled units already (Cholesky of covariance);
            # we additionally scale by the global stressed-vol multiplier
            noise_component = corr_Z[:, f_idx] * vol_scale_global
            path_shocks[:, f_idx] = base_component + noise_component

        # Dollar loss per path = sum_f sensitivity[f] * shock[f]
        # sensitivity[f] = $ P&L per unit of shock in factor-native units
        # Negate because sensitivity is expressed as positive-when-loss-inducing
        sens_vec = np.array(
            [sensitivities[f] for f in FACTOR_NAMES], dtype=np.float64
        )
        # portfolio_pnl[i] = sens_vec . path_shocks[i]  (dot product)
        portfolio_pnl = path_shocks @ sens_vec   # shape (n_paths,)

        # Loss = negative P&L (positive loss = portfolio lost money)
        losses = -portfolio_pnl

        # ── Step 8: Metrics ───────────────────────────────────────────────────
        metrics = self._compute_metrics(losses)

        logger.info(
            "MC run complete: n_paths=%d  E[loss]=$%,.0f  VaR95=$%,.0f  ES95=$%,.0f",
            self.n_paths,
            metrics["expected_loss"],
            metrics["var_95"],
            metrics["es_95"],
        )

        return MonteCarloResult(
            n_paths=self.n_paths,
            pre_nav=pre_nav,
            losses=losses.tolist(),
            var_95=metrics["var_95"],
            var_99=metrics["var_99"],
            es_95=metrics["es_95"],
            es_99=metrics["es_99"],
            expected_loss=metrics["expected_loss"],
            worst_loss=metrics["worst_loss"],
            best_outcome=metrics["best_outcome"],
        )
