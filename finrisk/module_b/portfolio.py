"""
Synthetic wholesale banking portfolio generator.

Design decisions
────────────────
- All positions are generated deterministically from a fixed seed (42) so
  every test run, demo, and stress scenario operates on the identical book.
  Pass a different seed to SyntheticPortfolio(seed=N) for sensitivity testing.
- Dataclasses are used instead of Pydantic for speed and to keep the hot
  valuation path free of validator overhead.  Callers should treat fields as
  immutable after construction.
- The portfolio is deliberately representative of a mid-size institutional
  book: IG-tilted bonds, leveraged loans, long-only equities, FX hedges,
  and duration-extension swaps.

Position counts
───────────────
  120 corporate bonds    (10 sectors × 12 bonds)
   80 loans              (4 rating buckets × 20 loans)
   60 equities           (20 tickers × 3 lots each)
   25 FX forwards        (5 pairs × 5 forward contracts)
   15 interest-rate swaps(3 tenor clusters × 5 swaps)
  ─── ───
  300 total

NAV target: ~$50 M  (bonds ≈ $21 M, loans ≈ $11 M, equities ≈ $9 M,
                      FX ≈ $5 M, swaps DV01-equiv ≈ $4 M)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

# ── Sector universe ────────────────────────────────────────────────────────────

SECTORS: list[str] = [
    "Technology",
    "Financials",
    "Health Care",
    "Energy",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Utilities",
    "Materials",
    "Communication Services",
]

# ── FX pair universe ──────────────────────────────────────────────────────────

FX_PAIRS: list[str] = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]

# Approximate spot rates (fictional but plausible, fixed)
_SPOT_RATES: dict[str, float] = {
    "EURUSD": 1.0850,
    "GBPUSD": 1.2650,
    "USDJPY": 148.50,
    "USDCHF": 0.9010,
    "AUDUSD": 0.6520,
}

# ── Equity ticker universe (20 names across 7 sectors) ────────────────────────

EQUITY_TICKERS: list[tuple[str, str]] = [
    # (ticker, sector)
    ("AAPL",  "Technology"),
    ("MSFT",  "Technology"),
    ("NVDA",  "Technology"),
    ("GOOGL", "Communication Services"),
    ("META",  "Communication Services"),
    ("JPM",   "Financials"),
    ("BAC",   "Financials"),
    ("GS",    "Financials"),
    ("JNJ",   "Health Care"),
    ("UNH",   "Health Care"),
    ("PFE",   "Health Care"),
    ("XOM",   "Energy"),
    ("CVX",   "Energy"),
    ("CAT",   "Industrials"),
    ("BA",    "Industrials"),
    ("WMT",   "Consumer Staples"),
    ("PG",    "Consumer Staples"),
    ("HD",    "Consumer Discretionary"),
    ("AMZN",  "Consumer Discretionary"),
    ("TSLA",  "Consumer Discretionary"),
]


# ── Position dataclasses ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Bond:
    """
    USD-denominated corporate bond.

    ``market_value`` is the current clean price expressed in USD.
    ``credit_spread_bps`` is the option-adjusted spread over the benchmark.
    ``convexity`` is expressed in years² (raw, not divided by 10000).
    """

    position_id: str
    sector: str
    issuer: str
    notional: float            # face value, USD
    coupon: float              # annual coupon rate, e.g. 0.045 = 4.5 %
    duration: float            # modified duration, years
    convexity: float           # convexity, years²
    credit_spread_bps: float   # OAS over risk-free, bps
    market_value: float        # current USD market value


@dataclass(frozen=True)
class Loan:
    """
    Leveraged / syndicated loan position.

    Expected loss is pre-calculated as PD × LGD × EAD and stored for speed.
    The stress engine recalculates EL under stressed PD and LGD.
    """

    position_id: str
    counterparty: str
    sector: str
    ead: float                 # exposure at default, USD
    pd: float                  # annual probability of default, e.g. 0.025 = 2.5 %
    lgd: float                 # loss given default, e.g. 0.40 = 40 %
    rating: str                # A | BBB | BB | B
    expected_loss: float       # = pd × lgd × ead, USD


@dataclass(frozen=True)
class Equity:
    """Long equity position."""

    position_id: str
    ticker: str
    sector: str
    shares: float              # number of shares held
    price: float               # current price, USD per share
    market_value: float        # = shares × price
    beta: float                # dollar beta vs S&P 500


@dataclass(frozen=True)
class FXForward:
    """
    Short-dated FX forward used for currency hedging.

    ``notional_usd`` is the USD equivalent of the forward notional.
    The forward PnL approximation uses a simple spot-delta.
    """

    position_id: str
    currency_pair: str         # e.g. 'EURUSD'
    notional_usd: float        # USD-equivalent notional
    spot_rate: float           # current spot mid
    forward_rate: float        # agreed forward rate


@dataclass(frozen=True)
class InterestRateSwap:
    """
    Plain-vanilla pay-fixed / receive-floating interest rate swap.

    DV01 is the dollar value of a 1-bps shift in the yield curve
    (positive = paying fixed, so rates up → swap loses value).
    """

    position_id: str
    notional: float            # notional principal, USD
    dv01: float                # dollar sensitivity to +1bp, USD
    term_years: int            # maturity in years
    fixed_rate: float          # pay-fixed rate, e.g. 0.0375 = 3.75 %
    sector: str                # counterparty sector tag


# ── Portfolio class ────────────────────────────────────────────────────────────

class SyntheticPortfolio:
    """
    Generates and holds the full ~300-position synthetic wholesale portfolio.

    All positions are created once at construction time using a deterministic
    random number generator seeded by ``seed``.  The same seed always produces
    identical positions, which is a deliberate design choice for reproducible
    stress-test results.

    Parameters
    ──────────
    seed : int
        Random seed.  Default 42.

    Attributes
    ──────────
    bonds    : list[Bond]
    loans    : list[Loan]
    equities : list[Equity]
    fx_forwards : list[FXForward]
    swaps    : list[InterestRateSwap]
    """

    # Rating → (PD lower, PD upper) annual default probabilities
    _RATING_PD: ClassVar[dict[str, tuple[float, float]]] = {
        "A":   (0.005, 0.010),
        "BBB": (0.010, 0.020),
        "BB":  (0.020, 0.035),
        "B":   (0.035, 0.050),
    }

    # Sector → credit spread range for bonds (bps)
    _SECTOR_SPREAD: ClassVar[dict[str, tuple[float, float]]] = {
        "Technology":             (50,  200),
        "Financials":             (60,  220),
        "Health Care":            (55,  190),
        "Energy":                 (80,  300),
        "Industrials":            (70,  250),
        "Consumer Discretionary": (90,  320),
        "Consumer Staples":       (55,  180),
        "Utilities":              (50,  160),
        "Materials":              (75,  260),
        "Communication Services": (65,  210),
    }

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self.bonds: list[Bond] = []
        self.loans: list[Loan] = []
        self.equities: list[Equity] = []
        self.fx_forwards: list[FXForward] = []
        self.swaps: list[InterestRateSwap] = []

        self._generate()

    # ── Generation helpers ────────────────────────────────────────────────────

    def _uniform(self, lo: float, hi: float) -> float:
        return float(self._rng.uniform(lo, hi))

    def _choice(self, seq: list) -> object:
        idx = int(self._rng.integers(0, len(seq)))
        return seq[idx]

    def _issuer_name(self, sector: str, idx: int) -> str:
        prefix = {
            "Technology":             "TechCorp",
            "Financials":             "FinBank",
            "Health Care":            "HealthGen",
            "Energy":                 "EnergyCo",
            "Industrials":            "IndustMfg",
            "Consumer Discretionary": "RetailCo",
            "Consumer Staples":       "StaplesBrand",
            "Utilities":              "PowerGrid",
            "Materials":              "MaterialsInc",
            "Communication Services": "MediaGroup",
        }.get(sector, "Corp")
        return f"{prefix}-{idx:03d}"

    # ── Bond generation ───────────────────────────────────────────────────────

    def _generate_bonds(self, n_per_sector: int = 12) -> None:
        """
        Generate 10 × 12 = 120 corporate bonds, 12 per sector.

        Duration is drawn from [2, 10] years; convexity is approximated as
        duration² / 100 + small noise (a standard market approximation for
        bullet bonds).  Market value is set so the total bond book ≈ $21 M.
        """
        target_total = 21_000_000.0
        # Average market value per bond
        avg_mv = target_total / (len(SECTORS) * n_per_sector)

        for sector in SECTORS:
            spread_lo, spread_hi = self._SECTOR_SPREAD[sector]
            for i in range(n_per_sector):
                pos_id = f"BOND-{sector[:3].upper()}-{i+1:03d}"
                issuer = self._issuer_name(sector, i + 1)
                # Market value: log-normally distributed around avg_mv
                mv = float(np.clip(
                    self._rng.lognormal(math.log(avg_mv), 0.5),
                    100_000, 5_000_000,
                ))
                # Notional = market value / price (price drawn near par)
                price_pct = self._uniform(0.85, 1.05)
                notional = mv / price_pct

                duration = self._uniform(2.0, 10.0)
                # Convexity ≈ D² / 100 for a par bond (first-order approximation)
                convexity = (duration ** 2) / 100.0 + self._uniform(0.0, 0.5)
                coupon = self._uniform(0.020, 0.065)
                credit_spread = self._uniform(spread_lo, spread_hi)

                self.bonds.append(Bond(
                    position_id=pos_id,
                    sector=sector,
                    issuer=issuer,
                    notional=round(notional, 2),
                    coupon=round(coupon, 4),
                    duration=round(duration, 3),
                    convexity=round(convexity, 4),
                    credit_spread_bps=round(credit_spread, 1),
                    market_value=round(mv, 2),
                ))

    # ── Loan generation ───────────────────────────────────────────────────────

    def _generate_loans(self, n_total: int = 80) -> None:
        """
        Generate 80 leveraged loans spread across 4 rating buckets.

        Rating distribution matches configs/portfolio.yaml:
          A: 10 loans, BBB: 20 loans, BB: 30 loans, B: 20 loans

        EAD per loan is drawn from $500k–$5M (realistic wholesale banking sizes).
        Total loan book target ≈ $11M; individual EAD sizes are those of a
        syndicated loan participation / revolving credit facility slice.
        Target average EAD ≈ $137k per loan (11M / 80), achieved by drawing
        from $50k–$300k to reflect participation sizes in a $50M portfolio.
        """
        # Counts by rating to sum to 80
        rating_counts = {"A": 10, "BBB": 20, "BB": 30, "B": 20}
        sectors_for_loans = SECTORS[:7]  # 7 sectors for loans

        for rating, count in rating_counts.items():
            pd_lo, pd_hi = self._RATING_PD[rating]
            for i in range(count):
                pos_id = f"LOAN-{rating}-{i+1:03d}"
                sector = sectors_for_loans[i % len(sectors_for_loans)]
                counterparty = f"{sector[:3].upper()}-Borrower-{i+1:03d}"
                # EAD: $50k–$300k per loan to keep total book ≈ $11M
                # (participation sizes in a $50M institutional book)
                ead = round(self._uniform(50_000, 300_000), 2)
                pd_val = round(self._uniform(pd_lo, pd_hi), 5)
                lgd = round(self._uniform(0.35, 0.45), 4)
                el = round(pd_val * lgd * ead, 2)

                self.loans.append(Loan(
                    position_id=pos_id,
                    counterparty=counterparty,
                    sector=sector,
                    ead=ead,
                    pd=pd_val,
                    lgd=lgd,
                    rating=rating,
                    expected_loss=el,
                ))

    # ── Equity generation ─────────────────────────────────────────────────────

    def _generate_equities(self, lots_per_ticker: int = 3) -> None:
        """
        Generate 3 lots × 20 tickers = 60 equity positions.

        Each lot of the same ticker has a slightly different average cost
        (simulating accumulation over time) but a single current price used
        for all market value calculations.  Beta is drawn from [0.8, 1.5].
        """
        for ticker, sector in EQUITY_TICKERS:
            # Plausible price range per sector
            price_ranges: dict[str, tuple[float, float]] = {
                "Technology": (100, 500),
                "Communication Services": (150, 400),
                "Financials": (30, 200),
                "Health Care": (60, 350),
                "Energy": (50, 150),
                "Industrials": (80, 300),
                "Consumer Staples": (40, 200),
                "Consumer Discretionary": (80, 400),
            }
            lo, hi = price_ranges.get(sector, (50, 300))
            price = round(self._uniform(lo, hi), 2)
            beta = round(self._uniform(0.80, 1.50), 3)

            for lot in range(lots_per_ticker):
                pos_id = f"EQ-{ticker}-LOT{lot+1}"
                # Shares per lot: target each position ~$150k in market value
                target_mv = self._uniform(80_000, 250_000)
                shares = round(target_mv / price, 0)
                mv = round(shares * price, 2)

                self.equities.append(Equity(
                    position_id=pos_id,
                    ticker=ticker,
                    sector=sector,
                    shares=shares,
                    price=price,
                    market_value=mv,
                    beta=beta,
                ))

    # ── FX forward generation ─────────────────────────────────────────────────

    def _generate_fx_forwards(self, n_per_pair: int = 5) -> None:
        """
        Generate 5 pairs × 5 contracts = 25 FX forwards.

        The forward rate is derived from spot via a random forward points
        spread (±150 pips, i.e. 0.0150) to model realistic hedging hedges.
        """
        for pair in FX_PAIRS:
            spot = _SPOT_RATES[pair]
            for i in range(n_per_pair):
                pos_id = f"FX-{pair}-{i+1:02d}"
                notional_usd = round(self._uniform(100_000, 500_000), 2)
                # Small forward premium/discount (±150 pips)
                fwd_pts = self._uniform(-0.0150, 0.0150)
                forward_rate = round(spot + fwd_pts, 5)

                self.fx_forwards.append(FXForward(
                    position_id=pos_id,
                    currency_pair=pair,
                    notional_usd=notional_usd,
                    spot_rate=round(spot, 5),
                    forward_rate=forward_rate,
                ))

    # ── Swap generation ───────────────────────────────────────────────────────

    def _generate_swaps(self, n_total: int = 15) -> None:
        """
        Generate 15 pay-fixed / receive-SOFR interest rate swaps.

        Tenor clusters: 5 short (2-3y), 5 medium (5-7y), 5 long (10-30y).
        DV01 is sized proportionally to a $50M portfolio: total swap-book DV01
        ≈ $10k–$50k across all 15 swaps ($500–$5k per position).  This keeps
        the swap contribution to total portfolio P&L proportional to the
        other asset classes.
        """
        tenor_groups = [
            (2, 3,  200,   1_500),   # short: DV01 $200–$1.5k
            (5, 7,  500,   3_000),   # medium
            (10, 30, 1_000, 5_000),  # long
        ]
        group_size = n_total // len(tenor_groups)
        extra = n_total - group_size * len(tenor_groups)

        swap_idx = 0
        for gi, (t_lo, t_hi, dv01_lo, dv01_hi) in enumerate(tenor_groups):
            count = group_size + (1 if gi < extra else 0)
            for i in range(count):
                swap_idx += 1
                pos_id = f"IRS-{swap_idx:03d}"
                term = int(self._rng.integers(t_lo, t_hi + 1))
                notional = round(self._uniform(1_000_000, 50_000_000), 2)
                dv01 = round(self._uniform(dv01_lo, dv01_hi), 2)
                # Fixed rate: 3–6% depending on tenor
                fixed_rate = round(self._uniform(0.030 + term * 0.001, 0.030 + term * 0.003), 4)
                sector = str(self._choice(SECTORS[:5]))  # counterparty sector

                self.swaps.append(InterestRateSwap(
                    position_id=pos_id,
                    notional=notional,
                    dv01=dv01,
                    term_years=term,
                    fixed_rate=fixed_rate,
                    sector=sector,
                ))

    # ── Master generate ───────────────────────────────────────────────────────

    def _generate(self) -> None:
        self._generate_bonds()       # 120
        self._generate_loans()       # 80
        self._generate_equities()    # 60
        self._generate_fx_forwards() # 25
        self._generate_swaps()       # 15

    # ── Portfolio-level methods ───────────────────────────────────────────────

    def compute_nav(self) -> float:
        """
        Compute total portfolio NAV as the sum of all position market values.

        For loans, NAV is approximated as EAD (book value), since loans are
        held-to-maturity and marked at par unless stressed.
        For FX forwards and swaps, NAV contribution is the current mark-to-market
        value (unrealised gain/loss), approximated as zero at inception and
        equal to notional_usd for FX and DV01 × current_rate for swaps.
        In practice, we include FX notional at par and swap DV01-equivalent.
        """
        bond_nav = sum(b.market_value for b in self.bonds)
        loan_nav = sum(l.ead for l in self.loans)
        equity_nav = sum(e.market_value for e in self.equities)
        fx_nav = sum(f.notional_usd for f in self.fx_forwards)
        # Swap NAV ≈ 0 at inception; for sizing purposes approximate as
        # (notional × term × 0.001) which gives a small but non-zero contribution
        swap_nav = sum(s.notional * s.term_years * 0.001 for s in self.swaps)
        return bond_nav + loan_nav + equity_nav + fx_nav + swap_nav

    def position_count(self) -> dict[str, int]:
        """Return count of positions by asset class."""
        return {
            "bonds": len(self.bonds),
            "loans": len(self.loans),
            "equities": len(self.equities),
            "fx_forwards": len(self.fx_forwards),
            "swaps": len(self.swaps),
            "total": (
                len(self.bonds) + len(self.loans) + len(self.equities)
                + len(self.fx_forwards) + len(self.swaps)
            ),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten all positions into a single DataFrame with one row per position.

        Columns present for all rows: position_id, asset_class, sector,
        market_value.  Asset-class-specific columns are NaN where not applicable.
        """
        rows: list[dict] = []

        for b in self.bonds:
            rows.append({
                "position_id":      b.position_id,
                "asset_class":      "Bond",
                "sector":           b.sector,
                "issuer":           b.issuer,
                "notional":         b.notional,
                "coupon":           b.coupon,
                "duration":         b.duration,
                "convexity":        b.convexity,
                "credit_spread_bps": b.credit_spread_bps,
                "market_value":     b.market_value,
                "rating":           None,
                "pd":               None,
                "lgd":              None,
                "ead":              None,
                "expected_loss":    None,
                "ticker":           None,
                "shares":           None,
                "price":            None,
                "beta":             None,
                "currency_pair":    None,
                "spot_rate":        None,
                "forward_rate":     None,
                "dv01":             None,
                "term_years":       None,
                "fixed_rate":       None,
            })

        for l in self.loans:
            rows.append({
                "position_id":      l.position_id,
                "asset_class":      "Loan",
                "sector":           l.sector,
                "issuer":           l.counterparty,
                "notional":         l.ead,
                "coupon":           None,
                "duration":         None,
                "convexity":        None,
                "credit_spread_bps": None,
                "market_value":     l.ead,
                "rating":           l.rating,
                "pd":               l.pd,
                "lgd":              l.lgd,
                "ead":              l.ead,
                "expected_loss":    l.expected_loss,
                "ticker":           None,
                "shares":           None,
                "price":            None,
                "beta":             None,
                "currency_pair":    None,
                "spot_rate":        None,
                "forward_rate":     None,
                "dv01":             None,
                "term_years":       None,
                "fixed_rate":       None,
            })

        for e in self.equities:
            rows.append({
                "position_id":      e.position_id,
                "asset_class":      "Equity",
                "sector":           e.sector,
                "issuer":           e.ticker,
                "notional":         e.market_value,
                "coupon":           None,
                "duration":         None,
                "convexity":        None,
                "credit_spread_bps": None,
                "market_value":     e.market_value,
                "rating":           None,
                "pd":               None,
                "lgd":              None,
                "ead":              None,
                "expected_loss":    None,
                "ticker":           e.ticker,
                "shares":           e.shares,
                "price":            e.price,
                "beta":             e.beta,
                "currency_pair":    None,
                "spot_rate":        None,
                "forward_rate":     None,
                "dv01":             None,
                "term_years":       None,
                "fixed_rate":       None,
            })

        for f in self.fx_forwards:
            rows.append({
                "position_id":      f.position_id,
                "asset_class":      "FXForward",
                "sector":           "FX",
                "issuer":           f.currency_pair,
                "notional":         f.notional_usd,
                "coupon":           None,
                "duration":         None,
                "convexity":        None,
                "credit_spread_bps": None,
                "market_value":     f.notional_usd,
                "rating":           None,
                "pd":               None,
                "lgd":              None,
                "ead":              None,
                "expected_loss":    None,
                "ticker":           None,
                "shares":           None,
                "price":            f.spot_rate,
                "beta":             None,
                "currency_pair":    f.currency_pair,
                "spot_rate":        f.spot_rate,
                "forward_rate":     f.forward_rate,
                "dv01":             None,
                "term_years":       None,
                "fixed_rate":       None,
            })

        for s in self.swaps:
            rows.append({
                "position_id":      s.position_id,
                "asset_class":      "IRSwap",
                "sector":           s.sector,
                "issuer":           f"Swap-{s.position_id}",
                "notional":         s.notional,
                "coupon":           None,
                "duration":         None,
                "convexity":        None,
                "credit_spread_bps": None,
                "market_value":     s.notional * s.term_years * 0.001,
                "rating":           None,
                "pd":               None,
                "lgd":              None,
                "ead":              None,
                "expected_loss":    None,
                "ticker":           None,
                "shares":           None,
                "price":            s.fixed_rate,
                "beta":             None,
                "currency_pair":    None,
                "spot_rate":        None,
                "forward_rate":     None,
                "dv01":             s.dv01,
                "term_years":       s.term_years,
                "fixed_rate":       s.fixed_rate,
            })

        return pd.DataFrame(rows)

    def nav_by_asset_class(self) -> dict[str, float]:
        """Return NAV broken down by asset class."""
        return {
            "bonds":       sum(b.market_value for b in self.bonds),
            "loans":       sum(l.ead for l in self.loans),
            "equities":    sum(e.market_value for e in self.equities),
            "fx_forwards": sum(f.notional_usd for f in self.fx_forwards),
            "swaps":       sum(s.notional * s.term_years * 0.001 for s in self.swaps),
        }

    def nav_by_sector(self) -> dict[str, float]:
        """Return NAV broken down by sector (FX forwards not included in sectors)."""
        result: dict[str, float] = {}
        for b in self.bonds:
            result[b.sector] = result.get(b.sector, 0.0) + b.market_value
        for l in self.loans:
            result[l.sector] = result.get(l.sector, 0.0) + l.ead
        for e in self.equities:
            result[e.sector] = result.get(e.sector, 0.0) + e.market_value
        for s in self.swaps:
            result[s.sector] = result.get(s.sector, 0.0) + s.notional * s.term_years * 0.001
        return result
