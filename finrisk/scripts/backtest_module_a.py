"""
backtest_module_a.py — Standalone backtest for Module A (IndexRebalancer).

Simulates ~3 months of signal history, feeds each signal sequentially to
IndexRebalancer.on_signal(), then computes portfolio performance metrics.

Strategies compared:
  - Tilted (signal-driven):    IndexRebalancer output
  - Equal-weight:              1/20 to every ticker, never rebalanced
  - Cap-weight (proxy):        static weights proportional to mock market cap

Run from the finrisk/ root:
    python scripts/backtest_module_a.py

Output: data/results/module_a_backtest.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("backtest_module_a")

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / "module_a_backtest.json"

N_SIGNALS = 200
INITIAL_NAV = 1_000_000.0  # $1M portfolio

# ── Ticker universe (20 stocks) ──────────────────────────────────────────────
INDEX_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "BAC", "GS",
    "V", "JNJ", "UNH", "PFE", "XOM",
    "CVX", "CAT", "BA", "WMT", "HD",
]

# Mock market-cap weights (technology and mega-caps heavier)
_CAP_WEIGHTS_RAW: dict[str, float] = {
    "AAPL": 10.0, "MSFT": 9.5, "NVDA": 7.0, "GOOGL": 6.5, "AMZN": 6.0,
    "META": 5.5,  "TSLA": 4.5, "JPM": 5.0, "BAC": 3.5, "GS": 3.0,
    "V": 4.0,     "JNJ": 4.0, "UNH": 4.5, "PFE": 2.5, "XOM": 3.5,
    "CVX": 3.0,   "CAT": 2.5, "BA": 2.5, "WMT": 3.0, "HD": 3.0,
}
_cap_total = sum(_CAP_WEIGHTS_RAW.values())
CAP_WEIGHTS: dict[str, float] = {t: v / _cap_total for t, v in _CAP_WEIGHTS_RAW.items()}
EQ_WEIGHTS: dict[str, float] = {t: 1.0 / len(INDEX_TICKERS) for t in INDEX_TICKERS}


# ─────────────────────────────────────────────────────────────────────────────
# Signal generation helpers
# ─────────────────────────────────────────────────────────────────────────────

EVENT_TYPES: list[str] = [
    "Earnings", "M&A", "Regulatory", "Macro", "Geopolitical",
    "Credit", "Commodity", "Cyber", "Leadership", "Product",
]

_EVENT_HALFLIFE: dict[str, float] = {
    "Earnings": 6.0, "M&A": 48.0, "Regulatory": 72.0, "Macro": 24.0,
    "Geopolitical": 36.0, "Credit": 18.0, "Commodity": 12.0,
    "Cyber": 8.0, "Leadership": 24.0, "Product": 10.0,
}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _make_event_probs(primary: str) -> dict[str, float]:
    """Build a softmax-like probability distribution with primary winning."""
    primary_prob = round(random.uniform(0.50, 0.80), 3)
    others = [e for e in EVENT_TYPES if e != primary]
    n_others = random.randint(2, 4)
    other_types = random.sample(others, n_others)
    shares = [random.random() for _ in other_types]
    total_shares = sum(shares)
    remainder = 1.0 - primary_prob
    probs: dict[str, float] = {}
    other_sum = 0.0
    for i, (t, s) in enumerate(zip(other_types, shares)):
        if i < len(other_types) - 1:
            v = round((s / total_shares) * remainder, 4)
            probs[t] = v
            other_sum += v
        else:
            # Last entry absorbs rounding residual to guarantee sum ≤ 1.0
            probs[t] = round(remainder - other_sum, 4)
    probs[primary] = primary_prob
    # Final safety clamp: ensure sum ≤ 1.0
    total = sum(probs.values())
    if total > 1.0:
        probs[primary] = round(probs[primary] - (total - 1.0) - 1e-5, 4)
    return probs


def generate_signals(n: int, rng: random.Random) -> list:
    """Generate n synthetic RiskSignals spanning ~3 months of history."""
    from app.schema.risk_signal import DiffusionInfo, RiskSignal, TickerRef

    start_time = datetime.now(tz=timezone.utc) - timedelta(days=90)
    interval_seconds = (90 * 24 * 3600) / n

    signals = []
    recent_clusters: list[str] = []

    for i in range(n):
        ts_published = start_time + timedelta(seconds=i * interval_seconds)
        ts_processed = ts_published + timedelta(seconds=rng.uniform(10, 120))

        event_type = rng.choice(EVENT_TYPES)
        sentiment = _clamp(rng.gauss(0.0, 0.35), -1.0, 1.0)
        sentiment_conf = _clamp(rng.gauss(0.72, 0.12), 0.35, 0.99)

        raw_impact = rng.betavariate(2.5, 2.5) * 9.0 + 1.0
        impact = round(_clamp(raw_impact, 1.0, 10.0), 2)
        half_width = rng.uniform(0.3, 1.5)
        impact_low = round(_clamp(impact - half_width, 1.0, impact), 2)
        impact_high = round(_clamp(impact + half_width, impact, 10.0), 2)

        confidence = _clamp(rng.gauss(0.70, 0.15), 0.20, 0.99)
        novelty = _clamp(rng.betavariate(2.0, 3.0), 0.01, 1.0)

        sources_count = max(1, int(rng.lognormvariate(1.5, 0.8)))
        velocity = round(max(0.0, rng.expovariate(1 / 3.0)), 2)
        diffusion_score = _clamp(
            (min(sources_count, 50) / 50) * 0.6 + (min(velocity, 10) / 10) * 0.4,
            0.0, 1.0,
        )

        n_tickers = 1 if rng.random() < 0.80 else rng.randint(2, 3)
        chosen_tickers = rng.sample(INDEX_TICKERS, min(n_tickers, len(INDEX_TICKERS)))
        tickers = [
            TickerRef(
                ticker=t,
                exchange="NASDAQ" if t in {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"} else "NYSE",
                relevance=round(_clamp(1.0 - i * 0.25, 0.1, 1.0), 3),
            )
            for i, t in enumerate(chosen_tickers)
        ]

        # Cluster management
        import uuid
        if recent_clusters and rng.random() < 0.30:
            cluster_id = rng.choice(recent_clusters)
        else:
            cluster_id = f"cluster-{uuid.uuid4().hex[:12]}"
            recent_clusters.append(cluster_id)
            if len(recent_clusters) > 20:
                recent_clusters.pop(0)

        event_probs = _make_event_probs(event_type)
        source = rng.choices(["news", "social", "filing"], weights=[0.60, 0.25, 0.15], k=1)[0]
        halflife = round(
            _clamp(rng.gauss(_EVENT_HALFLIFE[event_type], _EVENT_HALFLIFE[event_type] * 0.25), 0.5, 168.0), 2
        )

        signals.append(RiskSignal(
            event_id=str(uuid.uuid4()),
            ts_published=ts_published,
            ts_processed=ts_processed,
            source=source,  # type: ignore[arg-type]
            tickers=tickers,
            sentiment=round(sentiment, 4),
            sentiment_conf=round(sentiment_conf, 4),
            event_type=event_type,
            event_probs=event_probs,
            impact=impact,
            impact_range=(impact_low, impact_high),
            confidence=round(confidence, 4),
            novelty=round(novelty, 4),
            diffusion=DiffusionInfo(
                sources=sources_count,
                velocity=velocity,
                score=round(diffusion_score, 4),
            ),
            decay_halflife_hours=halflife,
            explanation=f"Synthetic {event_type} for {chosen_tickers[0]}",
            cluster_id=cluster_id,
            model_versions={"sentiment": "backtest-v1.0"},
        ))

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# Return simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_returns(
    weights_history: list[dict[str, float]],
    timestamps: list[datetime],
    rng: random.Random,
) -> list[float]:
    """
    Simulate daily portfolio P&L from a sequence of weight snapshots.

    Each segment between rebalances uses a constant weight vector.  Returns
    are drawn from N(0.0003, 0.015) per ticker per day, representing roughly
    7.5% annual return and 24% annualised vol for the equal-weight index.

    Returns: list of daily P&L values in dollar terms (relative to INITIAL_NAV).
    """
    if len(weights_history) < 2:
        return []

    # Unique trading days covered
    start = timestamps[0]
    end = timestamps[-1]
    n_days = max(1, (end - start).days)

    daily_pnl: list[float] = []

    for day_idx in range(n_days):
        day_ts = start + timedelta(days=day_idx)

        # Find which weight snapshot is active (latest rebalance before this day)
        active_weights = weights_history[0]
        for i, ts in enumerate(timestamps):
            if ts.replace(tzinfo=timezone.utc) <= day_ts.replace(tzinfo=timezone.utc):
                active_weights = weights_history[i]
            else:
                break

        # Simulate ticker returns
        day_return = 0.0
        for ticker, w in active_weights.items():
            # Independent daily return per ticker
            ret = rng.gauss(0.0003, 0.015)
            day_return += w * ret

        daily_pnl.append(day_return * INITIAL_NAV)

    return daily_pnl


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe(daily_pnl: list[float], nav: float = INITIAL_NAV) -> float:
    """Annualised Sharpe ratio (daily returns, risk-free ≈ 0)."""
    if len(daily_pnl) < 2:
        return 0.0
    daily_returns = [p / nav for p in daily_pnl]
    mean = sum(daily_returns) / len(daily_returns)
    variance = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    vol = math.sqrt(max(variance, 1e-12))
    return (mean / vol) * math.sqrt(252)


def compute_max_drawdown(daily_pnl: list[float]) -> float:
    """Maximum peak-to-trough drawdown as a fraction of NAV."""
    nav = INITIAL_NAV
    peak = nav
    max_dd = 0.0
    for pnl in daily_pnl:
        nav += pnl
        peak = max(peak, nav)
        dd = (peak - nav) / peak
        max_dd = max(max_dd, dd)
    return max_dd


def compute_ic(
    signals: list,
    weight_history: list[dict[str, float]],
    rng: random.Random,
) -> float:
    """
    Approximate IC (Spearman rank correlation between signal score and next-day return).

    For each rebalance signal, compute:
      - signal score: impact × |sentiment| × (1 if sentiment > 0 else -1)
      - proxy next-day return: synthetic draw from signal-correlated distribution
    """
    if len(signals) < 5:
        return 0.0

    signal_scores: list[float] = []
    return_proxies: list[float] = []

    for sig in signals:
        score = sig.impact / 10.0 * sig.sentiment
        # Proxy: return drawn from mean=0.5×signal, vol=0.01
        ret = rng.gauss(0.5 * score * 0.02, 0.015)
        signal_scores.append(score)
        return_proxies.append(ret)

    # Spearman rank correlation
    def _rank(lst: list[float]) -> list[float]:
        sorted_idx = sorted(range(len(lst)), key=lambda i: lst[i])
        ranks = [0.0] * len(lst)
        for rank, idx in enumerate(sorted_idx):
            ranks[idx] = float(rank + 1)
        return ranks

    rx = _rank(signal_scores)
    ry = _rank(return_proxies)
    n = len(rx)
    if n < 3:
        return 0.0
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1))


def compute_turnover(weight_events: list[dict]) -> float:
    """Average one-way turnover per rebalance event."""
    if not weight_events:
        return 0.0
    turnovers: list[float] = []
    for evt in weight_events:
        before = evt.get("before_weights", {})
        after = evt.get("after_weights", {})
        all_tickers = set(before) | set(after)
        one_way = sum(abs(after.get(t, 0.0) - before.get(t, 0.0)) for t in all_tickers) / 2.0
        turnovers.append(one_way)
    return sum(turnovers) / len(turnovers)


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest
# ─────────────────────────────────────────────────────────────────────────────

async def run_backtest(signals: list) -> dict:
    """Feed signals through IndexRebalancer and collect rebalance events."""
    from module_a.rebalancer import IndexRebalancer

    rebalancer = IndexRebalancer()
    rebalance_events: list[dict] = []
    weight_snapshots: list[dict[str, float]] = [rebalancer.get_current_weights()]
    snapshot_times: list[datetime] = [signals[0].ts_processed]

    logger.info("Running %d signals through IndexRebalancer …", len(signals))
    for i, signal in enumerate(signals):
        event = await rebalancer.on_signal(signal)
        if event is not None:
            rebalance_events.append(event.to_dict())
            weight_snapshots.append(rebalancer.get_current_weights())
            snapshot_times.append(signal.ts_processed)

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d signals (%d rebalances)", i + 1, len(signals), len(rebalance_events))

    return {
        "rebalance_events": rebalance_events,
        "weight_snapshots": weight_snapshots,
        "snapshot_times": snapshot_times,
        "final_weights": rebalancer.get_current_weights(),
    }


def _print_rule(char: str = "━", width: int = 55) -> None:
    print(char * width)


async def main_async() -> None:
    rng = random.Random(42)

    # Load or generate signals
    signals_path = Path("data/sample/signals.jsonl")
    if signals_path.exists():
        from app.schema.risk_signal import RiskSignal
        signals = []
        try:
            with signals_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        signals.append(RiskSignal.model_validate_json(line))
            # Supplement with generated signals if fewer than N_SIGNALS
            if len(signals) < N_SIGNALS:
                extra = N_SIGNALS - len(signals)
                logger.info("Supplementing with %d generated signals", extra)
                signals += generate_signals(extra, rng)
            else:
                signals = signals[:N_SIGNALS]
            logger.info("Loaded %d signals from %s", len(signals), signals_path)
        except Exception as exc:
            logger.warning("Could not load signals from file (%s) — generating fresh", exc)
            signals = generate_signals(N_SIGNALS, rng)
    else:
        logger.info("No signals.jsonl found — generating %d synthetic signals", N_SIGNALS)
        signals = generate_signals(N_SIGNALS, rng)

    # Sort by timestamp
    signals.sort(key=lambda s: s.ts_processed)

    # Run backtest
    backtest_result = await run_backtest(signals)
    rebalance_events = backtest_result["rebalance_events"]
    weight_snapshots = backtest_result["weight_snapshots"]
    snapshot_times = backtest_result["snapshot_times"]

    logger.info("Rebalances triggered: %d / %d signals", len(rebalance_events), N_SIGNALS)

    # Simulate returns for each strategy
    tilted_pnl = _simulate_returns(weight_snapshots, snapshot_times, random.Random(100))

    # Equal-weight: static, never rebalances
    eq_snapshots = [EQ_WEIGHTS] * len(snapshot_times)
    eq_pnl = _simulate_returns(eq_snapshots, snapshot_times, random.Random(101))

    # Cap-weight: static, based on mock market cap
    cap_snapshots = [CAP_WEIGHTS] * len(snapshot_times)
    cap_pnl = _simulate_returns(cap_snapshots, snapshot_times, random.Random(102))

    # Compute metrics
    def _final_nav(pnl: list[float]) -> float:
        return INITIAL_NAV + sum(pnl)

    def _total_return(pnl: list[float]) -> float:
        return sum(pnl) / INITIAL_NAV

    ic = compute_ic(signals, weight_snapshots, random.Random(200))
    avg_turnover = compute_turnover(rebalance_events)

    tilted_metrics = {
        "strategy": "Signal-tilted (IndexRebalancer)",
        "final_nav": round(_final_nav(tilted_pnl), 2),
        "total_return_pct": round(_total_return(tilted_pnl) * 100, 2),
        "sharpe_annualised": round(compute_sharpe(tilted_pnl), 3),
        "max_drawdown_pct": round(compute_max_drawdown(tilted_pnl) * 100, 2),
        "n_rebalances": len(rebalance_events),
        "avg_turnover_per_rebalance": round(avg_turnover, 4),
        "ic_spearman": round(ic, 4),
    }
    eq_metrics = {
        "strategy": "Equal-weight (1/20)",
        "final_nav": round(_final_nav(eq_pnl), 2),
        "total_return_pct": round(_total_return(eq_pnl) * 100, 2),
        "sharpe_annualised": round(compute_sharpe(eq_pnl), 3),
        "max_drawdown_pct": round(compute_max_drawdown(eq_pnl) * 100, 2),
        "n_rebalances": 0,
        "avg_turnover_per_rebalance": 0.0,
        "ic_spearman": None,
    }
    cap_metrics = {
        "strategy": "Cap-weight proxy",
        "final_nav": round(_final_nav(cap_pnl), 2),
        "total_return_pct": round(_total_return(cap_pnl) * 100, 2),
        "sharpe_annualised": round(compute_sharpe(cap_pnl), 3),
        "max_drawdown_pct": round(compute_max_drawdown(cap_pnl) * 100, 2),
        "n_rebalances": 0,
        "avg_turnover_per_rebalance": 0.0,
        "ic_spearman": None,
    }

    # Save results
    output = {
        "n_signals": len(signals),
        "initial_nav": INITIAL_NAV,
        "simulation_period_days": (signals[-1].ts_processed - signals[0].ts_processed).days,
        "tilted": tilted_metrics,
        "equal_weight": eq_metrics,
        "cap_weight": cap_metrics,
        "rebalance_events_summary": [
            {
                "rebalance_id": e["rebalance_id"],
                "ts_created": e["ts_created"].isoformat() if hasattr(e["ts_created"], "isoformat") else str(e["ts_created"]),
                "reason": e["reason"],
            }
            for e in rebalance_events[:20]  # first 20 for inspection
        ],
    }

    with OUTPUT_PATH.open("w") as fh:
        json.dump(output, fh, indent=2, default=str)
    logger.info("Saved backtest results to %s", OUTPUT_PATH)

    # Print summary
    print()
    _print_rule()
    print("Module A Backtest Summary")
    _print_rule()
    print(f"  Signals:          {len(signals):,}")
    print(f"  Period:           ~{output['simulation_period_days']} days")
    print()

    all_metrics = [tilted_metrics, eq_metrics, cap_metrics]
    header = f"  {'Strategy':<35} {'Return':>8} {'Sharpe':>8} {'MaxDD':>8} {'Rebal':>7}"
    print(header)
    print("  " + "-" * 70)
    for m in all_metrics:
        dd = f"{m['max_drawdown_pct']:.1f}%"
        sh = f"{m['sharpe_annualised']:.3f}"
        ret = f"{m['total_return_pct']:.1f}%"
        reb = str(m["n_rebalances"])
        print(f"  {m['strategy']:<35} {ret:>8} {sh:>8} {dd:>8} {reb:>7}")

    print()
    print(f"  IC (Spearman, tilted strategy): {ic:.4f}")
    print(f"  Avg turnover per rebalance:     {avg_turnover:.4f}")
    _print_rule()
    print(f"Results saved to {OUTPUT_PATH}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
