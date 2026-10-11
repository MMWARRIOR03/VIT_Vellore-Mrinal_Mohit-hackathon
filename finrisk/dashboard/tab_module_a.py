"""
Tab 3 — Module A: Index Rebalancer

Shows current portfolio weights, weight history, sentiment heatmap,
recent rebalance events, and synthetic NAV comparison.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.charts import make_weight_history_chart
from dashboard.data_loader import DataLoader


# FinRisk-20 universe — must match configs/index.yaml exactly
_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "BAC", "GS",
    "V", "JNJ", "UNH", "PFE", "XOM",
    "CVX", "CAT", "BA", "WMT", "HD",
]
_EQ_WEIGHT = round(1.0 / len(_UNIVERSE), 6)


def render_module_a(loader: DataLoader) -> None:
    """Render the Module A: Index Rebalancer tab."""

    # ── Row 1: Current portfolio weights bar chart ────────────────────────────
    st.markdown("#### Current Portfolio Weights")

    current_weights = loader.get_current_weights()

    # Ensure every universe ticker is present
    weights_full: dict[str, float] = {
        t: current_weights.get(t, _EQ_WEIGHT) for t in _UNIVERSE
    }

    tickers = list(weights_full.keys())
    weights_vals = list(weights_full.values())
    deltas = [w - _EQ_WEIGHT for w in weights_vals]

    colors = ["#22C55E" if d >= 0 else "#EF4444" for d in deltas]
    text_labels = [f"{w*100:.2f}%" for w in weights_vals]

    fig_weights = go.Figure()
    fig_weights.add_trace(
        go.Bar(
            y=tickers,
            x=weights_vals,
            orientation="h",
            marker_color=colors,
            text=text_labels,
            textposition="outside",
            name="Current Weight",
        )
    )
    fig_weights.add_vline(
        x=_EQ_WEIGHT,
        line_dash="dash",
        line_color="#94A3B8",
        annotation_text=f"Equal weight ({_EQ_WEIGHT*100:.1f}%)",
        annotation_font_color="#94A3B8",
        annotation_position="top left",
    )
    fig_weights.update_layout(
        height=540,
        xaxis_title="Weight",
        xaxis_tickformat=".1%",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="#1E293B"),
        showlegend=False,
        margin=dict(r=120),
    )
    st.plotly_chart(fig_weights, use_container_width=True)

    # ── Row 2: Weight history stacked area ────────────────────────────────────
    st.markdown("#### Weight History")

    history = loader.get_weight_history()
    fig_history = make_weight_history_chart(history, _UNIVERSE)
    st.plotly_chart(fig_history, use_container_width=True)

    # ── Row 3: Sentiment heatmap ──────────────────────────────────────────────
    st.markdown("#### Ticker Sentiment Heatmap (last 50 signals)")

    ticker_stats = loader.get_ticker_stats()

    if not ticker_stats.empty:
        # Align to universe order; fill missing tickers with 0
        stat_map = dict(zip(ticker_stats["ticker"], ticker_stats["mean_sentiment"]))
        heat_tickers = [t for t in _UNIVERSE if t in stat_map]
        heat_vals = [[stat_map[t]] for t in heat_tickers]

        if heat_tickers:
            fig_heat = go.Figure(
                go.Heatmap(
                    z=heat_vals,
                    x=["Mean Sentiment"],
                    y=heat_tickers,
                    colorscale=[
                        [0.0, "#7F1D1D"],
                        [0.25, "#EF4444"],
                        [0.5, "#FFFFFF"],
                        [0.75, "#22C55E"],
                        [1.0, "#064E3B"],
                    ],
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=[[f"{stat_map[t]:.3f}"] for t in heat_tickers],
                    texttemplate="%{text}",
                    showscale=True,
                    colorbar=dict(
                        tickfont=dict(color="#E2E8F0"),
                        title=dict(text="Score", font=dict(color="#E2E8F0")),
                    ),
                )
            )
            fig_heat.update_layout(
                height=max(320, 32 * len(heat_tickers) + 80),
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
                font=dict(color="#E2E8F0"),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False, autorange="reversed"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No ticker sentiment data yet.")
    else:
        st.info("No ticker stats yet. Signals will populate this when the engine runs.")

    # ── Row 4: Recent rebalance events ────────────────────────────────────────
    st.markdown("#### Recent Rebalance Events")

    events = loader.get_rebalance_events(limit=20)

    if not events:
        st.info("No rebalance events yet. The index rebalancer triggers when high-confidence signals arrive.")
    else:
        rows: list[dict] = []
        for ev in events:
            ts = str(ev["ts_created"])[:19].replace("T", " ")
            delta = ev.get("delta", {})
            if delta:
                biggest = sorted(delta.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                changes = ", ".join(f"{k}: {v*100:+.2f}%" for k, v in biggest)
            else:
                changes = "—"
            rows.append(
                {
                    "Time (UTC)": ts,
                    "Trigger": ev.get("reason", "—")[:60],
                    "Top Weight Changes": changes,
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time (UTC)": st.column_config.TextColumn(width="medium"),
                "Trigger": st.column_config.TextColumn(width="large"),
                "Top Weight Changes": st.column_config.TextColumn(width="large"),
            },
        )

    # ── Row 5: Synthetic NAV comparison ──────────────────────────────────────
    st.markdown("#### Illustrative Tilted vs Equal-Weight NAV ($10M synthetic)")
    st.caption("Illustrative returns are deterministic placeholders, not historical market performance.")

    _NAV_BASE = 10_000_000.0

    if not events:
        # Flat line — no rebalances yet
        now_ts = datetime.now(timezone.utc)
        fig_nav = go.Figure()
        fig_nav.add_trace(
            go.Scatter(
                x=[now_ts, now_ts],
                y=[_NAV_BASE, _NAV_BASE],
                mode="lines",
                name="Equal Weight",
                line=dict(color="#94A3B8", dash="dot", width=2),
            )
        )
        fig_nav.add_annotation(
            text="No rebalance events yet",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#6B7280"),
        )
        fig_nav.update_layout(
            height=320,
            plot_bgcolor="#0F172A",
            paper_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
        )
        st.plotly_chart(fig_nav, use_container_width=True)
        return

    # Build synthetic NAV series from rebalance deltas
    events_asc = list(reversed(events))  # oldest first
    timestamps: list[str] = []
    tilted_nav: list[float] = []
    eq_nav: list[float] = []

    nav_tilt = _NAV_BASE
    nav_eq = _NAV_BASE

    for ev in events_asc:
        ts = str(ev["ts_created"])[:19]
        delta = ev.get("delta", {})
        meta  = ev.get("meta", {})
        after = meta.get("after_weights", {})
        before = meta.get("before_weights", {})

        # Compute active return: sum over tickers of (weight_delta × sentiment_proxy)
        # Overweight tickers that had positive sentiment → tilted outperforms equal-weight
        # Use weight delta sign as direction: positive delta = bet on that ticker
        # Assume average stock return of 0.05%/day; active return = weighted deviation
        active_return = 0.0
        for ticker, w_after in after.items():
            w_before = before.get(ticker, _EQ_WEIGHT)
            w_delta = w_after - w_before
            # Positive tilt on a ticker contributes positively if that ticker "outperforms"
            # We use a random seed per ticker for reproducible synthetic returns
            import hashlib
            seed_val = int(hashlib.md5(f"{ticker}{ts}".encode()).hexdigest()[:8], 16)
            import random
            rng_local = random.Random(seed_val)
            ticker_return = rng_local.gauss(0.0002, 0.012)  # ~5% annualised daily
            active_return += w_delta * ticker_return

        nav_tilt = nav_tilt * (1 + active_return)

        timestamps.append(ts)
        tilted_nav.append(nav_tilt)
        eq_nav.append(nav_eq)  # eq weight doesn't change

    fig_nav = go.Figure()
    fig_nav.add_trace(
        go.Scatter(
            x=timestamps,
            y=[v / 1e6 for v in tilted_nav],
            mode="lines+markers",
            name="Tilted NAV",
            line=dict(color="#22C55E", width=2),
        )
    )
    fig_nav.add_trace(
        go.Scatter(
            x=timestamps,
            y=[v / 1e6 for v in eq_nav],
            mode="lines",
            name="Equal Weight NAV",
            line=dict(color="#94A3B8", dash="dot", width=2),
        )
    )
    fig_nav.update_layout(
        height=320,
        xaxis_title="Rebalance Event",
        yaxis_title="NAV ($M)",
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_nav, use_container_width=True)
