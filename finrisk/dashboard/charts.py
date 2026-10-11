"""
Reusable Plotly chart factories for the FinRisk dashboard.

All functions:
- Accept empty / None inputs and return a blank placeholder figure.
- Return go.Figure so callers can use st.plotly_chart(fig, use_container_width=True).
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# ── Colour helpers ────────────────────────────────────────────────────────────

def sentiment_color(val: float) -> str:
    """
    Map sentiment score in [-1, 1] to a hex colour string.
    -1 → red (#EF4444), 0 → neutral (#9CA3AF), +1 → green (#22C55E).
    """
    val = max(-1.0, min(1.0, float(val)))
    if val < 0:
        # Interpolate from neutral grey to red
        t = abs(val)
        r = int(156 + t * (239 - 156))
        g = int(163 + t * (68 - 163))
        b = int(175 + t * (68 - 175))
    else:
        # Interpolate from neutral grey to green
        t = val
        r = int(156 + t * (34 - 156))
        g = int(163 + t * (197 - 163))
        b = int(175 + t * (94 - 175))
    return f"#{r:02X}{g:02X}{b:02X}"


# ── Waterfall chart ───────────────────────────────────────────────────────────

def make_waterfall_chart(
    breakdown: dict[str, float],
    title: str = "P&L Breakdown by Asset Class",
) -> go.Figure:
    """
    Waterfall chart showing gain/loss contribution per asset class.

    ``breakdown`` maps label → P&L (negative = loss, positive = gain).
    Bars ending below zero are red; above zero are green.
    """
    if not breakdown:
        fig = go.Figure()
        fig.add_annotation(
            text="No breakdown data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#6B7280"),
        )
        fig.update_layout(title=title, height=320)
        return fig

    labels = list(breakdown.keys())
    values = list(breakdown.values())
    total = sum(values)

    # Friendly label mapping
    label_map = {
        "bonds_pnl": "Bonds",
        "loans_pnl": "Loans",
        "equities_pnl": "Equities",
        "fx_pnl": "FX",
        "swaps_pnl": "Swaps",
    }
    display_labels = [label_map.get(k, k.replace("_", " ").title()) for k in labels]
    display_labels.append("Total")

    measure = ["relative"] * len(values) + ["total"]
    y_values = values + [total]

    marker_colors = []
    for v in values:
        marker_colors.append("#EF4444" if v < 0 else "#22C55E")
    marker_colors.append("#EF4444" if total < 0 else "#22C55E")

    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=display_labels,
            y=[v / 1_000_000 for v in y_values],
            text=[f"${v/1e6:+.2f}M" for v in y_values],
            textposition="outside",
            connector={"line": {"color": "#374151"}},
            increasing={"marker": {"color": "#22C55E"}},
            decreasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#22C55E" if total >= 0 else "#EF4444"}},
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title="P&L ($M)",
        height=380,
        showlegend=False,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
    )
    return fig


# ── Monte Carlo loss distribution ─────────────────────────────────────────────

def make_loss_distribution_chart(
    losses: list[float],
    var_95: float | None,
    es_95: float | None,
    det_loss: float | None,
) -> go.Figure:
    """
    Histogram of Monte Carlo simulated losses with VaR/ES overlay lines.

    Losses are in USD (will be displayed in $M).  Negative values = losses.
    """
    fig = go.Figure()

    if not losses:
        fig.add_annotation(
            text="No Monte Carlo data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#6B7280"),
        )
        fig.update_layout(title="Monte Carlo Loss Distribution", height=340)
        return fig

    losses_m = [abs(v) / 1_000_000 for v in losses]  # convert to $M, show as positive loss

    fig.add_trace(
        go.Histogram(
            x=losses_m,
            nbinsx=60,
            name="Simulated Loss",
            marker_color="#3B82F6",
            opacity=0.75,
        )
    )

    # Vertical reference lines
    annotations = []
    shapes = []

    def _vline(x_val: float, color: str, label: str, dash: str = "dash"):
        shapes.append(
            dict(
                type="line",
                x0=x_val, x1=x_val,
                y0=0, y1=1,
                yref="paper",
                line=dict(color=color, width=2, dash=dash),
            )
        )
        annotations.append(
            dict(
                x=x_val, y=0.95,
                xref="x", yref="paper",
                text=label,
                showarrow=False,
                font=dict(color=color, size=11),
                textangle=-90,
                xanchor="left",
            )
        )

    if var_95 is not None:
        _vline(abs(var_95) / 1e6, "#F59E0B", f"VaR 95%: ${abs(var_95)/1e6:.2f}M", dash="dash")
    if es_95 is not None:
        _vline(abs(es_95) / 1e6, "#EF4444", f"ES 95%: ${abs(es_95)/1e6:.2f}M", dash="dot")
    if det_loss is not None:
        _vline(abs(det_loss) / 1e6, "#60A5FA", f"Det. loss: ${abs(det_loss)/1e6:.2f}M", dash="longdash")

    fig.update_layout(
        title="Monte Carlo Loss Distribution",
        xaxis_title="Loss ($M)",
        yaxis_title="Frequency",
        height=360,
        showlegend=False,
        shapes=shapes,
        annotations=annotations,
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
    )
    return fig


# ── Weight history stacked area ───────────────────────────────────────────────

def make_weight_history_chart(
    history: list[dict],
    tickers: list[str],
) -> go.Figure:
    """
    Stacked area chart of portfolio weights over time.

    ``history`` is a list of dicts: {ts_created: str, weights: dict[str, float]}.
    ``tickers`` determines the ordering and colour assignment.
    """
    fig = go.Figure()

    if not history or not tickers:
        fig.add_annotation(
            text="No rebalance history yet — equal weight displayed when events arrive",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="#6B7280"),
        )
        fig.update_layout(title="Portfolio Weight History", height=360)
        return fig

    import pandas as pd
    rows = []
    for h in history:
        ts = h["ts_created"]
        w = h["weights"]
        for ticker in tickers:
            rows.append({"ts": ts, "ticker": ticker, "weight": w.get(ticker, 0.0)})

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])

    colors = px.colors.qualitative.Plotly
    for i, ticker in enumerate(tickers):
        sub = df[df["ticker"] == ticker].sort_values("ts")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["ts"],
                y=sub["weight"] * 100,  # convert to percentage
                name=ticker,
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
            )
        )

    fig.update_layout(
        title="Portfolio Weight History",
        xaxis_title="Time",
        yaxis_title="Weight (%)",
        yaxis=dict(gridcolor="#1E293B", ticksuffix="%", range=[0, 105]),
        height=360,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="#1E293B"),
    )
    return fig


# ── Gauge chart ───────────────────────────────────────────────────────────────

def make_gauge_chart(
    value: float,
    min_val: float,
    max_val: float,
    title: str,
) -> go.Figure:
    """
    Circular gauge chart (used for breach impact score).

    Colours: green 0–4, yellow 4–7, red 7–10.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"color": "#E2E8F0"}},
            number={"font": {"color": "#E2E8F0"}, "suffix": "/10"},
            gauge={
                "axis": {
                    "range": [min_val, max_val],
                    "tickcolor": "#94A3B8",
                    "tickfont": {"color": "#94A3B8"},
                },
                "bar": {"color": "#3B82F6"},
                "bgcolor": "#1E293B",
                "borderwidth": 2,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [min_val, 4], "color": "#064E3B"},
                    {"range": [4, 7], "color": "#78350F"},
                    {"range": [7, max_val], "color": "#7F1D1D"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
    )
    return fig


# ── Sector heatmap ────────────────────────────────────────────────────────────

def make_sector_heatmap(sector_pnl: dict[str, float]) -> go.Figure:
    """
    Heatmap showing P&L contribution by sector.

    ``sector_pnl`` maps sector name → P&L value (USD).
    """
    fig = go.Figure()

    if not sector_pnl:
        fig.add_annotation(
            text="No sector data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#6B7280"),
        )
        fig.update_layout(title="Sector P&L Heatmap", height=340)
        return fig

    sectors = list(sector_pnl.keys())
    values_m = [sector_pnl[s] / 1_000_000 for s in sectors]

    # Format to 2D for heatmap (1 column: "P&L")
    fig.add_trace(
        go.Heatmap(
            z=[[v] for v in values_m],
            x=["P&L ($M)"],
            y=sectors,
            colorscale=[
                [0.0, "#7F1D1D"],
                [0.4, "#EF4444"],
                [0.5, "#FFFFFF"],
                [0.6, "#22C55E"],
                [1.0, "#064E3B"],
            ],
            zmid=0,
            text=[[f"${v:.2f}M"] for v in values_m],
            texttemplate="%{text}",
            showscale=True,
            colorbar=dict(
                tickfont=dict(color="#E2E8F0"),
                title=dict(text="$M", font=dict(color="#E2E8F0")),
            ),
        )
    )
    fig.update_layout(
        title="Sector P&L Contribution",
        height=max(340, 40 * len(sectors) + 100),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig


# ── Sentiment timeline ────────────────────────────────────────────────────────

def make_sentiment_timeline(
    df,  # pd.DataFrame with columns: ts_processed, sentiment, event_type, ticker_csv
    title: str = "Sentiment Over Time",
) -> go.Figure:
    """
    Line chart of sentiment scores coloured by event_type.
    Handles empty DataFrame gracefully.
    """
    import pandas as pd

    fig = go.Figure()

    if df is None or (hasattr(df, "__len__") and len(df) == 0):
        fig.add_annotation(
            text="No signal data available for this filter",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#6B7280"),
        )
        fig.update_layout(title=title, height=320)
        return fig

    colors = px.colors.qualitative.Plotly
    event_types = df["event_type"].unique() if "event_type" in df.columns else []
    color_map = {et: colors[i % len(colors)] for i, et in enumerate(event_types)}

    for et in event_types:
        sub = df[df["event_type"] == et].sort_values("ts_processed")
        fig.add_trace(
            go.Scatter(
                x=sub["ts_processed"],
                y=sub["sentiment"],
                mode="lines+markers",
                name=et,
                line=dict(color=color_map.get(et, "#3B82F6"), width=1.5),
                marker=dict(size=5),
            )
        )

    fig.add_hline(y=0, line_dash="dot", line_color="#6B7280", line_width=1)
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Sentiment",
        yaxis_range=[-1.1, 1.1],
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B"),
    )
    return fig
