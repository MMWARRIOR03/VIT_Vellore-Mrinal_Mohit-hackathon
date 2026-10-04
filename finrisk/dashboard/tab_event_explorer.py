"""
Tab 2 — Event Explorer

Interactive exploration of signals by ticker and time range.
Charts: sentiment timeline, event-type distribution, diffusion score,
event cluster table.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.charts import make_sentiment_timeline
from dashboard.data_loader import DataLoader


_TIME_RANGES = {
    "Last 1 hour": timedelta(hours=1),
    "Last 6 hours": timedelta(hours=6),
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days": timedelta(days=7),
}


def _signals_to_df(signals: list[dict]) -> pd.DataFrame:
    """Flatten signal list to a DataFrame for chart consumption."""
    rows: list[dict] = []
    for sig in signals:
        ts = sig.get("ts_processed") or sig.get("ts_published", "")
        tickers_raw = sig.get("tickers", [])
        if isinstance(tickers_raw, list):
            ticker_str = ", ".join(
                (t.get("ticker", "") if isinstance(t, dict) else str(t))
                for t in tickers_raw
            )
        else:
            ticker_str = str(tickers_raw)

        diffusion = sig.get("diffusion", {}) or {}
        diffusion_score = (
            diffusion.get("score", 0.0) if isinstance(diffusion, dict) else 0.0
        )

        rows.append(
            {
                "ts_processed": ts,
                "ticker_csv": ticker_str,
                "event_type": sig.get("event_type", "Unknown"),
                "sentiment": float(sig.get("sentiment", 0.0)),
                "impact": float(sig.get("impact", 1.0)),
                "confidence": float(sig.get("confidence", 0.0)),
                "diffusion_score": float(diffusion_score),
                "cluster_id": sig.get("cluster_id", "")[:12],
                "source": sig.get("source", "—"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "ts_processed" in df.columns:
        df["ts_processed"] = pd.to_datetime(df["ts_processed"], utc=True, errors="coerce")
    return df


def render_event_explorer(loader: DataLoader) -> None:
    """Render the Event Explorer tab."""

    # ── Filter controls ───────────────────────────────────────────────────────
    all_tickers = loader.get_all_tickers()
    ticker_options = ["All"] + all_tickers

    ctrl1, ctrl2 = st.columns([2, 2])
    with ctrl1:
        selected_ticker = st.selectbox(
            "Ticker filter",
            options=ticker_options,
            index=0,
            key="explorer_ticker",
        )
    with ctrl2:
        selected_range = st.selectbox(
            "Time range",
            options=list(_TIME_RANGES.keys()),
            index=2,
            key="explorer_range",
        )

    # ── Fetch and filter data ─────────────────────────────────────────────────
    ticker_arg = None if selected_ticker == "All" else selected_ticker
    raw_signals = loader.get_recent_signals(limit=500, ticker=ticker_arg)
    df = _signals_to_df(raw_signals)

    cutoff = datetime.now(timezone.utc) - _TIME_RANGES[selected_range]

    if not df.empty and "ts_processed" in df.columns:
        df = df[df["ts_processed"] >= cutoff]

    if df.empty:
        st.info("No signals match the current filter. Try a wider time range or 'All' tickers.")
        return

    # ── Row 1: Sentiment timeline ─────────────────────────────────────────────
    st.markdown("#### Sentiment Timeline")
    fig_sentiment = make_sentiment_timeline(df, title="Sentiment Score by Event Type")
    st.plotly_chart(fig_sentiment, use_container_width=True)

    # ── Row 2: Event type distribution + Diffusion score ─────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Event Type Distribution")
        type_counts = df["event_type"].value_counts().reset_index()
        type_counts.columns = ["event_type", "count"]

        if not type_counts.empty:
            fig_pie = go.Figure(
                go.Pie(
                    labels=type_counts["event_type"],
                    values=type_counts["count"],
                    hole=0.45,
                    textinfo="label+percent",
                    textfont=dict(color="#E2E8F0"),
                    marker=dict(
                        colors=px.colors.qualitative.Plotly,
                        line=dict(color="#0F172A", width=1),
                    ),
                )
            )
            fig_pie.update_layout(
                height=340,
                paper_bgcolor="#0F172A",
                font=dict(color="#E2E8F0"),
                showlegend=True,
                legend=dict(font=dict(color="#E2E8F0")),
                margin=dict(t=30, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No event type data.")

    with col_right:
        st.markdown("#### Diffusion Score Over Time")
        df_diff = df[df["diffusion_score"] > 0].sort_values("ts_processed")

        if not df_diff.empty:
            fig_diff = go.Figure()
            fig_diff.add_trace(
                go.Scatter(
                    x=df_diff["ts_processed"],
                    y=df_diff["diffusion_score"],
                    mode="lines",
                    fill="tozeroy",
                    name="Diffusion",
                    line=dict(color="#60A5FA", width=1.5),
                    fillcolor="rgba(96,165,250,0.2)",
                )
            )
            fig_diff.update_layout(
                height=340,
                xaxis_title="Time",
                yaxis_title="Diffusion Score",
                yaxis_range=[0, 1.05],
                plot_bgcolor="#0F172A",
                paper_bgcolor="#0F172A",
                font=dict(color="#E2E8F0"),
                xaxis=dict(gridcolor="#1E293B"),
                yaxis=dict(gridcolor="#1E293B"),
                showlegend=False,
                margin=dict(t=30, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_diff, use_container_width=True)
        else:
            st.info("No diffusion data for this filter.")

    # ── Row 3: Event cluster table ────────────────────────────────────────────
    st.markdown("#### Event Clusters")

    if "cluster_id" in df.columns and not df["cluster_id"].eq("").all():
        cluster_df = (
            df.groupby("cluster_id")
            .agg(
                ticker=("ticker_csv", "first"),
                event_type=("event_type", "first"),
                signal_count=("cluster_id", "count"),
                latest_impact=("impact", "last"),
                latest_ts=("ts_processed", "max"),
            )
            .reset_index()
            .sort_values("latest_ts", ascending=False)
            .head(20)
        )
        cluster_df["latest_ts"] = cluster_df["latest_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        cluster_df.columns = [
            "Cluster ID", "Ticker(s)", "Event Type",
            "Signal Count", "Latest Impact", "Latest Time",
        ]
        st.dataframe(
            cluster_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cluster ID": st.column_config.TextColumn("Cluster ID", width="medium"),
                "Signal Count": st.column_config.NumberColumn("Signal Count", width="small"),
                "Latest Impact": st.column_config.ProgressColumn(
                    "Latest Impact", min_value=1.0, max_value=10.0, format="%.1f"
                ),
            },
        )
    else:
        st.info("No cluster data available.")
