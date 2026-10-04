"""
Tab 1 — Live Risk Feed

Displays the 50 most recent RiskSignals in a formatted table with
KPI metrics at the top. Auto-refreshed by the parent app.py loop.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.charts import sentiment_color
from dashboard.data_loader import DataLoader


def _parse_tickers(tickers_raw) -> str:
    """Convert tickers field (list of dicts or list of strings) to comma-joined string."""
    if not tickers_raw:
        return ""
    if isinstance(tickers_raw, list):
        parts: list[str] = []
        for item in tickers_raw:
            if isinstance(item, dict):
                parts.append(item.get("ticker", ""))
            elif isinstance(item, str):
                parts.append(item)
        return ", ".join(p for p in parts if p)
    return str(tickers_raw)


def render_live_feed(loader: DataLoader) -> None:
    """Render the Live Risk Feed tab."""

    # ── Top KPI row ───────────────────────────────────────────────────────────
    stats = loader.get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Signals Today", stats["total_signals_today"])
    with col2:
        st.metric(
            "Mean Impact (last 50)",
            f"{stats['mean_impact']:.1f} / 10",
        )
    with col3:
        st.metric("Active Tickers", stats["active_tickers"])
    with col4:
        latest = stats["latest_ts"]
        if latest:
            # Trim to HH:MM:SS for readability
            try:
                ts_str = str(latest)[:19].replace("T", " ")
            except Exception:
                ts_str = str(latest)
            st.metric("Last Signal", ts_str)
        else:
            st.metric("Last Signal", "—")

    st.divider()

    # ── Signal table ──────────────────────────────────────────────────────────
    signals = loader.get_recent_signals(limit=50)

    if not signals:
        st.info("No signals yet. Start the API server and signal generator to see live data.")
        return

    # Build display rows
    rows: list[dict] = []
    for sig in signals:
        ts_raw = sig.get("ts_processed") or sig.get("ts_published", "")
        try:
            ts_str = str(ts_raw)[:19].replace("T", " ")
        except Exception:
            ts_str = str(ts_raw)

        tickers_str = _parse_tickers(sig.get("tickers", []))
        sentiment = float(sig.get("sentiment", 0.0))
        impact = float(sig.get("impact", 1.0))
        confidence = float(sig.get("confidence", 0.0))

        rows.append(
            {
                "Time (UTC)": ts_str,
                "Ticker(s)": tickers_str,
                "Event Type": sig.get("event_type", "—"),
                "Sentiment": round(sentiment, 3),
                "Impact": round(impact, 1),
                "Confidence": round(confidence, 3),
                "Source": sig.get("source", "—"),
                # hidden columns used by expanders
                "_event_id": sig.get("event_id", ""),
                "_explanation": sig.get("explanation") or "",
                "_cluster_id": sig.get("cluster_id", ""),
            }
        )

    df_display = pd.DataFrame(rows)

    # Columns visible in the dataframe widget
    visible_cols = [
        "Time (UTC)", "Ticker(s)", "Event Type", "Sentiment", "Impact", "Confidence", "Source"
    ]

    st.dataframe(
        df_display[visible_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Time (UTC)": st.column_config.TextColumn("Time (UTC)", width="medium"),
            "Ticker(s)": st.column_config.TextColumn("Ticker(s)", width="small"),
            "Event Type": st.column_config.TextColumn("Event Type", width="medium"),
            "Sentiment": st.column_config.NumberColumn(
                "Sentiment",
                min_value=-1.0,
                max_value=1.0,
                format="%.3f",
                width="small",
            ),
            "Impact": st.column_config.ProgressColumn(
                "Impact",
                min_value=1.0,
                max_value=10.0,
                format="%.1f",
                width="medium",
            ),
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                width="medium",
            ),
            "Source": st.column_config.TextColumn("Source", width="small"),
        },
    )

    # ── Per-signal expanders ──────────────────────────────────────────────────
    st.markdown("#### Signal Detail")

    shown = 0
    for row in rows[:10]:  # show expanders for the 10 most recent
        event_id = row["_event_id"]
        explanation = row["_explanation"]
        cluster_id = row["_cluster_id"]
        label = f"▸ {row['Time (UTC)']}  •  {row['Ticker(s)']}  •  {row['Event Type']}"

        with st.expander(label, expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Sentiment", f"{row['Sentiment']:+.3f}")
            c2.metric("Impact", f"{row['Impact']:.1f}/10")
            c3.metric("Confidence", f"{row['Confidence']:.0%}")

            if explanation:
                st.markdown(f"**Analysis:** {explanation}")
            if event_id:
                st.caption(f"Event ID: `{event_id}`   Cluster: `{cluster_id}`")

        shown += 1

    if len(rows) > 10:
        st.caption(f"Showing expanders for the 10 most recent signals. {len(rows) - 10} more in the table above.")
