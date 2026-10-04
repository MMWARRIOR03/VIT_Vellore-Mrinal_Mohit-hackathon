"""
Tab 4 — Module B: Stress Tester

Displays the most recent stress test result (pre/post NAV, waterfall,
Monte Carlo distribution, sector heatmap, reverse stress gauge) and a
historical runs table. Also exposes a manual stress test trigger.
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from dashboard.charts import (
    make_gauge_chart,
    make_loss_distribution_chart,
    make_sector_heatmap,
    make_waterfall_chart,
)
from dashboard.data_loader import DataLoader


def _fmt_m(val: float | None, prefix: str = "$") -> str:
    """Format a dollar value in $M with sign."""
    if val is None:
        return "—"
    return f"{prefix}{val / 1_000_000:+.2f}M" if val < 0 else f"{prefix}{val / 1_000_000:.2f}M"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val * 100:+.2f}%"


def render_module_b(loader: DataLoader) -> None:
    """Render the Module B: Stress Tester tab."""

    stress_runs = loader.get_stress_runs(limit=10)

    # ── Manual stress test button (top of tab) ────────────────────────────────
    st.markdown("#### Manual Stress Test")

    with st.expander("Run a manual stress scenario", expanded=False):
        from dashboard.manual_stress import run_manual_stress
        from module_b.scenarios import ScenarioEngine

        try:
            scenario_engine = ScenarioEngine()
            available_scenarios = scenario_engine.available_scenarios()
        except Exception:
            available_scenarios = [
                "Earnings", "Geopolitical", "Credit", "Regulatory",
                "M&A", "Product", "Macro", "Natural Disaster",
            ]

        m_col1, m_col2, m_col3 = st.columns([2, 2, 1])
        with m_col1:
            manual_event_type = st.selectbox(
                "Event Type",
                options=available_scenarios,
                key="manual_event_type",
            )
        with m_col2:
            manual_impact = st.slider(
                "Impact Score",
                min_value=1.0,
                max_value=10.0,
                value=8.0,
                step=0.5,
                key="manual_impact",
            )
        with m_col3:
            manual_sentiment = st.slider(
                "Sentiment",
                min_value=-1.0,
                max_value=1.0,
                value=-0.5,
                step=0.1,
                key="manual_sentiment",
            )

        if st.button("▶ Run Stress Test", type="primary", key="run_manual_stress"):
            with st.spinner("Running stress test…"):
                stress_result, reverse_result = run_manual_stress(
                    event_type=manual_event_type,
                    impact=manual_impact,
                    sentiment=manual_sentiment,
                )

            if stress_result is None:
                st.error("Stress test failed to produce a result.")
            else:
                st.success(
                    f"Stress test complete — "
                    f"ΔNAV: **{_fmt_m(stress_result.delta_nav)}** "
                    f"({_fmt_pct(stress_result.delta_pct)})"
                )
                # Store result in session state so it shows in the main panel
                st.session_state["manual_stress_result"] = stress_result
                st.session_state["manual_reverse_result"] = reverse_result
                st.rerun()

    # Prefer manual result if available
    manual_sr = st.session_state.get("manual_stress_result")
    manual_rr = st.session_state.get("manual_reverse_result")

    # ── Determine display result: manual or most recent DB run ────────────────
    if manual_sr is not None:
        # Reconstruct display dict from StressResult dataclass
        current = {
            "scenario_name": manual_sr.event_type,
            "ts_created": manual_sr.ts_created.isoformat() if hasattr(manual_sr.ts_created, "isoformat") else str(manual_sr.ts_created),
            "results": {
                "pre_nav": manual_sr.pre_nav,
                "post_nav": manual_sr.post_nav,
                "delta_nav": manual_sr.delta_nav,
                "delta_pct": manual_sr.delta_pct,
                "breakdown_by_asset_class": manual_sr.breakdown_by_asset_class,
                "breakdown_by_sector": manual_sr.breakdown_by_sector,
                "mc_expected_loss": manual_sr.mc_expected_loss,
                "mc_worst_loss": manual_sr.mc_worst_loss,
            },
            "meta": {
                "event_type": manual_sr.event_type,
                "impact": manual_sr.impact,
            },
            "pnl_impact": manual_sr.delta_nav,
            "var_95": manual_sr.mc_var_95,
            "cvar_95": manual_sr.mc_es_95,
        }
        reverse_result = manual_rr
    elif stress_runs:
        current = stress_runs[0]
        reverse_result = None
    else:
        current = None
        reverse_result = None

    if current is None:
        st.info(
            "No stress test results yet. "
            "Stress tests are triggered automatically when a signal with "
            "impact ≥ 7 arrives, or you can run one manually above."
        )
        return

    st.divider()
    st.markdown("#### Latest Stress Run")

    # ── Scenario card ─────────────────────────────────────────────────────────
    meta = current.get("meta", {})
    results = current.get("results", {})
    event_type = meta.get("event_type") or current.get("scenario_name", "—")
    impact_val = meta.get("impact", 0.0)
    ts_str = str(current.get("ts_created", ""))[:19].replace("T", " ")

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.info(f"**Scenario:** {event_type}")
    with info_col2:
        st.info(f"**Impact:** {impact_val:.1f}/10")
    with info_col3:
        st.info(f"**Run at:** {ts_str} UTC")

    # ── Row 1: NAV metrics ────────────────────────────────────────────────────
    st.markdown("##### Before / After NAV")

    pre_nav = results.get("pre_nav") or current.get("pre_nav")
    post_nav = results.get("post_nav") or current.get("post_nav")
    delta_nav = results.get("delta_nav") or current.get("pnl_impact")
    delta_pct = results.get("delta_pct") or current.get("delta_pct")

    nav_c1, nav_c2, nav_c3, nav_c4 = st.columns(4)

    with nav_c1:
        st.metric(
            "Pre-NAV",
            f"${pre_nav / 1e6:.2f}M" if pre_nav else "—",
        )
    with nav_c2:
        st.metric(
            "Post-NAV",
            f"${post_nav / 1e6:.2f}M" if post_nav else "—",
        )
    with nav_c3:
        delta_fmt = f"${delta_nav / 1e6:+.2f}M" if delta_nav is not None else "—"
        st.metric(
            "ΔNAV",
            delta_fmt,
            delta=delta_fmt if delta_nav is not None else None,
        )
    with nav_c4:
        pct_fmt = f"{delta_pct * 100:+.2f}%" if delta_pct is not None else "—"
        st.metric(
            "Δ%",
            pct_fmt,
            delta=pct_fmt if delta_pct is not None else None,
        )

    # ── Row 2: Asset class waterfall ──────────────────────────────────────────
    st.markdown("##### Asset Class P&L Waterfall")

    breakdown_ac = results.get("breakdown_by_asset_class", {})
    if isinstance(breakdown_ac, str):
        try:
            breakdown_ac = json.loads(breakdown_ac)
        except Exception:
            breakdown_ac = {}

    fig_waterfall = make_waterfall_chart(
        breakdown_ac, title="P&L Breakdown by Asset Class"
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # ── Row 3: Monte Carlo loss distribution ──────────────────────────────────
    var_95 = current.get("var_95")
    es_95 = current.get("cvar_95")
    mc_exp_loss = results.get("mc_expected_loss")
    mc_worst = results.get("mc_worst_loss")

    if var_95 is not None:
        st.markdown("##### Monte Carlo Loss Distribution")
        # Reconstruct a synthetic loss sample for the histogram using
        # var_95, es_95, and expected_loss as distribution anchors
        import numpy as np

        rng = np.random.default_rng(seed=42)
        # Approximate: losses ~ truncated normal anchored to expected_loss
        exp_loss = mc_exp_loss or abs(delta_nav or 0)
        worst = mc_worst or abs(var_95) * 1.5 if var_95 else abs(exp_loss) * 2
        std_est = abs(var_95 - exp_loss) / 1.645 if var_95 and exp_loss else abs(exp_loss) * 0.3
        losses = rng.normal(loc=-abs(exp_loss), scale=max(abs(std_est), 1.0), size=2000).tolist()

        fig_mc = make_loss_distribution_chart(
            losses=losses,
            var_95=-abs(var_95) if var_95 else None,
            es_95=-abs(es_95) if es_95 else None,
            det_loss=delta_nav,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # ── Row 4: Sector exposure heatmap ────────────────────────────────────────
    breakdown_sector = results.get("breakdown_by_sector", {})
    if isinstance(breakdown_sector, str):
        try:
            breakdown_sector = json.loads(breakdown_sector)
        except Exception:
            breakdown_sector = {}

    if breakdown_sector:
        st.markdown("##### Sector P&L Contribution")
        fig_sector = make_sector_heatmap(breakdown_sector)
        st.plotly_chart(fig_sector, use_container_width=True)

    # ── Row 5: Reverse stress result ─────────────────────────────────────────
    if reverse_result is not None:
        st.markdown("##### Reverse Stress Test")

        st.info(f"🔍 **{reverse_result.narrative}**")

        gauge_col, _ = st.columns([1, 1])
        with gauge_col:
            fig_gauge = make_gauge_chart(
                value=reverse_result.breach_impact,
                min_val=1.0,
                max_val=10.0,
                title="Breach Threshold Impact",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Row 6: Historical stress runs table ───────────────────────────────────
    st.markdown("##### Historical Stress Runs (last 10)")

    if stress_runs:
        hist_rows: list[dict] = []
        for run in stress_runs:
            r_meta = run.get("meta", {})
            r_results = run.get("results", {})
            hist_rows.append(
                {
                    "Time (UTC)": str(run.get("ts_created", ""))[:19].replace("T", " "),
                    "Scenario": run.get("scenario_name", "—"),
                    "ΔNAV": _fmt_m(run.get("pnl_impact")),
                    "Δ%": _fmt_pct(r_results.get("delta_pct")),
                    "VaR 95%": _fmt_m(run.get("var_95")),
                    "ES 95%": _fmt_m(run.get("cvar_95")),
                }
            )

        st.dataframe(
            pd.DataFrame(hist_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time (UTC)": st.column_config.TextColumn(width="medium"),
                "Scenario": st.column_config.TextColumn(width="medium"),
                "ΔNAV": st.column_config.TextColumn(width="small"),
                "Δ%": st.column_config.TextColumn(width="small"),
                "VaR 95%": st.column_config.TextColumn(width="small"),
                "ES 95%": st.column_config.TextColumn(width="small"),
            },
        )
    else:
        st.info("No historical stress runs in the database yet.")
