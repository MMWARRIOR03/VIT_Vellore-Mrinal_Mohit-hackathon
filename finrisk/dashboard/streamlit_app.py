"""
FinRisk Intelligence Dashboard — main Streamlit entry point.

Run with:
    streamlit run dashboard/streamlit_app.py --server.port 8501

from the finrisk/ directory.

Auto-refresh: uses streamlit-autorefresh if installed, falls back to
a manual time.sleep + st.rerun() approach.

All data reads directly from DuckDB — no FastAPI calls needed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ── sys.path setup ────────────────────────────────────────────────────────────
# Keep the project root before Streamlit's script directory. This ensures that
# imports such as `app.schema` resolve to finrisk/app/, never to this dashboard.
_FINRISK_ROOT = Path(__file__).resolve().parents[1]
_root_string = str(_FINRISK_ROOT)
if _root_string in sys.path:
    sys.path.remove(_root_string)
sys.path.insert(0, _root_string)

# Change working directory to finrisk/ so relative paths (e.g. ./data/finrisk.duckdb) work
os.chdir(_FINRISK_ROOT)

# ── Page config — MUST be the first Streamlit call ────────────────────────────
st.set_page_config(
    page_title="FinRisk Intelligence",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="collapsed",
)

# ── Autorefresh ───────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=5_000, limit=None, key="global_autorefresh")
    _USING_AUTOREFRESH = True
except ImportError:
    _USING_AUTOREFRESH = False

# ── Imports (after sys.path is set) ───────────────────────────────────────────
from dashboard.data_loader import DataLoader
from dashboard.tab_live_feed import render_live_feed
from dashboard.tab_event_explorer import render_event_explorer
from dashboard.tab_module_a import render_module_a
from dashboard.tab_module_b import render_module_b

# ── Global style injection ────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Dark sidebar */
    [data-testid="stSidebar"] { background-color: #0F172A; }
    /* Slightly tighten metric labels */
    [data-testid="stMetricLabel"] { font-size: 0.75rem; }
    /* Remove default top padding */
    .block-container { padding-top: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 📊")
with col_title:
    st.markdown("# FinRisk Intelligence")
    st.caption("Real-time financial risk monitoring · auto-refreshes every 5s")

if not _USING_AUTOREFRESH:
    st.caption("ℹ️ Install `streamlit-autorefresh` for automatic refresh. Refresh manually or add it to requirements.")

st.divider()

# ── DataLoader (shared across all tabs) ──────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_loader() -> DataLoader:
    """Single DataLoader instance for the session."""
    return DataLoader()


loader = get_loader()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_feed, tab_explorer, tab_module_a, tab_module_b = st.tabs(
    [
        "📡 Live Risk Feed",
        "🔍 Event Explorer",
        "⚖️ Module A: Index Rebalancer",
        "🧨 Module B: Stress Tester",
    ]
)

with tab_feed:
    render_live_feed(loader)

with tab_explorer:
    render_event_explorer(loader)

with tab_module_a:
    render_module_a(loader)

with tab_module_b:
    render_module_b(loader)

# ── Manual fallback refresh ───────────────────────────────────────────────────
if not _USING_AUTOREFRESH:
    if st.button("🔄 Refresh now", key="manual_refresh"):
        st.rerun()
