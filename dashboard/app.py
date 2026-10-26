"""Retail Churn Intelligence - Customer Churn Prediction & BI System (Streamlit).
 
    streamlit run dashboard/app.py        (or double-click run_dashboard.bat on Windows)
"""
from __future__ import annotations
 
import os
import sqlite3
import sys
from pathlib import Path
 
import streamlit as st
 
HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
 
st.set_page_config(page_title="Retail Churn Intelligence", page_icon=str(HERE / "assets" / "icon.svg"),
                   layout="wide", initial_sidebar_state="expanded")
 
# Streamlit Cloud: pick up the Claude key from secrets without ever showing it.
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:  # no secrets file locally - AI features fall back gracefully
    pass
 
from theme import inject_css  # noqa: E402
 
inject_css()
st.logo(str(HERE / "assets" / "logo.svg"), size="large", icon_image=str(HERE / "assets" / "icon.svg"))
 
 
@st.cache_data(show_spinner=False)
def customer_count() -> int | None:
    """Row count of the `customers` table (01_create_schema.sql), or None if the DB isn't found."""
    candidates = [os.environ.get("CHURN_DB_PATH")] + [
        str(HERE.parent / "data" / "retail_churn.db"),
        str(HERE.parent / "retail_churn.db"),
        str(HERE.parent / "data" / "churn.db"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
                    return con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            except sqlite3.Error:
                continue
    return None
 
 
views = HERE / "views"
nav = st.navigation({
    "Command center": [
        st.Page(views / "overview.py", title="Overview", icon=":material/space_dashboard:",
                url_path="overview", default=True),
        st.Page(views / "targets.py", title="Retention targets", icon=":material/target:",
                url_path="targets"),
        st.Page(views / "scenarios.py", title="Scenario Lab", icon=":material/science:",
                url_path="scenarios"),
        st.Page(views / "economics.py", title="Campaign economics", icon=":material/payments:",
                url_path="economics"),
    ],
    "Insights": [
        st.Page(views / "customers.py", title="Customer & churn analysis", icon=":material/groups:",
                url_path="analysis"),
        st.Page(views / "drivers.py", title="Churn drivers & segments", icon=":material/psychology:",
                url_path="drivers"),
        st.Page(views / "models.py", title="Model performance", icon=":material/model_training:",
                url_path="models"),
    ],
    "AI & tools": [
        st.Page(views / "copilot.py", title="AI Copilot & SQL Lab", icon=":material/auto_awesome:",
                url_path="copilot"),
        st.Page(views / "predict.py", title="Score new customers", icon=":material/upload_file:",
                url_path="predict"),
    ],
})
 
with st.sidebar:
    ai = "Claude connected" if os.environ.get("ANTHROPIC_API_KEY") else "AI: template mode"
    n = customer_count()
    dataset = f"Retail customers &middot; {n:,} records" if n else "Retail customer dataset"
    st.markdown(
        "<div class='side-foot'><span class='pulse' style='display:inline-block;margin-right:6px'>"
        f"</span>Model online &middot; XGBoost<br>{ai}<br><br>{dataset}. "
        "Every score is out-of-fold: produced by a model that never saw that customer.</div>",
        unsafe_allow_html=True)
 
nav.run()
