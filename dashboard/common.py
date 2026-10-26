"""Shared data loading, economics state and chart styling for the retail churn dashboard."""
from __future__ import annotations
 
import json
import sys
from dataclasses import replace
from pathlib import Path
 
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
 
from src import configs  # noqa: E402
from src.data_preprocessing import named_queries, run_named_query  # noqa: E402
from src.economics import (expected_loss, expected_net_gain, optimal_threshold,  # noqa: E402
                           threshold_curve)
from src.predict import load_bundle, shap_frame  # noqa: E402
 
# --------------------------------------------------------------------------- #
# Retail value basis. Telco used monthly_charges; retail has no recurring bill,
# so a customer's monthly value is their average spend per month, derived from
# customer_features.spend_per_year (see 01_create_schema.sql). The economics
# module still receives a "monthly" series, so src/economics.py is unchanged.
# --------------------------------------------------------------------------- #
CHURN_COL = "churn"              # 0/1 label used throughout the dashboard
VALUE_COL = "monthly_spend"      # spend_per_year / 12
 
# --------------------------------------------------------------------------- #
# Palette (dark theme) - validated colour-blind-safe categorical order in its
# dark-surface steps, blue<->red diverging for SHAP, status colours for risk.
# All series clear 3:1 contrast on the #0a0f1e background; text clears 4.5:1.
# --------------------------------------------------------------------------- #
INK, INK_2, INK_3 = "#e8ecf6", "#a4adc4", "#7c87a3"
GRID, SURFACE, CARD = "rgba(164,173,196,0.12)", "#0a0f1e", "#131b30"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
BAR, BAR_MUTED = SERIES[0], "#35527d"
STAYED, CHURNED = SERIES[0], SERIES[1]
RISK_UP, RISK_DOWN = "#e66767", "#3987e5"
RISK_COLORS = {"HIGH": "#d03b3b", "MEDIUM": "#ec835a", "LOW": "#0ca30c"}
RISK_ORDER = ["HIGH", "MEDIUM", "LOW"]
# One hue, dark -> brighter. Capped at a mid-blue so white cell labels stay readable.
SEQUENTIAL = [[0, "#16233d"], [1, "#2a78d6"]]
 
 
def style(fig: go.Figure, height: int = 340, title: str | None = None,
          legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, title=title,
        margin=dict(l=8, r=16, t=48 if title else 16, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", color=INK_2, size=12),
        title_font=dict(family="Space Grotesk, Inter, sans-serif", size=15, color=INK),
        title_x=0, title_xanchor="left",
        hoverlabel=dict(bgcolor=CARD, font_color=INK, bordercolor="#2c3a5c",
                        font_family="Inter, sans-serif"),
        showlegend=legend, bargap=0.35,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title_text="",
                    font=dict(color=INK_2)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="")
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="")
    return fig
 
 
def gauge(probability: float, threshold: float | None = None, height: int = 230,
          title: str = "Churn probability") -> go.Figure:
    """Radial risk gauge: bands for LOW / MEDIUM / HIGH, a marker for the contact threshold."""
    level = configs.risk_level(probability)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number=dict(suffix="%", valueformat=".0f", font=dict(size=42, color=INK,
                                                             family="Space Grotesk, sans-serif")),
        title=dict(text=title, font=dict(size=13, color=INK_2)),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%", tickcolor=INK_3, tickwidth=1,
                      tickfont=dict(color=INK_3, size=10)),
            bar=dict(color=RISK_COLORS[level], thickness=0.28),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 30], color="rgba(12,163,12,0.10)"),
                   dict(range=[30, 60], color="rgba(236,131,90,0.12)"),
                   dict(range=[60, 100], color="rgba(208,59,59,0.14)")],
            threshold=None if threshold is None else dict(
                line=dict(color=INK, width=2), thickness=0.8, value=threshold * 100),
        )))
    fig.update_layout(height=height, margin=dict(l=24, r=24, t=48, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"))
    return fig
 
 
def hbar(labels, values, text=None, highlight_max=True, value_format="{:.1f}%",
         hover=None) -> go.Figure:
    values = list(values)
    colors = [BAR if (not highlight_max or v == max(values)) else BAR_MUTED for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=list(labels), orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=text if text is not None else [value_format.format(v) for v in values],
        textposition="outside", cliponaxis=False,
        hovertemplate=hover or "%{y}: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, max(values) * 1.22])   # room for the value labels
    return fig
 
 
def chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displayModeBar": False})
 
 
def show_sql(name: str) -> None:
    with st.expander("SQL behind this chart"):
        st.code(named_queries()[name], language="sql")
 
 
def money(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
 
 
def money_short(v: float) -> str:
    """Compact money for KPI tiles: $820.4K, $1.2M."""
    sign, a = ("-" if v < 0 else ""), abs(v)
    if a >= 1e6:
        return f"{sign}${a / 1e6:.2f}M"
    if a >= 1e5:
        return f"{sign}${a / 1e3:,.0f}K"
    if a >= 1e4:
        return f"{sign}${a / 1e3:.1f}K"
    return f"{sign}${a:,.0f}"
 
 
def md(text: str) -> str:
    """Escape dollar signs: Streamlit markdown renders text between two '$' as LaTeX."""
    return text.replace("$", r"\$")
 
 
# --------------------------------------------------------------------------- #
# Retail value basis helpers
# --------------------------------------------------------------------------- #
def _add_retail_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee `churn` and `monthly_spend` exist, whatever the scored file calls them."""
    df = df.copy()
    if CHURN_COL not in df.columns and "target_churn" in df.columns:
        df[CHURN_COL] = df["target_churn"]
    if VALUE_COL not in df.columns:
        if "spend_per_year" in df.columns:
            annual = df["spend_per_year"]
        else:  # same rule as the customer_features view
            years = df["years_as_customer"].where(df["years_as_customer"] > 0)
            annual = (df["total_spend"] / years).fillna(df["total_spend"])
        df[VALUE_COL] = annual / 12
    return df
 
 
# --------------------------------------------------------------------------- #
# Cached artifacts
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_scored() -> pd.DataFrame:
    return _add_retail_columns(pd.read_csv(configs.SCORED_CSV))
 
 
@st.cache_data(show_spinner=False)
def load_report() -> dict:
    return json.loads(configs.METRICS_JSON.read_text(encoding="utf-8"))
 
 
@st.cache_data(show_spinner=False)
def load_threshold_report() -> dict:
    return json.loads(configs.THRESHOLD_JSON.read_text(encoding="utf-8"))
 
 
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(configs.REPORTS_DIR / name)
 
 
@st.cache_data(show_spinner=False)
def load_llm_examples() -> list:
    return json.loads(configs.LLM_EXAMPLES_JSON.read_text(encoding="utf-8"))
 
 
@st.cache_data(show_spinner=False)
def query(name: str) -> pd.DataFrame:
    return run_named_query(name)
 
 
@st.cache_resource(show_spinner=False)
def bundle() -> dict:
    return load_bundle()
 
 
@st.cache_resource(show_spinner=False)
def explainer():
    from src.explain import ChurnExplainer
    b = bundle()
    return ChurnExplainer(b["pipeline"], b["background"])
 
 
@st.cache_data(show_spinner=False)
def shap_values() -> pd.DataFrame:
    return shap_frame(load_scored())
 
 
# --------------------------------------------------------------------------- #
# Campaign economics shared across pages (set on the Economics page)
# --------------------------------------------------------------------------- #
def economics() -> configs.Economics:
    return st.session_state.get("economics", configs.ECONOMICS)
 
 
@st.cache_data(show_spinner=False)
def _optimal_threshold(offer_cost, save_rate, horizon, margin) -> float:
    s = load_scored()
    econ = configs.Economics(offer_cost, save_rate, horizon, margin)
    return optimal_threshold(s[CHURN_COL], s["churn_probability"], s[VALUE_COL], econ)
 
 
def threshold() -> float:
    """Money-optimal threshold for the current assumptions.
 
    With the default assumptions this is the threshold the pipeline chose on
    training out-of-fold scores; when a user changes the assumptions it is
    re-optimised on the out-of-fold scores of all customers.
    """
    econ = economics()
    if econ == configs.ECONOMICS:
        return float(bundle()["threshold"])
    return _optimal_threshold(econ.offer_cost, econ.save_rate, econ.horizon_months,
                              econ.gross_margin)
 
 
def with_economics(scored: pd.DataFrame) -> pd.DataFrame:
    """Recompute money columns and the contact flag under the current assumptions."""
    econ, t = economics(), threshold()
    out = _add_retail_columns(scored)
    out["customer_value"] = econ.customer_value(out[VALUE_COL])
    out["expected_loss"] = expected_loss(out["churn_probability"], out[VALUE_COL], econ)
    out["expected_net_gain"] = expected_net_gain(out["churn_probability"], out[VALUE_COL], econ)
    out["contact_recommended"] = out["churn_probability"] >= t
    return out
 
 
def active_customers() -> pd.DataFrame:
    """Customers who have not churned, highest expected loss first."""
    s = with_economics(load_scored())
    return s[s[CHURN_COL] == 0].sort_values("expected_loss", ascending=False)
 
 
def curve_for(econ: configs.Economics) -> pd.DataFrame:
    s = load_scored()
    return threshold_curve(s[CHURN_COL], s["churn_probability"], s[VALUE_COL], econ)
 
 
def economics_caption() -> str:
    e, t = economics(), threshold()
    return md(f"Assumptions: offer ${e.offer_cost:,.0f} per contact, {e.save_rate:.0%} of "
              f"contacted churners saved, value = {e.horizon_months} months of average spend x "
              f"{e.gross_margin:.0%} margin. Contact threshold {t:.2f}. Change them on the "
              "Campaign economics page.")
 
 
def economics_changed() -> bool:
    return economics() != configs.ECONOMICS
 
 
def replace_economics(**kwargs) -> None:
    st.session_state["economics"] = replace(economics(), **kwargs)
 
 
def risk_badge(level: str) -> str:
    icon = {"HIGH": ":material/error:", "MEDIUM": ":material/warning:", "LOW": ":material/check_circle:"}
    color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
    return f":{color[level]}-badge[{icon[level]} {level}]"
