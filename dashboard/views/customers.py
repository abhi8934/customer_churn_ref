import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
from common import (BAR, CHURNED, SEQUENTIAL, STAYED, chart, hbar, load_scored, query, show_sql,
                    style)
from src.configs import FEATURE_LABELS, TARGET
from theme import hero, section
 
hero("Customer & churn analysis",
     "Who churns, sliced every way that matters. Every segment chart on this page is a named "
     "query in sql/03_business_queries.sql, run live against the SQLite analytical layer - "
     "open 'SQL behind this chart' to see it.",
     eyebrow="SQL-powered business intelligence",
     chips=["7 segment views", "Satisfaction x recency heatmap", "Churn timing", "Spending & demographics"])
 
kpi = query("kpi_overview").iloc[0]
overall = kpi["churn_rate_pct"]
 
# label shown in the UI -> named query in sql/03_business_queries.sql
DIMENSIONS = {
    "Satisfaction score": "churn_by_satisfaction",
    "Years as customer": "churn_by_tenure_bucket",
    "Purchase recency": "churn_by_recency_bucket",
    "Income band": "churn_by_income_band",
    "Age group": "churn_by_age_group",
    "Support contacts": "churn_by_support_contacts",
    "Promotion response": "churn_by_promotion_response",
}
dim = st.segmented_control("Break churn down by", list(DIMENSIONS), default="Satisfaction score",
                           required=True)
name = DIMENSIONS[dim]
df = query(name)
left, right = st.columns([3, 2])
with left:
    fig = hbar(df["segment"].astype(str), df["churn_rate_pct"],
               hover="%{y}: %{x:.1f}% churn<extra></extra>")
    fig.add_vline(x=overall, line_dash="dot", line_color="#8a8984",
                  annotation_text=f"all customers {overall:.1f}%", annotation_position="top")
    fig.update_xaxes(ticksuffix="%")
    chart(style(fig, 360, f"Churn rate by {dim.lower()}", legend=False))
with right:
    st.dataframe(df, hide_index=True)
    show_sql(name)
 
st.divider()
left, right = st.columns(2)
with left:
    # Satisfaction x recency: low satisfaction + long silence is the classic retail danger zone.
    # Row/column order comes from the ORDER BY in the SQL, so buckets stay in natural order.
    cx = query("churn_satisfaction_x_recency")
    rows_order = list(cx["satisfaction"].unique())
    cols_order = list(cx["recency_bucket"].unique())
    pivot = cx.pivot(index="satisfaction", columns="recency_bucket",
                     values="churn_rate_pct").loc[rows_order, cols_order]
    counts = cx.pivot(index="satisfaction", columns="recency_bucket",
                      values="customers").loc[rows_order, cols_order]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index, zmin=0, zmax=100,
        colorscale=SEQUENTIAL, colorbar=dict(ticksuffix="%", len=0.8),
        text=[[f"{v:.0f}%<br>n={n:,}" for v, n in zip(r1, r2)]
              for r1, r2 in zip(pivot.values, counts.values)],
        texttemplate="%{text}",
        hovertemplate="Satisfaction %{y} + last purchase %{x}: %{z:.1f}% churn<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    fig.update_xaxes(showgrid=False, title_text="Days since last purchase")
    chart(style(fig, 340, "Satisfaction x purchase recency: churn rate", legend=False))
    show_sql("churn_satisfaction_x_recency")
with right:
    # Cumulative share of all churners by customer age (whole years as a customer).
    ty = query("churn_by_tenure_year")
    fig = go.Figure(go.Scatter(x=ty["years_as_customer"], y=ty["cumulative_share_of_churners_pct"],
                               mode="lines+markers", line=dict(color=BAR, width=2.5),
                               hovertemplate="year %{x}: %{y:.1f}% of churners<extra></extra>"))
    for yr in (1, 3, 5):
        hit = ty.loc[ty["years_as_customer"] == yr, "cumulative_share_of_churners_pct"]
        if not hit.empty:
            v = hit.iloc[0]
            fig.add_annotation(x=yr, y=v, text=f"{v:.0f}% by year {yr}", showarrow=True,
                               arrowhead=0, ax=50, ay=10, font=dict(size=11))
    fig.update_yaxes(ticksuffix="%", range=[0, 102])
    fig.update_xaxes(title_text="Years as customer")
    chart(style(fig, 340, "When customers leave: cumulative share of churners", legend=False))
    show_sql("churn_by_tenure_year")
 
st.divider()
section("Spending behaviour and demographics")
scored = load_scored()
left, right = st.columns(2)
with left:
    spend = scored["spend_per_year"]
    size = max((spend.max() - spend.min()) / 20, 1e-9)  # shared bins so the two groups line up
    fig = go.Figure()
    for label, value, color in (("Stayed", 0, STAYED), ("Churned", 1, CHURNED)):
        fig.add_trace(go.Histogram(
            x=scored.loc[scored[TARGET] == value, "spend_per_year"], name=label,
            histnorm="percent", xbins=dict(start=spend.min(), end=spend.max() + size, size=size),
            marker_color=color,
            hovertemplate=f"{label}: %{{y:.1f}}% of group<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.08)
    fig.update_xaxes(title_text=FEATURE_LABELS["spend_per_year"] + " ($)", tickprefix="$")
    fig.update_yaxes(title_text="% of group", ticksuffix="%")
    chart(style(fig, 340, "Yearly spend: stayed vs churned"))
with right:
    rows = []
    # Age bands, then every categorical model feature (gender, promotion response, email opt-in)
    age_band = pd.cut(scored["age"], bins=[0, 24, 34, 44, 54, 64, 120],
                      labels=["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"])
    for key, rate in (scored.groupby(age_band, observed=True)[TARGET].mean() * 100).items():
        rows.append((f"Age: {key}", rate))
    for col in ("gender", "promotion_response", "email_opt_in"):
        for key, rate in (scored.groupby(col)[TARGET].mean() * 100).items():
            rows.append((f"{FEATURE_LABELS[col]}: {key}", rate))
    fig = hbar([r[0] for r in rows], [r[1] for r in rows], highlight_max=False,
               hover="%{y}: %{x:.1f}% churn<extra></extra>")
    fig.add_vline(x=overall, line_dash="dot", line_color="#8a8984",
                  annotation_text=f"all customers {overall:.1f}%", annotation_position="top")
    fig.update_xaxes(ticksuffix="%")
    chart(style(fig, 340, "Churn rate by demographic group", legend=False))
    st.caption("Compare each group against the dotted all-customer line; "
               "groups far from it are where churn is concentrated.")
