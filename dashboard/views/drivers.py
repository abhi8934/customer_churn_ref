import plotly.graph_objects as go
import streamlit as st
 
from common import (BAR, INK_3, SERIES, chart, hbar, load_csv, load_report, load_scored,
                    money, style)
from src.configs import FEATURE_LABELS
from theme import hero, section
 
report = load_report()
all_scored = load_scored()
hero("Churn drivers & segments",
     f"Why customers leave, from SHAP on the champion model ({report['champion']}) - computed "
     f"out-of-fold for all {len(all_scored):,} customers and summed back from encoded columns to "
     "the business feature they came from - cross-checked with permutation importance.",
     eyebrow="Explainable AI",
     chips=["SHAP", "Permutation importance", "Dependence plots", "k-means segments"])
 
left, right = st.columns(2)
with left:
    imp = load_csv("global_importance.csv").head(12)
    fig = hbar(imp["label"], imp["mean_abs_shap"], highlight_max=False, value_format="{:.2f}",
               hover="%{y}: %{x:.3f}<extra></extra>")
    fig.update_xaxes(title_text="mean |SHAP| (log-odds)")
    chart(style(fig, 440, "SHAP: average impact on the prediction", legend=False))
with right:
    perm = load_csv("permutation_importance.csv").head(12)
    fig = go.Figure(go.Bar(
        x=perm["importance_mean"], y=[FEATURE_LABELS.get(f, f) for f in perm["feature"]],
        orientation="h", marker_color=BAR,
        error_x=dict(type="data", array=perm["importance_std"], color=INK_3, thickness=1),
        hovertemplate="%{y}: ROC-AUC drops %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="Drop in test ROC-AUC when shuffled")
    chart(style(fig, 440, "Permutation importance (test set)", legend=False))
 
top3 = ", ".join(str(x).lower() for x in imp["label"].head(3))
st.caption("Two independent methods - SHAP (how much each feature moves individual predictions) "
           "and permutation importance (how much accuracy is lost without it). SHAP ranks "
           f"{top3} highest. Correlated features (e.g. total spend, spend per year and "
           "purchases) share credit under permutation, so they can rank lower there than in SHAP.")
 
st.divider()
section("How the main drivers behave", "each point is one customer")
scored = all_scored.sample(min(3000, len(all_scored)), random_state=42)
GROUP = "promotion_response"          # colour dependence plots by this categorical feature
groups = sorted(scored[GROUP].dropna().unique(), key=str)
left, right = st.columns(2)
for col, feature, title in ((left, "satisfaction_score",
                             "Satisfaction effect by promotion response"),
                            (right, "last_purchase_days_ago",
                             "Purchase-recency effect by promotion response")):
    fig = go.Figure()
    for i, grp in enumerate(groups):
        d = scored[scored[GROUP] == grp]
        fig.add_trace(go.Scattergl(
            x=d[feature], y=d[f"shap__{feature}"], mode="markers", name=str(grp),
            marker=dict(color=SERIES[i % len(SERIES)], size=5, opacity=0.5),
            hovertemplate=f"{grp}<br>{FEATURE_LABELS[feature]} %{{x}}<br>"
                          "SHAP %{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title_text=FEATURE_LABELS[feature])
    fig.update_yaxes(title_text="SHAP (pushes to churn ->)")
    with col:
        chart(style(fig, 380, title))
st.caption("Each point is one customer; above zero means the feature raises their churn risk, "
           "below zero means it protects them. The point where a curve crosses zero is the "
           "value at which that feature stops hurting retention.")
 
st.divider()
section("Behavioural segments",
        "k-means on spend per year, purchases per year and days since last purchase")
seg = load_csv("segment_profile.csv")
left, right = st.columns([2, 3])
with left:
    fig = hbar(seg["segment"], seg["churn_rate"] * 100, hover="%{y}: %{x:.1f}% churn<extra></extra>")
    fig.update_xaxes(ticksuffix="%")
    chart(style(fig, 300, "Churn rate by segment", legend=False))
with right:
    show = seg.assign(
        share=seg["share_of_customers"], churn=seg["churn_rate"],
        expected_loss=seg["expected_loss_active"].map(money))[
        ["segment", "customers", "share", "churn", "avg_spend_per_year",
         "avg_purchases_per_year", "avg_days_since_purchase", "active_customers",
         "expected_loss"]]
    st.dataframe(show, hide_index=True, column_config={
        "segment": "Segment", "customers": "Customers",
        "share": st.column_config.NumberColumn("Share", format="percent"),
        "churn": st.column_config.NumberColumn("Churn rate", format="percent"),
        "avg_spend_per_year": st.column_config.NumberColumn("Avg spend / yr", format="dollar"),
        "avg_purchases_per_year": st.column_config.NumberColumn("Avg purchases / yr",
                                                                format="%.1f"),
        "avg_days_since_purchase": st.column_config.NumberColumn("Avg days since purchase",
                                                                 format="%.0f"),
        "active_customers": "Active", "expected_loss": "Expected loss (active)"})
    st.caption("Segments give the campaign team a vocabulary: the model scores individuals, "
               "segments decide which offer template and channel to use.")
