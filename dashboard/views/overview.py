import plotly.graph_objects as go
import streamlit as st

from common import (RISK_COLORS, RISK_ORDER, active_customers, chart, economics,
                    economics_caption, hbar, load_csv, load_report, money, money_short, query,
                    show_sql, style, threshold)
from src import configs as config
from theme import callout, hero, section

report = load_report()
champ = report["champion"]
auc = report["models"][champ]["test_at_0.5"]["roc_auc"]

active = active_customers()          # sorted by expected loss, highest first
flagged = active[active["contact_recommended"]]
kpi = query("kpi_overview").iloc[0]
econ, t = economics(), threshold()

# Retail: revenue is annual spend per customer (no monthly bill)
revenue_at_risk = float((active["churn_probability"] * active["spend_per_year"]).sum())
campaign_value = float(flagged["expected_net_gain"].sum())
loss_share = flagged["expected_loss"].sum() / active["expected_loss"].sum()

hero("Customer Retention Command Center",
     "Which customers are about to stop buying, why, and what it is worth to keep them - "
     "predicted by a machine-learning model, explained customer by customer, and priced in "
     "dollars.",
     eyebrow="Churn intelligence · live model", live=True,
     chips=[f"<b>{champ}</b> · ROC-AUC {auc:.3f}",
            f"<b>{int(kpi['customers']):,}</b> customers",
            f"Contact threshold <b>{t:.2f}</b>",
            "Out-of-fold scores", "SHAP explanations", "Claude briefs"])

cols = st.columns(5)
cols[0].metric("Customers", f"{int(kpi['customers']):,}", border=True)
cols[1].metric("Churn rate", f"{kpi['churn_rate_pct']:.1f}%", border=True,
               help=f"Customers labelled as churned: {int(kpi['churned']):,}. They account for "
                    f"{kpi['revenue_churn_pct']:.1f}% of total customer spend.")
cols[2].metric("To contact", f"{len(flagged):,}", border=True,
               help=f"Active customers with churn probability >= {t:.2f}, the threshold that "
                    "maximises campaign value under the current assumptions.")
cols[3].metric("At risk / yr", money_short(revenue_at_risk), border=True,
               help=f"Revenue at risk: {money(revenue_at_risk)} per year - sum over active "
                    "customers of P(churn) x annual spend.")
cols[4].metric("Net value", money_short(campaign_value), border=True,
               help="Expected campaign value of contacting them: sum of P(churn) x save rate x "
                    "customer value - offer cost.")

callout(f"<b>{len(flagged):,} of {len(active):,} active customers</b> "
        f"({len(flagged) / len(active):.0%}) carry <b>{loss_share:.0%} of the expected margin "
        f"loss</b>. Contacting them costs about {money(len(flagged) * econ.offer_cost)} and is "
        f"expected to return <b>{money(campaign_value)}</b> net over {econ.horizon_months} months.")
st.caption(economics_caption())

section("Launchpad", "jump straight to the job you came to do")
cards = [
    ("views/targets.py", ":material/target:", "Call list",
     "Ranked customers with SHAP reasons, next-best actions and AI-written briefs."),
    ("views/scenarios.py", ":material/science:", "Scenario Lab",
     "Simulate loyalty offers, discounts or win-back campaigns and see churn avoided and ROI."),
    ("views/copilot.py", ":material/auto_awesome:", "AI Copilot",
     "Ask the data in plain English, or run your own SQL in the SQL Lab."),
    ("views/predict.py", ":material/upload_file:", "Score a file",
     "Upload new customers, get risk scores, reasons and a data-health check."),
]
for col, (page, icon, title, desc) in zip(st.columns(4), cards):
    with col.container(key=f"card_{title.lower().replace(' ', '_')}"):
        st.markdown(f"##### {icon} {title}")
        st.caption(desc)
        st.page_link(page, label="Open", icon=":material/arrow_forward:")

section("Where the risk sits")
left, mid, right = st.columns(3)
with left:
    by_risk = (active.groupby("risk_level")
               .agg(customers=("customer_id", "size"), expected_loss=("expected_loss", "sum"))
               .reindex(RISK_ORDER).fillna(0))
    fig = go.Figure(go.Bar(
        x=[f"{r}<br>n={int(n):,}" for r, n in zip(by_risk.index, by_risk["customers"])],
        y=by_risk["expected_loss"],
        marker=dict(color=[RISK_COLORS[r] for r in by_risk.index], line=dict(width=0)),
        text=[money(v) for v in by_risk["expected_loss"]],
        textposition="outside", cliponaxis=False, textfont=dict(size=12),
        hovertemplate="%{x}: %{y:$,.0f} expected loss<extra></extra>"))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f",
                     range=[0, by_risk["expected_loss"].max() * 1.25])
    fig.update_xaxes(tickangle=0)
    chart(style(fig, 360, "Expected margin loss by risk level", legend=False))
    hi, med = config.RISK_BANDS[0][0], config.RISK_BANDS[1][0]   # bands come from the config
    st.caption(f"Active customers. HIGH >= {hi:.0%}, MEDIUM {med:.0%}-{hi:.0%}, "
               f"LOW < {med:.0%} churn probability.")

with mid:
    df = query("churn_by_satisfaction")
    fig = hbar(df["segment"].astype(str), df["churn_rate_pct"],
               hover="%{y}: %{x:.1f}% churn<extra></extra>")
    fig.update_xaxes(ticksuffix="%")
    chart(style(fig, 360, "Churn rate by satisfaction score", legend=False))
    show_sql("churn_by_satisfaction")

with right:
    imp = load_csv("global_importance.csv").head(8)
    fig = hbar(imp["label"], imp["mean_abs_shap"], highlight_max=False, value_format="{:.2f}",
               hover="%{y}: %{x:.3f}<extra></extra>")
    chart(style(fig, 360, "Top churn drivers (mean |SHAP|)", legend=False))

section("Call these customers first", "ranked by expected loss = P(churn) x customer value")
top = flagged.head(10)[["customer_id", "churn_probability", "expected_loss", "satisfaction_score",
                        "years_as_customer", "spend_per_year", "last_purchase_days_ago",
                        "reason_1", "reason_2", "recommended_action"]]
st.dataframe(
    top, hide_index=True,
    column_config={
        "customer_id": "Customer",
        "churn_probability": st.column_config.ProgressColumn(
            "P(churn)", format="percent", min_value=0, max_value=1),
        "expected_loss": st.column_config.NumberColumn("Expected loss", format="dollar"),
        "satisfaction_score": st.column_config.NumberColumn("Satisfaction"),
        "years_as_customer": st.column_config.NumberColumn("Years as customer", format="%.1f"),
        "spend_per_year": st.column_config.NumberColumn("Spend / yr", format="dollar"),
        "last_purchase_days_ago": st.column_config.NumberColumn("Days since purchase"),
        "reason_1": "Main reason",
        "reason_2": "Second reason",
        "recommended_action": "Recommended action",
    })
st.page_link("views/targets.py", label="Full retention list with explanations",
             icon=":material/arrow_forward:")