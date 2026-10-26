import json
 
import plotly.graph_objects as go
import streamlit as st
 
from common import (INK_3, RISK_DOWN, RISK_ORDER, RISK_UP, active_customers, bundle, chart,
                    economics, economics_caption, gauge, md, money, money_short, risk_badge,
                    shap_values, style, threshold)
from src.configs import FEATURE_LABELS
from src.explain import explain_customer
from src.llm import build_context, generate_brief, llm_available
from src.scenarios import next_best_actions
from theme import action_card, hero, section
 
hero("Retention targets",
     "Active customers ranked by expected loss (churn probability x customer value), so the team "
     "contacts the customers where the money is first. Click a row for the full story.",
     eyebrow="Who to contact, why, and what to offer",
     chips=["SHAP reasons", "Next-best action", "AI-written brief", "CSV export"])
st.caption(economics_caption())
 
active = active_customers()
 
f1, f2, f3, f4 = st.columns([1.2, 1.4, 1.6, 1])
risk = f1.pills("Risk level", RISK_ORDER, selection_mode="multi", default=["HIGH", "MEDIUM"])
promos = f2.multiselect("Promotion response", sorted(active["promotion_response"].dropna().unique()))
segments = f3.multiselect("Segment", sorted(active["segment"].unique()))
only_contact = f4.toggle("Recommended for contact only", value=True)
 
view = active
if risk:
    view = view[view["risk_level"].isin(risk)]
if promos:
    view = view[view["promotion_response"].isin(promos)]
if segments:
    view = view[view["segment"].isin(segments)]
if only_contact:
    view = view[view["contact_recommended"]]
 
econ = economics()
m = st.columns(4)
m[0].metric("Customers in list", f"{len(view):,}", border=True)
m[1].metric("Expected margin loss", money_short(view["expected_loss"].sum()), border=True)
m[2].metric("Campaign cost", money_short(len(view) * econ.offer_cost), border=True)
m[3].metric("Net value", money_short(view["expected_net_gain"].sum()), border=True,
            help="Expected net value of contacting everyone in the list.")
 
table = view[["customer_id", "churn_probability", "risk_level", "expected_loss",
              "expected_net_gain", "satisfaction_score", "years_as_customer", "spend_per_year",
              "last_purchase_days_ago", "reason_1", "reason_2", "reason_3",
              "recommended_action", "segment"]].reset_index()
event = st.dataframe(
    table.drop(columns="index"), hide_index=True, height=360, on_select="rerun",
    selection_mode="single-row", key="targets_table",
    column_config={
        "customer_id": "Customer",
        "churn_probability": st.column_config.ProgressColumn("P(churn)", format="percent",
                                                             min_value=0, max_value=1),
        "risk_level": "Risk",
        "expected_loss": st.column_config.NumberColumn("Expected loss", format="dollar"),
        "expected_net_gain": st.column_config.NumberColumn("Net gain if contacted", format="dollar"),
        "satisfaction_score": st.column_config.NumberColumn("Satisfaction"),
        "years_as_customer": st.column_config.NumberColumn("Years as customer", format="%.1f"),
        "spend_per_year": st.column_config.NumberColumn("Spend / yr", format="dollar"),
        "last_purchase_days_ago": st.column_config.NumberColumn("Days since purchase"),
        "reason_1": "Reason 1", "reason_2": "Reason 2", "reason_3": "Reason 3",
        "recommended_action": "Recommended action", "segment": "Segment",
    })
st.download_button("Download this list (CSV)", table.drop(columns="index").to_csv(index=False),
                   file_name="retention_targets.csv", mime="text/csv", icon=":material/download:")
 
if table.empty:
    st.stop()
 
# --------------------------------------------------------------------------- #
# Customer detail
# --------------------------------------------------------------------------- #
section("Customer 360", "click a row above, or pick a customer")
selected_rows = event.selection.rows if event and event.selection else []
default_id = table.loc[selected_rows[0], "customer_id"] if selected_rows else table["customer_id"].iloc[0]
ids = table["customer_id"].tolist()
customer_id = st.selectbox("Customer", ids, index=ids.index(default_id),
                           label_visibility="collapsed")
idx = table.loc[table["customer_id"] == customer_id, "index"].iloc[0]
row = active.loc[idx]
shap_row = shap_values().loc[idx]
explanation = explain_customer(shap_row, row, bundle()["reference"])
 
st.subheader(f"{customer_id}  {risk_badge(row['risk_level'])}")
g, left, right = st.columns([1.15, 1.45, 2])
with g:
    st.plotly_chart(gauge(row["churn_probability"], threshold()), width="stretch", theme=None,
                    config={"displayModeBar": False})
    st.metric("Expected loss", money(row["expected_loss"]), border=True,
              help="Churn probability x margin value of this customer over the planning horizon.")
with left:
    st.markdown(md(
        f"**{row['years_as_customer']:.1f} years** as a customer, "
        f"**${row['spend_per_year']:,.2f}/year** spend across "
        f"**{int(row['num_of_purchases'])} purchases**, last purchase "
        f"**{int(row['last_purchase_days_ago'])} days ago**. Satisfaction "
        f"**{row['satisfaction_score']:g}**, {int(row['num_of_support_contacts'])} support "
        f"contact{'s' if int(row['num_of_support_contacts']) != 1 else ''}. "
        f"Segment: *{row['segment']}*."))
    st.markdown("**Why they are at risk**")
    st.markdown(md("\n".join(f"{i}. {f['text']}"
                             for i, f in enumerate(explanation["risk_factors"], 1))))
    if explanation["protective_factors"]:
        st.markdown("**What keeps them**")
        st.markdown(md("\n".join(f"- {f['text']}" for f in explanation["protective_factors"])))
    st.markdown("**Recommended action**")
    for a_ in explanation["actions"]:
        st.markdown(f"- {a_}")
 
with right:
    top = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index).head(10)[::-1]
    labels = [f"{FEATURE_LABELS.get(f, f)} = {row[f]:.2f}" if isinstance(row[f], float)
              else f"{FEATURE_LABELS.get(f, f)} = {row[f]}" for f in top.index]
    fig = go.Figure(go.Bar(
        x=top.values, y=labels, orientation="h",
        marker_color=[RISK_UP if v > 0 else RISK_DOWN for v in top.values],
        hovertemplate="%{y}<br>impact %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=INK_3, line_width=1)
    fig.update_xaxes(title_text="SHAP impact (red = raises risk)")
    chart(style(fig, 400, "What drives this prediction", legend=False))
 
# --------------------------------------------------------------------------- #
# Next best action: every lever this customer is eligible for, re-scored
# --------------------------------------------------------------------------- #
section("Next best action", "each offer tested on this customer with the live model")
 
 
@st.cache_data(show_spinner=False)
def _actions(customer: str, econ_values: tuple):
    from src.configs import Economics
    return next_best_actions(active.loc[idx], bundle()["pipeline"], econ=Economics(*econ_values))
 
 
actions = _actions(customer_id, (econ.offer_cost, econ.save_rate, econ.horizon_months,
                                 econ.gross_margin))
if actions.empty:
    st.caption("No modelled offer applies to this customer.")
else:
    shown = actions.head(4)
    best_value = (shown["net_if_accepted"].idxmax()
                  if shown["net_if_accepted"].max() > 0 else None)
    for rank, (col, (i, a_)) in enumerate(zip(st.columns(4), shown.iterrows()), 1):
        label = "Biggest risk drop" if rank == 1 else f"Option {rank}"
        if i == best_value:
            label += " · best value"
        with col:
            action_card(label, a_["action"], row["churn_probability"],
                        max(row["churn_probability"] - a_["risk_drop"], 0.0),
                        a_["margin_saved"], a_["offer_cost"], a_["net_if_accepted"],
                        highlight=i == best_value)
    st.caption("Ranked by how much each offer lowers this customer's risk; the green card is the "
               "one that pays for itself best if accepted. Risk after the offer = today's score "
               "minus the change the production model predicts for the new profile - a model-based "
               "estimate, not a guarantee. Use the Scenario Lab to price an offer across many "
               "customers.")
 
# --------------------------------------------------------------------------- #
# LLM brief
# --------------------------------------------------------------------------- #
section("Retention brief", "plain-English summary and a ready-to-send message")
 
 
@st.cache_data(show_spinner=False)
def _brief(ctx_json: str, use_llm: bool) -> dict:
    return generate_brief(json.loads(ctx_json), use_llm=use_llm)
 
 
ctx = build_context(row, explanation)
ctx_json = json.dumps(ctx, sort_keys=True)
if llm_available():
    if st.button("Write brief with Claude", icon=":material/auto_awesome:", type="primary"):
        st.session_state[f"brief_{customer_id}"] = True
    use_llm = st.session_state.get(f"brief_{customer_id}", False)
else:
    use_llm = False
    st.caption("No ANTHROPIC_API_KEY configured - showing the deterministic template. "
               "Add the key to generate the brief with Claude.")
 
with st.spinner("Writing brief..."):
    brief = _brief(ctx_json, use_llm)
c1, c2 = st.columns(2)
with c1.container(key="glass_brief_manager"):
    st.markdown("**For the account manager**")
    st.markdown(md(brief["summary"]))
with c2.container(key="glass_brief_customer"):
    st.markdown("**Message to the customer**")
    st.markdown(md(brief["customer_message"]))
st.caption(f"Source: {brief['source']}. The brief uses only the facts below - the model's "
           "probability, SHAP reasons and approved playbook actions.")
with st.expander("Facts sent to the LLM"):
    st.json(ctx)
