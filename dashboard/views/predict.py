from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    INK_3, RISK_DOWN, RISK_UP, bundle, chart, explainer, gauge, md, money,
    money_short, risk_badge, style,
)
from src.data_preprocessing import load_raw_csv, prepare_features
from src.predict import OUTPUT_COLUMNS, score_features, shap_frame
from theme import hero, section

# Support either src/configs.py or src/config.py.
try:
    config = importlib.import_module("src.configs")
except ModuleNotFoundError:
    config = importlib.import_module("src.config")


def cfg(name, default=None):
    return getattr(config, name, default)


def normalize_retail_columns(df):
    """Accept the original retail column names or their snake_case equivalents."""
    aliases = {
        "customer_id": "Customer_ID", "age": "Age", "gender": "Gender",
        "annual_income": "Annual_Income", "total_spend": "Total_Spend",
        "years_as_customer": "Years_as_Customer", "num_of_purchases": "Num_of_Purchases",
        "average_transaction_amount": "Average_Transaction_Amount",
        "num_of_returns": "Num_of_Returns",
        "num_of_support_contacts": "Num_of_Support_Contacts",
        "satisfaction_score": "Satisfaction_Score",
        "last_purchase_days_ago": "Last_Purchase_Days_Ago",
        "email_opt_in": "Email_Opt_In", "promotion_response": "Promotion_Response",
        "target_churn": "Target_Churn",
    }
    rename = {c: aliases[c.lower()] for c in df.columns if c.lower() in aliases}
    return df.rename(columns=rename)


def required_upload_columns():
    required = cfg("RAW_REQUIRED_COLUMNS", [
        "Customer_ID", "Age", "Gender", "Annual_Income", "Total_Spend",
        "Years_as_Customer", "Num_of_Purchases", "Average_Transaction_Amount",
        "Num_of_Returns", "Num_of_Support_Contacts", "Satisfaction_Score",
        "Last_Purchase_Days_Ago", "Email_Opt_In", "Promotion_Response", "Target_Churn",
    ])
    # Target_Churn is optional when scoring previously unseen customers.
    return [c for c in required if c not in {"Target_Churn", "target_churn", cfg("TARGET", "target_churn")}]


def score_retail(raw, model_bundle):
    features = prepare_features(raw)
    if features.empty:
        st.error("No customer rows remained after preprocessing.")
        st.stop()
    with st.spinner(f"Scoring {len(features):,} retail customers..."):
        scored = score_features(features, model_bundle, explainer=explainer())
    return features, scored


try:
    model_bundle = bundle()
except Exception as exc:
    st.error(f"Could not load the trained model bundle: {exc}")
    st.info("Train the retail churn model and check the model path in configs.py.")
    st.stop()

trained_at = model_bundle.get("trained_at", "unknown")
try:
    trained_label = pd.Timestamp(trained_at).strftime("%Y-%m-%d %H:%M UTC")
except Exception:
    trained_label = str(trained_at)

hero(
    "Score retail customers",
    f"Upload retail customer records or build a single customer profile. "
    f"{model_bundle.get('model_name', 'Retail churn model')} scores customers "
    "using the existing preprocessing and prediction pipeline.",
    eyebrow="Retail churn scoring",
    chips=[f"Contact threshold {float(model_bundle.get('threshold', 0.30)):.2f}",
           "Churn probability", "Risk level and retention economics"],
)
st.caption(f"Model training timestamp: {trained_label}")

tab_upload, tab_single = st.tabs([
    ":material/upload_file: Upload a customer file",
    ":material/person: Single retail customer",
])

with tab_upload:
    try:
        sample = normalize_retail_columns(load_raw_csv())
        target_names = {"Target_Churn", "target_churn", cfg("TARGET", "target_churn")}
        sample = sample.drop(columns=[c for c in target_names if c in sample.columns]).head(25)
        st.download_button(
            "Download retail sample CSV (25 customers)", sample.to_csv(index=False),
            file_name="sample_retail_customers.csv", mime="text/csv",
            icon=":material/download:",
        )
    except Exception as exc:
        st.caption(f"Sample CSV unavailable: {exc}")

    upload = st.file_uploader(
        "Upload a CSV using the retail columns defined in configs.py. Target_Churn is optional.",
        type=["csv"], key="retail_customer_upload",
    )
    if upload is not None:
        try:
            raw = normalize_retail_columns(pd.read_csv(upload))
            missing = [c for c in required_upload_columns() if c not in raw.columns]
            if missing:
                st.error("Missing required retail columns: " + ", ".join(missing))
                st.stop()

            features, scored = score_retail(raw, model_bundle)
            cols = st.columns(4)
            cols[0].metric("Customers scored", f"{len(scored):,}", border=True)
            cols[1].metric(
                "High risk",
                f"{int((scored['risk_level'] == 'HIGH').sum()):,}"
                if "risk_level" in scored else "—", border=True,
            )
            cols[2].metric(
                "Contact recommended",
                f"{int(scored['contact_recommended'].fillna(False).astype(bool).sum()):,}"
                if "contact_recommended" in scored else "—", border=True,
            )
            cols[3].metric(
                "Expected loss",
                money_short(pd.to_numeric(scored["expected_loss"], errors="coerce").sum())
                if "expected_loss" in scored else "—", border=True,
            )

            section("Customer scores")
            preferred = cfg("SCORE_TABLE_COLUMNS", OUTPUT_COLUMNS)
            display_cols = [c for c in preferred if c in scored.columns] or list(scored.columns)
            st.dataframe(scored[display_cols], hide_index=True, use_container_width=True)
            st.download_button(
                "Download retail churn scores", scored[display_cols].to_csv(index=False),
                file_name="retail_churn_scores.csv", mime="text/csv",
                icon=":material/download:", type="primary",
            )

            # Drift monitoring is optional; it must not prevent scoring.
            try:
                from src.monitoring import drift_report, overall_status
                ref_path = cfg("FEATURES_CSV")
                if ref_path and Path(ref_path).exists():
                    drift = drift_report(pd.read_csv(ref_path), features)
                    st.subheader("Data drift check")
                    st.metric("Feature drift status", overall_status(drift))
                    st.dataframe(drift, hide_index=True, use_container_width=True)
            except Exception as exc:
                st.caption(f"Optional drift report unavailable: {exc}")
        except (ValueError, KeyError) as exc:
            st.error(f"Could not process this retail CSV: {exc}")
        except Exception as exc:
            st.error(f"Scoring failed. Check model/preprocessing compatibility with configs.py. Details: {exc}")

with tab_single:
    with st.form("retail_single_customer"):
        st.subheader("Retail customer profile")
        c1, c2, c3 = st.columns(3)
        customer_id = c1.text_input("Customer ID", value="WHAT-IF-001")
        age = c1.number_input("Age", min_value=18, max_value=110, value=35)
        gender = c1.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
        annual_income = c1.number_input("Annual income", min_value=0.0, value=60000.0, step=1000.0)
        years = c1.number_input("Years as customer", min_value=0.0, value=3.0, step=0.5)

        total_spend = c2.number_input("Total spend", min_value=0.0, value=2500.0, step=100.0)
        purchases = c2.number_input("Number of purchases", min_value=0, value=30, step=1)
        avg_transaction = c2.number_input("Average transaction amount", min_value=0.0, value=83.33, step=5.0)
        returns = c2.number_input("Number of returns", min_value=0, value=2, step=1)
        support = c2.number_input("Support contacts", min_value=0, value=1, step=1)

        satisfaction = c3.slider("Satisfaction score", min_value=1, max_value=10, value=7)
        last_purchase = c3.number_input("Days since last purchase", min_value=0, value=20, step=1)
        email_opt_in = c3.selectbox("Email opt-in", ["Yes", "No"])
        promotion = c3.selectbox("Promotion response", ["Responded", "Ignored", "Not offered"])
        submitted = st.form_submit_button("Score this customer", type="primary")

    if submitted:
        raw_single = pd.DataFrame([{
            "Customer_ID": customer_id.strip() or "WHAT-IF-001",
            "Age": age, "Gender": gender, "Annual_Income": annual_income,
            "Total_Spend": total_spend, "Years_as_Customer": years,
            "Num_of_Purchases": purchases, "Average_Transaction_Amount": avg_transaction,
            "Num_of_Returns": returns, "Num_of_Support_Contacts": support,
            "Satisfaction_Score": satisfaction, "Last_Purchase_Days_Ago": last_purchase,
            "Email_Opt_In": email_opt_in, "Promotion_Response": promotion,
        }])
        try:
            _, scored = score_retail(raw_single, model_bundle)
            row = scored.iloc[0]
            left, middle, right = st.columns([1.0, 1.5, 2.2])
            with left:
                if "churn_probability" in row:
                    st.plotly_chart(
                        gauge(float(row["churn_probability"]), float(model_bundle.get("threshold", 0.30))),
                        use_container_width=True, theme=None,
                        config={"displayModeBar": False},
                    )
            with middle:
                level = row.get("risk_level", "UNKNOWN")
                st.markdown(f"#### {risk_badge(level)} risk")
                if "expected_loss" in row:
                    st.markdown(md(f"Expected margin loss **{money(row['expected_loss'])}**"))
                if "expected_net_gain" in row:
                    st.markdown(md(f"Expected net gain **{money(row['expected_net_gain'])}**"))
                if "contact_recommended" in row:
                    recommendation = "recommended" if bool(row["contact_recommended"]) else "not recommended"
                    st.markdown("**Retention contact:** " + recommendation)
                if "recommended_action" in row:
                    st.markdown("**Recommended action:** " + str(row["recommended_action"]))
                if "top_reasons" in row:
                    st.markdown("**Top reasons**")
                    st.write(row["top_reasons"])
            with right:
                try:
                    shap_row = shap_frame(scored).iloc[0]
                    labels = cfg("FEATURE_LABELS", {})
                    top = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index).head(10)[::-1]
                    fig = go.Figure(go.Bar(
                        x=top.values,
                        y=[labels.get(str(f), str(f).replace("_", " ").title()) for f in top.index],
                        orientation="h",
                        marker_color=[RISK_UP if v > 0 else RISK_DOWN for v in top.values],
                        hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
                    ))
                    fig.add_vline(x=0, line_color=INK_3, line_width=1)
                    fig.update_xaxes(title_text="SHAP impact (positive values increase model output)")
                    chart(style(fig, 380, "What drives this prediction", legend=False))
                except Exception as exc:
                    st.caption(f"Feature-impact chart unavailable: {exc}")
            st.subheader("Score details")
            st.dataframe(scored, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not score this customer: {exc}")
