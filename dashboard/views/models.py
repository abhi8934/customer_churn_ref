import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from common import BAR, INK_3, SEQUENTIAL, SERIES, chart, load_csv, load_report, style
from src import configs as config
from theme import hero, section

report = load_report()
champ = report["champion"]
models = report["models"]
TARGET = config.TARGET
hero("Model performance",
     f"{len(models)} model families, each tuned with a {config.SEARCH_ITERATIONS}-draw randomised "
     f"search and {config.CV_FOLDS}-fold stratified cross-validation on "
     f"{report['dataset']['train']:,} training customers. The champion is picked on CV "
     f"{config.SELECTION_METRIC.replace('_', '-').upper()}; the "
     f"{report['dataset']['test']:,}-customer test split was scored once, at the end.",
     eyebrow="Rigorous, reproducible evaluation",
     chips=[f"Champion <b>{champ}</b>",
            f"Test ROC-AUC <b>{models[champ]['test_at_0.5']['roc_auc']:.3f}</b>",
            "Calibrated probabilities", "Honest negative results"])

preds = load_csv("test_predictions.csv")
base_rate = 1 - preds[TARGET].mean()   # accuracy of "predict everyone stays"

rows = []
for name, m in models.items():
    d, o = m["test_at_0.5"], m["test_at_optimal"]
    rows.append({
        "Model": ("* " if name == champ else "") + name,
        "CV ROC-AUC": f"{m['cv']['roc_auc_mean']:.4f} ± {m['cv']['roc_auc_std']:.3f}",
        "Test ROC-AUC": d["roc_auc"], "Test PR-AUC": d["pr_auc"], "Brier": d["brier"],
        "Accuracy @0.5": d["accuracy"], "Precision @0.5": d["precision"],
        "Recall @0.5": d["recall"], "F1 @0.5": d["f1"],
        "Threshold t*": m["optimal_threshold"], "Recall @t*": o["recall"],
        "Net value @t*": m["campaign_test_at_optimal"]["net_value"],
    })
pct = st.column_config.NumberColumn(format="percent")
st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
    "Test ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
    "Test PR-AUC": st.column_config.NumberColumn(format="%.3f"),
    "Brier": st.column_config.NumberColumn(format="%.3f"),
    "Accuracy @0.5": pct, "Precision @0.5": pct, "Recall @0.5": pct, "F1 @0.5": pct,
    "Threshold t*": st.column_config.NumberColumn(format="%.2f"), "Recall @t*": pct,
    "Net value @t*": st.column_config.NumberColumn(format="dollar")})
st.caption(f"Accuracy is shown for completeness only: predicting 'stays' for everyone scores "
           f"{base_rate:.0%}. t* = money-optimal threshold chosen on out-of-fold training "
           "predictions; net value is the campaign result on the test set at t*.")

left, right = st.columns(2)
roc, pr = go.Figure(), go.Figure()
for i, name in enumerate(models):
    fpr, tpr, _ = roc_curve(preds[TARGET], preds[name])
    prec, rec, _ = precision_recall_curve(preds[TARGET], preds[name])
    width = 3 if name == champ else 1.5
    roc.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines",
                             line=dict(color=SERIES[i % len(SERIES)], width=width)))
    pr.add_trace(go.Scatter(x=rec, y=prec, name=name, mode="lines",
                            line=dict(color=SERIES[i % len(SERIES)], width=width)))
roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                         line=dict(color=INK_3, width=1, dash="dot"), hoverinfo="skip"))
roc.update_xaxes(title_text="False positive rate")
roc.update_yaxes(title_text="True positive rate")
pr.update_xaxes(title_text="Recall")
pr.update_yaxes(title_text="Precision")
inside = dict(orientation="v", bgcolor="rgba(10,15,30,0.80)", xanchor="right", x=0.99)
with left:   # the lower-right corner of a ROC plot is always empty
    chart(style(roc, 400, "ROC curves (test set)").update_layout(
        legend=dict(**inside, yanchor="bottom", y=0.02)))
with right:  # ...and so is the upper-right corner of a precision-recall plot
    chart(style(pr, 400, "Precision-recall curves (test set)").update_layout(
        legend=dict(**inside, yanchor="top", y=0.98)))

left, right = st.columns(2)
with left:
    t = models[champ]["optimal_threshold"]
    cm = models[champ]["test_at_optimal"]
    cm5 = models[champ]["test_at_0.5"]
    z = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted stay", "Predicted churn"], y=["Actually stayed", "Actually churned"],
        colorscale=SEQUENTIAL, showscale=False,
        text=[[f"{v:,}" for v in r] for r in z], texttemplate="%{text}",
        textfont=dict(size=16), hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    chart(style(fig, 300, f"{champ}: confusion matrix at t* = {t:.2f}", legend=False))
    st.caption(f"At the default 0.5 the same model catches {cm5['tp']} churners "
               f"(recall {cm5['recall']:.0%}); at t* it catches {cm['tp']} "
               f"(recall {cm['recall']:.0%}) at the price of {cm['fp'] - cm5['fp']} more false "
               "alarms - worth it when a false alarm costs one offer and a miss costs a customer.")
with right:
    frac, mean_pred = calibration_curve(preds[TARGET], preds[champ], n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                             line=dict(color=INK_3, width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers", name=champ,
                             line=dict(color=BAR, width=2.5), marker=dict(size=9)))
    fig.update_xaxes(title_text="Predicted churn probability", range=[0, 1])
    fig.update_yaxes(title_text="Observed churn rate", range=[0, 1])
    chart(style(fig, 300, "Calibration: are the probabilities honest?"))
    st.caption("Expected loss = probability x value only makes sense if a predicted 40% "
               "really means ~40%. Deciles close to the diagonal = trustworthy money numbers.")

section("Design decisions, tested")
left, right = st.columns(2)
with left:
    st.markdown("**Class imbalance: re-weight or not?**")
    imb = pd.DataFrame(report["imbalance_experiment"])
    st.dataframe(imb, hide_index=True, column_config={
        "cv_roc_auc": st.column_config.NumberColumn("CV ROC-AUC", format="%.4f"),
        "cv_brier": st.column_config.NumberColumn("CV Brier", format="%.4f"),
        "mean_predicted_prob": st.column_config.NumberColumn("Mean predicted", format="percent"),
        "actual_churn_rate": st.column_config.NumberColumn("Actual rate", format="percent"),
        "optimal_threshold": st.column_config.NumberColumn("t*", format="%.2f"),
        "test_net_value": st.column_config.NumberColumn("Test net value", format="dollar")})
    st.caption("Re-weighting changes the probability scale: the average prediction moves away "
               "from the real churn rate and the Brier score worsens, while ranking (ROC-AUC) "
               "and, once each variant gets its own money-optimal threshold, net value should "
               "stay roughly the same. Expected loss = probability x value needs honest "
               "probabilities, so the models are trained unweighted and imbalance is handled at "
               "the threshold. Check the table above against your own results.")
with right:
    st.markdown("**Does the feature engineering help?**")
    abl = pd.DataFrame(report["feature_ablation"])
    st.dataframe(abl, hide_index=True, column_config={
        "cv_roc_auc": st.column_config.NumberColumn("CV ROC-AUC", format="%.4f"),
        "cv_roc_auc_std": st.column_config.NumberColumn("± std", format="%.4f"),
        "cv_pr_auc": st.column_config.NumberColumn("CV PR-AUC", format="%.4f")})
    delta = abl["cv_roc_auc"].iloc[1] - abl["cv_roc_auc"].iloc[0]
    eng = ", ".join(config.FEATURE_LABELS[f].lower() for f in config.ENGINEERED_NUMERIC[:3])
    st.caption(f"Engineered features ({eng}, ...) change CV ROC-AUC by {delta:+.4f} for the "
               "champion. "
               + ("A small but consistent gain." if delta > 0.001 else
                  "That is no gain: boosted trees can often learn these combinations from the "
                  "raw columns. The features stay because they make segments and explanations "
                  "easier to read, not because they add accuracy."))

with st.expander("Best hyper-parameters per model"):
    st.json({n: m["best_params"] for n, m in models.items()})