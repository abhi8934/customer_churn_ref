"""Run the whole project end to end:
    python -m src.pipeline
    python -m src.pipeline --no-llm

Raw csv -> SQLite (clean + quality gate) -> features -> 5 tuned models -> 
champion -> money-optimal threshold -> out-of-fold scores + SHAP reasons for
every customer ->segments -> final model bundle -> figures.
"""

from __future__ import annotations
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from src import configs
from src.data_preprocessing import build_database, load_features, write_scores
from src.economics import threshold_curve
from src.explain import ChurnExplainer, explain_customer
from src.feature_engineering import reference_stats, split_xy
from src.llm import build_context, generate_brief
from src.predict import assemble_scores, build_bundle, save_bundle, shap_frame
from src.segmentation import segment_customers
from src.train_model import cv_splitter, save_report, train_and_compare


def _step(msg: str) -> float:
    print(f"\n ==> {msg}")
    return time.perf_counter()

def score_customer_base(features: pd.DataFrame, champion_pipeline, threshold:float,
                        reference: dict) -> pd.DataFrame:
    """Out-of-fold probability and SHAP reasons for every customer.

    Each customer is scored and explained by a model that never saw them, so
    the risk list in the dashboard is as honest as the test-set metrics.
    """
    X, y = split_xy(features)
    proba = np.zeros(len(X))
    shap_value = pd.DataFrame(0.0, index=X.index, columns=configs.MODEL_FEATURES)
    shap_base = np.zeros(len(X))
    for train_idx, hold_idx in cv_splitter().split(X, y):
        model = clone(champion_pipeline).fit(X.iloc[train_idx],  y.iloc[train_idx])
        hold = X.iloc[hold_idx]
        proba[hold_idx] = model.predict_proba(hold)[:, 1]
        values, base = ChurnExplainer(model, X.iloc[train_idx]).explain(hold)
        shap_value.iloc[hold_idx] = values.to_numpy()
        shap_base[hold_idx] = base
    return assemble_scores(features, proba, shap_value, shap_base, threshold, reference)

def write_retention_briefs(scored: pd.DataFrame, reference: dict, use_llm: bool, n: int = 3) -> None:    
    """Example LLM briefs for the highest-loss active customers in the contact list."""    
    shap_values = shap_frame(scored)    
    flagged = scored[(scored[configs.TARGET] == 0) & scored["contact_recommended"]]
    examples = []    
    for idx, row in flagged.sort_values("expected_loss", ascending=False).head(n).iterrows():        
        ctx = build_context(row, explain_customer(shap_values.loc[idx], row, reference))        
        examples.append({"context": ctx, "brief": generate_brief(ctx, use_llm=use_llm)})
        print(f"  {ctx['customer_id']}: {examples[-1]['brief']['source']}")
    configs.LLM_EXAMPLES_JSON.write_text(json.dumps(examples, indent=2), encoding="utf-8")

def run(use_llm: bool = True) -> None:
    t0 = time.perf_counter()

    _step("1/7 Building SQLite analytical layer + data-quality gate")
    build_database()
    features = load_features()
    features.to_csv(configs.FEATURES_CSV, index=False)
    print(f"  {len(features):,} customers, {len(configs.MODEL_FEATURES)} model features")

    _step("2/7 Tuning and comparing 5 models (5-fold CV on the training split)")
    out = train_and_compare(features)
    report, champion = out["report"], out["champion"]
    X_train, X_test, y_train, y_test = out["splits"]
    out["test_predictions"].to_csv(configs.REPORTS_DIR / "test_predictions.csv", index=False)
    out["permutation_importance"].to_csv(configs.REPORTS_DIR / "permutation_importance.csv", index=False)
    print(f"  champion: {champion.name} (CV ROC-AUC {champion.cv['roc_auc_mean']:.4f}),"
          f" money-optimal threshold {champion.threshold:.2f}")

    _step("3/7 Threshold economics")
    economics = {
        "economics": configs.ECONOMICS.as_dict(),
        "model": champion.name,
        "chosen_threshold": champion.threshold,
        "chosen_on": "out-of-fold predictions on the training split",
        "curve_train_oof": threshold_curve(y_train, champion.oof_proba,
                                           X_train["spend_per_year"]).to_dict("records"),
        "curve_test": threshold_curve(y_test, champion.test_proba,
                                      X_test["spend_per_year"]).to_dict("records"),
        "policy_comparison_test": report["policy_comparison_test"],
    }
    configs.THRESHOLD_JSON.write_text(json.dumps(economics, indent=2), encoding="utf-8")
    for row in report["policy_comparison_test"]:
        print(f"  {row['policy']:<40} contacted {row['contacted']:>4}  net ${row['net_value']:>9,.0f}")

    _step("4/7 Scoring every customer out-of-fold, with SHAP reasons")
    reference = reference_stats(features)
    scored = score_customer_base(features, champion.pipeline, champion.threshold, reference)
    shap_values = shap_frame(scored)
    importance = (shap_values.abs().mean().sort_values(ascending=False)
                  .rename("mean_abs_shap").rename_axis("feature").reset_index())
    importance["label"] = importance["feature"].map(configs.FEATURE_LABELS)
    importance.to_csv(configs.GLOBAL_IMPORTANCE_CSV, index=False)

    _step("5/7 Customer segmentation")
    labels, profile = segment_customers(features)
    silhouette = profile.attrs["silhouette"]
    scored["segment"] = labels.to_numpy()
    seg = (scored.groupby("segment")
           .agg(active_customers=(configs.TARGET, lambda s: int((s == 0).sum())),
                avg_churn_probability=("churn_probability", "mean"),
                expected_loss_active=("expected_loss",
                                      lambda s: s[scored.loc[s.index, configs.TARGET] == 0].sum()))
           .reset_index())
    profile = profile.merge(seg, on="segment")
    profile.to_csv(configs.REPORTS_DIR / "segment_profile.csv", index=False)
    print(f"  silhouette {silhouette:.3f}")
    print(profile[["segment", "customers", "churn_rate", "avg_years_as_customer",
               "avg_spend_per_year"]].to_string(index=False))

    scored.round(5).to_csv(configs.SCORED_CSV, index=False)
    write_scores(scored)
    active = scored[scored[configs.TARGET] == 0]
    flagged = active[active["contact_recommended"]]
    print(f"  active customers {len(active):,}; recommended for contact {len(flagged):,};"
          f" expected margin loss (active) ${active['expected_loss'].sum():,.0f}")

    _step("6/7 Final model: champion refit on all customers")
    X_all, y_all = split_xy(features)
    final = clone(champion.pipeline).fit(X_all, y_all)
    bundle = build_bundle(final, champion.name, champion.threshold, reference, X_all,
                          metrics=report["models"][champion.name])
    save_bundle(bundle)
    print(f"  saved {configs.MODEL_PATH.relative_to(configs.PROJECT_ROOT)}")

    _step("7/7 Retention briefs for the top-3 customers by expected loss")
    write_retention_briefs(scored, reference, use_llm)

    save_report(report)
    try:  # figures and the PDF report are nice-to-have; never fail the pipeline on them
        from src.visualization import make_all_figures
        make_all_figures()
        from src.report import build_report   # needs reportlab (requirements-dev.txt)
        build_report()
    except Exception as exc:
        print(f"  (figures/report skipped: {exc})")

    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-llm", action="store_true", help="use the template instead of Claude")
    parser.add_argument("--briefs-only", action="store_true",
                        help="only regenerate reports/llm_examples.json from the saved scores "
                             "(e.g. after adding an ANTHROPIC_API_KEY)")
    args = parser.parse_args()
    if args.briefs_only:
        from src.predict import load_bundle
        write_retention_briefs(pd.read_csv(configs.SCORED_CSV), load_bundle()["reference"],
                               use_llm=not args.no_llm)
    else:
        run(use_llm=not args.no_llm)
