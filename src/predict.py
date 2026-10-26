"""Score customers: churn probability, risk level, expected loss, reasons, action.

Usage (raw file in the original IBM Telco format):

    python -m src.predict --input new_customers.csv --output scored.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import configs
from src.data_preprocessing import load_raw_csv, prepare_features
from src.economics import expected_loss, expected_net_gain
from src.explain import ChurnExplainer, explain_customer

SHAP_PREFIX = "shap__"


# --------------------------------------------------------------------------- #
# Model bundle: everything needed to score and explain, in one file
# --------------------------------------------------------------------------- #
def build_bundle(pipeline, model_name: str, threshold: float, reference: dict,
                 background: pd.DataFrame, metrics: dict) -> dict:
    return {
        "pipeline": pipeline,
        "model_name": model_name,
        "threshold": float(threshold),
        "economics": configs.ECONOMICS.as_dict(),
        "reference": reference,
        "background": background[configs.MODEL_FEATURES].sample(
            min(200, len(background)), random_state=configs.RANDOM_STATE),
        "features": list(configs.MODEL_FEATURES),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_bundle(bundle: dict, path: Path = configs.MODEL_PATH) -> None:
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path = configs.MODEL_PATH) -> dict:
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def assemble_scores(features: pd.DataFrame, proba: np.ndarray, shap_values: pd.DataFrame | None,
                    shap_base: np.ndarray | None, threshold: float,
                    reference: dict) -> pd.DataFrame:
    """Attach probabilities, money and (optionally) reasons to a feature table."""
    out = features.copy().reset_index(drop=True)
    out["churn_probability"] = proba
    out["risk_level"] = [configs.risk_level(p) for p in proba]
    out["customer_value"] = configs.ECONOMICS.customer_value(out["spend_per_year"])
    out["expected_loss"] = expected_loss(proba, out["spend_per_year"])
    out["expected_net_gain"] = expected_net_gain(proba, out["spend_per_year"])
    out["contact_recommended"] = proba >= threshold

    if shap_values is not None:
        shap_values = shap_values.reset_index(drop=True)
        explanations = [explain_customer(shap_values.loc[i], out.loc[i], reference)
                        for i in range(len(out))]
        for k in range(3):
            out[f"reason_{k + 1}"] = [e["risk_factors"][k]["text"] if len(e["risk_factors"]) > k
                                      else "" for e in explanations]
        out["protective_factor"] = [e["protective_factors"][0]["text"] if e["protective_factors"]
                                    else "" for e in explanations]
        out["top_reasons"] = ["; ".join(f["text"] for f in e["risk_factors"][:3]) for e in explanations]
        out["recommended_action"] = [" | ".join(e["actions"]) for e in explanations]
        out["shap_base"] = shap_base
        for col in shap_values.columns:
            out[SHAP_PREFIX + col] = shap_values[col].to_numpy()
    return out


def score_features(features: pd.DataFrame, bundle: dict, with_reasons: bool = True,
                   explainer: ChurnExplainer | None = None) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    proba = pipeline.predict_proba(features[config.MODEL_FEATURES])[:, 1]
    shap_values = shap_base = None
    if with_reasons:
        explainer = explainer or ChurnExplainer(pipeline, bundle["background"])
        shap_values, shap_base = explainer.explain(features)
    scored = assemble_scores(features, proba, shap_values, shap_base,
                             bundle["threshold"], bundle["reference"])
    return scored.sort_values("expected_loss", ascending=False, ignore_index=True)


def score_raw(raw: pd.DataFrame, bundle: dict | None = None, with_reasons: bool = True) -> pd.DataFrame:
    """Raw Telco-format rows in, scored and explained rows out."""
    bundle = bundle or load_bundle()
    return score_features(prepare_features(raw), bundle, with_reasons)


def shap_frame(scored: pd.DataFrame) -> pd.DataFrame:
    """Recover the per-feature SHAP table stored on a scored frame."""
    cols = [c for c in scored.columns if c.startswith(SHAP_PREFIX)]
    return scored[cols].rename(columns=lambda c: c[len(SHAP_PREFIX):])


OUTPUT_COLUMNS = [configs.ID_COL, "churn_probability", "risk_level", "expected_loss",
                  "expected_net_gain", "contact_recommended", "reason_1", "reason_2",
                  "reason_3", "recommended_action", "years_as_customer",
                  "spend_per_year", "satisfaction_score", "last_purchase_days_ago"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a customer file for churn risk.")
    parser.add_argument("--input", required=True, type=Path, help="CSV in IBM Telco format")
    parser.add_argument("--output", type=Path, default=Path("scored_customers.csv"))
    args = parser.parse_args()

    scored = score_raw(load_raw_csv(args.input))
    scored[OUTPUT_COLUMNS].to_csv(args.output, index=False)
    flagged = int(scored["contact_recommended"].sum())
    print(f"Scored {len(scored):,} customers -> {args.output}  ({flagged:,} recommended for contact)")
    print(scored[OUTPUT_COLUMNS[:6]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
