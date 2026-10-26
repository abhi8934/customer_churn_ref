"""Why is this customer at risk? SHAP values turned into business language.

Pipeline: fitted model -> SHAP value per *encoded* column -> summed back to the
source feature (all `contract_*` columns become "contract") -> the features
pushing this customer towards churn, phrased the way an account manager would
say them, plus a playbook action for each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import configs
from src.feature_engineering import output_feature_groups, percentile_of


class ChurnExplainer:
    """SHAP explainer for a fitted preprocessing + model pipeline.

    Values are reported per source feature, in the model's raw output space
    (log-odds for logistic regression and boosting; probability for forests).
    Positive = pushes towards churn.
    """

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame, max_background: int = 200):
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.features = list(configs.MODEL_FEATURES)

        groups = output_feature_groups(self.preprocess)
        self._aggregate = np.zeros((len(groups), len(self.features)))
        for i, group in enumerate(groups):
            self._aggregate[i, self.features.index(group)] = 1.0

        if isinstance(self.model, LogisticRegression):
            sample = background[self.features].sample(
                min(max_background, len(background)), random_state=configs.RANDOM_STATE)
            self._explainer = shap.LinearExplainer(self.model, self.preprocess.transform(sample))
            self._is_tree = False
        else:
            self._explainer = shap.TreeExplainer(self.model)
            self._is_tree = True

    def explain(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return (SHAP values per source feature, base value per row)."""
        encoded = self.preprocess.transform(X[self.features])
        if self._is_tree:
            values = self._explainer.shap_values(encoded, check_additivity=False)
        else:
            values = self._explainer.shap_values(encoded)
        base = np.asarray(self._explainer.expected_value, dtype=float)

        values = np.asarray(values)
        if values.ndim == 3:                     # (rows, cols, classes) -> churn class
            values = values[:, :, 1]
        if base.ndim and base.size > 1:
            base = base[1]
        grouped = values @ self._aggregate
        return (pd.DataFrame(grouped, columns=self.features, index=X.index),
                np.full(len(X), float(base)))


# --------------------------------------------------------------------------- #
# Plain-language phrasing
# --------------------------------------------------------------------------- #


def describe(feature: str, row: pd.Series, reference: dict[str, list[float]]) -> str:
    """One factual phrase for a feature's value on this customer."""
    v = row[feature]

    def pct() -> float:
        return percentile_of(float(v), reference[feature])

    if feature == "age":
        return f"Age {int(v)}"
    if feature == "gender":
        return f"Gender: {v}"
    if feature == "annual_income":
        return f"Annual income of ${v:,.0f}"
    if feature == "total_spend":
        return f"Low lifetime spend (${v:,.0f})" if pct() < 40 else f"Lifetime spend of ${v:,.0f}"
    if feature == "years_as_customer":
        return "Newer customer (under a year)" if v < 1 else f"{v:g} years as a customer"
    if feature == "num_of_purchases":
        n = int(v)
        return f"Only {n} purchase{'s' if n != 1 else ''} to date" if pct() < 30 else f"{n} purchases to date"
    if feature == "average_transaction_amount":
        return f"Average basket of ${v:,.2f}"
    if feature == "num_of_returns":
        n = int(v)
        return f"{n} return{'s' if n != 1 else ''}"
    if feature == "num_of_support_contacts":
        n = int(v)
        return (f"{n} support contacts, more than {pct():.0f}% of customers"
                if pct() >= 70 else f"{n} support contact{'s' if n != 1 else ''}")
    if feature == "satisfaction_score":
        return (f"Low satisfaction score ({v:g})" if pct() < 30
                else f"Satisfaction score of {v:g}")
    if feature == "last_purchase_days_ago":
        d = int(v)
        return (f"No purchase in {d} days, longer than {pct():.0f}% of customers"
                if pct() >= 70 else f"Last purchase {d} days ago")
    if feature == "return_rate":
        return (f"High return rate ({v:.0%}), above {pct():.0f}% of customers"
                if pct() >= 70 else f"Return rate of {v:.0%}")
    if feature == "purchases_per_year":
        return (f"Infrequent buyer ({v:.1f} purchases/year)" if pct() < 30
                else f"{v:.1f} purchases per year")
    if feature == "spend_per_year":
        return f"Low annual spend (${v:,.0f}/year)" if pct() < 40 else f"Spends ${v:,.0f} per year"
    if feature == "support_contact_rate":
        return (f"Contacts support on {v:.0%} of purchases, above {pct():.0f}% of customers"
                if pct() >= 70 else f"Contacts support on {v:.0%} of purchases")
    if feature == "rescency_ratio":          # days since last purchase / usual gap between purchases
        if v >= 1.5:
            return f"Overdue: {v:.1f}x their usual gap between purchases"
        if v <= 0.8:
            return "Buying faster than their usual rhythm"
        return "Roughly on their usual purchase cycle"
    if feature == "spend_to_income_ratio":
        return f"Spends {v:.1%} of their income with us"
    if feature == "email_opt_in":
        return "Opted in to email" if str(v).lower() in ("yes", "true", "1") else "Not opted in to email"
    if feature == "promotion_response":
        return {
            "Responded":    "Responds to promotions",
            "Ignored":      "Ignores promotional messages",
            "Unsubscribed": "Unsubscribed from promotions",
        }.get(str(v), f"Promotion response: {v}")
    return f"{configs.FEATURE_LABELS.get(feature, feature)}: {v}"


# --------------------------------------------------------------------------- #
# Retention playbook: which action answers which risk factor
# --------------------------------------------------------------------------- #
_PLAYBOOK = [
    (("last_purchase_days_ago", "rescency_ratio"),
     lambda r: r["rescency_ratio"] >= 1.5,
     "Win-back offer: personalised discount on their usual category"),
    (("satisfaction_score",),
     lambda r: r["satisfaction_score"] <= 5,            # adjust to your score scale
     "Service-recovery call to understand the issue, plus an apology voucher"),
    (("num_of_support_contacts", "support_contact_rate"),
     lambda r: r["num_of_support_contacts"] >= 3,
     "Escalate open support issues to a senior agent"),
    (("return_rate", "num_of_returns"),
     lambda r: r["return_rate"] >= 0.2,
     "Review return reasons; offer fit/size guidance or a free-returns perk"),
    (("purchases_per_year", "num_of_purchases", "spend_per_year", "total_spend"),
     lambda r: True,
     "Loyalty-points bonus on the next purchase to lift frequency"),
    (("email_opt_in",),
     lambda r: str(r["email_opt_in"]).lower() in ("no", "false", "0"),
     "Reach via SMS/app push and invite them to opt in to email with a small incentive"),
    (("promotion_response",),
    lambda r: r["promotion_response"] == "Unsubscribed",
    "Don't send promotions: reach out via a service call or app message instead (check consent first)"),
    (("promotion_response",),
    lambda r: r["promotion_response"] == "Ignored",
    "Change channel or message: try SMS/app push with a personalised, category-specific offer"),
    (("promotion_response",),
    lambda r: r["promotion_response"] == "Responded",
    "They respond to offers: a targeted discount on their usual category is the best bet"),
]
DEFAULT_ACTION = "Personal check-in call from the customer-success team"


def recommend_actions(risk_features: list[str], row: pd.Series, k: int = 2) -> list[str]:
    actions: list[str] = []
    for feature in risk_features:
        for triggers, applies, action in _PLAYBOOK:
            if feature in triggers and applies(row) and action not in actions:
                actions.append(action)
    return actions[:k] or [DEFAULT_ACTION]

HIDDEN_FEATURES = {"gender"}

def explain_customer(shap_row: pd.Series, row: pd.Series, reference: dict[str, list[float]],
                     k: int = 4) -> dict:
    """Top risk factors, top protective factors and recommended actions."""
    ranked = shap_row.drop(labels=HIDDEN_FEATURES, errors="ignore").sort_values(ascending=False)
    risk = [f for f, s in ranked.items() if s > 0.01][:k]
    protective = [f for f, s in ranked[::-1].items() if s < -0.01][:2]
    return {
        "risk_factors": [{"feature": f, "text": describe(f, row, reference),
                          "impact": float(shap_row[f])} for f in risk],
        "protective_factors": [{"feature": f, "text": describe(f, row, reference),
                                "impact": float(shap_row[f])} for f in protective],
        "actions": recommend_actions(risk, row),
    }