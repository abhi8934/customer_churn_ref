"""Scenario Lab: model-based what-if analysis of retail retention interventions.
 
Each intervention changes the *profile* of the customers it applies to (e.g.
a lapsed customer is won back, a dissatisfied customer gets service recovery)
and the champion model re-scores them. The drop in churn probability,
multiplied by customer value and by the share of customers who accept the
offer, is the expected benefit; the offer cost is paid for every customer who
accepts.
 
Caveat, shown in the UI too: this is what the *model* expects, based on how
similar customers behave today. It is correlational, not a proven causal
effect - an A/B test is the way to confirm it.
"""
from __future__ import annotations
 
from dataclasses import dataclass
from typing import Callable
 
import numpy as np
import pandas as pd
from src import configs
from src.configs import Economics, ECONOMICS
 
# Category / scale values used by the interventions - match these to your dataset.
EMAIL_YES, EMAIL_NO = "Yes", "No"
SATISFACTION_MAX = 5          # top of the satisfaction scale
SATISFACTION_LOW = 2          # at or below this a customer counts as dissatisfied
LAPSED_DAYS = 90              # no purchase for this long = lapsed
WIN_BACK_DAYS = 30            # a won-back customer is treated as having bought this recently
SUPPORT_HEAVY = 3             # this many support contacts = needs service recovery attention
 
 
# --------------------------------------------------------------------------- #
# Derived features - a Python mirror of the engineered features in
# configs.ENGINEERED_NUMERIC (tests/test_scenarios.py should check they agree
# with the feature pipeline). Division by zero gives 0.
# --------------------------------------------------------------------------- #
def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num.astype(float) / den.astype(float).where(den != 0)).fillna(0.0)
 
 
def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    purchases = out["num_of_purchases"]
    years = out["years_as_customer"]
 
    out["return_rate"] = _ratio(out["num_of_returns"], purchases)
    out["purchases_per_year"] = _ratio(purchases, years)
    out["spend_per_year"] = _ratio(out["total_spend"], years)
    out["support_contact_rate"] = _ratio(out["num_of_support_contacts"], purchases)
    # last_purchase_days_ago / (365 / purchases_per_year), written to avoid dividing by zero
    out["rescency_ratio"] = out["last_purchase_days_ago"].astype(float) \
        * out["purchases_per_year"] / 365
    out["spend_to_income_ratio"] = _ratio(out["total_spend"], out["annual_income"])
    return out
 
 
# --------------------------------------------------------------------------- #
# Interventions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Intervention:
    key: str
    label: str
    description: str
    cost_label: str
    default_cost: float
    eligible: Callable[[pd.DataFrame], pd.Series]
    apply: Callable[[pd.DataFrame, dict], pd.DataFrame]
    # cost per accepting customer, given the rows and the scenario parameters
    cost: Callable[[pd.DataFrame, dict], pd.Series]
 
 
def _set(**values):
    def apply(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
        return rows.assign(**values)
    return apply
 
 
def _win_back(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
    return rows.assign(last_purchase_days_ago=rows["last_purchase_days_ago"].clip(
        upper=params.get("win_back_days", WIN_BACK_DAYS)))
 
 
def _service_recovery(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
    return rows.assign(satisfaction_score=(rows["satisfaction_score"] + 1).clip(
        upper=SATISFACTION_MAX))
 
 
def _loyalty_discount(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The customer makes one extra purchase, at a discount, right now."""
    price = rows["average_transaction_amount"] * (1 - params.get("discount", 0.15))
    return rows.assign(total_spend=(rows["total_spend"] + price).round(2),
                       num_of_purchases=rows["num_of_purchases"] + 1,
                       last_purchase_days_ago=0)
 
 
INTERVENTIONS: dict[str, Intervention] = {iv.key: iv for iv in [
    Intervention(
        "win_back", "Win-back voucher for lapsed customers",
        f"Customers with no purchase for {LAPSED_DAYS}+ days get a voucher that brings them "
        "back to buy.",
        "Voucher value ($ per customer)", 15.0,
        eligible=lambda d: d["last_purchase_days_ago"] >= LAPSED_DAYS,
        apply=_win_back,
        cost=lambda d, p: pd.Series(p.get("win_back", 15.0), index=d.index)),
    Intervention(
        "email_opt_in", "Get customers to opt in to email",
        "Customers who have not opted in are offered a small incentive to join the email list.",
        "Sign-up incentive ($ per customer)", 3.0,
        eligible=lambda d: d["email_opt_in"] == EMAIL_NO,
        apply=_set(email_opt_in=EMAIL_YES),
        cost=lambda d, p: pd.Series(p.get("email_opt_in", 3.0), index=d.index)),
    Intervention(
        "service_recovery", "Service recovery for unhappy customers",
        "Dissatisfied or support-heavy customers get a personal follow-up and a goodwill "
        "credit; satisfaction is assumed to improve by one point.",
        "Goodwill credit ($ per customer)", 25.0,
        eligible=lambda d: (d["satisfaction_score"] <= SATISFACTION_LOW)
        | (d["num_of_support_contacts"] >= SUPPORT_HEAVY),
        apply=_service_recovery,
        cost=lambda d, p: pd.Series(p.get("service_recovery", 25.0), index=d.index)),
    Intervention(
        "loyalty_discount", "Loyalty discount on the next purchase",
        "Targeted customers get a percentage off an immediate purchase; the lost margin on "
        "that order is the cost.",
        "Discount (% of the average order)", 0.15,
        eligible=lambda d: d["average_transaction_amount"] > 0,
        apply=_loyalty_discount,
        cost=lambda d, p: d["average_transaction_amount"] * p.get("discount", 0.15)),
]}
 
 
# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _predict(pipeline, features: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(features[configs.MODEL_FEATURES])[:, 1]
 
 
def apply_interventions(features: pd.DataFrame, keys: list[str],
                        params: dict | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (changed profiles, eligible-for-anything mask, offer cost per customer)."""
    params = params or {}
    changed = features.copy()
    eligible = pd.Series(False, index=features.index)
    cost = pd.Series(0.0, index=features.index)
    for key in keys:
        iv = INTERVENTIONS[key]
        mask = iv.eligible(changed)
        if mask.any():
            cost[mask] += iv.cost(changed.loc[mask], params)
            changed.loc[mask] = iv.apply(changed.loc[mask], params)
            eligible |= mask
            changed = recompute_derived(changed)   # later interventions see updated features
    return recompute_derived(changed), eligible, cost
 
 
def simulate(features: pd.DataFrame, pipeline, keys: list[str], acceptance: float,
             params: dict | None = None, econ: Economics = ECONOMICS) -> tuple[dict, pd.DataFrame]:
    """Expected effect of offering `keys` to every customer in `features`."""
    params = {"horizon": econ.horizon_months, **(params or {})}
    base = _predict(pipeline, features)
    changed, eligible, cost = apply_interventions(features, keys, params)
    after = np.where(eligible, _predict(pipeline, changed), base)
 
    # Customer value is based on the customer's spend *before* the intervention.
    value = econ.customer_value(features["spend_per_year"])
    drop = base - after
    detail = pd.DataFrame({
        configs.ID_COL: features[configs.ID_COL].to_numpy(),
        "eligible": eligible.to_numpy(),
        "p_before": base, "p_after": after, "risk_drop": drop,
        "value": value, "offer_cost": cost.to_numpy(),
        "expected_benefit": acceptance * drop * value,
        "expected_cost": acceptance * cost.to_numpy(),
    })
    detail["expected_net"] = detail["expected_benefit"] - detail["expected_cost"]
 
    n_eligible = int(eligible.sum())
    benefit, spend = detail["expected_benefit"].sum(), detail["expected_cost"].sum()
    summary = {
        "customers": len(features),
        "eligible": n_eligible,
        "accepting": acceptance * n_eligible,
        "churners_before": float(base.sum()),
        "churners_after": float((base - acceptance * drop).sum()),
        "churners_avoided": float(acceptance * drop.sum()),
        "margin_retained": float(benefit),
        "offer_cost": float(spend),
        "net_value": float(benefit - spend),
        "roi": float((benefit - spend) / spend) if spend else float("nan"),
        "avg_risk_before": float(base[eligible.to_numpy()].mean()) if n_eligible else float("nan"),
        "avg_risk_after": float(after[eligible.to_numpy()].mean()) if n_eligible else float("nan"),
    }
    return summary, detail
 
 
def next_best_actions(row: pd.Series, pipeline, params: dict | None = None,
                      econ: Economics = ECONOMICS) -> pd.DataFrame:
    """Every intervention this customer is eligible for, ranked by risk reduction."""
    one = row.to_frame().T.infer_objects()
    rows = []
    for key, iv in INTERVENTIONS.items():
        summary, detail = simulate(one, pipeline, [key], acceptance=1.0, params=params, econ=econ)
        if not summary["eligible"]:
            continue
        d = detail.iloc[0]
        rows.append({"action": iv.label, "key": key, "p_before": d["p_before"],
                     "p_after": d["p_after"], "risk_drop": d["risk_drop"],
                     "margin_saved": d["expected_benefit"], "offer_cost": d["offer_cost"],
                     "net_if_accepted": d["expected_net"]})
    if not rows:
        return pd.DataFrame(columns=["action", "key", "p_before", "p_after", "risk_drop",
                                     "margin_saved", "offer_cost", "net_if_accepted"])
    return pd.DataFrame(rows).sort_values("risk_drop", ascending=False, ignore_index=True)
