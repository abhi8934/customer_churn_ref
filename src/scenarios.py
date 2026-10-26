"""Scenario Lab: model-based what-if analysis of retention interventions (retail).

Each intervention changes the *profile* of the customers it applies to (e.g.
an ignored promotion becomes a responded one) and the champion model re-scores
them. The drop in churn probability, multiplied by customer value and by the
share of customers who accept the offer, is the expected benefit; the offer
cost is paid for every customer who accepts.

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


# --------------------------------------------------------------------------- #
# Dataset-specific assumptions - check these against your raw CSV.
# --------------------------------------------------------------------------- #
OPT_IN_TOKENS = {"true", "yes", "1", "1.0"}
PROMO_IGNORED = "Ignored"         # promotion_response values
PROMO_RESPONDED = "Responded"
MAX_SATISFACTION = 10             # satisfaction_score scale is 1..MAX_SATISFACTION
LOW_SATISFACTION = 4              # <= this counts as unhappy
HIGH_SUPPORT_CONTACTS = 3         # >= this many support contacts counts as friction
LAPSED_DAYS = 90                  # no purchase for this long = lapsed


# --------------------------------------------------------------------------- #
# Derived features - a Python mirror of the ENGINEERED_NUMERIC features in
# configs.py. They must match whatever your feature pipeline / SQL view does
# (including how it handles zeros), otherwise "before" scores will not match
# the scored file.
# --------------------------------------------------------------------------- #
def _div(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den, with 0 wherever the denominator is 0 or missing."""
    out = num.astype(float) / den.astype(float).where(den > 0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    purchases = out["num_of_purchases"]
    years = out["years_as_customer"]

    out["return_rate"] = _div(out["num_of_returns"], purchases)
    out["purchases_per_year"] = _div(purchases, years)
    out["spend_per_year"] = _div(out["total_spend"], years)
    out["support_contact_rate"] = _div(out["num_of_support_contacts"], purchases)
    # last_purchase_days_ago / (365 / purchases_per_year)
    out["rescency_ratio"] = out["last_purchase_days_ago"].astype(float) * out["purchases_per_year"] / 365
    out["spend_to_income_ratio"] = _div(out["total_spend"], out["annual_income"])
    return out


def _opted_in(col: pd.Series) -> pd.Series:
    return col.astype(str).str.strip().str.lower().isin(OPT_IN_TOKENS)


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


def _service_recovery(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
    lift = params.get("satisfaction_lift", 2)
    return rows.assign(satisfaction_score=(rows["satisfaction_score"] + lift).clip(upper=MAX_SATISFACTION))


def _reactivate(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Customer comes back and makes one more average-sized purchase."""
    days = params.get("reactivation_days", 14)
    return rows.assign(
        last_purchase_days_ago=np.minimum(rows["last_purchase_days_ago"], days),
        num_of_purchases=rows["num_of_purchases"] + 1,
        total_spend=rows["total_spend"] + rows["average_transaction_amount"],
    )


_NO_TO_YES = {"false": "true", "no": "yes", "n": "y", "0": "1", "0.0": "1.0"}


def _opt_in(rows: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Flip 'not opted in' to 'opted in', keeping the column's own format."""
    def flip(v):
        if isinstance(v, (bool, np.bool_)):
            return True
        if isinstance(v, (int, float, np.integer, np.floating)):
            return 1
        s = str(v).strip()
        yes = _NO_TO_YES.get(s.lower(), "yes")
        return yes.upper() if s.isupper() else yes.capitalize() if s[:1].isupper() else yes
    return rows.assign(email_opt_in=rows["email_opt_in"].map(flip))


INTERVENTIONS: dict[str, Intervention] = {iv.key: iv for iv in [
    Intervention(
        "email_opt_in", "Get customers to opt in to email",
        "Customers who are not opted in to email are invited to join the mailing list.",
        "Incentive ($ per customer)", 2.0,
        eligible=lambda d: ~_opted_in(d["email_opt_in"]),
        apply=_opt_in,
        cost=lambda d, p: pd.Series(p.get("email_opt_in", 2.0), index=d.index)),
    Intervention(
        "promo_engagement", "Re-engage with a personalised promotion",
        "Customers who ignore promotions get a tailored offer they respond to.",
        "Promotion cost ($ per customer)", 8.0,
        eligible=lambda d: d["promotion_response"] == PROMO_IGNORED,
        apply=_set(promotion_response=PROMO_RESPONDED),
        cost=lambda d, p: pd.Series(p.get("promo_engagement", 8.0), index=d.index)),
    Intervention(
        "service_recovery", "Proactive service recovery",
        "Unhappy or support-heavy customers get a personal call and a goodwill gesture.",
        "Cost of call + gesture ($ per customer)", 15.0,
        eligible=lambda d: (d["satisfaction_score"] <= LOW_SATISFACTION)
                           | (d["num_of_support_contacts"] >= HIGH_SUPPORT_CONTACTS),
        apply=_service_recovery,
        cost=lambda d, p: pd.Series(p.get("service_recovery", 15.0), index=d.index)),
    Intervention(
        "reactivation_offer", "Win-back discount for lapsed customers",
        "Customers who have not bought for a long time get a discount on their next basket.",
        "Discount (% of an average basket)", 0.15,
        eligible=lambda d: d["last_purchase_days_ago"] >= LAPSED_DAYS,
        apply=_reactivate,
        cost=lambda d, p: d["average_transaction_amount"] * p.get("reactivation_offer", 0.15)),
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
        if not mask.any():
            continue

        subset = changed.loc[mask].copy()
        updated = iv.apply(subset, params)
        cost.loc[mask] += iv.cost(subset, params)

        # Update only the columns the intervention actually changes.
        for col in updated.columns:
            if not updated[col].equals(subset[col]):
                changed.loc[mask, col] = updated[col]

        eligible |= mask

    return recompute_derived(changed), eligible, cost


def simulate(features: pd.DataFrame, pipeline, keys: list[str], acceptance: float,
             params: dict | None = None, econ: Economics = ECONOMICS) -> tuple[dict, pd.DataFrame]:
    """Expected effect of offering `keys` to every customer in `features`."""
    params = {"horizon": econ.horizon_months, **(params or {})}
    base = _predict(pipeline, features)
    changed, eligible, cost = apply_interventions(features, keys, params)
    after = np.where(eligible, _predict(pipeline, changed), base)

    # Margin we keep if the customer stays, from their (unchanged) yearly spend.
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