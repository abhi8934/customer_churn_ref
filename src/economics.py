"""Turn churn probabilities into a retention-campaign decision priced in money.

Accuracy treats every mistake the same. A retention campaign does not:

* contacting a customer costs `offer_cost`, whether or not they would have left;
* contacting a customer who *would* have left saves them with probability
  `save_rate`, which is worth their margin over the planning horizon;
* missing a churner costs nothing extra today, but the revenue walks away.

So the right threshold is the one that maximises net value, and that depends
on the price list in `configs.Economics`, not on the 0.5 default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.configs import ECONOMICS, Economics

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def campaign_outcome(y_true, proba, spend_per_year, *, threshold: float | None = None,
                     target=None, econ: Economics = ECONOMICS) -> dict:
    """Realised value of contacting everyone above `threshold` (or a given mask)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    value = econ.customer_value(spend_per_year)
    if target is None:
        target = p >= threshold
    target = np.asarray(target, dtype=bool)

    caught = target & (y == 1)
    benefit = float((econ.save_rate * value * caught).sum())
    cost = float(econ.offer_cost * target.sum())
    return {
        "threshold": threshold,
        "contacted": int(target.sum()),
        "churners_caught": int(caught.sum()),
        "churners_total": int(y.sum()),
        "recall": float(caught.sum() / max(y.sum(), 1)),
        "precision": float(caught.sum() / max(target.sum(), 1)),
        "expected_customers_saved": float(econ.save_rate * caught.sum()),
        "retained_value": benefit,
        "campaign_cost": cost,
        "net_value": benefit - cost,
    }


def threshold_curve(y_true, proba, spend_per_year, econ: Economics = ECONOMICS,
                    thresholds=THRESHOLDS) -> pd.DataFrame:
    """Net value of the campaign at every candidate threshold."""
    return pd.DataFrame([campaign_outcome(y_true, proba, spend_per_year,
                                          threshold=t, econ=econ) for t in thresholds])


def optimal_threshold(y_true, proba, spend_per_year, econ: Economics = ECONOMICS) -> float:
    curve = threshold_curve(y_true, proba, spend_per_year, econ)
    return float(curve.loc[curve["net_value"].idxmax(), "threshold"])


def expected_value_target(proba, spend_per_year, econ: Economics = ECONOMICS) -> np.ndarray:
    """Per-customer rule: contact when the *expected* saving beats the offer cost.

    p * save_rate * value > offer_cost. Unlike a single threshold this lets a
    high-value customer qualify at a lower probability than a low-value one.
    It needs calibrated probabilities, which is why the models are trained
    without class re-weighting.
    """
    p = np.asarray(proba, dtype=float)
    return p * econ.save_rate * econ.customer_value(spend_per_year) > econ.offer_cost


def expected_loss(proba, spend_per_year, econ: Economics = ECONOMICS) -> np.ndarray:
    """Margin we expect to lose if we do nothing: P(churn) x customer value."""
    return np.asarray(proba, dtype=float) * econ.customer_value(spend_per_year)


def expected_net_gain(proba, spend_per_year, econ: Economics = ECONOMICS) -> np.ndarray:
    """Expected value of contacting this customer: p x save_rate x value - cost."""
    return expected_loss(proba, spend_per_year, econ) * econ.save_rate - econ.offer_cost


def expected_campaign_value(proba, spend_per_year, target, econ: Economics = ECONOMICS) -> float:
    """Forward-looking campaign value on customers whose outcome is unknown yet."""
    gain = expected_net_gain(proba, spend_per_year, econ)
    return float(gain[np.asarray(target, dtype=bool)].sum())
