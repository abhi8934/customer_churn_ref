"""Metrics for an imbalanced, money-driven classification problem.

Only ~27% of customers churn, so a model that predicts "stays" for everyone is
73% accurate and worthless. Accuracy is reported for completeness; the numbers
that matter are recall/precision at the chosen threshold, ROC-AUC and PR-AUC
(ranking quality), Brier score (are the probabilities honest?) and net value.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def lift_at(y_true, proba, top_fraction: float = 0.10) -> float:
    """Churn rate in the top-scored `top_fraction` of customers vs. the base rate."""
    y = np.asarray(y_true, dtype=int)
    order = np.argsort(-np.asarray(proba, dtype=float))
    top = y[order[: max(1, int(len(y) * top_fraction))]]
    return float(top.mean() / y.mean())
