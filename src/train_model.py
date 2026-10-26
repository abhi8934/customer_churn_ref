"""Train, tune and compare five classifiers; pick a champion; price its decisions.
 
Protocol (nothing here ever looks at the test set until the very end):
 
1. Stratified 80/20 train/test split.
2. For each model family: randomised hyper-parameter search, 5-fold stratified
   CV on the training set, optimising ROC-AUC.
3. Out-of-fold (OOF) training predictions of the tuned model -> the
   money-optimal threshold for that model.
4. Only then: score the untouched test set, at 0.5 and at the chosen threshold.
 
Class imbalance is handled at the *decision* (threshold) rather than by
re-weighting the training data - re-weighting inflates the probabilities, and
the expected-loss ranking needs them honest. `imbalance_experiment` checks that
claim instead of assuming it.
 
Customer value for the economics comes from `spend_per_year` (see
`configs.Economics.customer_value`).
"""
from __future__ import annotations
 
import json
from dataclasses import dataclass, field
 
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
 
from src import configs
from src.economics import campaign_outcome, expected_value_target, optimal_threshold
from src.evaluate_model import classification_metrics, lift_at
from src.feature_engineering import build_pipeline, split_xy
 
RS = configs.RANDOM_STATE
 
# Column that drives customer value in the campaign economics (was monthly_charges in Telco).
VALUE_COL = "spend_per_year"
 
 
def model_space() -> dict[str, tuple]:
    """Estimator + hyper-parameter distributions for each model family."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=5000),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RS),
            {"model__max_depth": randint(3, 12),
             "model__min_samples_leaf": randint(5, 120),
             "model__criterion": ["gini", "entropy"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=400, n_jobs=1, random_state=RS),
            {"model__max_depth": [4, 6, 8, 10, 12, None],
             "model__min_samples_leaf": randint(1, 40),
             "model__max_features": ["sqrt", 0.3, 0.5]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RS),
            {"model__n_estimators": randint(100, 500),
             "model__learning_rate": loguniform(0.01, 0.2),
             "model__max_depth": randint(2, 5),
             "model__subsample": uniform(0.6, 0.4),
             "model__min_samples_leaf": randint(5, 60)},
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=RS),
            {"model__n_estimators": randint(100, 600),
             "model__learning_rate": loguniform(0.01, 0.2),
             "model__max_depth": randint(2, 6),
             "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.5, 0.5),
             "model__min_child_weight": randint(1, 20),
             "model__reg_lambda": loguniform(0.1, 20)},
        ),
    }
 
 
def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=configs.CV_FOLDS, shuffle=True, random_state=RS)
 
 
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "brier": "neg_brier_score"}
 
 
def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
 
 
@dataclass
class ModelResult:
    name: str
    pipeline: object
    best_params: dict
    cv: dict
    oof_proba: np.ndarray
    test_proba: np.ndarray
    threshold: float
    test_default: dict = field(default_factory=dict)
    test_tuned: dict = field(default_factory=dict)
    campaign_default: dict = field(default_factory=dict)
    campaign_tuned: dict = field(default_factory=dict)
 
    def summary(self) -> dict:
        return _jsonable({
            "best_params": {k.replace("model__", ""): v for k, v in self.best_params.items()},
            "cv": self.cv,
            "optimal_threshold": self.threshold,
            "test_at_0.5": self.test_default,
            "test_at_optimal": self.test_tuned,
            "campaign_test_at_0.5": self.campaign_default,
            "campaign_test_at_optimal": self.campaign_tuned,
        })
 
 
def tune(name: str, estimator, params: dict, X: pd.DataFrame, y: pd.Series,
         n_iter: int = configs.SEARCH_ITERATIONS) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        build_pipeline(estimator), params, n_iter=n_iter, scoring=SCORING, refit="roc_auc",
        cv=cv_splitter(), n_jobs=-1, random_state=RS, error_score="raise")
    search.fit(X, y)
    return search
 
 
def evaluate(name: str, search: RandomizedSearchCV, X_train, y_train, X_test, y_test) -> ModelResult:
    i = search.best_index_
    res = search.cv_results_
    cv = {
        "roc_auc_mean": res["mean_test_roc_auc"][i],
        "roc_auc_std": res["std_test_roc_auc"][i],
        "pr_auc_mean": res["mean_test_pr_auc"][i],
        "pr_auc_std": res["std_test_pr_auc"][i],
        "brier_mean": -res["mean_test_brier"][i],
    }
 
    oof = cross_val_predict(clone(search.best_estimator_), X_train, y_train, cv=cv_splitter(),
                            method="predict_proba", n_jobs=-1)[:, 1]
    threshold = optimal_threshold(y_train, oof, X_train[VALUE_COL])
    test_proba = search.best_estimator_.predict_proba(X_test)[:, 1]
    value = X_test[VALUE_COL]
 
    result = ModelResult(name, search.best_estimator_, search.best_params_, cv, oof,
                         test_proba, threshold)
    result.test_default = classification_metrics(y_test, test_proba, 0.5)
    result.test_default["lift_top_10pct"] = lift_at(y_test, test_proba, 0.10)
    result.test_tuned = classification_metrics(y_test, test_proba, threshold)
    result.campaign_default = campaign_outcome(y_test, test_proba, value, threshold=0.5)
    result.campaign_tuned = campaign_outcome(y_test, test_proba, value, threshold=threshold)
    return result
 
 
def policy_comparison(y_test, proba, value, threshold: float) -> list[dict]:
    """What each targeting rule would have earned on the held-out customers."""
    rows = [
        ("Contact nobody", np.zeros(len(proba), dtype=bool)),
        ("Contact everyone", np.ones(len(proba), dtype=bool)),
        ("Default threshold (0.50)", proba >= 0.5),
        (f"Money-optimal threshold ({threshold:.2f})", proba >= threshold),
        ("Expected-value rule (per customer)", expected_value_target(proba, value)),
    ]
    return [{"policy": label, **campaign_outcome(y_test, proba, value, target=mask)}
            for label, mask in rows]
 
 
def imbalance_experiment(results: dict[str, ModelResult], X_train, y_train, X_test, y_test) -> list[dict]:
    """Same tuned models, with and without class re-weighting."""
    rows = []
    for name in ("Logistic Regression", "XGBoost"):
        base = results[name].pipeline
        pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        variants = {"none": clone(base), "balanced": clone(base)}
        if name == "XGBoost":
            variants["balanced"].set_params(model__scale_pos_weight=pos_weight)
        else:
            variants["balanced"].set_params(model__class_weight="balanced")
        for weighting, pipe in variants.items():
            cv = cross_validate(pipe, X_train, y_train, cv=cv_splitter(), scoring=SCORING, n_jobs=-1)
            oof = cross_val_predict(pipe, X_train, y_train, cv=cv_splitter(),
                                    method="predict_proba", n_jobs=-1)[:, 1]
            t = optimal_threshold(y_train, oof, X_train[VALUE_COL])
            proba = pipe.fit(X_train, y_train).predict_proba(X_test)[:, 1]
            rows.append({
                "model": name, "class_weighting": weighting,
                "cv_roc_auc": cv["test_roc_auc"].mean(),
                "cv_brier": -cv["test_brier"].mean(),
                "mean_predicted_prob": float(oof.mean()),
                "actual_churn_rate": float(y_train.mean()),
                "optimal_threshold": t,
                "test_net_value": campaign_outcome(y_test, proba, X_test[VALUE_COL],
                                                   threshold=t)["net_value"],
            })
    return _jsonable(rows)
 
 
def feature_ablation(champion: ModelResult, X_train, y_train) -> list[dict]:
    """Does the feature engineering earn its keep? Same model, two feature sets."""
    rows = []
    for label, numeric in (("Raw columns only", configs.BASE_NUMERIC),
                           ("Raw + engineered features", configs.NUMERIC_FEATURES)):
        pipe = build_pipeline(clone(champion.pipeline.named_steps["model"]),
                              numeric, configs.CATEGORICAL_FEATURES)
        cv = cross_validate(pipe, X_train[numeric + configs.CATEGORICAL_FEATURES], y_train,
                            cv=cv_splitter(), scoring=SCORING, n_jobs=-1)
        rows.append({"feature_set": label, "n_features": len(numeric) + len(configs.CATEGORICAL_FEATURES),
                     "cv_roc_auc": cv["test_roc_auc"].mean(), "cv_roc_auc_std": cv["test_roc_auc"].std(),
                     "cv_pr_auc": cv["test_pr_auc"].mean()})
    return _jsonable(rows)
 
 
def permutation_importances(champion: ModelResult, X_test, y_test) -> pd.DataFrame:
    """Model-agnostic cross-check on SHAP: ROC-AUC drop when a feature is shuffled."""
    imp = permutation_importance(champion.pipeline, X_test, y_test, scoring="roc_auc",
                                 n_repeats=10, random_state=RS, n_jobs=-1)
    return (pd.DataFrame({"feature": X_test.columns, "importance_mean": imp.importances_mean,
                          "importance_std": imp.importances_std})
            .sort_values("importance_mean", ascending=False, ignore_index=True))
 
 
def train_and_compare(features: pd.DataFrame, verbose: bool = True) -> dict:
    X, y = split_xy(features)
    ids = features[configs.ID_COL]
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=configs.TEST_SIZE, stratify=y, random_state=RS)
 
    results: dict[str, ModelResult] = {}
    for name, (estimator, params) in model_space().items():
        search = tune(name, estimator, params, X_train, y_train)
        results[name] = evaluate(name, search, X_train, y_train, X_test, y_test)
        if verbose:
            r = results[name]
            print(f"  {name:<20} CV ROC-AUC {r.cv['roc_auc_mean']:.4f} ± {r.cv['roc_auc_std']:.4f}"
                  f" | test ROC-AUC {r.test_default['roc_auc']:.4f}"
                  f" | t*={r.threshold:.2f} net ${r.campaign_tuned['net_value']:,.0f}")
 
    champion_name = max(results, key=lambda n: results[n].cv[f"{configs.SELECTION_METRIC}_mean"])
    champion = results[champion_name]
 
    report = {
        "dataset": {"customers": int(len(X)), "train": int(len(X_train)), "test": int(len(X_test)),
                    "churn_rate": float(y.mean()), "n_model_features": len(configs.MODEL_FEATURES)},
        "economics": configs.ECONOMICS.as_dict(),
        "selection_metric": f"cv_{configs.SELECTION_METRIC}",
        "champion": champion_name,
        "models": {n: r.summary() for n, r in results.items()},
        "policy_comparison_test": _jsonable(policy_comparison(
            y_test.to_numpy(), champion.test_proba, X_test[VALUE_COL].to_numpy(),
            champion.threshold)),
    }
    if verbose:
        print("  running class-imbalance experiment ...")
    report["imbalance_experiment"] = imbalance_experiment(results, X_train, y_train, X_test, y_test)
    if verbose:
        print("  running feature ablation ...")
    report["feature_ablation"] = feature_ablation(champion, X_train, y_train)
 
    test_predictions = pd.DataFrame({configs.ID_COL: id_test.to_numpy(), "churn": y_test.to_numpy(),
                                     VALUE_COL: X_test[VALUE_COL].to_numpy()})
    for n, r in results.items():
        test_predictions[n] = r.test_proba
 
    return {
        "report": report,
        "results": results,
        "champion": champion,
        "splits": (X_train, X_test, y_train, y_test),
        "test_predictions": test_predictions,
        "permutation_importance": permutation_importances(champion, X_test, y_test),
    }
 
 
def save_report(report: dict) -> None:
    configs.METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
