""" Model-side feature handeling

Row level business features (average spend, service count, price per service,
...) are computed in SQL - see sql/01_create_scheme.sql. What remains here is 
everything that must be *fitted*: scaling and one-hot encoding. Keeping those
inside a sci-kit learn Pipeline means they are learned on the training fold only
and cannot leak information from the test set.
"""

from __future__ import annotations

import numpy as np 
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import configs

def build_preprocessor(numeric: list[str] = configs.NUMERIC_FEATURES,
                        categorical: list[str] = configs.CATEGORICAL_FEATURES) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), list(numeric)),
            ("cat", OneHotEncoder(handle_unknown="ignore",
                                  drop='if_binary', sparse_output=False), list(categorical)),
        ],
        verbose_feature_names_out=False,
    )

def build_pipeline(model, numeric: list[str] = configs.NUMERIC_FEATURES,
                   categorical: list[str] = configs.CATEGORICAL_FEATURES) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor(numeric, categorical)),
                     ("model", model)])

def output_feature_groups(preprocessor: ColumnTransformer) -> list[str]:
    """ For every column the fitted preprocessor emits, the source feature it came from.

    SHAP values are additive, so summing the one-hot columns of 'contract'
    gives the contribution of "contract type" as a whole.  That is the level a
    business user reasons at.
    """
    groups: list[str] = []
    for name, transformers, columns in preprocessor.transformers_:
        if name == "num":
            groups.extend(columns)
        elif name == "cat":
            drop_idx = transformers.drop_idx_
            for i, (feature, categories) in enumerate(zip(columns, transformers.categories_)):
                dropped = drop_idx[i] if drop_idx is not None else None
                groups.extend(feature for j in range(len(categories)) if j!=dropped)
    return groups

def split_xy(features: pd.DataFrame) -> tuple[pd.DataFrame]:
    return features[configs.MODEL_FEATURES], features[configs.TARGET].astype(int)

def reference_stats(features: pd.DataFrame) -> dict[str, list[float]]:
    """0th...100th percentiles of each numeric feature on the training base, for
    percentile context in explanations ("higher than 85% of customers"). """
    grid = np.linspace(0, 100, 101)
    return {c: np.percentile(features[c].astype(float), grid).tolist()
            for c in configs.NUMERIC_FEATURES}

def percentile_of(value: float, percentile: list[float]) -> float:
    """Approximate share (0-100) of the reference base at or below `value`."""
    return float(np.interp(value, percentile,np.linspace(0, 100, len(percentile))))