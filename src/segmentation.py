"""Behavioural customer segments (unsupervised), profiled against churn.

Segmentation has no "right answer" on its own, so here it is judged by the one
thing that matters to this project: do the segments separate churn risk and
revenue at risk well enough to plan different campaigns for them?
"""
from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src import configs

SEGMENT_FEATURES = ["years_as_customer", "spend_per_year", "purchases_per_year"]


def _name(profile: pd.Series, years_as_customer_mid: float, spend_mid: float) -> str:
    years_as_customer = "Loyal" if profile["years_as_customer"] >= years_as_customer_mid else "New"
    spend = "high-spend" if profile["spend_per_year"] >= spend_mid else "low-spend"
    return f"{years_as_customer}, {spend}"


def segment_customers(features: pd.DataFrame, k: int = 4) -> tuple[pd.Series, pd.DataFrame]:
    """Return (segment label per customer, segment profile table)."""
    X = StandardScaler().fit_transform(features[SEGMENT_FEATURES])
    km = KMeans(n_clusters=k, n_init=20, random_state=configs.RANDOM_STATE).fit(X)

    frame = features[SEGMENT_FEATURES + [configs.TARGET]].copy()
    frame["cluster"] = km.labels_
    centroids = frame.groupby("cluster")[SEGMENT_FEATURES].mean()

    years_as_customer_mid = features["years_as_customer"].median()
    spend_mid = features["spend_per_year"].median()
    names = {c: _name(centroids.loc[c], years_as_customer_mid, spend_mid) for c in centroids.index}
    # Two clusters can land in the same quadrant; tell them apart by puchase frequency.
    seen: dict[str, list[int]] = {}
    for c, n in names.items():
        seen.setdefault(n, []).append(c)
    for n, clusters in seen.items():
        if len(clusters) > 1:
            for c in clusters:
                names[c] = f"{n} ({centroids.loc[c, 'purchases_per_year']:.1f} purchases/yr)"

    labels = frame["cluster"].map(names).rename("segment")
    profile = (frame.assign(segment=labels)
               .groupby("segment")
               .agg(customers=(configs.TARGET, "size"),
                    churn_rate=(configs.TARGET, "mean"),
                    avg_years_as_customer=("years_as_customer", "mean"),
                    avg_spend_per_year=("spend_per_year", "mean"),
                    avg_purchases_per_year=("purchases_per_year", "mean"))
               .sort_values("churn_rate", ascending=False))
    profile["share_of_customers"] = profile["customers"] / profile["customers"].sum()
    profile.attrs["silhouette"] = float(silhouette_score(X, km.labels_, sample_size=3000,
                                                         random_state=configs.RANDOM_STATE))
    return labels, profile.reset_index()


def choose_k(features: pd.DataFrame, ks=range(2, 8)) -> pd.DataFrame:
    """Elbow + silhouette table for the notebook."""
    X = StandardScaler().fit_transform(features[SEGMENT_FEATURES])
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=configs.RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_, sample_size=3000,
                                                    random_state=configs.RANDOM_STATE)})
    return pd.DataFrame(rows)