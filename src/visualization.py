"""Static figures for the README and notebooks, built from saved artifacts.

    python -m src.visualization     # regenerate reports/figures/*.png

Reads only files written by the pipeline, so charts can be restyled without
retraining anything.
"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src import configs as config  # noqa: E402
from src.data_preprocessing import run_named_query  # noqa: E402

TARGET = config.TARGET

# --------------------------------------------------------------------------- #
# Palette (validated categorical order; blue<->red diverging; one-hue sequential)
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STAYED, CHURNED = SERIES[0], SERIES[1]
RISK_UP, RISK_DOWN = "#e34948", "#2a78d6"   # diverging poles for SHAP direction
BAR = SERIES[0]
BAR_MUTED = "#b7d3f6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "font.family": ["Segoe UI", "Arial", "DejaVu Sans"],
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "axes.axisbelow": True, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})

money = FuncFormatter(lambda v, _: f"-${abs(v) / 1000:,.0f}k" if v < 0 else f"${v / 1000:,.0f}k")


def _save(fig, name: str) -> None:
    fig.savefig(config.FIGURES_DIR / name)
    plt.close(fig)


def _hbar(ax, labels, values, highlight_max=True, fmt="{:.0f}%", ref=None):
    colors = [BAR if (not highlight_max or v == max(values)) else BAR_MUTED for v in values]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for yi, v in zip(y, values):
        ax.text(v, yi, "  " + fmt.format(v), va="center", ha="left", fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.22)
    if ref is not None:
        ax.axvline(ref, color=INK_3, linewidth=1, linestyle=(0, (3, 3)))


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def fig_churn_by_segment() -> None:
    overall = run_named_query("kpi_overview")["churn_rate_pct"].iloc[0]
    panels = [("churn_by_satisfaction", "Satisfaction score"),
              ("churn_by_tenure_bucket", "Years as customer"),
              ("churn_by_recency_bucket", "Days since last purchase"),
              ("churn_by_income_band", "Income band"),
              ("churn_by_age_group", "Age group"),
              ("churn_by_support_contacts", "Support contacts")]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    for ax, (query, title) in zip(axes.flat, panels):
        df = run_named_query(query)
        _hbar(ax, df["segment"].astype(str), df["churn_rate_pct"].tolist(), ref=overall)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    fig.suptitle(f"Churn rate by customer segment  (dotted line = all customers, {overall:.1f}%)",
                 x=0.01, ha="left", fontsize=14, fontweight="semibold", color=INK)
    fig.tight_layout()
    _save(fig, "eda_churn_by_segment.png")


def fig_tenure_curve() -> None:
    df = run_named_query("churn_by_tenure_year")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(df["years_as_customer"], df["churn_rate_pct"], color=BAR)
    a1.set_title("Churn rate by years as customer")
    a1.set_xlabel("Years as customer")
    a1.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    a2.plot(df["years_as_customer"], df["cumulative_share_of_churners_pct"], color=BAR)
    a2.set_title("Cumulative share of all churners")
    a2.set_xlabel("Years as customer")
    a2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    for years in (1, 3, 5):
        hit = df.loc[df["years_as_customer"] == years, "cumulative_share_of_churners_pct"]
        if hit.empty:
            continue
        v = hit.iloc[0]
        a2.plot(years, v, "o", color=BAR, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        a2.annotate(f"{v:.0f}% by year {years}", (years, v), xytext=(10, -4),
                    textcoords="offset points", fontsize=9, color=INK)
    fig.tight_layout()
    _save(fig, "eda_tenure_curve.png")


def fig_spend_distribution(features: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, title in ((axes[0], "spend_per_year", "Spend per year ($)"),
                           (axes[1], "last_purchase_days_ago", "Days since last purchase")):
        bins = np.linspace(features[col].min(), features[col].max(), 36)
        for label, value, color in (("Stayed", 0, STAYED), ("Churned", 1, CHURNED)):
            ax.hist(features.loc[features[TARGET] == value, col], bins=bins, density=True,
                    histtype="step", linewidth=2, color=color, label=label)
        ax.set_title(f"{title}: churned vs. stayed")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "eda_distributions.png")


def fig_correlation(features: pd.DataFrame) -> None:
    cols = config.NUMERIC_FEATURES + [TARGET]
    corr = features[cols].corr()
    labels = [config.FEATURE_LABELS.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), labels)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title("Correlation between numeric features and churn")
    _save(fig, "eda_correlation.png")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def fig_roc_pr(report: dict, preds: pd.DataFrame) -> None:
    names = list(report["models"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for i, name in enumerate(names):
        fpr, tpr, _ = roc_curve(preds[TARGET], preds[name])
        prec, rec, _ = precision_recall_curve(preds[TARGET], preds[name])
        auc = report["models"][name]["test_at_0.5"]["roc_auc"]
        ap = report["models"][name]["test_at_0.5"]["pr_auc"]
        width = 2.6 if name == report["champion"] else 1.6
        color = SERIES[i % len(SERIES)]
        a1.plot(fpr, tpr, color=color, linewidth=width, label=f"{name} ({auc:.3f})")
        a2.plot(rec, prec, color=color, linewidth=width, label=f"{name} ({ap:.3f})")
    a1.plot([0, 1], [0, 1], color=INK_3, linewidth=1, linestyle=(0, (3, 3)))
    a2.axhline(preds[TARGET].mean(), color=INK_3, linewidth=1, linestyle=(0, (3, 3)))
    a1.set(title="ROC curve (test set) - legend: ROC-AUC", xlabel="False positive rate",
           ylabel="True positive rate (recall)")
    a2.set(title="Precision-recall curve - legend: PR-AUC", xlabel="Recall", ylabel="Precision")
    a1.legend(loc="lower right")
    a2.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "model_roc_pr.png")


def fig_calibration(report: dict, preds: pd.DataFrame) -> None:
    name = report["champion"]
    frac, mean_pred = calibration_curve(preds[TARGET], preds[name], n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=INK_3, linewidth=1, linestyle=(0, (3, 3)), label="Perfect calibration")
    ax.plot(mean_pred, frac, "-o", color=BAR, markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=2, label=name)
    ax.set(title="Calibration (test set)", xlabel="Predicted churn probability",
           ylabel="Observed churn rate")
    ax.legend(loc="upper left")
    _save(fig, "model_calibration.png")


def fig_threshold_economics() -> None:
    econ = json.loads(config.THRESHOLD_JSON.read_text(encoding="utf-8"))
    test = pd.DataFrame(econ["curve_test"])
    oof = pd.DataFrame(econ["curve_train_oof"])
    t_star = econ["chosen_threshold"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    # put the training curve on the test set's scale (it covers ~4x as many customers)
    scale = test["churners_total"].iloc[0] / oof["churners_total"].iloc[0]
    a1.plot(oof["threshold"], oof["net_value"] * scale, color=SERIES[1], linewidth=1.6,
            label="Training OOF (used to choose)")
    a1.plot(test["threshold"], test["net_value"], color=BAR, label="Test set (held out)")
    a1.axhline(0, color=INK_3, linewidth=1)
    for t, label in ((0.5, "0.50 default"), (t_star, f"{t_star:.2f} chosen")):
        v = test.loc[np.isclose(test["threshold"], t), "net_value"].iloc[0]
        a1.axvline(t, color=INK_3, linewidth=1, linestyle=(0, (3, 3)))
        a1.plot(t, v, "o", color=BAR, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        a1.annotate(f"{label}\n${v:,.0f}", (t, v), xytext=(6, 8), textcoords="offset points",
                    fontsize=9, color=INK)
    a1.yaxis.set_major_formatter(money)
    a1.set(title="Campaign net value vs. decision threshold", xlabel="Contact if P(churn) >=")
    a1.legend(loc="lower left")

    pol = pd.DataFrame(econ["policy_comparison_test"])
    colors = [BAR if v == pol["net_value"].max() else BAR_MUTED for v in pol["net_value"]]
    y = np.arange(len(pol))[::-1]
    a2.barh(y, pol["net_value"], color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    a2.set_yticks(y, pol["policy"])
    a2.axvline(0, color=INK_3, linewidth=1)
    a2.grid(axis="y", visible=False)
    a2.tick_params(axis="y", length=0)
    for yi, v in zip(y, pol["net_value"]):
        a2.text(v if v > 0 else 0, yi, f"  ${v:,.0f}", va="center", ha="left", fontsize=9, color=INK)
    a2.xaxis.set_major_locator(MaxNLocator(nbins=5))   # scale differs per dataset, so no fixed step
    a2.xaxis.set_major_formatter(money)
    a2.set_xlim(min(0, pol["net_value"].min() * 1.1), pol["net_value"].max() * 1.45)
    a2.set_title("Net value by targeting rule (test set)")
    fig.tight_layout()
    _save(fig, "threshold_economics.png")


def fig_global_importance() -> None:
    imp = pd.read_csv(config.GLOBAL_IMPORTANCE_CSV).head(12)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    y = np.arange(len(imp))[::-1]
    ax.barh(y, imp["mean_abs_shap"], color=BAR, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(y, imp["label"])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set(title="What drives churn predictions (mean |SHAP|, all customers)",
           xlabel="Average impact on the prediction (log-odds)")
    _save(fig, "shap_global_importance.png")


def fig_shap_beeswarm() -> None:
    """SHAP value per customer for the top features, coloured by feature value."""
    scored = pd.read_csv(config.SCORED_CSV)
    imp = pd.read_csv(config.GLOBAL_IMPORTANCE_CSV).head(10)
    sample = scored.sample(min(2000, len(scored)), random_state=config.RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    rng = np.random.default_rng(config.RANDOM_STATE)
    for i, feature in enumerate(imp["feature"][::-1]):
        vals = sample[f"shap__{feature}"].to_numpy()
        raw = sample[feature]
        if not pd.api.types.is_numeric_dtype(raw):  # categorical: riskier level = "high"
            rank = sample.groupby(feature)[f"shap__{feature}"].mean().rank()
            color_val = raw.map((rank - 1) / max(len(rank) - 1, 1)).to_numpy()
        else:
            color_val = raw.rank(pct=True).to_numpy()
        ax.scatter(vals, i + rng.uniform(-0.28, 0.28, len(vals)), c=color_val, cmap="RdBu_r",
                   s=6, alpha=0.6, linewidths=0, vmin=0, vmax=1)
    ax.axvline(0, color=INK_3, linewidth=1)
    ax.set_yticks(range(len(imp)), imp["label"][::-1])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set(title="SHAP values per customer (red = high value / riskier level)",
           xlabel="Impact on churn log-odds  (<- keeps customer | pushes to churn ->)")
    _save(fig, "shap_beeswarm.png")


def fig_customer_explanation() -> None:
    """Waterfall-style bar chart of the reasons for the highest-loss active customer."""
    scored = pd.read_csv(config.SCORED_CSV)
    row = scored[scored[TARGET] == 0].sort_values("expected_loss", ascending=False).iloc[0]
    shap_cols = [c for c in scored.columns if c.startswith("shap__")]
    s = row[shap_cols].astype(float)
    s.index = [c[6:] for c in shap_cols]
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(8)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    y = np.arange(len(top))
    ax.barh(y, top.values, color=[RISK_UP if v > 0 else RISK_DOWN for v in top.values],
            height=0.6, edgecolor=SURFACE, linewidth=2)
    labels = [f"{config.FEATURE_LABELS.get(f, f)} = {row[f]}" for f in top.index]
    ax.set_yticks(y, labels)
    ax.axvline(0, color=INK_3, linewidth=1)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set(title=f"Why {row[config.ID_COL]} is at risk: P(churn) = {row['churn_probability']:.0%}",
           xlabel="SHAP impact (red pushes towards churn, blue keeps the customer)")
    _save(fig, "shap_customer_example.png")


def fig_segments() -> None:
    prof = pd.read_csv(config.REPORTS_DIR / "segment_profile.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    _hbar(ax, prof["segment"], (prof["churn_rate"] * 100).tolist())
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_title("Churn rate by behavioural segment (k-means)")
    _save(fig, "segments.png")


def make_all_figures() -> None:
    features = pd.read_csv(config.FEATURES_CSV)
    report = json.loads(config.METRICS_JSON.read_text(encoding="utf-8"))
    preds = pd.read_csv(config.REPORTS_DIR / "test_predictions.csv")
    fig_churn_by_segment()
    fig_tenure_curve()
    fig_spend_distribution(features)
    fig_correlation(features)
    fig_roc_pr(report, preds)
    fig_calibration(report, preds)
    fig_threshold_economics()
    fig_global_importance()
    fig_shap_beeswarm()
    fig_customer_explanation()
    fig_segments()
    print(f"  figures -> {config.FIGURES_DIR.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    make_all_figures()