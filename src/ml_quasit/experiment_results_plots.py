"""Figures: RFECV curve, correlation heatmap, feature importance, SHAP."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler

from ml_quasit.experiment_config import ExperimentConfig


def plot_rfecv_curve(selector, config: ExperimentConfig) -> Path:
    out = config.results_dir / "Figure_S1_RFECV_Curve.png"
    scores = -1 * selector.cv_results_["mean_test_score"]
    n_start = selector.min_features_to_select
    plt.figure(figsize=(9, 6), dpi=300)
    plt.plot(
        range(n_start, len(scores) + n_start),
        scores,
        color="teal",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    plt.axvline(
        x=selector.n_features_,
        color="red",
        linestyle="--",
        label=f"Optimal: {selector.n_features_} features",
    )
    ylabel = "RMSE (nm)" if config.target_column == "EMISSION_PEAK_NM" else "RMSE (eV)"
    plt.title("RFECV: Optimal Number of Molecular Descriptors", fontsize=14)
    plt.xlabel("Number of Features Kept", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_correlation_heatmap(X_train_final: pd.DataFrame, config: ExperimentConfig) -> Path:
    suffix = "EQE" if config.task == "eqe" else (
        "nm" if config.target_column == "EMISSION_PEAK_NM" else "RFECV_eV"
    )
    out = config.results_dir / f"Figure_2_Correlation_Heatmap_Clean_{suffix}.png"
    plt.figure(figsize=(12, 10), dpi=300)
    corr = X_train_final.corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        annot_kws={"size": 7},
        cbar_kws={"label": "Pearson Correlation"},
    )
    title = (
        "Correlation Matrix of Selected Molecular Descriptors (EQE)"
        if config.task == "eqe"
        else "Correlation Matrix of Selected Molecular Descriptors"
    )
    plt.title(title, fontsize=16)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: pd.Index,
    config: ExperimentConfig,
) -> Path:
    suffix = "EQE" if config.task == "eqe" else (
        "RF_WL" if config.target_column == "EMISSION_PEAK_NM" else "RFECV_eV"
    )
    out = config.results_dir / f"Figure_3_Feature_Importance_{suffix}.png"
    order = np.argsort(importances)
    plt.figure(figsize=(10, 8), dpi=300)
    plt.barh(feature_names[order], importances[order], color="teal", edgecolor="black")
    plt.xlabel("Relative Importance Score", fontsize=12)
    plt.ylabel("Feature / Molecular Descriptor", fontsize=12)
    title = (
        "Top Predictors for Perovskite EQE (%)"
        if config.task == "eqe"
        else "Top Predictors for Perovskite Emission Peak"
    )
    plt.title(title, fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


def plot_shap_emission(
    models: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: StandardScaler,
    config: ExperimentConfig,
) -> list[Path]:
    """SHAP beeswarm plots for emission models (scaled features)."""
    saved = []
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    background = shap.sample(X_train_s, 50)

    for name, model in models.items():
        plt.figure(figsize=(10, 8), dpi=300)
        if name in ("Random Forest", "CatBoost", "XGBoost"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_train_s)
            shap_values.data = X_train.values
            shap_values.feature_names = X_train.columns.tolist()
            shap.plots.beeswarm(shap_values, max_display=15, show=False)
        else:
            explainer = shap.KernelExplainer(model.predict, background)
            raw = explainer.shap_values(X_test_s)
            shap.summary_plot(
                raw,
                features=X_test,
                feature_names=X_test.columns.tolist(),
                max_display=20,
                show=False,
            )
        tuned = "_Tuned" if config.hyperparameter_tuning else ""
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        out = config.results_dir / f"Figure_SHAP_{safe}_RFECV_eV{tuned}.png"
        plt.title(
            f"SHAP Summary: Drivers of Perovskite Emission Peak ({name})",
            fontsize=14,
            pad=20,
        )
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        saved.append(out)
    return saved


def plot_shap_eqe(models: dict, X_train_map: dict, config: ExperimentConfig) -> list[Path]:
    saved = []
    for name, model in models.items():
        plt.figure(figsize=(10, 8), dpi=300)
        X_tr = X_train_map[name]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_tr)
        shap.plots.beeswarm(shap_values, max_display=15, show=False)
        safe = name.replace(" ", "_")
        out = config.results_dir / f"Figure_SHAP_{safe}_EQE.png"
        plt.title(f"SHAP Summary: Drivers of Perovskite EQE ({name})", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        saved.append(out)
    return saved
