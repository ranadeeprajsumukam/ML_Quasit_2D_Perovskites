"""End-to-end ML pipeline orchestration."""

from __future__ import annotations

import warnings

import pandas as pd

from ml_quasit.experiment_config import ExperimentConfig, load_config
from ml_quasit.train_test_feature_selection import (
    SplitData,
    select_features,
    train_test_split_data,
)
from ml_quasit.rdkit_descriptor_features import build_feature_matrix
from ml_quasit.regression_model_trainer import train_emission_models, train_eqe_models
from ml_quasit.experiment_results_plots import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_rfecv_curve,
    plot_shap_emission,
    plot_shap_eqe,
)

warnings.filterwarnings("ignore")


def _feature_importances(selector, support) -> tuple:
    if hasattr(selector, "estimator_"):
        imp = selector.estimator_.feature_importances_
    elif hasattr(selector, "feature_importances_"):
        imp = selector.feature_importances_[support]
    else:
        imp = selector.feature_importances_
    return imp


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Execute the full workflow for one experiment configuration."""
    print(f"\n=== Running pipeline: {config.name} ===\n")
    dataset_final = build_feature_matrix(config)
    print(f"Feature matrix shape: {dataset_final.shape}")

    split = train_test_split_data(dataset_final, config)
    X_train_final, X_test_final, selector, support = select_features(split, config)

    if config.feature_selection == "rfecv" and hasattr(selector, "cv_results_"):
        plot_rfecv_curve(selector, config)

    plot_correlation_heatmap(X_train_final, config)
    plot_feature_importance(
        _feature_importances(selector, support),
        X_train_final.columns,
        config,
    )

    if config.task == "eqe":
        models, benchmark = train_eqe_models(
            X_train_final,
            X_test_final,
            split.y_train,
            split.y_test,
            config,
            split.X_train_char,
            split.X_test_char,
        )
        if not config.eqe_catboost_architecture_only:
            train_map = {}
            string_cols = X_train_final.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()
            cat_features = [c for c in X_train_final.columns if c in string_cols]
            num_features = [c for c in X_train_final.columns if c not in string_cols]
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            X_num = pd.DataFrame(
                imputer.fit_transform(X_train_final[num_features]),
                columns=num_features,
            )
            X_cat = X_train_final.copy()
            for col in cat_features:
                X_cat[col] = X_cat[col].fillna("Unknown").astype(str)
            for name in models:
                train_map[name] = X_cat if "CatBoost" in name else X_num
            plot_shap_eqe(models, train_map, config)
        else:
            X_full = pd.concat(
                [X_train_final.fillna(0), split.X_train_char], axis=1
            )
            for col in split.X_train_char.columns:
                X_full[col] = X_full[col].astype(str)
            plot_shap_eqe(
                models, {"CatBoost Architecture": X_full}, config
            )
    else:
        models, benchmark, scaler = train_emission_models(
            X_train_final,
            X_test_final,
            split.y_train,
            split.y_test,
            config,
        )
        if scaler is not None:
            plot_shap_emission(
                models, X_train_final, X_test_final, scaler, config
            )

    out_csv = config.results_dir / "benchmark_results.csv"
    benchmark.to_csv(out_csv, index=False)
    print("\n--- Benchmark Results ---")
    print(benchmark.to_string(index=False))
    print(f"\nResults saved to: {config.results_dir}")
    return benchmark


def main(config_path: str) -> None:
    """CLI entry: python -m ml_quasit.run_experiment configs/foo.yaml"""
    config = load_config(config_path)
    run_experiment(config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ml_quasit.run_experiment <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
