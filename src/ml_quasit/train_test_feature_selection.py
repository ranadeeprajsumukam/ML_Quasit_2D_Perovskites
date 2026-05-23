"""Correlation pruning, RFECV, SelectFromModel, and RFE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE, SelectFromModel, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ml_quasit.experiment_config import ExperimentConfig

sklearn.set_config(enable_metadata_routing=True)

CHAR_COLUMNS_EQE = ["Additive", "Additive ratio", "HTL", "ETL"]


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    stratify_col: pd.Series
    X_train_char: pd.DataFrame | None = None
    X_test_char: pd.DataFrame | None = None


def group_rare_categories(series: pd.Series, min_count: int = 3) -> pd.Series:
    counts = series.value_counts()
    rare = counts[counts < min_count].index
    return series.replace(rare, "Other")


def build_eqe_stratify_column(X: pd.DataFrame) -> pd.Series:
    """Composite stratification profile for small EQE datasets."""
    htl = group_rare_categories(X["HTL"].astype(str))
    etl = group_rare_categories(X["ETL"].astype(str))
    additive = group_rare_categories(X["Additive"].astype(str))
    mixed = (X["IS_MIXED_SPACERS_SPACER"] > 0).astype(int).astype(str)
    add_presence = (X["Additive ratio"] > 0).astype(int).astype(str)
    profile = mixed + "_" + additive + "_" + add_presence + "_" + htl + "_" + etl
    counts = profile.value_counts()
    rare = counts[counts < 5].index
    return profile.replace(rare, "Rare_Combined_Profile")


def train_test_split_data(
    dataset_final: pd.DataFrame, config: ExperimentConfig
) -> SplitData:
    """Split features/target with task-appropriate stratification."""
    drop_cols = list(config.drop_columns) + [config.target_column]
    X = dataset_final.drop(columns=drop_cols, errors="ignore")
    y = dataset_final[config.target_column]

    if config.task == "eqe" and config.eqe_split_numeric_categorical:
        stratify_col = build_eqe_stratify_column(X)
    else:
        stratify_col = (X["IS_MIXED_SPACERS_SPACER"] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col,
    )

    char_train = char_test = None
    if config.eqe_split_numeric_categorical:
        char_train = X_train[CHAR_COLUMNS_EQE].copy()
        char_test = X_test[CHAR_COLUMNS_EQE].copy()
        X_train = X_train.drop(columns=CHAR_COLUMNS_EQE)
        X_test = X_test.drop(columns=CHAR_COLUMNS_EQE)

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        stratify_col=stratify_col,
        X_train_char=char_train,
        X_test_char=char_test,
    )


def drop_correlated_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    corr = X_train.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return (
        X_train.drop(columns=to_drop),
        X_test.drop(columns=to_drop),
        to_drop,
    )


def select_features(
    split: SplitData, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.DataFrame, object, np.ndarray]:
    """
    Run correlation filter + configured feature selector.

    Returns X_train_final, X_test_final, selector object, selected indices.
    """
    X_train = split.X_train.copy()
    X_test = split.X_test.copy()

    if config.variance_threshold is not None:
        vt = VarianceThreshold(threshold=config.variance_threshold)
        X_train = pd.DataFrame(
            vt.fit_transform(X_train),
            columns=X_train.columns[vt.get_support()],
            index=X_train.index,
        )
        X_test = X_test[X_train.columns]

    X_train_uncorr, X_test_uncorr, dropped = drop_correlated_features(
        X_train, X_test, config.correlation_threshold
    )
    print(f"Dropping {len(dropped)} highly correlated features.")

    weights = compute_sample_weight(
        class_weight="balanced",
        y=split.stratify_col.loc[X_train_uncorr.index],
    )

    selector_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=config.rfecv_max_depth,
        min_samples_leaf=config.rfecv_min_samples_leaf,
        random_state=42,
    )

    if config.feature_selection == "rfecv":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = list(
            skf.split(
                X_train_uncorr,
                split.stratify_col.loc[X_train_uncorr.index],
            )
        )
        selector_model.set_fit_request(sample_weight=True)
        selector = RFECV(
            estimator=selector_model,
            step=1,
            cv=cv_splits,
            scoring="neg_root_mean_squared_error",
            min_features_to_select=5,
            n_jobs=-1,
        )
        print("Starting RFECV feature selection...")
        selector.fit(X_train_uncorr, split.y_train, sample_weight=weights)
        support = selector.support_
        print(f"Optimal number of features: {selector.n_features_}")
    elif config.feature_selection == "rfe":
        X_enc = X_train_uncorr.copy()
        X_test_enc = X_test_uncorr.copy()
        string_cols = X_enc.select_dtypes(include=["object", "string"]).columns
        if len(string_cols):
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            X_enc[string_cols] = enc.fit_transform(X_enc[string_cols].astype(str))
            X_test_enc[string_cols] = enc.transform(
                X_test_enc[string_cols].astype(str)
            )
        selector = RFE(
            estimator=selector_model,
            n_features_to_select=config.rfe_n_features,
            step=2,
        )
        print(f"Starting RFE (n_features={config.rfe_n_features})...")
        selector.fit(X_enc, split.y_train)
        support = selector.support_
    else:
        selector_model.fit(X_train_uncorr, split.y_train, sample_weight=weights)
        selector = SelectFromModel(selector_model, prefit=True)
        support = selector.get_support()
        selector = selector_model

    best_features = X_train_uncorr.columns[support]
    print(f"Selected {len(best_features)} features.")
    return (
        X_train_uncorr[best_features],
        X_test_uncorr[best_features],
        selector,
        support,
    )
