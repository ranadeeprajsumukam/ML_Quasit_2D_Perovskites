"""Model training, hyperparameter tuning, and evaluation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from ml_quasit.experiment_config import ExperimentConfig

warnings.filterwarnings("ignore")


def _metric_label(config: ExperimentConfig) -> str:
    if config.task == "eqe":
        return "EQE %"
    if config.target_column == "EMISSION_PEAK_NM":
        return "nm"
    return "eV"


def evaluate_models(
    models: dict,
    X_test_map: dict[str, pd.DataFrame],
    y_test: pd.Series,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Evaluate fitted models and return a benchmark table."""
    unit = _metric_label(config)
    results = []
    for name, model in models.items():
        X_te = X_test_map.get(name, X_test_map["default"])
        y_pred = model.predict(X_te)
        results.append(
            {
                "Model": name,
                "R2 Score": r2_score(y_test, y_pred),
                f"RMSE ({unit})": np.sqrt(mean_squared_error(y_test, y_pred)),
                f"MAE ({unit})": mean_absolute_error(y_test, y_pred),
            }
        )
    df = pd.DataFrame(results)
    rmse_col = f"RMSE ({unit})"
    return df.sort_values(by=rmse_col, ascending=True).reset_index(drop=True)


def train_emission_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ExperimentConfig,
) -> tuple[dict, pd.DataFrame, StandardScaler | None]:
    """Train emission models with optional RandomizedSearchCV tuning."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if config.hyperparameter_tuning:
        param_grids = {
            "Random Forest": {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
            },
            "XGBoost": {
                "n_estimators": [100, 300, 500],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
            },
            "CatBoost": {
                "iterations": [300, 500, 800],
                "depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
            },
            "SVR (RBF)": {
                "C": [0.1, 1, 10, 50],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "epsilon": [0.01, 0.1, 0.2],
            },
        }
        base = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42),
            "CatBoost": CatBoostRegressor(random_seed=42, verbose=0),
            "SVR (RBF)": SVR(kernel="rbf"),
        }
        models = {}
        for name, est in base.items():
            print(f"Tuning {name}...")
            search = RandomizedSearchCV(
                est,
                param_grids[name],
                n_iter=15,
                cv=5,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train_s, y_train)
            models[name] = search.best_estimator_
            print(f"  Best params: {search.best_params_}")
        gpr = GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
            random_state=42,
            n_restarts_optimizer=5,
        )
        gpr.fit(X_train_s, y_train)
        models["Gaussian Process"] = gpr
    else:
        models = {
            "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
            "CatBoost": CatBoostRegressor(
                iterations=500, learning_rate=0.05, depth=5, random_seed=42, verbose=0
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42
            ),
            "SVR (RBF)": SVR(kernel="rbf", C=10.0, epsilon=0.01),
            "Gaussian Process": GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1),
                random_state=42,
                n_restarts_optimizer=5,
            ),
        }
        for model in models.values():
            model.fit(X_train_s, y_train)

    test_map = {name: X_test_s for name in models}
    benchmark = evaluate_models(models, test_map, y_test, config)
    return models, benchmark, scaler


def train_eqe_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ExperimentConfig,
    X_train_char: pd.DataFrame | None = None,
    X_test_char: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Train EQE models (tuned tree models or CatBoost architecture-only)."""
    if config.eqe_catboost_architecture_only:
        if X_train_char is None:
            raise ValueError("eqe_catboost_architecture_only requires character columns")
        X_train_full = pd.concat([X_train.fillna(0), X_train_char], axis=1)
        X_test_full = pd.concat([X_test.fillna(0), X_test_char], axis=1)
        cat_features = list(X_train_char.columns)
        for col in cat_features:
            X_train_full[col] = X_train_full[col].astype(str)
            X_test_full[col] = X_test_full[col].astype(str)

        model = CatBoostRegressor(
            cat_features=cat_features, verbose=0, random_seed=42
        )
        cat_grid = {
            "iterations": [300, 500, 800, 1000],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "depth": [4, 5, 6, 7],
            "l2_leaf_reg": [3, 5, 10, 15],
        }
        print("Tuning CatBoost with device architecture...")
        model.randomized_search(
            cat_grid, X=X_train_full, y=y_train, cv=5, n_iter=10, verbose=False, plot=False
        )
        models = {"CatBoost Architecture": model}
        test_map = {"CatBoost Architecture": X_test_full}
        return models, evaluate_models(models, test_map, y_test, config)

    string_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    cat_features = [c for c in X_train.columns if c in string_cols]
    num_features = [c for c in X_train.columns if c not in string_cols]

    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()
    for col in cat_features:
        X_train_cat[col] = X_train_cat[col].fillna("Unknown").astype(str)
        X_test_cat[col] = X_test_cat[col].fillna("Unknown").astype(str)

    imputer = SimpleImputer(strategy="median")
    X_train_num = pd.DataFrame(
        imputer.fit_transform(X_train[num_features]), columns=num_features
    )
    X_test_num = pd.DataFrame(
        imputer.transform(X_test[num_features]), columns=num_features
    )

    models = {}
    test_map = {}

    print("Tuning XGBoost...")
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42),
        {
            "n_estimators": [100, 300, 500, 800],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 5, 6, 7],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 2, 4],
        },
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        random_state=42,
    )
    xgb_search.fit(X_train_num, y_train)
    models["Tuned XGBoost"] = xgb_search.best_estimator_
    test_map["Tuned XGBoost"] = X_test_num

    print("Tuning CatBoost...")
    cat = CatBoostRegressor(cat_features=cat_features, verbose=0, random_seed=42)
    cat.randomized_search(
        {
            "iterations": [300, 500, 800],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "depth": [4, 5, 6, 8],
            "l2_leaf_reg": [1, 3, 5, 9],
            "random_strength": [0.1, 1, 5],
        },
        X=X_train_cat,
        y=y_train,
        cv=5,
        n_iter=20,
        verbose=False,
        plot=False,
    )
    models["Tuned CatBoost"] = cat
    test_map["Tuned CatBoost"] = X_test_cat

    print("Tuning Random Forest...")
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [200, 400, 600],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 1.0],
        },
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        random_state=42,
    )
    rf_search.fit(X_train_num, y_train)
    models["Tuned Random Forest"] = rf_search.best_estimator_
    test_map["Tuned Random Forest"] = X_test_num

    return models, evaluate_models(models, test_map, y_test, config)
