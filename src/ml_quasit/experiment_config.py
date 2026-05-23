"""Experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ExperimentConfig:
    """Settings for one notebook / experiment variant."""

    name: str
    task: str  # emission_ev | emission_wl | eqe
    target_column: str
    drop_columns: list[str]
    excel_sheet: str
    data_file: Path

    add_hydrogens: bool = False
    feature_selection: str = "rfecv"  # rfecv | select_from_model | rfe
    hyperparameter_tuning: bool = False
    correlation_threshold: float = 0.90
    fill_na_with_zero: bool = True
    eqe_preprocessing: bool = False
    eqe_split_numeric_categorical: bool = False
    eqe_catboost_architecture_only: bool = False
    rfecv_max_depth: int = 5
    rfecv_min_samples_leaf: int = 6
    rfe_n_features: int = 20
    variance_threshold: float | None = None
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs")

    @property
    def results_dir(self) -> Path:
        path = self.output_dir / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config and resolve paths relative to the repo root."""
    config_path = Path(path)
    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    data_file = Path(raw.get("data_file", "data/SF_DATA_EMISSION_PEAK.xlsx"))
    if not data_file.is_absolute():
        data_file = REPO_ROOT / data_file

    output_dir = Path(raw.get("output_dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    return ExperimentConfig(
        name=raw["name"],
        task=raw["task"],
        target_column=raw["target_column"],
        drop_columns=list(raw.get("drop_columns", [])),
        excel_sheet=raw["excel_sheet"],
        data_file=data_file,
        add_hydrogens=raw.get("add_hydrogens", False),
        feature_selection=raw.get("feature_selection", "rfecv"),
        hyperparameter_tuning=raw.get("hyperparameter_tuning", False),
        correlation_threshold=raw.get("correlation_threshold", 0.90),
        fill_na_with_zero=raw.get("fill_na_with_zero", True),
        eqe_preprocessing=raw.get("eqe_preprocessing", False),
        eqe_split_numeric_categorical=raw.get("eqe_split_numeric_categorical", False),
        eqe_catboost_architecture_only=raw.get(
            "eqe_catboost_architecture_only", False
        ),
        rfecv_max_depth=raw.get("rfecv_max_depth", 5),
        rfecv_min_samples_leaf=raw.get("rfecv_min_samples_leaf", 6),
        rfe_n_features=raw.get("rfe_n_features", 20),
        variance_threshold=raw.get("variance_threshold"),
        output_dir=output_dir,
    )
