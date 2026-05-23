"""Load Excel datasets and apply task-specific cleaning."""

from __future__ import annotations

import pandas as pd

from ml_quasit.experiment_config import ExperimentConfig
from ml_quasit.iupac_to_smiles import add_smiles_columns

COLUMNS_TO_OMIT = [
    "REFERENCE_DOI",
    "PRIMARY_ORGANIC_SPACER_IUPAC",
    "SECONDARY_ORGANIC_SPACER_IUPAC",
    "PRIMARY_ORGANIC_SPACER_SMILES",
    "SECONDARY_ORGANIC_SPACER_SMILES",
    "PRIMARY_ORGANIC_SPACER_MOL",
    "SECONDARY_ORGANIC_SPACER_MOL",
    "PRIMARY_ORGANIC_SPACER_STRUCT",
    "SECONDARY_ORGANIC_SPACER_STRUCT",
]


def load_raw_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Read the Excel sheet defined in the pipeline config."""
    if not config.data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {config.data_file}\n"
            "Place SF_DATA_EMISSION_PEAK.xlsx in the data/ folder "
            "(see data/README.md)."
        )
    return pd.read_excel(config.data_file, sheet_name=config.excel_sheet)


def preprocess_eqe(dataset: pd.DataFrame) -> pd.DataFrame:
    """Clean categorical and numeric columns for EQE workflows."""
    out = dataset.copy()
    out.replace("NaN", pd.NA, inplace=True)
    out.replace("N/A", pd.NA, inplace=True)
    out["Additive"] = out["Additive"].fillna("None").astype(str)
    out["HTL"] = out["HTL"].fillna("Unknown_HTL").astype(str)
    out["ETL"] = out["ETL"].fillna("Unknown_ETL").astype(str)
    out["Processing_Temp_C"] = pd.to_numeric(out["Processing_Temp_C"])
    out["Processing_Temp_C"] = out["Processing_Temp_C"].fillna(
        out["Processing_Temp_C"].median()
    )
    out["Annealing_Time_min"] = pd.to_numeric(out["Annealing_Time_min"])
    out["Annealing_Time_min"] = out["Annealing_Time_min"].fillna(
        out["Annealing_Time_min"].median()
    )
    return out


def encode_solvent_one_hot(dataset: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the SOLVENT column."""
    from sklearn.preprocessing import OneHotEncoder

    out = dataset.copy()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    solvent_df = out[["SOLVENT"]]
    encoded_array = encoder.fit_transform(solvent_df)
    encoded_df = pd.DataFrame(
        encoded_array, columns=encoder.get_feature_names_out(["SOLVENT"])
    )
    return pd.concat([out.drop("SOLVENT", axis=1), encoded_df], axis=1)


def prepare_dataset(config: ExperimentConfig) -> pd.DataFrame:
    """Load, add SMILES, encode solvent, and drop intermediate columns."""
    dataset = load_raw_dataset(config)
    if config.eqe_preprocessing:
        dataset = preprocess_eqe(dataset)
    dataset = add_smiles_columns(dataset)
    dataset = encode_solvent_one_hot(dataset)
    return dataset
