"""RDKit molecular descriptors and dataset featurization."""

from __future__ import annotations

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from ml_quasit.experiment_config import ExperimentConfig
from ml_quasit.excel_dataset_loader import COLUMNS_TO_OMIT, prepare_dataset

_calc = MoleculeDescriptors.MolecularDescriptorCalculator(
    [x[0] for x in Descriptors._descList]
)
DESC_NAMES = _calc.GetDescriptorNames()


def get_2d_descriptors(smiles: str | float | None, add_hydrogens: bool = False):
    """Compute RDKit 2D descriptors for a SMILES string."""
    if pd.isna(smiles) or str(smiles).strip() == "":
        return [None] * len(DESC_NAMES)
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return [None] * len(DESC_NAMES)
        if add_hydrogens:
            mol = Chem.AddHs(mol)
        return _calc.CalcDescriptors(mol)
    except Exception:
        return [None] * len(DESC_NAMES)


def attach_molecular_descriptors(
    dataset: pd.DataFrame, add_hydrogens: bool = False
) -> pd.DataFrame:
    """Merge primary/secondary RDKit descriptor blocks onto the dataset."""
    primary_smiles = dataset["PRIMARY_ORGANIC_SPACER_SMILES"].dropna().unique().tolist()
    secondary_smiles = (
        dataset["SECONDARY_ORGANIC_SPACER_SMILES"].dropna().unique().tolist()
    )
    unique_smiles = list(set(primary_smiles + secondary_smiles))

    features_dict = {
        sm: get_2d_descriptors(sm, add_hydrogens=add_hydrogens)
        for sm in unique_smiles
    }
    df_unique = pd.DataFrame.from_dict(
        features_dict, orient="index", columns=DESC_NAMES
    )
    df_primary = df_unique.add_prefix("Pri_")
    df_secondary = df_unique.add_prefix("Sec_")

    out = dataset.merge(
        df_primary,
        how="left",
        left_on="PRIMARY_ORGANIC_SPACER_SMILES",
        right_index=True,
    )
    out = out.merge(
        df_secondary,
        how="left",
        left_on="SECONDARY_ORGANIC_SPACER_SMILES",
        right_index=True,
    )
    return out


def build_feature_matrix(config: ExperimentConfig) -> pd.DataFrame:
    """Full featurization pipeline from raw Excel to model-ready matrix."""
    dataset = prepare_dataset(config)
    dataset = attach_molecular_descriptors(
        dataset, add_hydrogens=config.add_hydrogens
    )
    dataset_final = dataset.drop(columns=COLUMNS_TO_OMIT, errors="ignore")
    if config.fill_na_with_zero:
        dataset_final = dataset_final.fillna(0)
    return dataset_final
