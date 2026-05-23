# ML_Quasit_2D_Perovskites

Machine learning models for quasi-2D perovskite blue LEDs: **photon energy (eV)**, **emission wavelength (nm)**, and **EQE (%)** prediction from molecular descriptors and device metadata.

## Repository layout

```
ML_Quasit_2D_Perovskites/
├── configs/                          # Experiment YAML files (grouped by target)
│   ├── emission_photon_energy/       # Predict PHOTON_ENERGY_EV
│   ├── emission_wavelength_nm/       # Predict EMISSION_PEAK_NM
│   └── eqe_percent/                  # Predict EQE_percent
├── data/                             # SF_DATA_EMISSION_PEAK.xlsx (see data/README.md)
├── notebooks/                        # Jupyter notebooks (names match configs/)
│   ├── emission_photon_energy/
│   ├── emission_wavelength_nm/
│   └── eqe_percent/
├── outputs/                          # Figures + benchmark CSVs (gitignored)
├── scripts/
│   ├── run_experiment.py             # CLI to run experiments from configs
│   └── import_notebooks_from_rana.py # Re-import notebooks from Documents/rana
└── src/ml_quasit/                    # Python package
    ├── experiment_config.py          # YAML settings → ExperimentConfig
    ├── excel_dataset_loader.py       # Excel load, EQE cleaning, solvent encoding
    ├── iupac_to_smiles.py            # IUPAC → SMILES conversion
    ├── rdkit_descriptor_features.py  # RDKit 2D molecular descriptors
    ├── train_test_feature_selection.py # Split, correlation filter, RFECV/RFE
    ├── regression_model_trainer.py   # CatBoost, XGBoost, RF, SVR, GPR training
    ├── experiment_results_plots.py   # Heatmaps, RFECV curve, SHAP figures
    └── run_experiment.py             # End-to-end experiment orchestration
```

## Python modules (what each file does)

| File | Responsibility |
|------|----------------|
| `experiment_config.py` | Load YAML experiment settings |
| `excel_dataset_loader.py` | Read Excel sheets and prepare raw tables |
| `iupac_to_smiles.py` | Convert organic spacer IUPAC names to SMILES |
| `rdkit_descriptor_features.py` | Compute and merge RDKit descriptor columns |
| `train_test_feature_selection.py` | Stratified split, drop correlated features, RFECV/RFE |
| `regression_model_trainer.py` | Train/tune regressors and build benchmark table |
| `experiment_results_plots.py` | Save publication-style figures |
| `run_experiment.py` | Wire all steps together for one config |

## Experiments (config + notebook pairs)

Each experiment has a matching **config** (`.yaml`) and **notebook** (`.ipynb`) with the same base name.

| Config & notebook | Original folder | Target | Feature selection | Tuning |
|-------------------|-----------------|--------|-------------------|--------|
| `emission_photon_energy/baseline_rfecv` | Emission | PHOTON_ENERGY_EV | RFECV | No |
| `emission_photon_energy/tuned_rfecv` | Emission_2 | PHOTON_ENERGY_EV | RFECV | Yes |
| `emission_photon_energy/tuned_select_from_model` | Emission_3 | PHOTON_ENERGY_EV | SelectFromModel | Yes |
| `emission_photon_energy/add_hydrogens_baseline_rfecv` | Emission_H_add | PHOTON_ENERGY_EV | RFECV + AddHs | No |
| `emission_photon_energy/add_hydrogens_tuned_rfecv` | Emission_H_add_Opty | PHOTON_ENERGY_EV | RFECV + AddHs | Yes |
| `emission_photon_energy/add_hydrogens_tuned_select_from_model` | Emission_H_add_opty_woRFECV | PHOTON_ENERGY_EV | SelectFromModel + AddHs | Yes |
| `emission_wavelength_nm/select_from_model_baseline` | Emission_WL | EMISSION_PEAK_NM | SelectFromModel | No |
| `emission_wavelength_nm/rfecv_baseline` | Emission_WL_2 | EMISSION_PEAK_NM | RFECV | No |
| `emission_wavelength_nm/rfecv_tuned` | Emission_WL_3 | EMISSION_PEAK_NM | RFECV | Yes |
| `eqe_percent/rfe_20_features_tuned` | EQE | EQE_percent | RFE (20) | Yes |
| `eqe_percent/catboost_only` | EQE 3_CB_alone | EQE_percent | SelectFromModel + HTL/ETL | CatBoost only |
| `eqe_percent/multi_model_tuned_plus_catboost` *(notebook only)* | EQE 2 | EQE_percent | SelectFromModel + HTL/ETL | Multi-model + CatBoost |

`multi_model_tuned_plus_catboost.ipynb` includes hyperparameter tuning for RF, XGBoost, CatBoost, SVR, and GPR, then a CatBoost device-architecture section. Only the CatBoost-only workflow is available via `eqe_percent/catboost_only.yaml` today.

## Setup

```bash
conda create -n ml-quasit python=3.11 -y
conda activate ml-quasit
pip install -r requirements.txt
pip install -e .
```

Copy your Excel file to `data/SF_DATA_EMISSION_PEAK.xlsx` (see [data/README.md](data/README.md)).

## Run experiments

```bash
python scripts/run_experiment.py --config configs/emission_photon_energy/baseline_rfecv.yaml
python scripts/run_experiment.py --all
```

Results are saved under `outputs/<experiment_name>/`.

## Notebooks

```
notebooks/
├── emission_photon_energy/     # PHOTON_ENERGY_EV (eV)
├── emission_wavelength_nm/     # EMISSION_PEAK_NM (nm)
└── eqe_percent/                # EQE_percent (%)
```

Example: open `notebooks/emission_photon_energy/baseline_rfecv.ipynb` — same experiment as `configs/emission_photon_energy/baseline_rfecv.yaml`.

To refresh from your local `Documents/rana` folders:

```bash
python scripts/import_notebooks_from_rana.py
```
