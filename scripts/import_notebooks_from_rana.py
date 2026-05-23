"""Copy Jupyter notebooks from original rana folders into this repository."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Destination names match configs/*.yaml and describe each experiment.
NOTEBOOK_SOURCES = {
    "notebooks/emission_photon_energy/baseline_rfecv.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission\Final_RFECV_eV_clean.ipynb"
    ),
    "notebooks/emission_photon_energy/tuned_rfecv.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission_2\Final_RFECV_eV_clean_opty.ipynb"
    ),
    "notebooks/emission_photon_energy/tuned_select_from_model.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission_3\Final_RFECV_eV_clean_opty_woRFECV.ipynb"
    ),
    "notebooks/emission_photon_energy/add_hydrogens_baseline_rfecv.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission_H_add\Final_RFECV_eV_clean_H_add.ipynb"
    ),
    "notebooks/emission_photon_energy/add_hydrogens_tuned_rfecv.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission_H_add_Opty\Final_RFECV_eV_clean_H_add_opty.ipynb"
    ),
    "notebooks/emission_photon_energy/add_hydrogens_tuned_select_from_model.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission_H_add_opty_woRFECV\Final_RFECV_eV_clean_H_add_opty_woRFECV.ipynb"
    ),
    "notebooks/emission_wavelength_nm/select_from_model_baseline.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission_WL\Final_WL_WoRFECV.ipynb"
    ),
    "notebooks/emission_wavelength_nm/rfecv_baseline.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission_WL_2\Final_WL_RFECV.ipynb"
    ),
    "notebooks/emission_wavelength_nm/rfecv_tuned.ipynb": Path(
        r"c:\Users\srika\Documents\rana\Emission_WL_3\Final_WL_RFECV_opty.ipynb"
    ),
    "notebooks/eqe_percent/rfe_20_features_tuned.ipynb": Path(
        r"c:\Users\srika\Documents\rana\EQE\EQE_1.ipynb"
    ),
    "notebooks/eqe_percent/multi_model_tuned_plus_catboost.ipynb": Path(
        r"c:\Users\srika\Documents\rana\EQE 2\EQE_2.ipynb"
    ),
    "notebooks/eqe_percent/catboost_only.ipynb": Path(
        r"c:\Users\srika\Documents\rana\EQE 3_CB_alone\EQE_2-Copy1.ipynb"
    ),
}

OLD_DATA_PATH = "C:/Users/srika/OneDrive/Desktop/SF_DATA_EMISSION_PEAK.xlsx"
NEW_DATA_PATH = "../../data/SF_DATA_EMISSION_PEAK.xlsx"



def patch_notebook_paths(nb_path: Path) -> None:
    """Replace hard-coded Desktop Excel path with repo-relative data path."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    changed = False
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        new_src = []
        for line in src:
            if OLD_DATA_PATH in line:
                line = line.replace(OLD_DATA_PATH, NEW_DATA_PATH)
                changed = True
            new_src.append(line)
        cell["source"] = new_src
    if changed:
        nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main() -> None:
    for dest_rel, src in NOTEBOOK_SOURCES.items():
        dest = REPO / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            print(f"SKIP (missing): {src}")
            continue
        shutil.copy2(src, dest)
        patch_notebook_paths(dest)
        print(f"Copied: {dest_rel}")


if __name__ == "__main__":
    main()
