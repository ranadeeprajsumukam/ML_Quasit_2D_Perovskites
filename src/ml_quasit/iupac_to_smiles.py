"""IUPAC-to-SMILES conversion and fallback lookup table."""

from __future__ import annotations

from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd

MASTER_SMILES_MAP: dict[str, str] = {
    "ethanamine": "CCN",
    "propan-1-amine": "CCCN",
    "propan-2-amine": "CC(N)C",
    "butan-1-amine": "CCCCN",
    "pentan-1-amine": "CCCCCN",
    "propane-1,3-diamine": "NCCCN",
    "pentane-1,5-diamine": "NCCCCCN",
    "N-(2-bromoethyl)propane-1,3-diamine": "BrCCNCCCN",
    "2-phenylethan-1-amine": "NCCc1ccccc1",
    "1-phenylethan-1-amine": "CC(N)c1ccccc1",
    "3,3-diphenylpropan-1-amine": "NCCC(c1ccccc1)c2ccccc2",
    "4-phenylbutan-1-amine": "NCCCCc1ccccc1",
    "4-phenylbutan-2-amine": "CC(N)CCc1ccccc1",
    "benzene-1,4-dimethanamine": "NCc1ccc(CN)cc1",
    "[1,1':3',1'':3'',1'''-quaterphenyl]-4-ylmethanamine": (
        "NCc1ccc(cc1)-c2cccc(c2)-c3cccc(c3)-c4ccccc4"
    ),
    "1-(naphthalen-1-yl)ethan-1-amine": "CC(N)c1cccc2ccccc12",
    "naphthalen-1-ylmethanamine": "NCc1cccc2ccccc12",
    "2-(2-fluorophenyl)ethan-1-amine": "NCCc1c(F)cccc1",
    "2-(3-fluorophenyl)ethan-1-amine": "NCCc1cc(F)ccc1",
    "2-(4-fluorophenyl)ethan-1-amine": "NCCc1ccc(F)cc1",
    "2-(2-methoxyphenyl)ethan-1-amine": "NCCc1c(OC)cccc1",
    "2-(3-methoxyphenyl)ethan-1-amine": "COc1cc(CCN)ccc1",
    "2-(4-methoxyphenyl)ethan-1-amine": "COc1ccc(CCN)cc1",
    "2-phenoxyethan-1-amine": "NCCOc1ccccc1",
    "2-(thiophen-2-yl)ethan-1-amine": "NCCc1cccs1",
    "2-[5-(2,2'-dimethyl-[1,1'-biphenyl]-4-yl)thiophen-2-yl]ethan-1-amine": (
        "Cc1ccccc1-c2c(C)cc(cc2)-c3ccc(CCN)s3"
    ),
    "2-[5-(3',5'-dimethyl-[1,1'-biphenyl]-4-yl)thiophen-2-yl]ethan-1-amine": (
        "Cc1cc(C)cc(c1)-c2ccc(cc2)-c3ccc(CCN)s3"
    ),
    "2-[2-(2-aminoethoxy)ethoxy]ethan-1-amine": "NCCOCCOCCN",
    "adamantan-1-amine": "NC12CC3CC(C1)CC(C3)C2",
    "ethanimidamide": "CC(=N)N",
    "guanidine": "NC(=N)N",
}


def cir_convert(iupac: str | float | None) -> str | None:
    """Resolve IUPAC name to SMILES (local map first, then CACTUS API)."""
    if pd.isna(iupac):
        return None
    key = str(iupac)
    if key in MASTER_SMILES_MAP:
        return MASTER_SMILES_MAP[key]
    try:
        url = (
            "http://cactus.nci.nih.gov/chemical/structure/"
            + quote(key)
            + "/smiles"
        )
        return urlopen(url, timeout=10).read().decode("utf8")
    except Exception:
        return "Could not find SMILES"


def add_smiles_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add primary/secondary organic spacer SMILES from IUPAC columns."""
    out = dataset.copy()
    out["PRIMARY_ORGANIC_SPACER_SMILES"] = out["PRIMARY_ORGANIC_SPACER_IUPAC"].apply(
        cir_convert
    )
    out["SECONDARY_ORGANIC_SPACER_SMILES"] = out[
        "SECONDARY_ORGANIC_SPACER_IUPAC"
    ].apply(cir_convert)
    return out
