"""Reads the prebuilt app data (models/appdata.joblib); no raw-spreadsheet reading happens here.

The contents are fixed at build time, so they are built ahead and only read here.

        Usage:  from src import appdata;  appdata.base()
"""
from __future__ import annotations

import os

import joblib

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "models", "appdata.joblib")
_A = None

_MISSING = (
    "models/appdata.joblib is missing. The app data is prebuilt; copy the models folder\n"
    "from the release."
)


def load():
    """The full prebuilt data; read once and reused."""
    global _A
    if _A is None:
        if not os.path.exists(_PATH):
            raise FileNotFoundError(_MISSING)
        _A = joblib.load(_PATH)
    return _A


def built():
    """Build date."""
    return load().get("built", "?")


def base():
    """Basic drug info (441 drugs, not filtered for TMDD)."""
    return load()["base"]


def invitro_idx():
    """NUM -> one in-vitro data row."""
    return load()["invitro_idx"]


def model_idx():
    """NUM -> one PK parameter row (CL, V1, V2, Q1, ka)."""
    return load()["model_idx"]


def iiv_by_num():
    """NUM -> (CL log-SD, V log-SD). Used for the inter-individual variability band."""
    return load().get("iiv_by_num", {})


def regimen_nums():
    """Set of NUMs that have an approved regimen."""
    return load()["regimen_nums"]


def regimen_for(num):
    """Representative approved regimen for the drug, or None."""
    return load()["regimen"].get(str(num))


def human_exposure(num):
    """Observed Cmax and Ctrough [µg/mL]; empty dict if none."""
    return load()["exposure"].get(str(num), {})


def regimen_routes(num):
    """Per-route regimens {"IV": regimen, "SC": regimen} for drugs having both; empty dict if none.

    The regimen for the route in use is the same as regimen_for.
    """
    return load().get("regimen_by_route", {}).get(str(num), {})


def exposure_for(num, route):
    """Observed Cmax and Ctrough [µg/mL] for that route only; empty dict if none."""
    return load().get("exposure_by_route", {}).get(str(num), {}).get(route, {})


def similarity_index():
    """Reference set (features, labels) for OOD and similar-drug checks."""
    a = load()
    return a["sim_feats"], a["sim_labels"]


def exclusion(num=None):
    """Why a drug was left out of training, listed separately for CL and V.

    Reasons
        "nonlinear"   TMDD field is filled: nonlinear, excluded on purpose
        "nonlinear*"  excluded after checking the log CL vs dose slope directly
        "unjudged"    TMDD could not be assessed (only CL is trimmed)
        "no_human"    no human observation, so no target value to train on
        "data_rule"   has a human value but fails the analyte/route/info-field rules

    With num, returns that drug's dict; otherwise the full dict.
    Drugs used in training have no key at all.
    """
    e = load().get("excl", {})
    return e if num is None else e.get(str(num), {})
