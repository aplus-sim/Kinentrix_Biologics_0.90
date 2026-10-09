"""New Drug tab — simulation settings.

Only the New Drug tab uses this module; the Existing Drugs tab is unchanged.

    SC bioavailability   F = 0.71   (median of 23 SC antibodies in published popPK models, IQR
                         0.64–0.76). The form has no F field, so SC doses are scaled by it.
    1-compartment        V = predicted V, CL = predicted CL.

1-compartment because the model predicts a single V.
"""
from __future__ import annotations

import config as C
from . import pksim

F_SC = getattr(C, "NEWDRUG_F_SC", 0.71)


# New Drug form category labels → reference-DB categories. Used only for the
# closest-drug / OOD search, which one-hot encodes the category: the form's labels
# ("mAb [human]", "ADC", ...) never matched the DB's ("mAb", "Conjugate", ...), so the
# search ignored the category and often returned ADCs for plain antibodies.
_CATEGORY = {"adc": "Conjugate", "bispecific": "bsAb", "missing": "missing"}


def db_category(form_label):
    s = str(form_label or "").strip()
    if s.lower().startswith("mab"):
        return "mAb"
    return _CATEGORY.get(s.lower(), s)


def similarity_row(base_row):
    """The New Drug input row as the closest-drug / OOD search should see it.

    - category label mapped to the DB's (db_category)
    - DAR 0 means "not an ADC": the DB leaves DAR blank for antibodies, and the search
      fills blanks with the DAR median of the ADCs (~4), so a typed 0 sat closer to
      low-DAR ADCs than to antibodies. 0 is passed as blank instead.
    """
    import numpy as np
    r = base_row.copy()
    r["MOLECULAR CATEGORY"] = db_category(r.get("MOLECULAR CATEGORY"))
    try:
        if float(r.get("DAR")) <= 0:
            r["DAR"] = np.nan
    except (TypeError, ValueError):
        r["DAR"] = np.nan
    return r


def is_sc(route):
    return str(route).upper().startswith("S")


def params(cl, vss, route):
    """Simulation parameters for a predicted CL [L/day] and V [L] — 1 compartment."""
    p = {"CL": cl, "V1": vss}
    if is_sc(route):
        p["ka"] = getattr(C, "DEFAULT_KA_SC", 0.25)
    return p


def terminal_thalf(cl, vss, route):
    """Terminal half-life [day] of the curve (disposition only, not ka) — ln2·V/CL."""
    import numpy as np
    r = pksim._rates({k: v for k, v in params(cl, vss, route).items() if k != "ka"})
    return float(np.log(2) / pksim._terminal_rate(r))


def dose(dose_mg, route):
    """Absorbed dose — SC doses scaled by the typical bioavailability."""
    return dose_mg * (F_SC if is_sc(route) else 1.0)


def simulate(cl, vss, regimen, n_doses, horizon):
    """Profile for the New Drug tab regimen dict (dose_mg, interval_days, infusion_days, route)."""
    return pksim.simulate(params(cl, vss, regimen["route"]),
                          dose(regimen["dose_mg"], regimen["route"]),
                          regimen["interval_days"], n_doses,
                          regimen["infusion_days"], regimen["route"], horizon)


def exposure(cl, vss, regimen):
    """Steady-state AUCτ / Cmax / Ctrough with the same settings as the curve."""
    return pksim.exposure_metrics(params(cl, vss, regimen["route"]),
                                  dose(regimen["dose_mg"], regimen["route"]),
                                  regimen["interval_days"], regimen.get("infusion_days", 0.0),
                                  regimen["route"],
                                  single_dose=bool(regimen.get("single_dose")))
