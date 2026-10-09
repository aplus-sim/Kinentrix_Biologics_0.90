"""Linear 1/2/3-compartment PK simulation (IV infusion / SC absorption, multiple dosing).

Units: CL,Q [L/day], V [L], dose [mg], time [day]
=> concentration C = A/V [mg/L = µg/mL].
ka is converted to 1/day on input.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _rates(p):
    """Parameter dict -> micro rate constants. p: CL,V1[,V2,Q1[,V3,Q2]], ka (optional)."""
    CL, V1 = p["CL"], p["V1"]
    k10 = CL / V1
    k12 = k21 = k13 = k31 = 0.0
    ncmt = 1
    if p.get("V2") and p.get("Q1"):
        k12 = p["Q1"] / V1
        k21 = p["Q1"] / p["V2"]
        ncmt = 2
    if p.get("V3") and p.get("Q2"):
        k13 = p["Q2"] / V1
        k31 = p["Q2"] / p["V3"]
        ncmt = 3
    return dict(k10=k10, k12=k12, k21=k21, k13=k13, k31=k31,
                ka=p.get("ka", 0.0) or 0.0, V1=V1, ncmt=ncmt)


def simulate(p, dose_mg, interval_days, n_doses, infusion_days=0.0,
             route="IV", horizon_days=None, npts=600):
    """Concentration-time profile for multiple doses. Returns (t[day], C[µg/mL])."""
    r = _rates(p)
    if horizon_days is None:
        horizon_days = interval_days * n_doses + max(interval_days, 14)
    # Interval <= 0 means a single dose; otherwise all n_doses would stack at t=0.
    dose_times = ([0.0] if interval_days <= 0
                  else [i * interval_days for i in range(n_doses)])
    is_sc = str(route).upper().startswith("S") or route in ("2", 2)

    # State: [Adepot, A1, A2, A3]
    def odes(t, y):
        Ad, A1, A2, A3 = y
        # Active IV infusion rate
        rate = 0.0
        if not is_sc and infusion_days > 0:
            for dt in dose_times:
                if dt <= t < dt + infusion_days:
                    rate += dose_mg / infusion_days
        ka = r["ka"]
        dAd = -ka * Ad
        dA1 = (rate + ka * Ad
               - (r["k10"] + r["k12"] + r["k13"]) * A1
               + r["k21"] * A2 + r["k31"] * A3)
        dA2 = r["k12"] * A1 - r["k21"] * A2
        dA3 = r["k13"] * A1 - r["k31"] * A3
        return [dAd, dA1, dA2, dA3]

    # Integrate piecewise between dosing events (handles bolus/depot input)
    breakpoints = sorted(set([0.0, horizon_days]
                             + dose_times
                             + [d + infusion_days for d in dose_times
                                if infusion_days > 0]))
    breakpoints = [b for b in breakpoints if b <= horizon_days]
    y = [0.0, 0.0, 0.0, 0.0]
    ts, cs = [], []
    for i in range(len(breakpoints) - 1):
        t0, t1 = breakpoints[i], breakpoints[i + 1]
        # Apply the dose at t0
        if t0 in dose_times:
            if is_sc:
                y[0] += dose_mg            # SC: bolus into depot
            elif infusion_days == 0:
                y[1] += dose_mg            # IV bolus: into central
            # IV infusion is handled by the rate term in odes
        seg = np.linspace(t0, t1, max(8, int(npts * (t1 - t0) / horizon_days) + 2))
        sol = solve_ivp(odes, (t0, t1), y, t_eval=seg, method="LSODA",
                        rtol=1e-6, atol=1e-9, max_step=max(interval_days/20, 0.05))
        y = list(sol.y[:, -1])
        ts.extend(sol.t.tolist())
        cs.extend((sol.y[1] / r["V1"]).tolist())
    t = np.array(ts)
    c = np.array(cs)
    # Remove duplicate time points
    order = np.argsort(t)
    return t[order], c[order]


def _terminal_rate(r):
    """Slowest elimination rate constant [1/day]: smallest eigenvalue of the compartment matrix (SC includes ka)."""
    k = np.array([[-(r["k10"] + r["k12"] + r["k13"]), r["k21"], r["k31"]],
                  [r["k12"], -r["k21"], 0.0],
                  [r["k13"], 0.0, -r["k31"]]])[:r["ncmt"], :r["ncmt"]]
    lam = float(np.min(np.abs(np.linalg.eigvals(k).real)))
    if r["ka"] > 0:
        lam = min(lam, r["ka"])
    return lam


def exposure_metrics(p, dose_mg, interval_days, infusion_days=0.0, route="IV",
                     single_dose=False, horizon_days=84.0, max_doses=150):
    """Exposure metrics (AUC [µg·day/mL], Cmax and Ctrough [µg/mL]).

    For repeated dosing, steady state: dosing continues past 7 terminal half-lives and AUCτ, Cmax and Ctrough are
    measured over the last dosing interval. For linear PK, AUCτ = Dose/CL must hold (check).
    With no dosing interval: single-dose AUC0–last and Cmax.
    """
    if single_dose or not interval_days or interval_days <= 0:
        t, c = simulate(p, dose_mg, horizon_days, 1, infusion_days, route,
                        horizon_days, npts=1200)
        return {"kind": "single", "AUC": float(_trapz(c, t)),
                "Cmax": float(c.max()), "Ctrough": None}
    lam = _terminal_rate(_rates(p))
    n = int(np.ceil(7 * np.log(2) / max(lam, 1e-9) / interval_days)) + 1
    n = min(max(n, 2), max_doses)
    t, c = simulate(p, dose_mg, interval_days, n, infusion_days, route,
                    n * interval_days, npts=300 * n)
    t0 = (n - 1) * interval_days
    k = t >= t0 - 1e-9
    tl, cl = t[k], c[k]
    # If the dose-count cap is hit, steady state may not be reached; the caller marks this with "≈".
    return {"kind": "ss", "AUC": float(_trapz(cl, tl)),
            "Cmax": float(cl.max()), "Ctrough": float(cl[-1]), "n_doses": n,
            "capped": n >= max_doses}


def params_from_prediction(cl_pred, v_pred, model_row=None):
    """Build simulation parameters from predicted CL/V: **1-compartment**.

    v_pred is the steady-state volume (Vss, mean of the reported V forms), so it is used directly
    as the 1-compartment V. Terminal half-life = ln2*Vss/CL follows directly from the two predictions.

    The published model's V2 and Q1 are not borrowed for a 2-compartment fit because:
      - its parameter units are mixed (L/h, mL), causing 24x or 1000x mismatches;
      - if its CL differs a lot from the predicted CL, the terminal phase disagrees with it;
      - it can contain typos, giving terminal half-lives of hundreds of days.
    Distribution kinetics of a new drug are unknown, so 1-compartment is honest and stable.
    The published 2-compartment profile is shown separately as the 'Model parameters' overlay (orange dashed line).

    The SC absorption rate ka is borrowed from the published value if present (units converted via _conv_ka);
    otherwise the caller fills in the default (config.DEFAULT_KA_SC).
    """
    p = {"CL": cl_pred, "V1": v_pred}
    if model_row is not None:
        ka = _conv_ka(model_row.get("ka"), model_row.get("ka_UNIT"))
        if ka:
            p["ka"] = ka
    return p


def params_from_model(model_row):
    """Extract simulation parameters directly from a published-model row (with unit conversion)."""
    p = {"CL": _conv_cl(model_row.get("CL"), model_row.get("CL_UNIT")),
         "V1": _conv_v(model_row.get("V1"), model_row.get("V1_UNIT"))}
    v2 = _conv_v(model_row.get("V2"), model_row.get("V2_UNIT"))
    q1 = _conv_cl(model_row.get("Q1"), model_row.get("Q1_UNIT"))
    v3 = _conv_v(model_row.get("V3"), model_row.get("V3_UNIT"))
    q2 = _conv_cl(model_row.get("Q2"), model_row.get("Q2_UNIT"))
    ka = _conv_ka(model_row.get("ka"), model_row.get("ka_UNIT"))
    if v2 and q1:
        p["V2"], p["Q1"] = v2, q1
    if v3 and q2:
        p["V3"], p["Q2"] = v3, q2
    if ka:
        p["ka"] = ka
    if not p["CL"] or not p["V1"]:
        return None
    return p


# ---- Unit conversion (for published-model parameters) ----
def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _norm_unit(unit):
    """Normalize a unit string: lowercase, strip spaces and asterisks, unify common notations.

        'day^-1' 'day-1' '/day' '1/day' 'd^-1'  -> '1/day'
        'h^-1'   'hr^-1' '/h'   '1/h'   'hour-1' -> '1/h'
        'L/h'    'l /hr'                          -> 'l/h'
    """
    u = str(unit).strip().lower().replace(" ", "").replace("*", "")
    u = u.replace("^-1", "-1").replace("hr", "h").replace("hour", "h") \
         .replace("minute", "min")
    if u in ("day-1", "/day", "1/day", "d-1", "/d", "1/d", "day", "d"):
        return "1/day"
    if u in ("h-1", "/h", "1/h", "h"):
        return "1/h"
    if u in ("min-1", "/min", "1/min", "min"):
        return "1/min"
    return u


# Time unit -> multiplier to /day
_PER_DAY = {"day": 1.0, "d": 1.0, "h": 24.0, "hr": 24.0, "min": 1440.0}


def _time_factor(u):
    """Time part (denominator) of a normalized unit as a multiplier to /day; 1.0 if unrecognized."""
    if u.endswith(("/day", "1/day")):
        return 1.0
    if u.endswith(("/h", "1/h")):
        return 24.0
    if u.endswith(("/min", "1/min")):
        return 1440.0
    return _PER_DAY.get(u.split("/")[-1], 1.0)


def _conv_cl(val, unit):
    """CL and Q to L/day; handles the volume part (mL/L) and the time part (h/day) separately."""
    v = _f(val)
    if v is None:
        return None
    u = _norm_unit(unit)
    if u.startswith("ml"):
        v = v / 1000.0
    return v * _time_factor(u)


def _conv_v(val, unit):
    v = _f(val)
    if v is None:
        return None
    return v / 1000.0 if _norm_unit(unit).startswith("ml") else v


def _conv_ka(val, unit):
    """Absorption rate to 1/day; handles variants such as 'h^-1', 'day^-1', '/h'."""
    v = _f(val)
    if v is None:
        return None
    return v * _time_factor(_norm_unit(unit))


# ---- Body-weight covariates (covariate structure of published popPK models) ----
_WT_KEYS = (("CL", "CL_WT"), ("V1", "V1_WT"), ("V2", "V2_WT"), ("Q1", "Q1_WT"))


def has_weight_covariate(model_row):
    """Whether the drug's published model has any body-weight covariate exponent (e.g. CL_WT)."""
    return any(_f(model_row.get(wt)) is not None for _, wt in _WT_KEYS)


def covariate_scaled_params(model_row, wt_kg, ref_wt=70.0):
    """Scale model_row (published popPK parameters) to an individual of weight wt_kg.

        param_i = param_pop · (wt_kg / ref_wt) ^ exponent

    Parameters without an exponent (e.g. missing Q1_WT) are left as is: this scaling reflects weight only,
    so missing information does not mean that parameter has zero inter-individual variability.
    Returns None if model_row cannot yield CL and V1 (params_from_model returns None).
    """
    base = params_from_model(model_row)
    if base is None:
        return None
    ratio = wt_kg / ref_wt
    out = dict(base)
    for key, wt_col in _WT_KEYS:
        if key not in out:
            continue
        exp = _f(model_row.get(wt_col))
        if exp is not None:
            out[key] = out[key] * (ratio ** exp)
    return out


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # sanity: 1-comp IV, CL=0.26 L/day, V=4.3 L, 1200mg q3w
    t, c = simulate({"CL": 0.26, "V1": 4.3}, dose_mg=1200,
                    interval_days=21, n_doses=4, infusion_days=1/24, route="IV")
    print("Cmax ug/mL:", round(float(c.max()), 1),
          "Ctrough ug/mL:", round(float(c[c > 0][-1]), 1))

    # unit-parsing checks
    assert _conv_ka(1.0, "h^-1") == 24.0 and _conv_ka(1.0, "1/day") == 1.0
    assert _conv_cl(1.0, "L/h") == 24.0 and _conv_v(1000.0, "mL") == 1.0
    print("unit parsing ok")
