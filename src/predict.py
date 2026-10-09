"""Prediction layer: loads a prebuilt model bundle; nothing is trained here.

Exposes the same names and return formats as predict_orig, so the UI code
works unchanged. The bundle holds six algorithms and their 5-fold
cross-validation results.

Allometry, traffic-light, similar-drug and OOD logic are reused from
predict_orig. Overridden here:
    build_features        builds the columns the bundle expects (incl. sequence descriptors)
    train                 reads the bundle and builds RES
    model_predictions     predicts with the bundle's models
    oof_model_predictions, apply_correction   same idea
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import config as C   # settings such as traffic-light cutoffs

# Model-independent parts are reused from predict_orig
from .predict_orig import (  # noqa: F401
    allometry_cl, allometry_v, build_ood_index,
    is_ood, build_sim_features, nearest_drugs, nearest_drugs_by_pk,
    _num, _str, _logsafe, _norm_route,
)

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "models" / "betabeta_v1.joblib"
LOOKUP = ROOT / "models" / "lookup.joblib"
NOSEQ = ROOT / "models" / "noseq_v1.joblib"     # sequence-free variant for the New Drug tab

KIND_OF = {"hybrid": "residual", "pure": "direct", "allometry": "baseline"}
NEEDED = ("sklearn", "catboost", "xgboost", "lightgbm")
CORE = tuple(getattr(C, "HYBRID_CORE", ("log_allo",)))

_B = None
_L = None


def _load():
    """Load the bundle and lookup table once."""
    global _B, _L
    if _B is None:
        if not BUNDLE.exists():
            raise FileNotFoundError(
                f"Model bundle not found: {BUNDLE}\n"
                "Copy the models folder from the release, or run `python build_models.py` "
                "in the build folder first.")
        miss = [m for m in NEEDED if importlib.util.find_spec(m) is None]
        if miss:
            import sys as _s
            raise ImportError(
                "Loading the model bundle needs these packages, missing in this Python: "
                + ", ".join(miss)
                + f"\n\nPython in use: {_s.executable}\n"
                f'  "{_s.executable}" -m pip install ' + " ".join(miss))
        _B = joblib.load(BUNDLE)
        _L = joblib.load(LOOKUP) if LOOKUP.exists() else {
            "ctd": {}, "ctd_cols": [], "monkey_rows": {}, "monkey_V_L": {}}
    return _B, _L


def jsname(key):
    """Bundle key to display name.  hybrid|CatBoost(...) -> CatBoost"""
    if key.startswith("allometry"):
        return "allometry_only"
    method, algo = key.split("|", 1)
    algo = algo.split("(")[0]
    return algo if method == "hybrid" else f"{algo}_pure"


def _key_of(row):
    return ("allometry|—" if row["방법"] == "allometry"
            else f"{row['방법']}|{row['알고리즘']}")


# ---------------------------------------------------------------- features
def build_features(base_row, invitro_row, log_allo, num=None):
    """Feature dict for one drug, using the bundle's column names.

    Sequence descriptors come from the lookup table. Drugs not in it (true new
    drugs) are left blank and filled by the pipeline's training-set median.
    """
    _, L = _load()
    iv = invitro_row if invitro_row is not None else {}
    num = str(num if num is not None else base_row.get("NUM", ""))

    mk_cl = _num(base_row.get("monkey_CL_Lday"))
    th = _num(base_row.get("monkey_t_half"))
    mv = L["monkey_V_L"].get(num)
    if mv is None:
        mv = _num(base_row.get("monkey_V_L"))
    nrow = L["monkey_rows"].get(num, 1)

    has_th = bool(np.isfinite(th) and th > 0)
    has_mv = bool(mv is not None and np.isfinite(mv) and mv > 0)
    f = {
        "MW": _num(base_row.get("MW")),
        "log_Kd": _logsafe(iv.get("Kd")),
        "log_allo": log_allo,
        "has_monkey": 1.0 if (np.isfinite(mk_cl) and mk_cl > 0) else 0.0,
        "log_mk_t_half": _logsafe(th),
        "t_half_imputed": 0.0 if has_th else 1.0,
        "mk_V": float(np.log(mv)) if has_mv else np.nan,
        "mk_n_rows": float(np.log(max(1, int(nrow)))),
    }
    cat = str(base_row.get("MOLECULAR CATEGORY", "")).strip().lower()
    for key, word in (("mc_mAb", "mab"), ("mc_conjugate", "conjugate"),
                      ("mc_bsAb", "bsab"), ("mc_fragment", "fragment"),
                      ("mc_fusion", "fusion protein")):
        f[key] = 1.0 if word in cat else 0.0
    f.update(L["ctd"].get(num, {}))
    return f


def _frame(feat, cols):
    """One-row frame in the bundle's column order; missing columns are NaN."""
    return pd.DataFrame([[feat.get(c, np.nan) for c in cols]], columns=cols)


def _complete(feat, cols):
    """True if all core columns are observed; decides whether to apply allometry."""
    for c in CORE:
        if c not in cols:
            continue
        v = pd.to_numeric(pd.Series([feat.get(c)]), errors="coerce").iloc[0]
        if not np.isfinite(v):
            return False
    return True


# ---------------------------------------------------------------- RES
def _variant_block(V, y, vt, pick):
    """Convert one bundle variant into the app's RES[tg] shape."""
    metrics, kinds, models, r2 = {}, {}, {}, {}
    for _, r in V["cv"].iterrows():
        key = _key_of(r)
        name = jsname(key)
        p = V["oof"].get(key)
        rmse = None
        if "rmse" in V:          # public build: precomputed on the full training set
            rmse = V["rmse"].get(key)
        elif p is not None:
            k = np.isfinite(p)
            if k.sum():
                rmse = float(np.sqrt(np.mean((p[k] - y[k]) ** 2)))
        metrics[name] = {"rmse": rmse, "gmfe": float(r["GMFE"]),
                         # lower score is better (the app sorts on it)
                         "score": -float(r["R2"]), "r2": float(r["R2"]),
                         "spearman": float(r["spearman"]),
                         "within2": float(r["within2"]),
                         "within3": float(r.get("within3", float("nan")))}
        kinds[name] = KIND_OF[r["방법"]]
        r2[name] = float(r["R2"])
        if key in V["models"]:
            models[name] = V["models"][key]

    # Final model: highest CV R2, unless the bundle names one
    best = jsname(pick) if pick and jsname(pick) in models else None
    if best is None:
        best = max(models, key=lambda n: r2.get(n, -9))
    return {
        "algo": best, "kind": kinds.get(best, "baseline"),
        "metrics": metrics, "models": models, "model_kind": kinds,
        "model": models.get(best),
        "cv_gmfe": metrics.get(best, {}).get("gmfe"),
        "cv_rmse": metrics.get(best, {}).get("rmse"),
        "cv_score": metrics.get(best, {}).get("score"),
        "baseline_rmse": metrics.get("allometry_only", {}).get("rmse"),
        "baseline_gmfe": metrics.get("allometry_only", {}).get("gmfe"),
        "all_results": {}, "columns": V["columns"], "variant": vt,
    }


def train(ds):
    """No training: read the bundle and return it in the shape the app expects.

    The name is kept because app.py calls it.
    """
    B, _ = _load()
    # public build keeps only the Top-5 rows; n_full is the size of the full training set
    n_of = {t: B[t].get("n_full", len(B[t]["nums"])) for t in ("CL", "V")}
    res = {"n_train": sum(n_of.values()),
           "n_train_by_target": n_of,
           "built": B["meta"]["built"], "meta": B["meta"]}

    for tg in ("CL", "V"):
        t = B[tg]
        vt = "esm" if "esm" in t["variants"] else "noesm"
        res[tg] = _variant_block(t["variants"][vt], t["y"], vt, t.get("main_key"))

    # Sequence-free variant, used only by the New Drug tab (model_predictions(noseq=True)).
    # Same drugs, labels and algorithms, trained without sequence columns.
    # If the file is missing, the original models are used.
    if NOSEQ.exists():
        NQ = joblib.load(NOSEQ)
        for tg in ("CL", "V"):
            if tg in NQ:
                res[tg]["noseq"] = _variant_block(NQ[tg], B[tg]["y"], "noseq",
                                                  NQ[tg].get("main_key"))

    # Out-of-fold predictions, shown for training drugs (not memorized values)
    oof = {}
    for tg in ("CL", "V"):
        t = B[tg]
        V = t["variants"][res[tg]["variant"]]
        for i, num in enumerate(t["nums"]):
            e = oof.setdefault(str(num), {})
            e[tg] = {jsname(k): float(np.exp(p[i]))
                     for k, p in V["oof"].items() if np.isfinite(p[i])}
    res["oof_table"] = oof
    return res, pd.DataFrame()


# ---------------------------------------------------------------- prediction
def _predict_one(res, target, name, feat, allo, blk=None):
    """Predict with one model; hybrid models add allometry as the base value.

    Allometry is applied only when all core columns are observed, as at build
    time; forcing it on drugs with missing data contaminates the base value.
    """
    blk = blk or res[target]
    ent = blk["models"].get(name)
    if ent is None:
        return None
    cols = blk["columns"]
    X = _frame(feat, cols)
    la = np.log(allo) if (allo and allo > 0 and np.isfinite(allo)) else np.nan
    if (blk["model_kind"].get(name) == "residual"
            and "m2" in ent and np.isfinite(la) and _complete(feat, cols)):
        return float(np.exp(float(ent["m2"].predict(X)[0]) + la))
    return float(np.exp(float(ent["m1"].predict(X)[0])))


def _entry(res, target, name, pred, allo, blk=None):
    blk = blk or res[target]
    m = blk["metrics"].get(name, {})
    adj = (float(np.log(pred / allo))
           if (pred and allo and allo > 0 and pred > 0) else None)
    return {"name": name, "kind": blk["model_kind"].get(name, "baseline"),
            "rmse": m.get("rmse"), "gmfe": m.get("gmfe"),
            "score": m.get("score"), "r2": m.get("r2"),
            "spearman": m.get("spearman"), "within2": m.get("within2"),
            "within3": m.get("within3"),
            "pred": pred, "adj": adj}


def model_predictions(res, target, feat, allo, noseq=False):
    """Per-model predictions plus CV metrics; allometry_only comes first.

    noseq=True (New Drug tab): predict with the sequence-free variant, since
    median-filling sequence columns in the original models biased CL high by
    1.32x on average. Falls back to the original models if there is no variant.
    """
    blk = noseq_block(res, target) if noseq else res[target]
    out = [_entry(res, target, "allometry_only", allo, allo, blk)]
    for name in blk["models"]:
        p = _predict_one(res, target, name, feat, allo, blk)
        if p is not None and np.isfinite(p):
            out.append(_entry(res, target, name, p, allo, blk))
    return out


def noseq_block(res, target):
    """Model block for the New Drug tab: sequence-free variant, else the original."""
    return res[target].get("noseq", res[target])


def oof_model_predictions(res, target, num, allo):
    """Out-of-fold predictions for a training drug (not self-memorized values)."""
    row = res.get("oof_table", {}).get(str(num), {}).get(target, {})
    return [_entry(res, target, name, pred, allo)
            for name, pred in row.items()]


def apply_correction(res, feat_cl, feat_v, allo_cl, allo_v):
    """Final CL/V prediction with the selected model."""
    out = {}
    for tg, feat, allo, pk, ak in (("CL", feat_cl, allo_cl,
                                    "CL_pred", "ml_adj_CL"),
                                   ("V", feat_v, allo_v,
                                    "V_pred", "ml_adj_V")):
        p = _predict_one(res, tg, res[tg]["algo"], feat, allo)
        if p is None or not np.isfinite(p):
            p = allo
        out[pk] = p
        out[ak] = (float(np.log(p / allo))
                   if (p and allo and allo > 0) else 0.0)
    return out

# ---------------------------------------------------------------- OOD reference set
def similarity_index(ds):
    """Reference drug list for OOD and similar-drug checks.

    Uses the drugs the models were actually trained on (union of the CL and V
    training sets), so drugs without monkey data are not all flagged as
    out-of-distribution.
    """
    B, _ = _load()
    train = []
    for tg in ("CL", "V"):
        train += [str(n) for n in B[tg]["nums"]]
    train = set(train)

    base = ds["base"]
    inv = ds["invitro"].set_index("NUM").to_dict("index")
    feats, labels = [], []
    for _, r in base.iterrows():
        num = str(r.get("NUM"))
        if num not in train:
            continue
        feats.append(build_sim_features(r, inv.get(r.get("NUM"))))
        labels.append({"NUM": r.get("NUM"), "INN": r.get("INN"),
                       "human_CL": r.get("human_CL_Lday"),
                       "human_V": r.get("human_V_L")})
    if not feats:      # no matches: fall back to the original method
        from .predict_orig import similarity_index as _orig
        return _orig(ds)
    return feats, labels


# ---------------------------------------------------------------- traffic light
def flag(pred_cl, obs_cl, pred_v, obs_v):
    """Traffic light, set separately for CL and V using fixed cutoffs (config.FLAG_CUT).

        CL   green within 2x, red beyond 3x
        V    green within 1.5x, red beyond 2x

    CL is looser because its observations span a far wider range than V; one
    shared cutoff would make the color mostly reflect CL vs V.
    """
    cut = getattr(C, "FLAG_CUT", {"CL": {"green": 2.0, "red": 3.0},
                                  "V": {"green": 1.5, "red": 2.0}})

    def fold(p, o):
        if p is None or o is None or not (np.isfinite(p) and np.isfinite(o))                 or p <= 0 or o <= 0:
            return None
        return max(p / o, o / p)

    out, worst = {}, -1
    for tg, p, o in (("CL", pred_cl, obs_cl), ("V", pred_v, obs_v)):
        f = fold(p, o)
        c = cut.get(tg, {"green": 1.5, "red": 3.0})
        if f is None:
            out[tg] = {"fold": None, "flag": "gray", "cut": c}
            continue
        r = 0 if f < c["green"] else (2 if f > c["red"] else 1)
        out[tg] = {"fold": f, "flag": ("green", "yellow", "red")[r], "cut": c}
        worst = max(worst, r)

    return {"CL": out["CL"], "V": out["V"],
            # legacy keys, still read by older callers
            "flag": ("green", "yellow", "red")[worst] if worst >= 0 else "gray",
            "fold_CL": out["CL"]["fold"], "fold_V": out["V"]["fold"],
            "cut": {tg: out[tg]["cut"] for tg in ("CL", "V")}}
