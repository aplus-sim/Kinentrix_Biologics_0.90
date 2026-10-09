"""Prediction pipeline: allometry -> tree-ML residual correction -> traffic-light flag.

ML target = log(observed Human / allometry prediction), with separate CL and V models.
Several tree models are compared by leave-one-out CV and the best is selected.
If ML cannot beat the allometry baseline (residual = 0) on a small sample,
it falls back to allometry-only.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import euclidean_distances

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# Feature definitions
#   log_mk_t_half : monkey half-life (day). Since t1/2 = 0.693*V/CL, together with
#                   log_allo (V-based) it adds monkey CL information.
#   SCIV          : approved route. Reported V for SC drugs is apparent (V/F), inflated
#                   by 1/F, so knowing the route lets the model correct that bias.
NUM_FEATS = ["MW", "DAR", "log_allo", "log_Kd", "log_IC50", "log_EC50", "log_mk_t_half"]
CAT_FEATS = ["Isotype", "MOLECULAR CATEGORY", "SCIV"]


# ---------------------------------------------------------------- allometry
def allometry_cl(monkey_cl_lday, monkey_bw):
    if monkey_cl_lday is None or not np.isfinite(monkey_cl_lday):
        return None
    bw = monkey_bw or C.DEFAULT_MONKEY_BW
    return monkey_cl_lday * (C.HUMAN_BW / bw) ** C.ALLO_EXP_CL


def allometry_v(monkey_v_l, monkey_bw):
    if monkey_v_l is None or not np.isfinite(monkey_v_l):
        return None
    bw = monkey_bw or C.DEFAULT_MONKEY_BW
    return monkey_v_l * (C.HUMAN_BW / bw) ** C.ALLO_EXP_V


# ---------------------------------------------------------------- feature build
def _logsafe(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.log(x) if (x is not None and np.isfinite(x) and x > 0) else np.nan


def build_features(base_row, invitro_row, log_allo):
    """Build the feature dict for one drug."""
    iv = invitro_row if invitro_row is not None else {}
    return {
        "MW": _num(base_row.get("MW")),
        "DAR": _num(base_row.get("DAR")),
        "log_allo": log_allo,
        "log_Kd": _logsafe(iv.get("Kd")),
        "log_IC50": _logsafe(iv.get("IC50")),
        "log_EC50": _logsafe(iv.get("EC50")),
        "log_mk_t_half": _logsafe(base_row.get("monkey_t_half")),
        "Isotype": _str(base_row.get("Isotype")),
        "MOLECULAR CATEGORY": _str(base_row.get("MOLECULAR CATEGORY")),
        "SCIV": _norm_route(base_row.get("SCIV")),
    }


def _norm_route(x):
    """Normalize route: IV / SC / IV_SC / missing."""
    s = _str(x).upper().replace(" ", "")
    if s in ("MISSING", ".", "NAN", ""):
        return "missing"
    if "IV" in s and "SC" in s:
        return "IV_SC"
    if "SC" in s:
        return "SC"
    if "IV" in s:
        return "IV"
    return "missing"


# ---------------------------------------------------------------- similar-drug lookup
# Similar drugs are found from intrinsic drug properties only (no allometry/monkey PK),
# since a new drug has no human PK yet and must be compared on molecular properties.
SIM_NUM_FEATS = ["MW", "DAR", "log_Kd", "log_IC50", "log_EC50"]
SIM_CAT_FEATS = ["Isotype", "MOLECULAR CATEGORY"]


def build_sim_features(base_row, invitro_row):
    """Feature dict for similarity search (build_features minus log_allo)."""
    iv = invitro_row if invitro_row is not None else {}
    return {
        "MW": _num(base_row.get("MW")),
        "DAR": _num(base_row.get("DAR")),
        "log_Kd": _logsafe(iv.get("Kd")),
        "log_IC50": _logsafe(iv.get("IC50")),
        "log_EC50": _logsafe(iv.get("EC50")),
        "Isotype": _str(base_row.get("Isotype")),
        "MOLECULAR CATEGORY": _str(base_row.get("MOLECULAR CATEGORY")),
    }


def similarity_index(ds):
    """Similarity features and labels for training drugs (observed Human CL/V); computed once and cached.

    labels[i]["NUM"] maps back to the training row, so a training drug can be
    excluded from its own OOD distance.
    """
    base = ds["base"]
    inv = ds["invitro"].set_index("NUM").to_dict("index")
    tr = base.dropna(subset=["monkey_CL_Lday", "monkey_V_L", "human_CL_Lday", "human_V_L"])
    feats, labels = [], []
    for _, r in tr.iterrows():
        feats.append(build_sim_features(r, inv.get(r["NUM"])))
        labels.append({"NUM": r.get("NUM"), "INN": r.get("INN"),
                       "human_CL": r.get("human_CL_Lday"), "human_V": r.get("human_V_L")})
    return feats, labels


def _sim_pipeline():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         SIM_NUM_FEATS),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         SIM_CAT_FEATS),
    ])


def nearest_drugs(train_feats, train_labels, query_feat, top_n=3):
    """Top_n training drugs closest to query_feat by standardized Euclidean distance.

    Distance uses MW/DAR/Kd/IC50/EC50 (standardized) plus Isotype/molecular category (one-hot).
    Returns [{"INN", "distance", "human_CL", "human_V"}], nearest first.
    """
    if not train_feats:
        return []
    Xtr = pd.DataFrame(train_feats)
    pipe = _sim_pipeline()
    Ztr = pipe.fit_transform(Xtr)
    Zq = pipe.transform(pd.DataFrame([query_feat]))
    d = euclidean_distances(Zq, Ztr)[0]
    order = np.argsort(d)[:top_n]
    return [{"INN": train_labels[i].get("INN"), "distance": float(d[i]),
             "human_CL": train_labels[i].get("human_CL"), "human_V": train_labels[i].get("human_V")}
            for i in order]


def _ood_pipeline():
    """Preprocessing for OOD distance; numeric features only if C.OOD_USE_CAT is False."""
    steps = [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                               ("sc", StandardScaler())]), SIM_NUM_FEATS)]
    if getattr(C, "OOD_USE_CAT", False):
        steps.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), SIM_CAT_FEATS))
    return ColumnTransformer(steps)


def build_ood_index(train_feats):
    """Density index over the training set's molecular-property space; the threshold is a quantile of its own distance distribution.

    Returns dict: pipeline (fitted), Z (training embedding), threshold, train_knn (mean 3-NN distance per drug).
    Self-distances are excluded.
    """
    if not train_feats:
        return None
    k = getattr(C, "OOD_K", 3)
    pipe = _ood_pipeline()
    Z = pipe.fit_transform(pd.DataFrame(train_feats))
    D = euclidean_distances(Z, Z)
    np.fill_diagonal(D, np.inf)                    # exclude self
    knn = np.sort(D, axis=1)[:, :k].mean(axis=1)
    thr = float(np.percentile(knn, getattr(C, "OOD_PCT", 65)))
    return {"pipe": pipe, "Z": Z, "threshold": thr, "train_knn": knn}


def ood_distance(ood, query_feat, exclude_idx=None):
    """Mean k-NN distance from query to the training set; exclude_idx drops that row
    (a training drug is excluded from itself, keeping an out-of-fold view)."""
    if not ood:
        return None
    k = getattr(C, "OOD_K", 3)
    Zq = ood["pipe"].transform(pd.DataFrame([query_feat]))
    d = euclidean_distances(Zq, ood["Z"])[0]
    if exclude_idx is not None and 0 <= exclude_idx < len(d):
        d = np.delete(d, exclude_idx)
    if len(d) < k:
        return None
    return float(np.sort(d)[:k].mean())


def is_ood(ood, query_feat, exclude_idx=None):
    """Whether the drug is outside the training distribution (ML unreliable). Returns (bool, distance, threshold)."""
    if not ood or not getattr(C, "OOD_FALLBACK", False):
        return False, None, None
    dist = ood_distance(ood, query_feat, exclude_idx)
    if dist is None:
        return False, None, ood["threshold"]
    return bool(dist > ood["threshold"]), dist, ood["threshold"]


def nearest_drugs_by_pk(train_labels, query_cl, query_v, top_n=3):
    """Top_n training drugs whose observed CL and V are closest to the predicted ones (PK-based similarity).

    Compares pharmacokinetics itself, not molecular properties: distance between the
    query's predicted CL/V and each drug's observed Human CL/V in log space, divided by
    the training std of log-CL and log-V so both contribute equally.
    Since distance is in std units, fold differences (max(p/o, o/p); 1.00x = exact match)
    are also returned for CL and V.
    Returns [{"INN", "distance", "fold_CL", "fold_V", "human_CL", "human_V"}], nearest first.
    """
    if not train_labels or query_cl is None or query_v is None \
            or not (query_cl > 0 and query_v > 0):
        return []
    rows = [(l, l.get("human_CL"), l.get("human_V")) for l in train_labels]
    rows = [(l, cl, v) for l, cl, v in rows
            if cl is not None and v is not None and cl > 0 and v > 0]
    if not rows:
        return []
    log_cl = np.array([np.log(cl) for _, cl, _ in rows])
    log_v = np.array([np.log(v) for _, _, v in rows])
    s_cl = float(np.std(log_cl)) or 1.0
    s_v = float(np.std(log_v)) or 1.0
    qcl, qv = np.log(query_cl), np.log(query_v)
    d = np.sqrt(((log_cl - qcl) / s_cl) ** 2 + ((log_v - qv) / s_v) ** 2)
    order = np.argsort(d)[:top_n]
    return [{"INN": rows[i][0].get("INN"), "distance": float(d[i]),
             "fold_CL": float(max(query_cl / rows[i][1], rows[i][1] / query_cl)),
             "fold_V": float(max(query_v / rows[i][2], rows[i][2] / query_v)),
             "human_CL": rows[i][1], "human_V": rows[i][2]} for i in order]


def _num(x):
    v = pd.to_numeric(x, errors="coerce")
    return float(v) if pd.notna(v) else np.nan


def _str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "missing"
    s = str(x).strip()
    return s if s else "missing"


def _make_pipeline(regressor):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM_FEATS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT_FEATS),
    ])
    return Pipeline([("pre", pre), ("reg", regressor)])


def _candidates():
    """Residual-correction candidates: learn y = log(Human/Allometry)."""
    cands = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=0),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=0),
        "GradientBoosting": GradientBoostingRegressor(random_state=0),
    }
    if _HAS_XGB:
        cands["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=0, verbosity=0)
    return cands


def _direct_candidates():
    """Pure (direct) candidates: learn y = log(Human) directly, without allometry.

    Same features (including log_allo), but the model predicts the absolute human
    CL/V instead of a multiplier on the allometry value.
    """
    cands = {
        "RandomForest_pure": RandomForestRegressor(n_estimators=300, random_state=0),
        "ExtraTrees_pure": ExtraTreesRegressor(n_estimators=300, random_state=0),
        "GradientBoosting_pure": GradientBoostingRegressor(random_state=0),
    }
    if _HAS_XGB:
        cands["XGBoost_pure"] = XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=0, verbosity=0)
    return cands


def _errmetrics(e):
    """Metrics from out-of-fold error e = predicted residual - true residual (= log(Pred/Obs)).

    rmse  : sqrt(mean(e^2))      - penalizes large errors (L2)
    mae   : mean(|e|)            - mean |log fold-error| (L1)
    gmfe  : exp(mae)             - geometric mean fold-error vs observed
    score : W_CLOSENESS*gmfe + W_RMSE*rmse - selection criterion (lower is better)
    """
    e = np.asarray(e, dtype=float)
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    gmfe = float(np.exp(mae))
    score = float(C.SEL_W_CLOSENESS * gmfe + C.SEL_W_RMSE * rmse)
    return {"rmse": rmse, "mae": mae, "gmfe": gmfe, "score": score}


def _loo_metrics(pipe, X, y):
    """LOO-CV error metrics plus out-of-fold predictions (each row predicted without being trained on).

    Returns (metrics_dict, oof_preds). oof_preds[i] is the prediction for X.iloc[i] from a
    model trained on all other rows (residual or direct target, depending on the family).
    Showing these gives training drugs unbiased predictions (avoids memorization in the
    Existing Drugs tab).
    """
    loo = LeaveOneOut()
    preds = np.zeros(len(y))
    for tr, te in loo.split(X):
        pipe.fit(X.iloc[tr], y[tr])
        preds[te] = pipe.predict(X.iloc[te])
    return _errmetrics(preds - y), preds


def _fit_target(X, y_resid, y_level):
    """Compare three approaches for one target (CL or V) and select the best.

    - allometry_only : baseline, no correction (e = 0 - y_resid)
    - residual ML    : learn y_resid = log(Human/Allometry); final = Allometry * exp(correction)
    - direct ML      : learn y_level = log(Human) without allometry; final = exp(prediction)

    In all three, the out-of-fold error e is on the log(Pred/Obs) scale, so GMFE/RMSE/the
    composite score compare them fairly. Every candidate is refit on all data and kept in
    `models`, so each method's CL/V prediction can be shown side by side per drug.
    """
    base_m = _errmetrics(-y_resid)  # residual = 0 (allometry-only): e = 0 - y_resid
    metrics = {"allometry_only": base_m}
    results = {"allometry_only": base_m["rmse"]}
    models = {}  # name -> pipeline fit on all data (for new-drug prediction)
    oof = {"allometry_only": np.zeros(len(y_resid))}  # name -> out-of-fold predictions (residual/direct, scale per family)
    kind = {"allometry_only": "baseline"}
    best_name, best_score = "allometry_only", base_m["score"]

    def _fit_family(candidates_fn, y, kind_label):
        nonlocal best_name, best_score
        for name, reg in candidates_fn().items():
            try:
                pipe = _make_pipeline(reg)
                mt, oof_raw = _loo_metrics(pipe, X, y)
                metrics[name] = mt
                results[name] = mt["rmse"]
                oof[name] = oof_raw
                fitted = _make_pipeline(candidates_fn()[name])  # refit on all data (fresh unfitted instance)
                fitted.fit(X, y)
                models[name] = fitted
                kind[name] = kind_label
                if mt["score"] < best_score:  # select by composite score
                    best_name, best_score = name, mt["score"]
            except Exception as e:
                results[name] = f"err:{e}"
                metrics[name] = None

    if len(y_resid) >= 4:  # minimum sample for CV
        _fit_family(_candidates, y_resid, "residual")
        _fit_family(_direct_candidates, y_level, "direct")

    final = models.get(best_name)  # None if allometry_only is selected
    bm = metrics[best_name]
    return {"algo": best_name, "kind": kind.get(best_name, "baseline"),
            "cv_rmse": bm["rmse"], "cv_gmfe": bm["gmfe"], "cv_score": bm["score"],
            "baseline_rmse": base_m["rmse"], "baseline_gmfe": base_m["gmfe"],
            "all_results": results, "metrics": metrics,
            "models": models, "model": final, "model_kind": kind, "oof": oof}


def model_predictions(res, target, feat, allo):
    """Corrected CL/V prediction and LOO-CV metrics per candidate model (plus allometry_only).

    Returns [{"name", "kind", "rmse", "gmfe", "score", "pred", "adj"}], with allometry_only
    always first. selected = res[target]["algo"].
    kind: "baseline" | "residual" (allometry + ML correction) | "direct" (pure ML, no allometry).
    adj is always log(pred/allo), so "% vs allometry" is meaningful for every method.
    """
    info = res.get(target, {})
    metrics = info.get("metrics", {})
    models = info.get("models", {})
    kind = info.get("model_kind", {})

    def _entry(name, pred, adj):
        m = metrics.get(name) or {}
        return {"name": name, "kind": kind.get(name, "baseline"),
                "rmse": m.get("rmse"), "gmfe": m.get("gmfe"),
                "score": m.get("score"), "pred": pred, "adj": adj}

    out = [_entry("allometry_only", allo, 0.0)]
    for name, pipe in models.items():
        try:
            raw = float(pipe.predict(pd.DataFrame([feat]))[0])
            if kind.get(name) == "direct":
                pred = float(np.exp(raw))
            else:
                pred = float(allo * np.exp(raw)) if allo is not None else None
            adj = float(np.log(pred / allo)) if (pred is not None and allo) else None
        except Exception:
            adj, pred = None, None
        out.append(_entry(name, pred, adj))
    return out


# ---------------------------------------------------------------- main trainer
def train(ds):
    """Train CL and V residual models on the training set. Returns the result dict and a per-drug prediction table."""
    base = ds["base"]
    inv = ds["invitro"].set_index("NUM").to_dict("index")
    rows = []
    for _, r in base.iterrows():
        num = r["NUM"]
        a_cl = allometry_cl(r.get("monkey_CL_Lday"), r.get("monkey_bw"))
        a_v = allometry_v(r.get("monkey_V_L"), r.get("monkey_bw"))
        if a_cl is None or a_v is None:
            continue
        feat_cl = build_features(r, inv.get(num), np.log(a_cl))
        feat_v = build_features(r, inv.get(num), np.log(a_v))
        rows.append({
            "NUM": num, "INN": r.get("INN"),
            "allo_CL": a_cl, "allo_V": a_v,
            "human_CL": r.get("human_CL_Lday"), "human_V": r.get("human_V_L"),
            "feat_CL": feat_cl, "feat_V": feat_v,
        })
    pred_df = pd.DataFrame(rows)

    # training set = drugs with observed human values
    tr = pred_df.dropna(subset=["human_CL", "human_V"])
    res = {"n_train": len(tr)}
    if len(tr) >= 4:
        Xcl = pd.DataFrame(list(tr["feat_CL"]))
        ycl_resid = np.log(tr["human_CL"].values / tr["allo_CL"].values)
        ycl_level = np.log(tr["human_CL"].values)
        Xv = pd.DataFrame(list(tr["feat_V"]))
        yv_resid = np.log(tr["human_V"].values / tr["allo_V"].values)
        yv_level = np.log(tr["human_V"].values)
        res["CL"] = _fit_target(Xcl, ycl_resid, ycl_level)
        res["V"] = _fit_target(Xv, yv_resid, yv_level)
        res["oof_table"] = _build_oof_table(tr, res)
    else:
        res["CL"] = {"algo": "allometry_only", "kind": "baseline",
                     "cv_rmse": None, "cv_gmfe": None,
                     "cv_score": None, "baseline_rmse": None, "baseline_gmfe": None,
                     "all_results": {}, "metrics": {}, "models": {}, "model": None,
                     "model_kind": {}, "oof": {}}
        res["V"] = dict(res["CL"])
        res["oof_table"] = {}
    return res, pred_df


def _build_oof_table(tr, res):
    """Out-of-fold (non-memorized) CL/V predictions per training drug. NUM -> {"CL": {name: pred}, "V": {...}}.

    The Existing Drugs tab shows these (model trained without that drug) instead of
    predictions from a model fit on all data, which would be memorized.
    """
    table = {}
    nums = tr["NUM"].values
    allo = {"CL": tr["allo_CL"].values, "V": tr["allo_V"].values}
    for i, num in enumerate(nums):
        entry = {}
        for target in ("CL", "V"):
            kind = res[target].get("model_kind", {})
            entry[target] = {}
            for name, raw_arr in res[target].get("oof", {}).items():
                raw = raw_arr[i]
                if kind.get(name, "baseline") == "direct":
                    pred = float(np.exp(raw))
                else:
                    pred = float(allo[target][i] * np.exp(raw))
                entry[target][name] = pred
        table[num] = entry
    return table


def oof_model_predictions(res, target, num, allo):
    """Out-of-fold CL/V prediction per candidate, computed with this drug left out of training.

    Used by the Existing Drugs tab, only for drugs in the training set (present in oof_table).
    Same format as model_predictions(): [{"name","kind","rmse","gmfe","score","pred","adj"}].
    """
    info = res.get(target, {})
    metrics = info.get("metrics", {})
    kind = info.get("model_kind", {})
    oof_row = res.get("oof_table", {}).get(num, {}).get(target, {})
    out = []
    for name, pred in oof_row.items():
        m = metrics.get(name) or {}
        adj = float(np.log(pred / allo)) if (pred is not None and allo) else None
        out.append({"name": name, "kind": kind.get(name, "baseline"),
                    "rmse": m.get("rmse"), "gmfe": m.get("gmfe"), "score": m.get("score"),
                    "pred": pred, "adj": adj})
    return out


def apply_correction(res, feat_cl, feat_v, allo_cl, allo_v):
    """Final CL/V prediction with the selected model. A direct (pure ML) model's absolute
    prediction is used as is, while adj stays log(pred/allo) so "% vs allometry" holds
    for every method.
    """
    def corr(target, feat, allo):
        m = res[target]["model"]
        if m is None:
            return allo, 0.0
        raw = float(m.predict(pd.DataFrame([feat]))[0])
        if res[target].get("kind") == "direct":
            pred = float(np.exp(raw))
        else:
            pred = float(allo * np.exp(raw))
        adj = float(np.log(pred / allo)) if allo else 0.0
        return pred, adj
    cl_pred, adj_cl = corr("CL", feat_cl, allo_cl)
    v_pred, adj_v = corr("V", feat_v, allo_v)
    return {"CL_pred": cl_pred, "V_pred": v_pred,
            "ml_adj_CL": adj_cl, "ml_adj_V": adj_v}


# ---------------------------------------------------------------- traffic light
def flag(pred_cl, obs_cl, pred_v, obs_v):
    """Traffic light based on CL and V fold-error."""
    def fold(p, o):
        if p is None or o is None or not (np.isfinite(p) and np.isfinite(o)) \
                or p <= 0 or o <= 0:
            return None
        return max(p / o, o / p)
    fcl, fv = fold(pred_cl, obs_cl), fold(pred_v, obs_v)
    if fcl is None or fv is None:
        return {"flag": "gray", "fold_CL": fcl, "fold_V": fv}
    if fcl < C.FLAG_GREEN and fv < C.FLAG_GREEN:
        f = "green"
    elif fcl > C.FLAG_RED and fv > C.FLAG_RED:
        f = "red"
    else:
        f = "yellow"
    return {"flag": f, "fold_CL": fcl, "fold_V": fv}


if __name__ == "__main__":
    import src.dataio as d
    ds = d.build_dataset()
    res, pred_df = train(ds)
    print("n_train:", res["n_train"])
    print("XGBoost available:", _HAS_XGB)
    for t in ("CL", "V"):
        print(f"\n[{t}] selected: {res[t]['algo']} ({res[t].get('kind')})  "
              f"cv_gmfe={res[t]['cv_gmfe']:.3f}  cv_rmse={res[t]['cv_rmse']:.3f}  "
              f"cv_score={res[t]['cv_score']:.3f}")
        kinds = res[t].get("model_kind", {})
        for name, m in res[t]["metrics"].items():
            if m:
                print(f"    {name:20s} [{kinds.get(name, 'baseline'):8s}] "
                      f"gmfe={m['gmfe']:.3f} rmse={m['rmse']:.3f} score={m['score']:.3f}")
