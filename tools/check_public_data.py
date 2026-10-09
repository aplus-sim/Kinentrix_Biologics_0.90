"""Check that the public data files carry per-drug values only for the kept drugs.

    python tools/check_public_data.py

Kept: the Top-5 (8 Pembrolizumab, 28 Canakinumab, 102 Dupilumab, 7 Nivolumab, 13 Avelumab)
and the two SC products their route toggle links to (449, 1). Every other drug may appear
only as a drop-down name in appdata["public"]["catalog"]. Exit code 1 on any violation.
"""
import os
import pickle
import re
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(ROOT, "models")
KEEP = {"8", "28", "102", "7", "13", "449", "1"}
bad = []


def fail(msg):
    bad.append(msg)
    print("FAIL", msg)


def is_num_key(k):
    return isinstance(k, str) and re.fullmatch(r"\d+", k) is not None


def walk(o, path):
    """Generic scan: NUM-keyed dicts, NUM columns and NUM fields must stay inside KEEP."""
    if hasattr(o, "predict") or hasattr(o, "transform"):      # fitted model objects
        return
    if isinstance(o, dict):
        if o and all(is_num_key(k) for k in o):
            extra = set(o) - KEEP
            if extra:
                fail(f"{path}: {len(extra)} NUM keys outside the kept set")
        for k, v in o.items():
            if k == "catalog" and path.endswith("public"):
                continue
            walk(v, f"{path}.{k}")
    elif isinstance(o, (list, tuple, set, frozenset)):
        items = list(o)
        if items and all(isinstance(x, str) and re.fullmatch(r"\d+", x) for x in items):
            if set(items) - KEEP:
                fail(f"{path}: NUM list has values outside the kept set")
        for i, v in enumerate(items):
            if isinstance(v, dict) and "NUM" in v and str(v["NUM"]) not in KEEP:
                fail(f"{path}[{i}]: NUM {v['NUM']} not kept")
            walk(v, f"{path}[{i}]")
    elif isinstance(o, pd.DataFrame):
        if "NUM" in o.columns and set(o["NUM"].astype(str)) - KEEP:
            fail(f"{path}: DataFrame has NUM rows outside the kept set")


A = joblib.load(os.path.join(M, "appdata.joblib"))
L = joblib.load(os.path.join(M, "lookup.joblib"))
B = joblib.load(os.path.join(M, "betabeta_v1.joblib"))
NQ = joblib.load(os.path.join(M, "noseq_v1.joblib"))
for name, obj in (("appdata", A), ("lookup", L), ("betabeta_v1", B), ("noseq_v1", NQ)):
    walk(obj, name)

# appdata: drug table, similarity set, names-only catalog
if set(A["base"]["NUM"].astype(str)) != KEEP:
    fail("appdata.base rows are not exactly the kept drugs")
if len(A["sim_feats"]) != len(A["sim_labels"]):
    fail("sim_feats / sim_labels length mismatch")
cat = A["public"]["catalog"]
if not all(isinstance(x, tuple) and len(x) == 2 and all(isinstance(s, str) for s in x) for x in cat):
    fail("public.catalog is not a names-only list of (label, INN)")

# bundles: per-drug arrays are kept rows only and stay aligned
for tg in ("CL", "V"):
    t = B[tg]
    n = len(t["nums"])
    if set(map(str, t["nums"])) - KEEP:
        fail(f"betabeta_v1.{tg}.nums outside the kept set")
    for k in ("INN", "y", "obs", "allo", "comp", "grade"):
        if len(t[k]) != n:
            fail(f"betabeta_v1.{tg}.{k} length {len(t[k])} != {n}")
    for vt, V in t["variants"].items():
        for key, p in V["oof"].items():
            if len(p) != n:
                fail(f"betabeta_v1.{tg}.{vt}.oof[{key}] length {len(p)} != {n}")
    q = NQ[tg]
    for key, p in q["oof"].items():
        if len(p) != n:
            fail(f"noseq_v1.{tg}.oof[{key}] length {len(p)} != {n}")
    if len(q["table"]):
        fail(f"noseq_v1.{tg}.table still has rows")

# names of the other drugs: allowed only in the catalog (appdata); never in lookup or bundles
kept_inn = set(A["base"].loc[A["base"]["NUM"].astype(str).isin(KEEP), "INN"])
others = {inn for _, inn in cat} - kept_inn
others = {s for s in others if len(s) >= 6}
for fname, obj in (("lookup", L), ("betabeta_v1", B), ("noseq_v1", NQ)):
    raw = pickle.dumps(obj)
    hits = [s for s in others if s.encode("utf-8") in raw]
    if hits:
        fail(f"{fname}: names of other drugs found, e.g. {hits[:3]}")
A2 = dict(A)
A2["public"] = {k: v for k, v in A["public"].items() if k != "catalog"}
raw = pickle.dumps(A2)
hits = [s for s in others if s.encode("utf-8") in raw]
if hits:
    fail(f"appdata (outside the catalog): names of other drugs found, e.g. {hits[:3]}")

print("kept NUMs:", sorted(KEEP, key=int))
print(f"catalog names: {len(cat)} (names only); per-target rows:",
      {tg: len(B[tg]["nums"]) for tg in ("CL", "V")}, "| full training sizes:",
      {tg: B[tg].get("n_full") for tg in ("CL", "V")})
print("PUBLIC DATA CHECK", "FAILED" if bad else "OK")
sys.exit(1 if bad else 0)
