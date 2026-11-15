# -*- coding: utf-8 -*-
"""Summarise the local reproduction run: AUROC/AUPRC grid, leakage drop,
seed-0 paired DeLong (EBM vs each) + Holm, fairness arms, conformal coverage."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.evaluation.metrics import delong_test, holm  # noqa: E402

R = Path("results")
raw = pd.read_csv(R / "raw_results.csv")
b = raw[(raw.stage == "benchmark") & (raw.subgroup == "ALL")]

PAPER = {  # paper Table 2 (mean over 5 seeds)
    ("T0", "logreg"): 0.8912, ("T0", "logreg_spline"): 0.8911,
    ("T0", "random_forest"): 0.8907, ("T0", "ebm"): 0.8927,
    ("T0", "xgboost"): 0.8930, ("T0", "lightgbm"): 0.8925,
    ("T0", "catboost"): 0.8933,
    ("T1", "logreg"): 0.8407, ("T1", "logreg_spline"): 0.8405,
    ("T1", "random_forest"): 0.8407, ("T1", "ebm"): 0.8430,
    ("T1", "xgboost"): 0.8430, ("T1", "lightgbm"): 0.8428,
    ("T1", "catboost"): 0.8439,
    ("T1_nostroke", "logreg"): 0.8331, ("T1_nostroke", "logreg_spline"): 0.8326,
    ("T1_nostroke", "random_forest"): 0.8332, ("T1_nostroke", "ebm"): 0.8354,
    ("T1_nostroke", "xgboost"): 0.8353, ("T1_nostroke", "lightgbm"): 0.8349,
    ("T1_nostroke", "catboost"): 0.8362,
}

def piv(metric):
    d = b[(b.metric == metric) & (b.split == "test")]
    return d.pivot_table(index="model", columns="tier", values="value")

auroc, auprc = piv("auroc"), piv("auprc")
order = ["logreg", "logreg_spline", "random_forest", "ebm", "xgboost",
         "lightgbm", "catboost"]
tiers = [t for t in ["T0", "T1", "T1_nostroke"] if t in auroc.columns]

rows = []
for m in [o for o in order if o in auroc.index]:
    r = {"model": m}
    for t in tiers:
        r[f"{t}_meas"] = round(auroc.loc[m, t], 4)
        r[f"{t}_paper"] = PAPER.get((t, m))
        r[f"{t}_d"] = (round(auroc.loc[m, t] - PAPER[(t, m)], 4)
                       if (t, m) in PAPER else None)
    if "T0" in tiers and "T1" in tiers:
        r["drop_meas"] = round(auroc.loc[m, "T0"] - auroc.loc[m, "T1"], 4)
        r["drop_paper"] = round(PAPER[("T0", m)] - PAPER[("T1", m)], 4)
    rows.append(r)
tab = pd.DataFrame(rows)
print("=== AUROC: đo được (1 seed) vs bài (5 seed) ===")
print(tab.to_string(index=False))

for t in tiers:
    v = auroc[t].dropna()
    print(f"\n[{t}] n={len(v)}  min={v.min():.4f} max={v.max():.4f} "
          f"band={v.max()-v.min():.4f}")
    if t == "T1":
        paper = [PAPER[(t, m)] for m in v.index if (t, m) in PAPER]
        print(f"      bài: min={min(paper):.4f} max={max(paper):.4f} "
              f"band={max(paper)-min(paper):.4f}")

print("\n=== AUPRC ===")
print(auprc.round(4).to_string())

# ---- paired DeLong at T1, EBM vs each (seed 0, test split) ---------------
def preds(model, tier="T1", seed=0):
    p = R / "predictions" / "brfss2022" / tier / model / f"seed{seed}.parquet"
    df = pd.read_parquet(p)
    te = df[df.split == "test"].sort_values("row_id")
    return te["y"].values, te["p"].values

if (R / "predictions/brfss2022/T1/ebm/seed0.parquet").exists():
    y, p_ebm = preds("ebm")
    comps = [m for m in order if m != "ebm"
             and (R / f"predictions/brfss2022/T1/{m}/seed0.parquet").exists()]
    res = []
    for m in comps:
        _, p = preds(m)
        a1, a2, pv = delong_test(y, p_ebm, p)
        res.append({"comparator": m, "auroc_ebm": a1, "auroc_comp": a2,
                    "delta": a1 - a2, "p_raw": pv})
    hol = holm([r["p_raw"] for r in res])
    for r, h in zip(res, hol):
        r["p_holm"] = h
        r["non_inferior_0.005"] = (r["delta"] > -0.005) and (h < 0.05)
    dl = pd.DataFrame(res)
    print("\n=== DeLong ghép cặp: EBM − đối thủ (T1, seed 0, test) ===")
    print(dl.round(5).to_string(index=False))

# ---- fairness -----------------------------------------------------------
for f in sorted((R / "fairness").glob("*.csv")):
    d = pd.read_csv(f)
    if {"arm", "tpr_gap"} <= set(d.columns):
        print(f"\n=== fairness: {f.name} ===")
        print(d.round(4).to_string(index=False))

# ---- conformal ----------------------------------------------------------
for f in sorted((R / "conformal").glob("*.csv")):
    d = pd.read_csv(f)
    keep = [c for c in d.columns if c in
            ("model", "method", "group_var", "group", "n", "coverage",
             "set_size", "deferral", "empty_rate")]
    print(f"\n=== conformal: {f.name} ===")
    print(d[keep].round(4).to_string(index=False))
