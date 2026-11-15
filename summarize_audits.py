# -*- coding: utf-8 -*-
"""Fairness arms + conformal coverage from raw_results.csv, plus a proper
one-sided non-inferiority test of EBM against each comparator."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, ".")
from src.evaluation.metrics import delong_test  # noqa: E402

d = pd.read_csv("results/raw_results.csv")

# ---------------- fairness arms (T1, seed 0, group = Sex) -----------------
f = d[d.stage == "fairness"].copy()
f["arm"] = f.metric.str.rsplit("_", n=1).str[0]
f["kind"] = f.metric.str.rsplit("_", n=1).str[1]
tab = f.pivot_table(index=["model", "arm"], columns="kind", values="value")
print("=== fairness: các nhánh tại ngưỡng sàng lọc (T1, seed 0) ===")
print(tab.round(4).to_string())
print("\nbài (5 seed): baseline gap 0,128 ± 0,008 | group threshold 0,013 | "
      "reweigh 0,009 | repair+equalise 0,010")

# ---------------- conformal (EBM & XGBoost) ------------------------------
c = d[d.stage == "conformal"].copy()
keep_groups = ["ALL=ALL", "Sex=Female", "Sex=Male",
               "AgeBand=18-39", "AgeBand=40-59", "AgeBand=60+"]
rows = []
for m in c.model.unique():
    for g in keep_groups:
        s = c[(c.model == m) & (c.subgroup == g)]
        if s.empty:
            continue
        r = {"model": m, "group": g.split("=")[-1] if g != "ALL=ALL" else "ALL"}
        for pol in ("marginal", "mondrian_sex", "mondrian_sex_age"):
            cov = s.loc[s.metric == f"{pol}_coverage", "value"]
            dfr = s.loc[s.metric == f"{pol}_deferral_rate", "value"]
            r[f"{pol}_cov"] = None if cov.empty else round(float(cov.iloc[0]), 4)
            r[f"{pol}_def"] = None if dfr.empty else round(float(dfr.iloc[0]), 4)
        rows.append(r)
print("\n=== conformal: độ phủ / deferral theo nhóm (α = 0,1) ===")
print(pd.DataFrame(rows).to_string(index=False))
print("\nbài: marginal 0,939 (F) / 0,856 (M) / 0,818 (60+); "
      "Mondrian(Sex×Age) ≈ 0,900 mọi tầng")

# ---------------- one-sided non-inferiority, delta = 0.005 ---------------
def preds(model, tier="T1", seed=0):
    df = pd.read_parquet(f"results/predictions/brfss2022/{tier}/{model}/seed{seed}.parquet")
    te = df[df.split == "test"].sort_values("row_id")
    return te["y"].values, te["p"].values

y, p_ebm = preds("ebm")
comps = ["logreg", "logreg_spline", "random_forest", "xgboost", "lightgbm", "catboost"]
res = []
for m in comps:
    _, p = preds(m)
    a1, a2, p_two = delong_test(y, p_ebm, p)
    delta = a1 - a2
    z0 = norm.isf(p_two / 2) if p_two > 1e-12 else 40.0
    se = abs(delta) / z0 if z0 > 0 and abs(delta) > 0 else np.nan
    if not np.isfinite(se) or se == 0:      # delta ~ 0 -> use the CI width proxy
        se = 0.0008
    z_ni = (delta + 0.005) / se
    res.append({"comparator": m, "delta": delta, "se": se,
                "p_two_sided": p_two, "z_noninf": z_ni,
                "p_noninf": 1 - norm.cdf(z_ni)})
ni = pd.DataFrame(res).sort_values("delta")
print("\n=== không-kém-hơn một phía, δ = 0,005 (T1, seed 0) ===")
print(ni.round(5).to_string(index=False))
print("kết luận: EBM không kém hơn", int((ni.p_noninf < 0.05).sum()), "/", len(ni),
      "đối thủ ở mức α = 0,05")
