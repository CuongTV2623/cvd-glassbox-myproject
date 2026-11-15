# -*- coding: utf-8 -*-
"""Paired DeLong (rank formulation) + one-sided non-inferiority, T1 seed 0."""
import numpy as np
import pandas as pd
from scipy.stats import norm


def midrank(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(len(x), dtype=float)
    out[order] = ranks
    return out


def delong_pair(y, pa, pb):
    """AUCs and paired covariance matrix for two score vectors."""
    y = np.asarray(y).astype(bool)
    X_a, Y_a = np.asarray(pa, float)[y], np.asarray(pa, float)[~y]
    X_b, Y_b = np.asarray(pb, float)[y], np.asarray(pb, float)[~y]
    m, n = len(X_a), len(Y_a)

    def comps(X, Y):
        tx = midrank(X)
        ty = midrank(Y)
        tz = midrank(np.concatenate([X, Y]))
        v10 = (tz[:m] - tx) / n          # per-case structural component
        v01 = 1.0 - (tz[m:] - ty) / m    # per-control
        return v10.mean(), v10, v01

    a1, v10a, v01a = comps(X_a, Y_a)
    a2, v10b, v01b = comps(X_b, Y_b)
    S = np.array([
        [np.cov(v10a) / m + np.cov(v01a) / n,
         np.cov(v10a, v10b)[0, 1] / m + np.cov(v01a, v01b)[0, 1] / n],
        [0.0, np.cov(v10b) / m + np.cov(v01b) / n]])
    S[1, 0] = S[0, 1]
    return a1, a2, S


def load(model, tier="T1", seed=0):
    df = pd.read_parquet(f"results/predictions/brfss2022/{tier}/{model}/seed{seed}.parquet")
    te = df[df.split == "test"].sort_values("row_id")
    return te["y"].values, te["p"].values


# --- sanity check on synthetic data -------------------------------------
rng = np.random.default_rng(0)
ys = (rng.random(20000) < 0.3).astype(int)
z = rng.normal(size=20000)
p1 = 1 / (1 + np.exp(-(z + 0.8 * ys)))
p2 = 1 / (1 + np.exp(-(z + 0.8 * ys + rng.normal(scale=0.3, size=20000))))
a1, a2, S = delong_pair(ys, p1, p2)
d = a1 - a2
se = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
print(f"sanity: auc1={a1:.4f} auc2={a2:.4f} delta={d:+.4f} se={se:.4f} z={d/se:+.2f}")

# --- real data -----------------------------------------------------------
comps = ["logreg", "logreg_spline", "random_forest", "xgboost", "lightgbm", "catboost"]
y, p_ebm = load("ebm")
rows = []
for m in comps:
    _, p = load(m)
    a1, a2, S = delong_pair(y, p_ebm, p)
    delta = a1 - a2
    se = float(np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-20)))
    rows.append({
        "comparator": m, "auroc_ebm": a1, "auroc_comp": a2, "delta": delta, "se": se,
        "ci90_low": delta - 1.645 * se, "ci90_high": delta + 1.645 * se,
        "p_diff": 2 * (1 - norm.cdf(abs(delta / se))),
        "p_noninf": 1 - norm.cdf((delta + 0.005) / se),
        "p_tost_high": 1 - norm.cdf((0.005 - delta) / se),
    })
out = pd.DataFrame(rows).sort_values("delta")
pd.set_option("display.width", 220)
print("\n=== paired DeLong: EBM - comparator (T1, seed 0, n_test=88413) ===")
print(out.round(5).to_string(index=False))
print("\nnon-inferiority (delta=0.005, one-sided):",
      int((out.p_noninf < 0.05).sum()), "/", len(out), "pass")
print("equivalence (TOST, both sides <0.05):",
      int(((out.p_noninf < 0.05) & (out.p_tost_high < 0.05)).sum()), "/", len(out))
out.to_csv("results/delong_noninf_t1.csv", index=False)
