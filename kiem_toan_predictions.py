# -*- coding: utf-8 -*-
"""Kiểm toán predictions: split, toàn vẹn dữ liệu, tính lại AUROC và ngưỡng."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.evaluation.metrics import roc_auc_score, threshold_for  # noqa: E402

RES = Path("results")
files = sorted((RES / "predictions").rglob("*.parquet"))
print(f"Tệp: {len(files)}\n")

raw = pd.read_csv(RES / "raw_results.csv")
bench = raw[(raw.stage == "benchmark") & (raw.subgroup == "ALL")]

rows, splits = [], {}
for p in files:
    d = pd.read_parquet(p)
    tier, model = p.parts[-3], p.parts[-2]
    va, te = d[d.split == "val"], d[d.split == "test"]
    splits[(tier, model)] = (set(va.row_id), set(te.row_id))

    # --- tính lại AUROC & ngưỡng từ chính tệp dự đoán
    auc = roc_auc_score(te.y.values, te.p.values)
    thr = threshold_for(va.y.values, va.p.values, "sens_at", 0.85)

    # --- so với những gì run_benchmark đã ghi
    def g(metric, split):
        s = bench[(bench.tier == tier) & (bench.model == model)
                  & (bench.metric == metric) & (bench.split == split)]
        return float(s.value.iloc[0]) if len(s) else np.nan
    auc_rec = g("auroc", "test")
    extra = bench[(bench.tier == tier) & (bench.model == model)
                  & (bench.metric == "screening_sensitivity")]["extra"]
    thr_rec = float(extra.iloc[0].split("=")[1]) if len(extra) else np.nan

    rows.append({
        "tầng": tier, "mô hình": model,
        "dòng": len(d), "val": len(va), "test": len(te),
        "val∩test": len(set(va.row_id) & set(te.row_id)),
        "prev val": round(va.y.mean(), 4), "prev test": round(te.y.mean(), 4),
        "p∈[0,1]": bool(d.p.between(0, 1).all()), "NaN p": int(d.p.isna().sum()),
        "AUROC tái tính": round(auc, 4), "AUROC đã ghi": round(auc_rec, 4),
        "khớp": "✓" if abs(auc - auc_rec) < 5e-5 else "✗",
        "ngưỡng tái tính": round(thr, 4), "ngưỡng đã ghi": round(thr_rec, 4),
        "khớp thr": "✓" if abs(thr - thr_rec) < 1e-4 else "✗",
    })

aud = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("=== 1. TOÀN VẸN TỆP VÀ TÍNH LẠI CHỈ SỐ ===")
print(aud.to_string(index=False))

# ---- 2. mọi model dùng CÙNG một split?
print("\n=== 2. CÙNG SPLIT GIỮA CÁC MÔ HÌNH? ===")
for tier in sorted({t for t, _ in splits}):
    sets = [(m, s) for (t, m), s in splits.items() if t == tier]
    base_m, (base_va, base_te) = sets[0]
    same = all(va == base_va and te == base_te for _, (va, te) in sets[1:])
    print(f"{tier:14s} {len(sets)} mô hình · cùng val: {same and all(s[0]=='x' or True for s in sets)}"
          f" · val={len(base_va):,} · test={len(base_te):,}"
          f" · trùng val↔test={len(base_va & base_te)}")

# ---- 3. nhãn y có nhất quán giữa các mô hình?
print("\n=== 3. NHÃN y NHẤT QUÁN GIỮA CÁC MÔ HÌNH (cùng row_id) ===")
for tier in sorted({t for t, _ in splits}):
    cols = {}
    for (t, m), _ in splits.items():
        if t != tier:
            continue
        d = pd.read_parquet(RES / "predictions" / "brfss2022" / t / m / "seed0.parquet")
        cols[m] = d.set_index("row_id").y.sort_index()
    first = list(cols)[0]
    ok = all(cols[m].equals(cols[first]) for m in list(cols)[1:])
    print(f"{tier:14s} {len(cols)} mô hình · y giống nhau: {ok}")

# ---- 4. kích thước split có khớp 60/20/20 và tổng cohort?
print("\n=== 4. KÍCH THƯỚC SPLIT ===")
n_val, n_test = 88414, 88413
n_train = int(bench[bench.metric == "n_train"].value.iloc[0])
total = n_train + n_val + n_test
print(f"train={n_train:,} val={n_val:,} test={n_test:,} → tổng={total:,} (cohort 442 067)")
print(f"tỷ lệ: {n_train/total:.3f} / {n_val/total:.3f} / {n_test/total:.3f}  (mục tiêu 0.600 / 0.200 / 0.200)")
