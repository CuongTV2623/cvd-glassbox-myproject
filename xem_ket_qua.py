# -*- coding: utf-8 -*-
"""Xem nhanh kết quả: bảng AUROC theo mô hình × tầng, so với bài báo.

Dùng:  ./.venv312/Scripts/python.exe xem_ket_qua.py
"""
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
PAPER = {  # Bảng 2 của bài báo (trung bình 5 seed)
    ("T0", "logreg"): 0.8912, ("T0", "logreg_spline"): 0.8911,
    ("T0", "random_forest"): 0.8907, ("T0", "ebm"): 0.8927,
    ("T0", "xgboost"): 0.8930, ("T0", "lightgbm"): 0.8925,
    ("T0", "catboost"): 0.8933,
    ("T1", "logreg"): 0.8407, ("T1", "logreg_spline"): 0.8405,
    ("T1", "random_forest"): 0.8407, ("T1", "ebm"): 0.8430,
    ("T1", "xgboost"): 0.8430, ("T1", "lightgbm"): 0.8428,
    ("T1", "catboost"): 0.8439,
    ("T1_portable", "logreg"): 0.8386, ("T1_portable", "ebm"): 0.8411,
    ("T1_nostroke", "ebm"): 0.8354, ("T1_nostroke", "xgboost"): 0.8353,
}

csv = REPO / "results" / "raw_results.csv"
if not csv.exists():
    raise SystemExit("Chưa có results/raw_results.csv — chạy run_benchmark trước.")

d = pd.read_csv(csv)
b = d[(d.stage == "benchmark") & (d.metric == "auroc") & (d.split == "test")]
if b.empty:
    raise SystemExit("Chưa có dòng AUROC nào trong results/raw_results.csv.")

piv = b.pivot_table(index="model", columns="tier", values="value", aggfunc="mean")
print("=== AUROC test (càng nhiều seed, giá trị càng ổn định) ===")
print(piv.round(4).to_string())

print("\n=== So với bài báo (chỉ các ô có số của bài) ===")
rows = []
for (tier, model), pv in PAPER.items():
    if model in piv.index and tier in piv.columns and pd.notna(piv.loc[model, tier]):
        got = piv.loc[model, tier]
        rows.append({"tầng": tier, "mô hình": model, "đo được": round(got, 4),
                     "bài báo": pv, "chênh": round(got - pv, 4)})
if rows:
    cmp = pd.DataFrame(rows).sort_values(["tầng", "chênh"])
    print(cmp.to_string(index=False))
    print(f"\nchênh trung bình: {cmp['chênh'].mean():+.4f} "
          f"| lớn nhất: {cmp['chênh'].abs().max():.4f}")
else:
    print("(chưa có ô nào trùng với bảng của bài)")

if "T0" in piv.columns and "T1" in piv.columns:
    drop = (piv["T0"] - piv["T1"]).dropna()
    if len(drop):
        print("\n=== Mức tụt khi bỏ 2 biến (T0 → T1) ===")
        print(drop.round(4).to_string())
        print(f"bài báo cáo 0,0494–0,0506 | đo được {drop.min():.4f}–{drop.max():.4f}")

summ = REPO / "results" / "summary.csv"
if summ.exists():
    print(f"\nTệp tổng hợp: {summ} ({len(pd.read_csv(summ))} dòng)")
