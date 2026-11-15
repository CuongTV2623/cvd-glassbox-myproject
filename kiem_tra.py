# -*- coding: utf-8 -*-
"""Kiểm tra môi trường, dữ liệu và tiến độ chạy. Gọi: python kiem_tra.py"""
import sys
from pathlib import Path

print("=== PHIÊN BẢN ===")
print(" ", sys.version.split()[0], "|", sys.executable)

print("\n=== THƯ VIỆN ===")
mods = ["numpy", "pandas", "sklearn", "scipy", "pyarrow", "yaml",
        "interpret", "xgboost", "lightgbm", "catboost", "pyreadstat"]
missing = []
for m in mods:
    try:
        mod = __import__(m)
        print(f"  {m:12s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        missing.append(m)
        print(f"  {m:12s} THIẾU ({type(e).__name__})")
try:
    import torch
    print(f"  {'torch':12s} {torch.__version__}  (cần cho mô hình mlp)")
except Exception:
    print(f"  {'torch':12s} THIẾU — mlp sẽ không chạy, các mô hình khác vẫn được")

print("\n=== DỮ LIỆU ===")
p = Path("data/analytic/brfss_2022_analytic.parquet")
if p.exists():
    import pandas as pd
    d = pd.read_parquet(p)
    print(f"  cohort: {len(d):,} dòng x {d.shape[1]} cột")
    print(f"  tỷ lệ mắc bệnh: {d.HadHeartAttack.mean():.4f}")
    for col in ("Sex", "AgeCategory"):
        if col in d.columns:
            print(f"  {col}: {d[col].nunique()} giá trị")
else:
    print("  THIẾU file data/analytic/brfss_2022_analytic.parquet")
    missing.append("dữ liệu")

print("\n=== TIẾN ĐỘ ===")
r = Path("results/raw_results.csv")
if r.exists():
    import pandas as pd
    d = pd.read_csv(r)
    b = d[d.stage == "benchmark"]
    cells = b[["tier", "model", "seed"]].drop_duplicates()
    print(f"  benchmark : {len(cells)} ô  (lưới đầy đủ là 18 ô)")
    print(f"  fairness  : {len(d[d.stage == 'fairness'])} dòng")
    print(f"  conformal : {len(d[d.stage == 'conformal'])} dòng")
    if len(cells):
        print("\n  các ô đã có:")
        for (t, m, s), _ in cells.groupby(["tier", "model", "seed"]):
            print(f"    {t:14s} {m:14s} seed {s}")
else:
    print("  chưa có results/raw_results.csv")

print("\n" + ("[!] Còn thiếu: " + ", ".join(missing) if missing else "[OK] Môi trường sẵn sàng."))
