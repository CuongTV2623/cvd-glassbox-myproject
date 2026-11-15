# -*- coding: utf-8 -*-
"""Xuất bộ file kết quả dễ đọc: index + fairness + conformal + biến theo tầng + zip predictions."""
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
RES = REPO / "results"
OUT = RES / "_exports"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))

raw = pd.read_csv(RES / "raw_results.csv")

# ---------------------------------------------------------------- 1. index
rows = []
for p in sorted((RES / "predictions").rglob("*.parquet")):
    d = pd.read_parquet(p)
    te = d[d.split == "test"]
    rows.append({
        "file": str(p.relative_to(RES)).replace("\\", "/"),
        "dòng": len(d), "val": int((d.split == "val").sum()),
        "test": int((d.split == "test").sum()),
        "tỷ lệ mắc (test)": round(float(te.y.mean()), 4),
        "xác suất TB (test)": round(float(te.p.mean()), 4),
        "cột": ", ".join(d.columns),
    })
idx = pd.DataFrame(rows)
idx.to_csv(OUT / "predictions_index.csv", index=False, encoding="utf-8-sig")
print(f"[index] {len(idx)} tệp → {OUT/'predictions_index.csv'}")

# ---------------------------------------------------------------- 2. zip
zp = RES / "predictions.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted((RES / "predictions").rglob("*.parquet")):
        z.write(p, p.relative_to(RES))
print(f"[zip] {zp} ({zp.stat().st_size/1e6:.1f} MB)")

# ------------------------------------------------------------ 3. fairness
f = raw[raw.stage == "fairness"].copy()
if not f.empty:
    SUF = {"_tpr_gap": "ΔTPR", "_sensitivity": "sensitivity",
           "_specificity": "specificity", "_fpr_gap": "ΔFPR", "_ppv_gap": "ΔPPV"}
    def split_metric(m):
        for s, name in SUF.items():
            if m.endswith(s):
                return m[: -len(s)], name
        return m, "khác"
    parsed = f.metric.map(split_metric)
    f["nhánh"] = [a for a, _ in parsed]
    f["chỉ số"] = [b for _, b in parsed]
    tab = f.pivot_table(index=["model", "nhánh"], columns="chỉ số",
                        values="value").reset_index()
    tab.columns.name = None
    tab = tab.rename(columns={"model": "mô hình"})
    tab = tab.round(4)
    tab.to_csv(OUT / "fairness_arms.csv", index=False, encoding="utf-8-sig")
    print(f"[fairness] {len(tab)} dòng → {OUT/'fairness_arms.csv'}")

# ----------------------------------------------------------- 4. conformal
c = raw[raw.stage == "conformal"].copy()
if not c.empty:
    keep = ["ALL=ALL", "Sex=Female", "Sex=Male",
            "AgeBand=18-39", "AgeBand=40-59", "AgeBand=60+"]
    c = c[c.subgroup.isin(keep)].copy()
    POL = ["mondrian_sex_age", "mondrian_sex", "marginal"]
    def split_pol(m):
        for pol in POL:
            if m.startswith(pol + "_"):
                return pol, m[len(pol) + 1:]
        return "khác", m
    parsed = c.metric.map(split_pol)
    c["chính sách"] = [a for a, _ in parsed]
    c["chỉ số"] = [b for _, b in parsed]
    wide = c.pivot_table(index=["model", "chính sách", "subgroup"],
                         columns="chỉ số", values="value").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={"model": "mô hình", "subgroup": "nhóm"})
    wide.to_csv(OUT / "conformal_by_group.csv", index=False, encoding="utf-8-sig")
    print(f"[conformal] {len(wide)} dòng → {OUT/'conformal_by_group.csv'}")

# --------------------------------------------------- 5. biến theo tầng
from src.data.variable_map import spec_for_year, tier_labels  # noqa: E402

spec = spec_for_year(2022)
mem = {t: set(tier_labels(t)) for t in ("T0", "T1", "T2", "T1_portable")}
mem["T1_nostroke"] = set(tier_labels("T1", drop_stroke=True))

rows = []
for raw_name, v in spec.items():
    if v.get("role") != "predictor":
        continue
    label = v["label"]
    rows.append({
        "biến gốc (BRFSS)": raw_name,
        "tên phân tích": label,
        "nhóm nội dung": v.get("family", ""),
        "kiểu": v.get("kind", ""),
        "cờ rò rỉ": v.get("leakage", ""),
        "T0": "✓" if label in mem["T0"] else "",
        "T1": "✓" if label in mem["T1"] else "",
        "T1-ns": "✓" if label in mem["T1_nostroke"] else "",
        "T1-portable": "✓" if label in mem["T1_portable"] else "",
        "T2": "✓" if label in mem["T2"] else "",
    })
mem_df = pd.DataFrame(rows).sort_values(["T0", "T1", "tên phân tích"], ascending=[False, False, True])
mem_df.to_csv(OUT / "bien_theo_tang.csv", index=False, encoding="utf-8-sig")
print(f"[biến] {len(mem_df)} biến → {OUT/'bien_theo_tang.csv'}")
print("số biến mỗi tầng:", {k: len(v) for k, v in mem.items()})

# ------------------------------------------------- 6. bảng gọn raw_results
small = raw[raw.metric.isin(["auroc", "auprc", "ece", "cal_slope", "brier",
                             "fit_seconds", "predict_seconds", "n_train"])]
small = small[small.split.isin(["test", "train"])]
small.to_csv(OUT / "raw_results_gon.csv", index=False, encoding="utf-8-sig")
print(f"[gọn] {len(small)} dòng → {OUT/'raw_results_gon.csv'}")
