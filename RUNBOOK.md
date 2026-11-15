# RUNBOOK — tự chạy thực nghiệm trên máy này

Hướng dẫn cho chính máy đã dựng sẵn (Windows, i5-12500H 12 nhân, 16 GB RAM, không GPU).
Mọi lệnh dưới đây **đã được chạy thử** và thời gian ghi kèm là số đo thực tế.

---

## 0. Máy đang có sẵn gì (đừng làm lại)

| Thành phần | Vị trí / giá trị |
|---|---|
| Python | `C:\Users\QUYEN\AppData\Local\Programs\Python\Python312` (3.12.10) |
| Môi trường ảo | `repo/.venv312` — numpy 2.4.6, pandas 2.3.3, scikit-learn 1.6.1, interpret 0.7.8, xgboost 2.1.4, lightgbm 4.7.0, catboost 1.2.10, pyreadstat 1.3.7, pyarrow 26.0.0, torch 2.15.0+cpu |
| Dữ liệu thô | `repo/data/raw/LLCP2022.XPT` (1,16 GB, tải từ CDC) |
| Cohort phân tích | `repo/data/analytic/brfss_2022_analytic.parquet` (442 067 × 44) |
| Kết quả đã có | `repo/results/raw_results.csv` — 19 ô benchmark (8 mô hình) + fairness + conformal |

Nếu môi trường bị xoá, xem mục 10 để dựng lại.

---

## 1. Mở đúng terminal

Dùng **Git Bash** (không phải PowerShell/cmd), rồi:

```bash
cd /c/Users/QUYEN/work/cvd-report/repo
```

## 2. Kiểm tra môi trường (≈10 giây)

```bash
./.venv312/Scripts/python.exe -V
./.venv312/Scripts/python.exe -c "import interpret, xgboost, pandas; print('deps ok')"
ls data/analytic/
```

Kết quả mong đợi: `Python 3.12.10`, `deps ok`, và tệp `brfss_2022_analytic.parquet`.

> Luôn gọi python bằng đường dẫn `./.venv312/Scripts/python.exe`. Nếu gõ `python` trần, máy sẽ dùng Python 3.14 và báo `No module named pandas`.

## 3. Chạy thử một ô (≈25 giây)

```bash
time ./.venv312/Scripts/python.exe -m experiments.run_benchmark \
  --config configs/default.yaml --tier T1 --model logreg --seed 0
```

Kết quả mong đợi:

```
[done] ('benchmark', 'brfss2022', 'T1', 'logreg', '0') auroc_test=0.8384 fit=8.7s
```

Nếu ô đó đã chạy rồi, chương trình in `[skip] ... already in results` — đây là cơ chế resume, không phải lỗi.

## 4. Bảng thời gian thực đo (để lên kế hoạch)

| Mô hình | Fit T0 (s) | Fit T1 (s) | Chấm 88 413 người (s) |
|---|---|---|---|
| lightgbm | 2 | 3 | 0,4 |
| xgboost | 7 | 9 | 0,3 |
| logreg | 8 | 9 | 1,7 |
| logreg_spline | 16 | 17 | 1,3 |
| catboost | 56 | 72 | 0,6 |
| random_forest | 124 | 104 | 3,0 |
| **ebm** | **148** | **161** | **0,1** |

Tổng cho lưới 7 mô hình × {T0, T1} + T1-nostroke + fairness + conformal: **≈22 phút** (thêm MLP thì ≈25 phút; MLP fit 52 giây/tầng).

## 5. Chạy cả lưới bằng một lệnh

```bash
bash run_night.sh
```

- Nhật ký: `results/overnight.log`
- Đứt giữa chừng: chạy lại đúng lệnh đó, ô nào xong sẽ `[skip]`
- Script chạy lần lượt: cohort (nếu thiếu) → 7 mô hình × T0/T1 → T1-nostroke (EBM, XGB) → fairness + conformal
- Muốn có luôn MLP: chạy thêm 2 lệnh ở mục 6 (torch đã cài sẵn)

## 6. Chạy thêm chỗ khác

```bash
# tầng khác: T1_portable (dùng cho kiểm chứng 2023), T1_nostroke, T2
./.venv312/Scripts/python.exe -m experiments.run_benchmark \
  --config configs/default.yaml --tier T1_portable --model ebm --seed 0

# seed khác (bài dùng 0..4; chạy thêm 1–4 để có độ lệch chuẩn như bài)
./.venv312/Scripts/python.exe -m experiments.run_benchmark \
  --config configs/default.yaml --tier T1 --model ebm --seed 1
```

Mô hình dùng được trên máy này (**8/10**): `logreg`, `logreg_spline`, `random_forest`,
`xgboost`, `lightgbm`, `catboost`, `ebm`, `mlp` (torch 2.15.0+cpu đã cài; MLP fit ≈52 giây/tầng).
Không chạy được: `tabpfn_v2`, `tabicl` — cần GPU 16–24 GB, máy có 4 GB VRAM.

## 7. Kiểm toán công bằng và bất định

```bash
./.venv312/Scripts/python.exe -m experiments.run_fairness \
  --config configs/default.yaml --tier T1 --model ebm --seed 0 --group Sex

./.venv312/Scripts/python.exe -m experiments.run_conformal \
  --config configs/default.yaml --tier T1 --model ebm --seed 0 --alpha 0.1
```

Hai lệnh này đọc dự đoán đã lưu trong `results/predictions/`, **không huấn luyện lại**.

## 8. Tổng hợp → bảng → hình

```bash
./.venv312/Scripts/python.exe -m analysis.aggregate_results --config configs/default.yaml
./.venv312/Scripts/python.exe -m analysis.make_tables       --config configs/default.yaml
./.venv312/Scripts/python.exe -m analysis.make_figures      --config configs/default.yaml
```

Kết quả (đã chạy thử ngày 10/11/2026): `results/summary.csv` (1 298 dòng), `results/delong.csv`
(13 so sánh), `paper/tables/`, `paper/figures/*.pdf`.

## 9. Đọc kết quả

```bash
./.venv312/Scripts/python.exe xem_ket_qua.py
```

In ra bảng AUROC theo mô hình × tầng, chênh lệch so với bài, và mức tụt khi bỏ hai biến.
Số hiện có (1 seed, không tuning): chênh trung bình **−0,0037**, lớn nhất 0,0061; mức tụt
0,0468–0,0487 so với 0,0494–0,0506 của bài.

## 10. Dựng lại từ đầu (nếu xoá mất)

```bash
# 1) dữ liệu (≈2 phút tải, 80,7 MB nén)
mkdir -p data/raw && cd data/raw
curl -L -o LLCP2022XPT.zip https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip
unzip -o LLCP2022XPT.zip && mv "LLCP2022.XPT " LLCP2022.XPT
cd ../..

# 2) cohort (≈6 phút, đọc 328 cột từ tệp 1,16 GB)
./.venv312/Scripts/python.exe -m src.data.build_dataset \
  --year 2022 --xpt data/raw/LLCP2022.XPT --out data/analytic
```

Kiểm tra: phải in `outcome-observed=442,067 (prev=0.0568)`.

## 11. Lỗi thường gặp

| Hiện tượng | Nguyên nhân và cách xử lý |
|---|---|
| `No module named pandas` | Dùng sai python — gọi `./.venv312/Scripts/python.exe` |
| `... missing — run src.data.build_dataset first` | Chưa có cohort; xem mục 10 |
| `[WinError 32] ... being used by another process` | Tệp đang mở (Excel/Word) hoặc tiến trình khác đang ghi; đóng rồi chạy lại |
| `OptimizeWarning: Unknown solver options: iprint` | Vô hại — scikit-learn 1.6 bỏ tham số cũ; kết quả không đổi |
| Chạy chậm bất thường | EBM ăn hết 12 nhân; kiểm tra Task Manager xem còn tiến trình cũ |
| Kết quả lệch bài ±0,003 | Bình thường với 1 seed và không tuning; muốn sát bài thì chạy thêm seed 1–4 |
| Muốn xoá kết quả để chạy lại | Xoá `results/raw_results.csv` (chỉ khi thật sự cần) |

## 12. Nếu muốn chạy trên Colab

Dùng `Colab_BRFSS_Leakage_Repro.ipynb` (đã để trong thư mục dự án). Colab free đủ cho
logreg/xgboost và cho phép chạm tới TabPFN v2; TabICL thì cần Colab Pro (A100 40 GB).
