# Bộ chạy thực nghiệm — Rò rỉ nhãn trong sàng lọc tim mạch (BRFSS 2022)

Tái lập thí nghiệm của bài **arXiv:2609.11838** — *Target leakage, not model class, explains
reported accuracy in survey-based cardiovascular screening* — trên dữ liệu CDC BRFSS 2022.

Bộ này **đã bao gồm dữ liệu đã xử lý** (9,8 MB), nên không cần tải 1,16 GB từ CDC.

---

## 1. Yêu cầu máy

| | Tối thiểu | Ghi chú |
|---|---|---|
| Hệ điều hành | Windows 10/11 · macOS · Linux | có script riêng cho từng loại |
| Python | **3.10 – 3.12** | 3.13/3.14 **không** cài được `pyarrow` bản cũ |
| RAM | 8 GB | 16 GB thì thoải mái |
| Ổ đĩa | 3 GB trống | venv ~1,5 GB + kết quả ~200 MB |
| GPU | **không cần** | 8 mô hình chạy CPU |

Kiểm tra Python đã có chưa:

```bat
py -0            :: Windows: liệt kê bản Python đã cài
python3 --version   :: macOS/Linux
```

Chưa có thì tải ở <https://www.python.org/downloads/release/python-3129/> (nhớ tick
*Add python.exe to PATH*).

---

## 2. Cài đặt (chỉ 1 lần, 5–10 phút)

**Windows:** bấm đúp **`CAI-DAT.bat`**

**macOS/Linux:**
```bash
cd GOI-THUC-NGHIEM
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Script sẽ tạo `.venv`, cài đúng phiên bản thư viện đã kiểm chứng, rồi tự kiểm tra lại.

---

## 3. Chạy thực nghiệm

**Windows:** bấm đúp **`CHAY-THUC-NGHIEM.bat`** → chọn số.

| Chức năng | Thời gian | Làm gì |
|---|---|---|
| **1. Kiểm tra** | 15 giây | xác nhận thư viện + dữ liệu + số ô đã chạy |
| **2. Một ô** | 10 giây – 3 phút | chạy 1 cặp (tầng × mô hình) để thử |
| **3. Cả lưới** | **~25 phút** | 8 mô hình × T0/T1 + ablation + fairness + conformal |
| **4. Xem kết quả** | tức thì | bảng AUROC so với bài báo |
| **5. Thêm seed 1–4** | ~8 phút/seed | để có độ lệch chuẩn như bài (bài dùng 5 seed) |
| **6. Kiểm toán** | ~6 phút | fairness 6 nhánh + conformal 3 chính sách |
| **7. Mở thư mục kết quả** | — | mở `results/` trong Explorer |

**macOS/Linux:** `bash chay_thuc_nghiem.sh` hoặc `bash run_night.sh`.

Chạy lại nhiều lần **không tốn thời gian**: ô nào đã có trong `results/raw_results.csv` thì
in `[skip]` và bỏ qua. Đứt giữa chừng chỉ cần chạy tiếp.

---

## 4. Kiểm tra mình chạy có đúng không

Mở chức năng 4, hoặc đối chiếu tay bảng sau (seed 0, tầng T1):

| Mô hình | AUROC máy này | AUROC trong bài | Lệch |
|---|---|---|---|
| logreg | 0,8384 | 0,8407 | −0,0023 |
| ebm | 0,8402 | 0,8430 | −0,0028 |
| xgboost | 0,8398 | 0,8430 | −0,0032 |
| catboost | 0,8414 | 0,8439 | −0,0025 |
| mlp | 0,8394 | 0,8432 | −0,0038 |

Chênh lệch trung bình −0,0037 là **bình thường**: bài dùng 5 seed và tuning Optuna, bộ này
dùng seed 0 và tham số mặc định. Điều cần khớp là **độ tụt T0→T1**: máy này 0,0468–0,0487,
bài 0,0494–0,0506.

Ba con số khác để đối chiếu:

- Độ tụt AUROC của EBM: **0,0477** (0,8879 → 0,8402)
- ΔTPR giới tính tại ngưỡng sàng lọc: **0,1468** (nữ 0,7456 · nam 0,8925)
- Coverage conformal biên ở nhóm 60+: **0,8183** → Mondrian(giới×tuổi): **0,9006**

---

## 5. Kết quả nằm ở đâu

```
results/
├── raw_results.csv          nguồn số duy nhất (1 408 dòng, long format)
├── summary.csv              bảng tổng hợp theo tầng × mô hình
├── delong.csv               so sánh cặp DeLong giữa các mô hình
├── predictions/             (tự sinh) dự đoán từng dòng: row_id, split, y, p, Sex, ...
├── fairness/                audit_*.csv + editlog_*.json từng nhánh sửa
├── conformal/               coverage_alpha0.1.csv từng chính sách
├── _exports/                bảng gọn đọc được bằng Excel
├── tham-chieu/              predictions.zip của lần chạy gốc (để đối chiếu)
└── overnight.log            log thời gian từng ô
```

`results/tham-chieu/predictions.zip` là dự đoán của lần chạy gốc — giải nén rồi so với
`results/predictions/` của mình để biết có khớp tuyệt đối hay không (nếu cùng seed và cùng
phiên bản thư viện thì phải khớp đến 4 chữ số).

---

## 6. Những gì **không** chạy được trong bộ này

| Thành phần | Lý do | Muốn chạy thì |
|---|---|---|
| `tabpfn_v2` | cần GPU ≥ 8 GB VRAM | Google Colab T4 (miễn phí) |
| `tabicl` | ngữ cảnh 265k dòng, cần GPU 24 GB | Colab Pro / A100 |
| LIME (nhánh faithfulness) | `lime==0.2.0.1` không build được trên Python ≥3.13 | Python 3.12 (bộ này đã là 3.12 → chạy được) |
| Kiểm chứng 2023 | cần tải thêm `LLCP2023XPT.zip` (400 MB) | xem mục 10 của `RUNBOOK.md` |

8/10 mô hình chạy được trên CPU. Kết luận chính của bài (rò rỉ nhãn giải thích độ chính xác,
không phải loại mô hình) **không phụ thuộc** hai mô hình nền tảng.

---

## 7. Lỗi thường gặp

| Thông báo | Cách sửa |
|---|---|
| `No module named pandas` | đang gọi Python hệ thống; dùng `.venv\Scripts\python.exe` |
| `pyarrow` cài lỗi | Python không phải 3.10–3.12 → cài lại 3.12 |
| `cohort ... missing` | thiếu `data/analytic/brfss_2022_analytic.parquet` |
| `[skip]` khi chạy | ô đã chạy rồi — không phải lỗi |
| `PermissionError: WinError 32` | đang mở `raw_results.csv` bằng Excel → đóng lại |
| Chạy chậm bất thường | EBM ăn hết nhân CPU; đóng ứng dụng nặng khác |

Chi tiết đầy đủ: `RUNBOOK.md` (12 mục). Hướng dẫn gốc của tác giả: `README-TAC-GIA.md`.

---

## 8. Nguồn và giấy phép

- Bài báo: <https://arxiv.org/abs/2609.11838> · mã nguồn: <https://github.com/raadbintareaf/cvd-glassbox-screening> (MIT) · lưu trữ: <https://doi.org/10.5281/zenodo.21918761>
- Dữ liệu: CDC BRFSS 2022, phạm vi công cộng, người trả lời đã được ẩn danh.
- Dữ liệu trong bộ này là **bản đã xử lý** (442 067 × 44); repo gốc không phân phối lại file CDC thô, ở đây cũng vậy.
