# -*- coding: utf-8 -*-
"""Sinh MANIFEST.txt cho gói thực nghiệm: liệt kê file + SHA-256."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).parent
lines = [
    "GOI-THUC-NGHIEM — Bản kê nội dung",
    "Bài: arXiv:2609.11838 — Target leakage, not model class, explains reported accuracy",
    "Dữ liệu: CDC BRFSS 2022 (442 067 người trả lời, 44 cột sau xử lý)",
    "",
    f"{'SHA-256':<16}{'KB':>9}  TỆP",
    "-" * 78,
]
total = 0
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or p.name == "MANIFEST.txt":
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    kb = p.stat().st_size / 1024
    total += p.stat().st_size
    lines.append(f"{h[:16]}  {kb:8.1f}  {p.relative_to(ROOT).as_posix()}")

lines += [
    "-" * 78,
    f"Tổng: {total/1e6:.1f} MB",
    "",
    "Kiểm tra toàn vẹn sau khi nhận (Windows PowerShell):",
    '  Get-FileHash -Algorithm SHA256 GOI-THUC-NGHIEM.zip',
    "",
    "Cấu trúc:",
    "  src/ experiments/ configs/ analysis/ tests/ scripts/   mã nguồn (MIT)",
    "  data/analytic/brfss_2022_analytic.parquet              dữ liệu đã xử lý, 9,8 MB",
    "  results/                                               kết quả lần chạy gốc",
    "  results/tham-chieu/predictions.zip                     dự đoán gốc để đối chiếu",
    "  README.md                                              hướng dẫn bắt đầu",
    "  CAI-DAT.bat · CHAY-THUC-NGHIEM.bat · chay_thuc_nghiem.sh   bộ chạy tự động",
    "  RUNBOOK.md                                             hướng dẫn chi tiết 12 mục",
]
(ROOT / "MANIFEST.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[-12:]))
print(f"\nĐã ghi {ROOT/'MANIFEST.txt'} ({len(lines)} dòng)")
