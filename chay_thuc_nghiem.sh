#!/usr/bin/env bash
# Chạy thực nghiệm trên macOS / Linux. Dùng: bash chay_thuc_nghiem.sh
set -uo pipefail
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then PY=./.venv/bin/python
elif [ -x ".venv312/bin/python" ]; then PY=./.venv312/bin/python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else echo "FATAL: không tìm thấy python3. Chạy mục 2 của README.md trước."; exit 1; fi

case "${1:-menu}" in
  1|check)
      echo "=== PHIÊN BẢN ==="; $PY -V
      echo "=== THƯ VIỆN ==="
      $PY -c "import numpy,pandas,sklearn,interpret,xgboost,lightgbm,catboost,pyarrow,pyreadstat;print('  tất cả OK')"
      $PY -c "import torch;print('  torch',torch.__version__)" 2>/dev/null || echo "  [cảnh báo] thiếu torch - mlp không chạy"
      echo "=== DỮ LIỆU ==="
      $PY -c "import pandas as pd;d=pd.read_parquet('data/analytic/brfss_2022_analytic.parquet');print('  cohort:',d.shape[0],'x',d.shape[1])"
      echo "=== Ô ĐÃ CHẠY ==="
      $PY -c "import pandas as pd;d=pd.read_csv('results/raw_results.csv');b=d[d.stage=='benchmark'];print('  benchmark:',b[['tier','model','seed']].drop_duplicates().shape[0],'ô')"
      ;;
  2|one)
      TIER="${2:-T1}"; MODEL="${3:-ebm}"
      $PY -m experiments.run_benchmark --config configs/default.yaml --tier "$TIER" --model "$MODEL" --seed 0
      ;;
  3|grid)
      bash run_night.sh
      echo "--- MLP (không có trong run_night.sh) ---"
      $PY -m experiments.run_benchmark --config configs/default.yaml --tier T0 --model mlp --seed 0
      $PY -m experiments.run_benchmark --config configs/default.yaml --tier T1 --model mlp --seed 0
      $PY xem_ket_qua.py
      ;;
  4|show)
      $PY xem_ket_qua.py
      ;;
  5|seeds)
      for s in 1 2 3 4; do for m in ebm xgboost logreg; do
        $PY -m experiments.run_benchmark --config configs/default.yaml --tier T1 --model $m --seed $s
      done; done
      $PY xem_ket_qua.py
      ;;
  6|audit)
      for m in ebm xgboost; do
        $PY -m experiments.run_fairness --config configs/default.yaml --tier T1 --model $m --seed 0 --group Sex
        $PY -m experiments.run_conformal --config configs/default.yaml --tier T1 --model $m --seed 0 --alpha 0.1
      done
      ;;
  *)
      cat <<EOF
==========================================================
  THỰC NGHIỆM: RÒ RỈ NHÃN TRONG SÀNG LỌC TIM MẠCH
==========================================================
  bash chay_thuc_nghiem.sh 1     kiểm tra môi trường + dữ liệu
  bash chay_thuc_nghiem.sh 2     chạy một ô (mặc định T1/ebm)
  bash chay_thuc_nghiem.sh 3     chạy cả lưới (~25 phút)
  bash chay_thuc_nghiem.sh 4     xem kết quả
  bash chay_thuc_nghiem.sh 5     chạy thêm seed 1-4
  bash chay_thuc_nghiem.sh 6     kiểm toán fairness + conformal

  Tuỳ chọn cho mục 2:  bash chay_thuc_nghiem.sh 2 T0 xgboost
EOF
      ;;
esac
