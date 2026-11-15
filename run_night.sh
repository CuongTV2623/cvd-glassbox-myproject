#!/usr/bin/env bash
# Overnight reproduction grid: T0 / T1 / T1-nostroke x CPU models x seed 0,
# then fairness + conformal on the saved predictions.
# Resumable: run_benchmark skips cells already present in results/raw_results.csv.
set -uo pipefail
cd "$(dirname "$0")"   # this script lives at the repo root

# Tự dò môi trường: .venv (Windows/Linux) → .venv312 → python hệ thống
if [ -x ".venv/Scripts/python.exe" ]; then PY=./.venv/Scripts/python.exe
elif [ -x ".venv/bin/python" ];        then PY=./.venv/bin/python
elif [ -x ".venv312/Scripts/python.exe" ]; then PY=./.venv312/Scripts/python.exe
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi
[ -x "$PY" ] || command -v "$PY" >/dev/null 2>&1 || { echo "FATAL: không tìm thấy python"; exit 1; }
CFG=configs/default.yaml
LOG=results/overnight.log
mkdir -p results
ts() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

say "=== start (python: $($PY -V 2>&1)) ==="

# ---- 0. cohort -------------------------------------------------------------
if [ ! -f data/analytic/brfss_2022_analytic.parquet ]; then
  say "building cohort 2022"
  $PY -m src.data.build_dataset --year 2022 --xpt data/raw/LLCP2022.XPT \
      --out data/analytic 2>&1 | tee -a "$LOG"
else
  say "cohort already present"
fi

# ---- 1. main grid: 7 CPU models x 2 tiers ----------------------------------
MODELS="logreg logreg_spline random_forest xgboost lightgbm catboost ebm"
for T in T0 T1; do
  for M in $MODELS; do
    say "benchmark $T/$M/seed0"
    $PY -m experiments.run_benchmark --config "$CFG" --tier "$T" --model "$M" \
        --seed 0 2>&1 | tail -3 | tee -a "$LOG"
  done
done

# ---- 2. T1-nostroke ablation (paper's finer tier), 2 models ----------------
for M in ebm xgboost; do
  say "benchmark T1_nostroke/$M/seed0"
  $PY -m experiments.run_benchmark --config "$CFG" --tier T1_nostroke \
      --model "$M" --seed 0 2>&1 | tail -3 | tee -a "$LOG"
done

# ---- 3. audits that read the saved predictions -----------------------------
for M in ebm xgboost; do
  say "fairness T1/$M (group=Sex)"
  $PY -m experiments.run_fairness --config "$CFG" --tier T1 --model "$M" \
      --seed 0 --group Sex 2>&1 | tail -4 | tee -a "$LOG"

  say "conformal T1/$M (alpha=0.1)"
  $PY -m experiments.run_conformal --config "$CFG" --tier T1 --model "$M" \
      --seed 0 --alpha 0.1 2>&1 | tail -4 | tee -a "$LOG"
done

say "=== DONE ==="
