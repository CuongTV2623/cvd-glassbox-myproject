@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Cai dat moi truong - Goi thuc nghiem BRFSS

echo ============================================================
echo   CAI DAT MOI TRUONG
echo   Goi thuc nghiem: ro ri nhan trong sang loc tim mach
echo ============================================================
echo.

set "PYCMD="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYCMD (
        py -%%V -c "import sys" >nul 2>&1
        if !errorlevel! equ 0 set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "import sys;raise SystemExit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1
    if !errorlevel! equ 0 set "PYCMD=python"
)
if not defined PYCMD (
    echo [LOI] Khong tim thay Python 3.10 - 3.12 tren may nay.
    echo.
    echo Tai Python 3.12 tai:
    echo    https://www.python.org/downloads/release/python-3129/
    echo Khi cai nho tick "Add python.exe to PATH", roi chay lai file nay.
    echo.
    pause
    exit /b 1
)
echo Tim thay Python: !PYCMD!
for /f "tokens=*" %%v in ('!PYCMD! -V 2^>^&1') do echo Phien ban: %%v
echo.

if exist ".venv\Scripts\python.exe" (
    echo Da co moi truong .venv - bo qua buoc tao.
) else (
    echo Dang tao moi truong ao .venv ...
    !PYCMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo [LOI] Tao .venv that bai.
        pause & exit /b 1
    )
)
set "PY=%CD%\.venv\Scripts\python.exe"

echo Dang nang cap pip ...
"%PY%" -m pip install --upgrade pip --quiet

echo.
echo Dang cai thu vien - mat 5-10 phut, dung tat cua so ...
"%PY%" -m pip install -r requirements-lock.txt --quiet
if !errorlevel! neq 0 (
    echo [CANH BAO] Ban ghim phien ban khong khop Python nay.
    echo Dang cai theo phien ban toi thieu ...
    "%PY%" -m pip install --quiet "numpy>=1.26" "pandas>=2.2" "scipy>=1.11" "scikit-learn>=1.5" "pyarrow>=15.0" pyyaml joblib matplotlib "interpret>=0.6.3" "xgboost>=2.1" "lightgbm>=4.3" "imbalanced-learn>=0.12" "optuna>=3.6" "shap>=0.46" "pyreadstat>=1.2" "catboost>=1.2"
)

echo Dang cai torch ban CPU - de chay mo hinh MLP ...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if !errorlevel! neq 0 echo [CANH BAO] torch cai loi - chi mo hinh mlp bi anh huong.

echo.
echo ================= KIEM TRA =================
"%PY%" -c "import numpy,pandas,sklearn,interpret,xgboost,lightgbm,catboost,pyarrow,pyreadstat;print('thu vien loi: OK')"
if !errorlevel! neq 0 echo [LOI] thieu thu vien - xem thong bao phia tren.
"%PY%" -c "import pandas as pd;d=pd.read_parquet('data/analytic/brfss_2022_analytic.parquet');print('du lieu:',d.shape[0],'dong x',d.shape[1],'cot')"
"%PY%" -c "import torch;print('torch:',torch.__version__)"
if !errorlevel! neq 0 echo [CANH BAO] thieu torch - mo hinh mlp se khong chay.
echo ===========================================
echo.
echo XONG. Bam dup file CHAY-THUC-NGHIEM.bat de bat dau.
echo.
pause
