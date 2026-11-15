@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title Chay thuc nghiem - ro ri nhan BRFSS

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [!] Chua cai moi truong. Hay chay CAI-DAT.bat truoc.
    pause & exit /b 1
)

set "ONESHOT=0"
if not "%~1"=="" (
    set "ONESHOT=1"
    set "CHOICE=%~1"
    goto RUN
)

:MENU
cls
echo ==========================================================
echo   THUC NGHIEM: RO RI NHAN TRONG SANG LOC TIM MACH
echo   Du lieu: BRFSS 2022 - 442 067 nguoi
echo ==========================================================
echo   1. Kiem tra moi truong va du lieu      (15 giay)
echo   2. Chay MOT o benchmark                (10 giay - 3 phut)
echo   3. Chay CA LUOI - 8 mo hinh            (~25 phut)
echo   4. Xem ket qua (AUROC so voi bai bao)  (tuc thi)
echo   5. Chay them seed 1-4                  (~8 phut/seed)
echo   6. Kiem toan fairness + conformal      (~6 phut)
echo   7. Mo thu muc ket qua
echo   0. Thoat
echo ==========================================================
set /p CHOICE=Chon [0-7]:
goto RUN

:RUN
if "%CHOICE%"=="1" goto CHECK
if "%CHOICE%"=="2" goto ONE
if "%CHOICE%"=="3" goto GRID
if "%CHOICE%"=="4" goto SHOW
if "%CHOICE%"=="5" goto SEEDS
if "%CHOICE%"=="6" goto AUDIT
if "%CHOICE%"=="7" goto OPEN
if "%CHOICE%"=="0" exit /b 0
echo Lua chon khong hop le.
call :ENDACT
goto MENU

:CHECK
echo.
"%PY%" kiem_tra.py
call :ENDACT
goto MENU

:ONE
echo.
echo Cac tang  : T0  T1  T1_nostroke  T1_portable  T2
set /p TIER=Nhap tang [T1]:
if "%TIER%"=="" set "TIER=T1"
echo Cac mo hinh: logreg logreg_spline random_forest xgboost lightgbm catboost ebm mlp
set /p MODEL=Nhap mo hinh [ebm]:
if "%MODEL%"=="" set "MODEL=ebm"
echo.
echo Dang chay %TIER% / %MODEL% / seed 0 ...
"%PY%" -m experiments.run_benchmark --config configs/default.yaml --tier %TIER% --model %MODEL% --seed 0
echo.
echo (Thay [skip] nghia la o nay da co ket qua - khong phai loi)
call :ENDACT
goto MENU

:GRID
echo.
echo Chay 18 o: T0/T1 x 8 mo hinh + ablation T1-nostroke x 2
echo Sau do: fairness + conformal cho ebm va xgboost
echo Uoc tinh 25 phut. Co the tat va chay lai - ket qua da co se duoc bo qua.
echo.
for %%T in (T0 T1) do (
    for %%M in (logreg logreg_spline random_forest xgboost lightgbm catboost ebm mlp) do (
        echo --- %%T / %%M ---
        "%PY%" -m experiments.run_benchmark --config configs/default.yaml --tier %%T --model %%M --seed 0
    )
)
echo --- ablation T1_nostroke ---
"%PY%" -m experiments.run_benchmark --config configs/default.yaml --tier T1_nostroke --model ebm --seed 0
"%PY%" -m experiments.run_benchmark --config configs/default.yaml --tier T1_nostroke --model xgboost --seed 0
echo.
echo === KIEM TOAN ===
for %%M in (ebm xgboost) do (
    "%PY%" -m experiments.run_fairness --config configs/default.yaml --tier T1 --model %%M --seed 0 --group Sex
    "%PY%" -m experiments.run_conformal --config configs/default.yaml --tier T1 --model %%M --seed 0 --alpha 0.1
)
echo.
echo XONG. Bam phim de xem ket qua.
if "%ONESHOT%"=="0" pause
"%PY%" xem_ket_qua.py
call :ENDACT
goto MENU

:SHOW
echo.
"%PY%" xem_ket_qua.py
echo.
call :ENDACT
goto MENU

:SEEDS
echo.
for %%S in (1 2 3 4) do (
    for %%M in (ebm xgboost logreg) do (
        for %%T in (T0 T1) do (
            echo --- seed %%S / %%T / %%M ---
            "%PY%" -m experiments.run_benchmark --config configs/default.yaml --tier %%T --model %%M --seed %%S
        )
    )
)
echo.
echo Xong 4 seed. Bam phim de xem lai bang ket qua.
if "%ONESHOT%"=="0" pause
"%PY%" xem_ket_qua.py
call :ENDACT
goto MENU

:AUDIT
echo.
for %%M in (ebm xgboost) do (
    echo --- fairness %%M ---
    "%PY%" -m experiments.run_fairness --config configs/default.yaml --tier T1 --model %%M --seed 0 --group Sex
    echo --- conformal %%M ---
    "%PY%" -m experiments.run_conformal --config configs/default.yaml --tier T1 --model %%M --seed 0 --alpha 0.1
)
echo.
call :ENDACT
goto MENU

:OPEN
start "" "%CD%\results"
call :ENDACT
goto MENU

:ENDACT
if "%ONESHOT%"=="1" exit /b 0
echo.
pause
exit /b 0
