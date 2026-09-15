@echo off
REM ============================================================
REM Junqi Engine 统一启动器
REM   run.bat gui
REM   run.bat calc
REM   run.bat benchmark --model models/best.pt
REM   run.bat train_rl --epochs 5 --games 500 --sims 20 --fresh
REM   run.bat gate --a hybrid2 --b expert2 --seeds 100
REM   run.bat test                (等价于 run_tests.bat)
REM
REM 为什么需要它：包级 import 会加载 torch/numpy。虚拟环境自 2026-09-15 起
REM 位于工程同级目录，直接用 PATH 上的 python 会报
REM "ModuleNotFoundError: No module named 'torch'"。
REM 本脚本自动定位正确解释器，避免踩这个坑。
REM ============================================================

setlocal
cd /d "%~dp0"

set "PY="
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not defined PY if exist "..\venv_junqi_engine\Scripts\python.exe" set "PY=..\venv_junqi_engine\Scripts\python.exe"

if not defined PY (
    echo ERROR: 未找到项目虚拟环境。已查找：
    echo     venv\Scripts\python.exe
    echo     ..\venv_junqi_engine\Scripts\python.exe
    echo.
    echo 请先创建：python -m venv ..\venv_junqi_engine ^&^& ..\venv_junqi_engine\Scripts\pip install -r requirements.txt
    exit /b 1
)

if /i "%~1"=="test" (
    "%PY%" -m pytest tests/ -v --tb=short
    endlocal
    exit /b %ERRORLEVEL%
)

if "%~1"=="" (
    echo 用法: run.bat ^<子命令^> [参数...]
    echo.
    "%PY%" -m junqi --help
    endlocal
    exit /b 0
)

"%PY%" -m junqi %*
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
