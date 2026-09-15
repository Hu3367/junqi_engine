@echo off
REM ============================================================
REM Junqi Engine Test Runner
REM 自动定位虚拟环境（2026-09-15 起虚拟环境已移出工程目录）
REM   优先级: 1) 工程内 venv\                    (旧布局 / 自建环境)
REM           2) 同级 ..\venv_junqi_engine\      (当前布局)
REM   都找不到时回退到 PATH 上的 python，并给出明确提示。
REM ============================================================

setlocal
cd /d "%~dp0"

set "PY="
if exist "venv\Scripts\python.exe" set "PY=venv\Scripts\python.exe"
if not defined PY if exist "..\venv_junqi_engine\Scripts\python.exe" set "PY=..\venv_junqi_engine\Scripts\python.exe"

echo ========================================
echo Junqi Engine Test Suite
echo ========================================
echo.

if defined PY (
    echo Using interpreter: %PY%
    echo Running tests...
    echo.
    "%PY%" -m pytest tests/ -v --tb=short
) else (
    echo WARNING: virtual environment not found.
    echo   looked for: venv\Scripts\python.exe
    echo   looked for: ..\venv_junqi_engine\Scripts\python.exe
    echo.
    echo Falling back to python on PATH ^(dependencies may be incomplete^)...
    echo.
    python -m pytest tests/ -v --tb=short
)

echo.
echo ========================================
echo Tests completed!
echo ========================================
endlocal
pause
