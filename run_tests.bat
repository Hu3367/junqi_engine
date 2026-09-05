@echo off
REM Junqi Engine Test Runner
REM This script runs all tests using the virtual environment's Python

cd /d "%~dp0"

echo ========================================
echo Junqi Engine Test Suite
echo ========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\python.exe" (
    echo Using virtual environment: venv\Scripts\python.exe
    echo Running tests...
    echo.
    
    "venv\Scripts\python.exe" -m pytest tests/ -v --tb=short
    
    echo.
    echo ========================================
    echo Tests completed!
    echo ========================================
) else (
    echo ERROR: Virtual environment not found!
    echo Please run: venv\Scripts\activate
    exit /b 1
)

pause
