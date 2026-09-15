@echo off
REM ============================================================
REM 激活项目虚拟环境（cmd.exe）
REM
REM   E:\Local code\军棋\junqi_engine> activate_env.cmd
REM
REM 虚拟环境自 2026-09-15 起位于工程**同级**目录 ..\venv_junqi_engine。
REM 本脚本自动定位，避免手敲相对路径写错（少一个点就会报"找不到路径"）。
REM ============================================================

setlocal
cd /d "%~dp0"

set "ACT="
if exist "venv\Scripts\activate.bat"                   set "ACT=venv\Scripts\activate.bat"
if not defined ACT if exist "..\venv_junqi_engine\Scripts\activate.bat" set "ACT=..\venv_junqi_engine\Scripts\activate.bat"

if not defined ACT (
    echo [ERROR] 未找到虚拟环境。已查找：
    echo     venv\Scripts\activate.bat
    echo     ..\venv_junqi_engine\Scripts\activate.bat
    echo.
    echo 请先创建：
    echo   python -m venv "%~dp0..\venv_junqi_engine"
    echo   "%~dp0..\venv_junqi_engine\Scripts\python.exe" -m pip install -r requirements.txt
    endlocal
    exit /b 1
)

REM 清理残留的旧激活（PATH 首项指向已删除目录会导致 python 落到别的解释器上）
if defined VIRTUAL_ENV (
    if not exist "%VIRTUAL_ENV%\Scripts\python.exe" (
        echo [!] 检测到残留的虚拟环境激活（路径已不存在）: %VIRTUAL_ENV%
        echo     已为本窗口清理 VIRTUAL_ENV。若 PATH 仍有残留，请重开一个 cmd 窗口。
        set "VIRTUAL_ENV="
    )
)

echo 激活虚拟环境: %ACT%
call "%ACT%"

echo.
echo 现在可以直接使用：
echo   python -m junqi gui
echo   python -m pytest tests/ -q
echo （也可以始终用 run.bat ^<子命令^>，无需先激活）

endlocal & cmd /k
