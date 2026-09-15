#!/usr/bin/env pwsh
# Mini-Junqi 快速测试（PowerShell）
#
# 2026-09-15：虚拟环境已移出工程目录，裸 `python` 会命中未装 torch 的解释器。
# 这里先定位项目虚拟环境，再调用它执行各步骤。

$ErrorActionPreference = 'Stop'

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "MINI-JUNQI QUICK TEST (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 定位解释器：工程内 venv\ 优先，其次同级 ..\venv_junqi_engine\
$repoRoot = Split-Path -Parent $PSScriptRoot
$candidates = @(
    (Join-Path $repoRoot 'venv\Scripts\python.exe'),
    (Join-Path (Split-Path -Parent $repoRoot) 'venv_junqi_engine\Scripts\python.exe')
)
$Py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $Py) {
    Write-Host "[ERROR] 未找到项目虚拟环境。已查找：" -ForegroundColor Red
    $candidates | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Write-Host "请参考 README「环境准备（必须）」，或改用 run.bat test。" -ForegroundColor Red
    exit 1
}

Write-Host "`n使用解释器: $Py" -ForegroundColor DarkGray
& $Py -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 该解释器缺少 torch，依赖不完整。" -ForegroundColor Red
    exit 1
}

Write-Host "`n[Step 1] Running system verification..." -ForegroundColor Yellow
& $Py scripts/verify_mini_junqi.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Verification failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[Step 2] Running quick experiment (50 games)..." -ForegroundColor Yellow
& $Py scripts/mini_junqi_experiment.py --games 50 --red-agent heuristic --black-agent random --scheme scheme_a

Write-Host "`n[Step 3] Analyzing results..." -ForegroundColor Yellow
& $Py scripts/analyze_results.py

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "TEST COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
