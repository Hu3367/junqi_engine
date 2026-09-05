#!/usr/bin/env pwsh
# Mini-Junqi Test Runner for PowerShell

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "MINI-JUNQI QUICK TEST (PowerShell)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "`n[Step 1] Running system verification..." -ForegroundColor Yellow
python scripts/verify_mini_junqi.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Verification failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[Step 2] Running quick experiment (50 games)..." -ForegroundColor Yellow
python scripts/mini_junqi_experiment.py --games 50 --red-agent heuristic --black-agent random --scheme scheme_a

Write-Host "`n[Step 3] Analyzing results..." -ForegroundColor Yellow
python scripts/analyze_results.py

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "TEST COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
