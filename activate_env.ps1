# ============================================================
# 激活项目虚拟环境（PowerShell）
#
#   PS E:\Local code\军棋\junqi_engine> .\activate_env.ps1
#
# 为什么需要它：虚拟环境自 2026-09-15 起位于工程**同级**目录
# （..\venv_junqi_engine），不是工程内。手敲相对路径很容易写成
# `.\venv_junqi_engine\...`（少一个点）而报 CommandNotFoundException。
# 本脚本按实际位置自动定位，并清理残留的旧激活状态。
# ============================================================

$ErrorActionPreference = 'Continue'

$repoRoot = $PSScriptRoot
if (-not $repoRoot) { $repoRoot = (Get-Location).Path }

$candidates = @(
    (Join-Path $repoRoot 'venv\Scripts\Activate.ps1'),
    (Join-Path (Split-Path -Parent $repoRoot) 'venv_junqi_engine\Scripts\Activate.ps1')
)

$activate = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $activate) {
    Write-Host "[ERROR] 未找到虚拟环境。已查找：" -ForegroundColor Red
    $candidates | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "请先创建：" -ForegroundColor Yellow
    Write-Host "  python -m venv `"$(Split-Path -Parent $repoRoot)\venv_junqi_engine`"" -ForegroundColor Yellow
    Write-Host "  `"$(Split-Path -Parent $repoRoot)\venv_junqi_engine\Scripts\python.exe`" -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# —— 清理残留的旧激活状态 ——
# 典型场景：venv 被移动/删除后，当前会话仍处于「已激活」状态（提示符显示 (venv)），
# 但 PATH 首项指向一个已不存在的目录，于是 `python` 静默落到别的解释器上，
# 表现为 `ModuleNotFoundError: No module named 'torch'`。这里先把它清干净。
$existing = $env:VIRTUAL_ENV
if ($existing) {
    if (-not (Test-Path (Join-Path $existing 'Scripts\python.exe'))) {
        Write-Host "[!] 检测到残留的虚拟环境激活（路径已不存在）：$existing" -ForegroundColor Yellow
        Write-Host "    正在清理该会话的 PATH 与 VIRTUAL_ENV ..." -ForegroundColor Yellow
        $deadScripts = Join-Path $existing 'Scripts'
        $parts = $env:PATH -split ';' | Where-Object {
            $_ -and ($_.TrimEnd('\') -ne $deadScripts.TrimEnd('\'))
        }
        $env:PATH = ($parts -join ';')
        Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
        Write-Host "    已清理。若提示符仍显示 (venv)，请重开一个 PowerShell 窗口。" -ForegroundColor Yellow
    }
}

Write-Host "激活虚拟环境: $activate" -ForegroundColor Cyan
& $activate

# —— 校验依赖完整性 ——
# $activate = <venv>\Scripts\Activate.ps1  →  解释器在 <venv>\Scripts\python.exe
$venvRoot = Split-Path -Parent (Split-Path -Parent $activate)
$py = Join-Path $venvRoot 'Scripts\python.exe'
if (Test-Path $py) {
    & $py -c "import torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] 该环境缺少 torch，依赖可能不完整。" -ForegroundColor Red
        Write-Host "       修复：& '$py' -m pip install -r requirements.txt" -ForegroundColor Red
    } else {
        Write-Host "依赖自检通过（torch 可用）。" -ForegroundColor Green
    }
} else {
    Write-Host "[WARN] 未找到 $py，跳过依赖自检。" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "现在可以直接使用：" -ForegroundColor Green
Write-Host "  python -m junqi gui" -ForegroundColor Gray
Write-Host "  python -m pytest tests/ -q" -ForegroundColor Gray
Write-Host "（也可以始终用 .\run.bat <子命令>，无需先激活）" -ForegroundColor Gray
