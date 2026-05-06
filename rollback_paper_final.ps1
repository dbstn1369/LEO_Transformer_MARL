# Restore paper-final state from backup_paper_snapshot_20260506/.
# Use when retrain produces worse results than the published paper figures.
#
# Usage:  .\rollback_paper_final.ps1
#
# Recovers: code (.py), checkpoints (.pt), reward histories (.npy),
#           figures (.png/.eps), CSV data, paper tex.

$ErrorActionPreference = "Stop"
$BACKUP = "backup_paper_snapshot_20260506"

if (-not (Test-Path $BACKUP)) {
    Write-Error "$BACKUP not found. Are you in the LEO_Transformer_MARL root?"
    exit 1
}

Write-Host "=== Rollback to paper-final state ($BACKUP) ===" -ForegroundColor Cyan

Write-Host "[1/5] Restoring code (.py)..."
Copy-Item "$BACKUP\code\*.py" -Destination . -Force

Write-Host "[2/5] Restoring checkpoints (.pt)..."
New-Item -ItemType Directory -Force -Path checkpoints | Out-Null
Copy-Item "$BACKUP\checkpoints\*.pt" -Destination checkpoints\ -Force

Write-Host "[3/5] Restoring reward histories (.npy)..."
New-Item -ItemType Directory -Force -Path logs | Out-Null
Copy-Item "$BACKUP\logs\*.npy" -Destination logs\ -Force

Write-Host "[4/5] Restoring figures + data..."
New-Item -ItemType Directory -Force -Path figures | Out-Null
New-Item -ItemType Directory -Force -Path data | Out-Null
Copy-Item "$BACKUP\figures\*" -Destination figures\ -Force
if (Test-Path "$BACKUP\data\*.csv") {
    Copy-Item "$BACKUP\data\*.csv" -Destination data\ -Force
}

Write-Host "[5/5] Restoring paper tex..."
if (Test-Path "$BACKUP\tex\*.tex") {
    Copy-Item "$BACKUP\tex\*.tex" -Destination . -Force
}

Write-Host ""
Write-Host "=== Rollback complete ===" -ForegroundColor Green
Write-Host "Verify: git status (expect modified .py, .pt, figures restored to paper-final)"
Write-Host "If satisfied:  git add -A; git commit -m 'Rollback to paper-final state'"
