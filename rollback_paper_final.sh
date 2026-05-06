#!/usr/bin/env bash
# Restore paper-final state from backup_paper_snapshot_20260506/.
# Use when retrain produces worse results than the published paper figures.
#
# Usage:  bash rollback_paper_final.sh
#
# Recovers: code (.py), checkpoints (.pt), reward histories (.npy),
#           figures (.png/.eps), CSV data, paper tex.
set -euo pipefail

BACKUP="backup_paper_snapshot_20260506"

if [ ! -d "$BACKUP" ]; then
    echo "ERROR: $BACKUP not found. Are you in the LEO_Transformer_MARL root?"
    exit 1
fi

echo "=== Rollback to paper-final state ($BACKUP) ==="

echo "[1/5] Restoring code (.py)..."
cp "$BACKUP/code/"*.py .

echo "[2/5] Restoring checkpoints (.pt)..."
mkdir -p checkpoints
cp "$BACKUP/checkpoints/"*.pt checkpoints/

echo "[3/5] Restoring reward histories (.npy)..."
mkdir -p logs
cp "$BACKUP/logs/"*.npy logs/

echo "[4/5] Restoring figures + data..."
mkdir -p figures data
cp "$BACKUP/figures/"* figures/
cp "$BACKUP/data/"*.csv data/ 2>/dev/null || true

echo "[5/5] Restoring paper tex..."
cp "$BACKUP/tex/"*.tex . 2>/dev/null || true

echo ""
echo "=== Rollback complete ==="
echo "Verify: git status (expect modified .py, .pt, figures restored to paper-final)"
echo "If satisfied:  git add -A && git commit -m 'Rollback to paper-final state'"
