#!/bin/bash
# Full pipeline: train 3 schemes → eval → figures
# Run with: bash run_full_pipeline.sh

cd "c:/Users/yoon/Documents/Python Scripts/LEO_Transformer_MARL"
PYTHON="C:/Users/yoon/anaconda3/python.exe"

echo "=== [1/4] Training Proposed (800ep, resume) ==="
$PYTHON -u train.py --episodes 800 --device cuda --resume 2>&1 | tee logs/train_proposed.log

echo "=== [2/4] Training MADRL (800ep) ==="
$PYTHON -u train_madrl.py --episodes 800 --device cuda 2>&1 | tee logs/train_madrl_run.log

echo "=== [3/4] Training GRLR (800ep) ==="
$PYTHON -u train_grlr.py --episodes 800 --device cuda 2>&1 | tee logs/train_grlr_run.log

echo "=== [4/4] Evaluation + Figures ==="
$PYTHON -u auto_finish.py 2>&1 | tee logs/auto_finish_run.log

echo "=== ALL COMPLETE ==="
