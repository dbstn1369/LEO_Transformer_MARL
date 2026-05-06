#!/bin/bash
# Run Proposed + MAAC extended training, then generate all figures
set -e
cd "c:/Users/yoon/Documents/Python Scripts/LEO_Transformer_MARL"

PYTHON="C:/Users/yoon/anaconda3/python.exe"

echo "=== Waiting for Proposed training to complete... ==="
# (already running in background, so just wait for it)
wait

echo "=== Step 1: MAAC resume (200 episodes) ==="
$PYTHON -u train_maac.py --episodes 200 --planes 18 --sats 18 --resume

echo "=== Step 2: Generate figures ==="
$PYTHON -u plot_paper_figures.py

echo "=== All done. Check figures/ folder ==="
