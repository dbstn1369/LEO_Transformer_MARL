#!/bin/bash
# Train all 3 DRL models with BC + PPO pipeline (fair comparison)
cd "c:/Users/yoon/Documents/Python Scripts/LEO_Transformer_MARL"

echo "=== Proposed (Transformer) ==="
C:/Users/yoon/anaconda3/python.exe -u train_bc_ppo.py --bc_episodes 30 --ppo_episodes 300 --device cuda --n_sessions 10 --model transformer

echo "=== MADRL (MAAC) ==="
C:/Users/yoon/anaconda3/python.exe -u train_bc_ppo.py --bc_episodes 30 --ppo_episodes 300 --device cuda --n_sessions 10 --model madrl --tag "_madrl"

echo "=== GRLR ==="
C:/Users/yoon/anaconda3/python.exe -u train_bc_ppo.py --bc_episodes 30 --ppo_episodes 300 --device cuda --n_sessions 10 --model grlr --tag "_grlr"

echo "=== All trained ==="
