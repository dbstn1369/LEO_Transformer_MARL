"""
Run full training + eval pipeline.
Execute directly: python run_all.py
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

def run(cmd, log_file):
    print(f">>> {' '.join(cmd)}")
    print(f"    Log: {log_file}")
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"    Exit code: {proc.returncode}")
    return proc.returncode

# 1. Proposed (FRESH retrain — no --resume; seed from cfg.SEED=7)
print("=== [1/4] Training Proposed (800ep, fresh) ===")
run([PYTHON, "-u", "train.py", "--episodes", "800", "--device", "cuda"],
    "logs/train_proposed.log")

# 2. MADRL
print("=== [2/4] Training MADRL (800ep) ===")
run([PYTHON, "-u", "train_madrl.py", "--episodes", "800", "--device", "cuda"],
    "logs/train_madrl_run.log")

# 3. GRLR
print("=== [3/4] Training GRLR (800ep) ===")
run([PYTHON, "-u", "train_grlr.py", "--episodes", "800", "--device", "cuda"],
    "logs/train_grlr_run.log")

# 4. Eval + Figures
print("=== [4/4] Evaluation + Figures ===")
run([PYTHON, "-u", "auto_finish.py"],
    "logs/auto_finish_run.log")

print("=== ALL COMPLETE ===")
