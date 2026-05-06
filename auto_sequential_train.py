"""Sequentially train Proposed, MADRL, GRLR with same BC+PPO pipeline."""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PYTHON = r"C:/Users/yoon/anaconda3/python.exe"

configs = [
    ("transformer", "", "best_transformer.pt"),
    ("madrl",       "_madrl", "best_maac.pt"),
    ("grlr",        "_grlr", "best_grlr.pt"),
]

for model, tag, ckpt in configs:
    print(f"\n=============================================")
    print(f"=== TRAINING: {model.upper()} ===")
    print(f"=============================================\n")
    cmd = [
        PYTHON, "-u", "train_bc_ppo.py",
        "--bc_episodes", "30",
        "--ppo_episodes", "300",
        "--device", "cuda",
        "--n_sessions", "10",
        "--model", model,
    ]
    if tag:
        cmd += ["--tag", tag]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[WARN] {model} training returned code {result.returncode}")

print("\n=== All 3 models trained ===")

# Auto-run finish pipeline (eval + figures)
print("\n=== Starting auto_finish pipeline ===\n")
subprocess.run([PYTHON, "-u", "auto_finish.py"])
