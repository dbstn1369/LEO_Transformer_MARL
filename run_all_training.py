"""
Run all three DRL model training sequentially on GPU.
Proposed (Transformer) → MADRL → GRLR, each 800 episodes.
"""
import subprocess
import sys
import time

PYTHON = r"C:/Users/yoon/anaconda3/python.exe"

tasks = [
    {
        "name": "Proposed (Transformer)",
        "cmd": [PYTHON, "train.py",
                "--episodes", "800", "--seed", "42", "--device", "cuda"],
    },
    {
        "name": "MADRL",
        "cmd": [PYTHON, "train_madrl.py",
                "--episodes", "800", "--seed", "43", "--device", "cuda"],
    },
    {
        "name": "GRLR",
        "cmd": [PYTHON, "train_grlr.py",
                "--episodes", "800", "--seed", "44", "--device", "cuda"],
    },
]

for task in tasks:
    print(f"\n{'='*60}")
    print(f"Starting: {task['name']}")
    print(f"Command: {' '.join(task['cmd'])}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(task["cmd"], cwd=".", timeout=72000)  # 20hr max
    elapsed = (time.time() - t0) / 60
    if result.returncode == 0:
        print(f"\n{task['name']} completed in {elapsed:.1f} min")
    else:
        print(f"\n{task['name']} FAILED (exit={result.returncode}) after {elapsed:.1f} min")
        sys.exit(1)

print(f"\n{'='*60}")
print("All training complete!")
print(f"{'='*60}")
