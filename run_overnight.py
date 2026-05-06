"""
Overnight training pipeline:
  1. Train Proposed (Transformer) - 500 ep, stable settings (lr=1e-4, entropy=0.01)
  2. Train MADRL (MLP baseline)    - 500 ep, standard settings
  3. Evaluate all 4 schemes       - 20 ep across 5 traffic rates
  4. Generate paper figures       - updates figures/*.png

Run:
    python run_overnight.py
Logs written to logs/overnight_*.log
"""

import subprocess
import sys
import time
import os

PYTHON = sys.executable
PYTHON_U = [sys.executable, "-u"]   # -u: unbuffered stdout → live log file updates
PLANES = "18"
SATS   = "18"

os.makedirs("logs", exist_ok=True)

def run(cmd, log_file, label):
    print(f"\n{'='*60}")
    print(f"[{label}] Starting: {' '.join(cmd)}")
    print(f"  Log: {log_file}")
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    with open(log_file, "w", buffering=1) as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
    print(f"[{label}] Finished in {elapsed/60:.1f} min  ->  {status}", flush=True)
    return proc.returncode == 0


# 1. Proposed: Transformer
# LR 1e-4: Stable learning rate
# Entropy 0.0: Disabled – avoids fighting direction-routing signal from hop_reduction feat
# N_SESSIONS 20: Denser reward signal (62% active agents vs 18.5% with 6 sessions)
# Direct-bypass in TransformerActor ensures fast per-neighbour feature learning
ok = run(
    PYTHON_U + ["train.py",
     "--episodes",   "500",
     "--planes",     PLANES,
     "--sats",       SATS,
     "--device",     "cuda",
     "--lr",         "1e-4",
     "--entropy",    "0.0",
     "--epochs",     "10",
     "--n_sessions", "20"],
    "logs/overnight_proposed.log",
    "Proposed (Transformer)",
)
if not ok:
    print("[ERROR] Proposed training failed. Check logs/overnight_proposed.log")
    sys.exit(1)

# 2. MADRL: MLP baseline, same settings
ok = run(
    PYTHON_U + ["train_madrl.py",
     "--episodes",   "500",
     "--planes",     PLANES,
     "--sats",       SATS,
     "--device",     "cuda",
     "--lr",         "1e-4",
     "--entropy",    "0.0",
     "--epochs",     "10",
     "--n_sessions", "20"],
    "logs/overnight_maac.log",
    "MADRL (MLP baseline)",
)
if not ok:
    print("[ERROR] MADRL training failed. Check logs/overnight_maac.log")
    sys.exit(1)

# 3. Evaluate: compare all 4 schemes across traffic rates
ok = run(
    PYTHON_U + ["evaluate.py",
     "--episodes", "20",
     "--planes",   PLANES,
     "--sats",     SATS,
     "--device",   "cuda"],
    "logs/overnight_evaluate.log",
    "Evaluation (all 4 schemes)",
)
if not ok:
    print("[WARNING] Evaluation failed. Check logs/overnight_evaluate.log")

# 4. Plot: regenerate all paper figures
ok = run(
    PYTHON_U + ["plot_paper_figures.py"],
    "logs/overnight_plot.log",
    "Plot paper figures",
)

print("\n" + "="*60)
print("Overnight pipeline complete!")
print("Check figures/ for updated PNG files.")
print("="*60)
