"""
Convergence study: train Proposed with different hyperparameters.

Produces 5 reward curves for Fig.1 convergence plot:
  (a) LR comparison:  LR = {1e-3, 1e-4, 1e-5}   (mini_batch=2048 fixed)
  (b) Batch comparison: MB = {256, 2048, 8192}    (LR=1e-4 fixed)
  Note: LR=1e-4 + MB=2048 is shared → 5 runs total.

Usage:
    python run_convergence.py [--episodes 800] [--device cuda]
"""
import os
import sys
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"C:\Users\yoon\anaconda3\python.exe"

# 5 unique (tag, lr, mini_batch) configurations
CONFIGS = [
    # (a) LR comparison (mini_batch=2048 fixed)
    ("lr1e-3",  1e-3, 2048),
    ("lr1e-4",  1e-4, 2048),   # shared with batch comparison
    ("lr1e-5",  1e-5, 2048),
    # (b) Batch comparison (lr=1e-4 fixed) — lr1e-4 already covers MB=2048
    ("mb256",   1e-4, 256),
    ("mb8192",  1e-4, 8192),
]


def run_one(tag, lr, mini_batch, episodes, device):
    """Train one configuration and save reward curve."""
    reward_file = os.path.join(BASE, "logs", f"train_rewards_{tag}.npy")
    if os.path.exists(reward_file):
        print(f"  [SKIP] {reward_file} already exists")
        return True

    cmd = [
        PYTHON, "-u", "train.py",
        "--episodes", str(episodes),
        "--planes", "18", "--sats", "18",
        "--device", device,
        "--lr", str(lr),
        "--entropy", "0.02",
        "--epochs", "10",
        "--n_sessions", "20",
        "--mini_batch", str(mini_batch),
        "--tag", tag,
    ]
    log_path = os.path.join(BASE, "logs", f"train_{tag}.log")

    print(f"\n{'='*60}")
    print(f"  Training: tag={tag}  LR={lr}  MB={mini_batch}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60

    if proc.returncode == 0:
        print(f"  Done in {elapsed:.1f} min (rc=0)")
        return True
    else:
        print(f"  FAILED in {elapsed:.1f} min (rc={proc.returncode})")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "checkpoints"), exist_ok=True)

    print(f"Convergence study: {len(CONFIGS)} runs x {args.episodes} episodes")
    print(f"Device: {args.device}")

    t_total = time.time()
    results = {}
    for tag, lr, mb in CONFIGS:
        ok = run_one(tag, lr, mb, args.episodes, args.device)
        results[tag] = ok

    elapsed_total = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed_total:.1f} min")
    for tag, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {tag}: {status}")
    print(f"{'='*60}")
    print("\nNext: python plot_paper_figures.py")


if __name__ == "__main__":
    main()
