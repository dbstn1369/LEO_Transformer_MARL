"""
Phase 2: Run after main training pipeline completes.
1. Train GRLR (800 ep)
2. Evaluate large-scale constellation (36x22 = 792 sats)
3. Re-evaluate standard with all 5 schemes (including fresh GRLR)
4. Generate all figures

Usage:
    python run_phase2.py [--device cuda]
"""
import os
import sys
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"C:\Users\yoon\anaconda3\python.exe"


def run(cmd_str, log_name):
    log_path = os.path.join(BASE, "logs", log_name)
    print(f"\n{'='*60}")
    print(f"  {log_name}")
    print(f"{'='*60}")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd_str, cwd=BASE, shell=True, stdout=f, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"  {status} in {elapsed:.1f} min")
    return proc.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=800, help="Training episodes")
    parser.add_argument("--eval_eps", type=int, default=20, help="Eval episodes per point")
    args = parser.parse_args()
    dev = args.device

    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)

    # 1. Train GRLR
    run(f"{PYTHON} -u train_grlr.py --episodes {args.episodes} --planes 18 --sats 18 "
        f"--device {dev} --lr 1e-4 --entropy 0.01 --epochs 10 --n_sessions 20",
        "train_grlr.log")

    # 2. Re-evaluate standard (18x18) with all 5 schemes including fresh GRLR
    run(f"{PYTHON} -u evaluate.py --episodes {args.eval_eps} --planes 18 --sats 18 "
        f"--device {dev} --tag standard --n_users 500,1000,1500,2000,2500,3000",
        "evaluate_standard.log")

    # 3. Evaluate large-scale (36x22 = 792 sats)
    run(f"{PYTHON} -u evaluate.py --episodes {args.eval_eps} --planes 36 --sats 22 "
        f"--device {dev} --tag large --n_users 2000,3000,4000,5000,6000,7000",
        "evaluate_large.log")

    # 4. Generate all figures
    run(f"{PYTHON} -u plot_paper_figures.py", "plot_phase2.log")

    print(f"\n{'='*60}")
    print("  PHASE 2 COMPLETE")
    print(f"{'='*60}")
    print("Figures:")
    print("  fig1_convergence.png    — Proposed vs MADRL convergence")
    print("  fig2_performance.png    — 4-panel (default)")
    print("  fig3_perf_standard.png  — 4-panel Standard (18x18, 324 sats)")
    print("  fig4_perf_large.png     — 4-panel Large-scale (36x22, 792 sats)")


if __name__ == "__main__":
    main()
