"""
Complete pipeline: runs ALL remaining phases after main training.
1. GRLR training (800 ep)
2. Convergence hyperparameter study (5 runs × 800 ep)
3. Evaluate Starlink-like (18×18)
4. Evaluate Mega-constellation (36×22)
5. Generate all figures

Usage:
    python run_all_phases.py [--device cuda]
"""
import os
import sys
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"C:\Users\yoon\anaconda3\python.exe"


def run(desc, cmd_list, log_name):
    log_path = os.path.join(BASE, "logs", log_name)
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  Log: logs/{log_name}")
    print(f"{'='*60}")
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd_list, cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    print(f"  {status} in {elapsed:.1f} min")
    return proc.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_eps", type=int, default=800)
    parser.add_argument("--eval_eps", type=int, default=20)
    parser.add_argument("--conv_eps", type=int, default=800, help="Episodes per convergence run")
    args = parser.parse_args()
    dev = args.device

    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "checkpoints"), exist_ok=True)

    t_total = time.time()

    # ── Phase 1: GRLR Training ────────────────────────────────────────
    run("Phase 1: GRLR Training",
        [PYTHON, "-u", "train_grlr.py",
         "--episodes", str(args.train_eps), "--planes", "18", "--sats", "18",
         "--device", dev, "--lr", "1e-4", "--entropy", "0.01",
         "--epochs", "10", "--n_sessions", "20"],
        "train_grlr.log")

    # ── Phase 2: Convergence Hyperparameter Study ─────────────────────
    conv_configs = [
        ("lr1e-3",  "1e-3", "2048"),
        ("lr1e-4",  "1e-4", "2048"),   # shared
        ("lr1e-5",  "1e-5", "2048"),
        ("mb256",   "1e-4", "256"),
        ("mb8192",  "1e-4", "8192"),
    ]
    for tag, lr, mb in conv_configs:
        reward_file = os.path.join(BASE, "logs", f"train_rewards_{tag}.npy")
        if os.path.exists(reward_file):
            print(f"\n  [SKIP] {tag} already exists")
            continue
        run(f"Phase 2: Convergence tag={tag} LR={lr} MB={mb}",
            [PYTHON, "-u", "train.py",
             "--episodes", str(args.conv_eps), "--planes", "18", "--sats", "18",
             "--device", dev, "--lr", lr, "--entropy", "0.01",
             "--epochs", "10", "--n_sessions", "20",
             "--mini_batch", mb, "--tag", tag],
            f"train_{tag}.log")

    # ── Phase 3: Evaluate Starlink-like ───────────────────────────────
    run("Phase 3: Evaluate Starlink-like (18×18)",
        [PYTHON, "-u", "evaluate.py",
         "--episodes", str(args.eval_eps), "--planes", "18", "--sats", "18",
         "--device", dev, "--tag", "starlink",
         "--n_users", "500,1000,1500,2000,2500,3000"],
        "evaluate_starlink.log")

    # ── Phase 4: Evaluate Mega-constellation ──────────────────────────
    run("Phase 4: Evaluate Mega-constellation (36×22)",
        [PYTHON, "-u", "evaluate.py",
         "--episodes", str(args.eval_eps), "--planes", "36", "--sats", "22",
         "--device", dev, "--tag", "mega",
         "--n_users", "2000,3000,4000,5000,6000,7000"],
        "evaluate_mega.log")

    # ── Phase 5: Generate All Figures ─────────────────────────────────
    run("Phase 5: Generate figures",
        [PYTHON, "-u", "plot_paper_figures.py"],
        "plot_final.log")

    elapsed_total = (time.time() - t_total) / 3600
    print(f"\n{'='*60}")
    print(f"  ALL PHASES COMPLETE in {elapsed_total:.1f} hours")
    print(f"{'='*60}")
    print("Figures:")
    print("  fig1_convergence      — Proposed vs MADRL")
    print("  fig_convergence_hp    — LR/batch comparison")
    print("  fig3_perf_starlink    — 4-panel Starlink-like")
    print("  fig4_perf_mega        — 4-panel Mega-constellation")


if __name__ == "__main__":
    main()
