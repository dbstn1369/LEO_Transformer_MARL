"""
Hyperparameter sweep runner.

Trains the proposed Transformer-based MADRL framework under multiple
hyperparameter configurations to populate Fig.~\\ref{fig:convergence_hp}.

All runs share the SAME environment, model, and reward function as the main
800-episode experiment. Only the swept hyperparameter (LR or mini-batch
size) varies. Each run uses the same number of episodes (default 100)
so that the curves are directly comparable.

Outputs:
  logs/train_rewards_hp_lr1e-3.npy        (LR sweep, batch=8192)
  logs/train_rewards_hp_lr1e-4.npy        (shared baseline)
  logs/train_rewards_hp_lr1e-5.npy
  logs/train_rewards_hp_mb256.npy         (batch sweep, lr=1e-4)
  logs/train_rewards_hp_mb2048.npy
  logs/train_rewards_hp_mb8192.npy        (= train_rewards_hp_lr1e-4.npy)

Usage:
    python run_hp_sweep.py [--episodes 100] [--device cuda] [--seed 42]

The script runs each configuration sequentially and writes a progress log
to logs/hp_sweep.log. To run as a detached background process, wrap with:
    python -c "import subprocess, sys; subprocess.Popen([sys.executable, '-u', 'run_hp_sweep.py'], creationflags=...)"
"""

import argparse
import os
import subprocess
import sys
import time

# (lr, batch_size, output_tag)
SWEEP_CONFIGS = [
    # LR sweep at batch=8192 (paper-default batch)
    (1e-3, 8192, "hp_lr1e-3"),
    (1e-4, 8192, "hp_lr1e-4"),    # shared with batch sweep mb8192
    (1e-5, 8192, "hp_lr1e-5"),
    # Batch-size sweep at lr=1e-4 (paper-default lr)
    (1e-4,  256, "hp_mb256"),
    (1e-4, 2048, "hp_mb2048"),
    # batch=8192 + lr=1e-4 already covered by hp_lr1e-4
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100,
                        help="Episodes per HP run (default 100)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip configs whose output file already exists")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    print(f"HP sweep: {len(SWEEP_CONFIGS)} configs, {args.episodes} ep each")
    total_start = time.time()

    for idx, (lr, batch, tag) in enumerate(SWEEP_CONFIGS):
        out_path = f"logs/train_rewards_{tag}.npy"
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{idx+1}/{len(SWEEP_CONFIGS)}] SKIP (exists): {tag}")
            continue

        print(f"[{idx+1}/{len(SWEEP_CONFIGS)}] Running {tag}: lr={lr}, batch={batch}")
        run_start = time.time()
        cmd = [
            sys.executable, "-u", "train.py",
            "--episodes", str(args.episodes),
            "--device", args.device,
            "--seed", str(args.seed),
            "--lr", str(lr),
            "--mini_batch", str(batch),
            "--tag", tag,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - run_start
        print(f"  Done in {elapsed/60:.1f} min, exit={result.returncode}")
        if result.returncode != 0:
            print(f"  ERROR — last 20 lines of stderr:")
            print("\n".join(result.stderr.splitlines()[-20:]))

    total_elapsed = time.time() - total_start
    print(f"\nHP sweep complete: total {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
