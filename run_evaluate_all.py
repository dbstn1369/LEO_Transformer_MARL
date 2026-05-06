"""
Run evaluation for two constellation scenarios (like reference Fig 10/11).

Scenario 1: Standard  — 18×18 = 324 sats,  |U| = 500–3000
Scenario 2: Large     — 36×22 = 792 sats,  |U| = 2000–7000

Then generate all figures.

Usage:
    python run_evaluate_all.py [--episodes 20] [--device cuda]
"""
import os
import sys
import subprocess
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PYTHON = r"C:\Users\yoon\anaconda3\python.exe"

SCENARIOS = [
    {
        "tag": "standard",
        "planes": 18,
        "sats": 18,
        "n_users": "500,1000,1500,2000,2500,3000",
        "desc": "Standard (18x18 = 324 sats)",
    },
    {
        "tag": "large",
        "planes": 36,
        "sats": 22,
        "n_users": "2000,3000,4000,5000,6000,7000",
        "desc": "Large-scale (36x22 = 792 sats)",
    },
]


def run_scenario(sc, episodes, device):
    cmd = [
        PYTHON, "-u", "evaluate.py",
        "--episodes", str(episodes),
        "--planes", str(sc["planes"]),
        "--sats", str(sc["sats"]),
        "--device", device,
        "--tag", sc["tag"],
        "--n_users", sc["n_users"],
    ]
    log_path = os.path.join(BASE, "logs", f"evaluate_{sc['tag']}.log")

    print(f"\n{'='*60}")
    print(f"  Scenario: {sc['desc']}")
    print(f"  Users: {sc['n_users']}")
    print(f"  Log: {log_path}")
    print(f"{'='*60}")

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60

    if proc.returncode == 0:
        print(f"  Done in {elapsed:.1f} min")
        return True
    else:
        print(f"  FAILED (rc={proc.returncode}), check {log_path}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)

    t_total = time.time()
    for sc in SCENARIOS:
        run_scenario(sc, args.episodes, args.device)

    # Generate all figures
    print("\n" + "="*60)
    print("  Generating figures...")
    print("="*60)
    proc = subprocess.run(
        [PYTHON, "-u", "plot_paper_figures.py"],
        cwd=BASE, capture_output=True, text=True
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)

    elapsed_total = (time.time() - t_total) / 60
    print(f"\nALL DONE in {elapsed_total:.1f} min")
    print("Figures -> ./figures/")


if __name__ == "__main__":
    main()
