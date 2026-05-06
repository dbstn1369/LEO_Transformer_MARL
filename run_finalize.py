"""
Finalize pipeline — waits for BOTH current training runs to complete,
then runs evaluate.py + plot_paper_figures.py.

Uses modification-time gating so old .npy files don't trigger early exit.
"""
import os, time, sys, subprocess
import numpy as np

PYTHON    = sys.executable
BASE      = os.path.dirname(os.path.abspath(__file__))
LOG_TF    = os.path.join(BASE, "logs", "train_rewards.npy")
LOG_MAAC  = os.path.join(BASE, "logs", "maac_rewards.npy")
TARGET    = 1000
START_TS  = time.time()   # any npy older than this is from a previous run

print(f"=== Finalize pipeline started (ts={START_TS:.0f}) ===")


def wait_for_new(npy_path, label, target=TARGET):
    """Wait until npy_path was written AFTER this script started AND has >= target entries."""
    print(f"Waiting for {label} ({target} eps, must be newer than now)...")
    while True:
        try:
            mtime = os.path.getmtime(npy_path)
            if mtime > START_TS:
                arr = np.load(npy_path)
                if len(arr) >= target:
                    print(f"  {label} done: {len(arr)} eps, best={arr.max():.2f}")
                    return arr
                else:
                    print(f"  ... {label}: {len(arr)}/{target} eps", flush=True)
            else:
                print(f"  ... {label}: waiting for new file (mtime too old)", flush=True)
        except Exception:
            pass
        time.sleep(60)


wait_for_new(LOG_TF,   "Proposed (Transformer)")
wait_for_new(LOG_MAAC, "MADRL")

print("\n=== Both training runs complete. Running evaluate.py ===")
ret = subprocess.run(
    [PYTHON, "-u", os.path.join(BASE, "evaluate.py"),
     "--episodes", "20", "--planes", "18", "--sats", "18", "--device", "cuda"],
    cwd=BASE,
)
if ret.returncode != 0:
    print("[WARN] evaluate.py failed; trying cpu fallback...")
    ret = subprocess.run(
        [PYTHON, "-u", os.path.join(BASE, "evaluate.py"),
         "--episodes", "20", "--planes", "18", "--sats", "18", "--device", "cpu"],
        cwd=BASE,
    )

print("\n=== Generating paper figures ===")
subprocess.run(
    [PYTHON, "-u", os.path.join(BASE, "plot_paper_figures.py"), "--no-gen"],
    cwd=BASE,
)

print("\n=== All done. Check figures/ and data/ folders ===")
