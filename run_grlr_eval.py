"""Run GRLR training + eval pipeline (detached-safe)."""
import subprocess, sys, os, gc
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
# Force garbage collection before starting
gc.collect()

def run(cmd, log_file):
    print(f">>> {' '.join(cmd)}", flush=True)
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"    Exit: {proc.returncode}", flush=True)

print("=== GRLR Training (800ep) ===", flush=True)
run([PYTHON, "-u", "train_grlr.py", "--episodes", "800", "--device", "cuda"],
    "logs/train_grlr_run.log")

print("=== Eval + Figures ===", flush=True)
run([PYTHON, "-u", "auto_finish.py"], "logs/auto_finish_run.log")

print("=== DONE ===", flush=True)
