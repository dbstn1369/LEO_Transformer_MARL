"""Run Mega eval + figure generation only."""
import subprocess, sys, os, gc
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable

def run(cmd, log_file):
    gc.collect()
    print(f">>> {' '.join(cmd)}", flush=True)
    with open(log_file, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"    Exit: {proc.returncode}", flush=True)

# Mega eval only (Starlink already done)
print("=== Mega Evaluation ===", flush=True)
run([PYTHON, "-u", "evaluate.py",
     "--episodes", "10", "--device", "cuda",
     "--planes", "36", "--sats", "22",
     "--tag", "mega",
     "--n_users", "2000,3000,4000,5000,6000,7000"],
    "logs/eval_mega.log")

# Copy starlink data for figures
import shutil
for suffix in ["fig1_delay_vs_users", "fig2_throughput", "fig3_stability", "fig5_plr_vs_users"]:
    src = f"data/{suffix}_starlink.csv"
    dst = f"data/{suffix}.csv"
    if os.path.exists(src):
        shutil.copy(src, dst)

# Generate figures
print("=== Generating Figures ===", flush=True)
run([PYTHON, "-u", "plot_paper_figures.py"], "logs/plot_figures.log")

print("=== DONE ===", flush=True)
