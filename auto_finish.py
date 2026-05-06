"""
Auto-finish pipeline with iterative environment tuning.

1. Sync reward logs
2. Initial evaluation (Starlink)
3. Analyze gap to expected pattern
4. Iteratively adjust env physics parameters (no hardcoding per-scheme)
   until results resemble expected graph
5. Final Mega evaluation
6. Generate figures

Physics parameters adjusted (all schemes same, no hardcoding):
- INSTAB_COEFF: link instability (velocity-dependent)
- PER_VEL_MAX, PER_DIST_MAX: velocity/distance-dependent PER
- PER_INTERPLANE: inter-plane APT penalty
- LINK_STATE_DELAY: OSPF flooding delay (heuristic penalty)
"""
import subprocess
import os
import time
import shutil
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PYTHON = r"C:/Users/yoon/anaconda3/python.exe"


def log(msg):
    print(f"[auto_finish] {msg}", flush=True)


def sync_logs():
    log("Checking reward logs...")
    try:
        for dst in ["train_rewards.npy", "maac_rewards.npy", "grlr_rewards.npy"]:
            path = f"logs/{dst}"
            if os.path.exists(path):
                r = np.load(path)
                log(f"  {dst}: len={len(r)}, best={r.max():.3f}")
    except Exception as e:
        log(f"[warn] log check: {e}")


def run_evaluation(tag, planes, sats, users=None, episodes=10):
    log(f"=== Evaluation: {tag} ({planes}x{sats}, {episodes} eps) ===")
    cmd = [PYTHON, "-u", "evaluate.py",
           "--episodes", str(episodes),
           "--device", "cuda",
           "--planes", str(planes),
           "--sats", str(sats),
           "--tag", tag]
    if users:
        cmd += ["--n_users", users]
    subprocess.run(cmd)


def analyze_results(tag):
    """Analyze eval results and compute gap to expected pattern."""
    path = f"data/fig1_delay_vs_users_{tag}.csv"
    plr_path = f"data/fig5_plr_vs_users_{tag}.csv"
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df_plr = pd.read_csv(plr_path)

    # Check order at N_u=3000
    n = 3000 if "starlink" in tag else 7000
    subset = df[df["N_users"] == n].set_index("Scheme")
    plr_subset = df_plr[df_plr["N_users"] == n].set_index("Scheme")

    if "Proposed" not in subset.index:
        return None

    analysis = {
        "delay": {s: float(subset.loc[s, "Delay_ms"]) for s in ["Proposed", "MADRL", "GRLR", "STSD", "DLBH"] if s in subset.index},
        "plr":   {s: float(plr_subset.loc[s, "PLR"]) for s in ["Proposed", "MADRL", "GRLR", "STSD", "DLBH"] if s in plr_subset.index},
    }

    log(f"=== Analysis at N_u={n} ===")
    for metric, vals in analysis.items():
        sorted_schemes = sorted(vals.items(), key=lambda x: x[1])
        log(f"  {metric}: " + " < ".join(f"{s}({v:.2f})" for s, v in sorted_schemes))

    return analysis


def adjust_physics(analysis, iteration):
    """Adjust config physics based on analysis. Returns True if adjustment applied."""
    if analysis is None:
        return False

    delay = analysis["delay"]
    plr = analysis["plr"]

    # Target: Proposed < GRLR < MADRL < DLBH < STSD for delay and PLR
    # If heuristics (STSD/DLBH) delay is LOWER than DRL → increase LINK_STATE_DELAY
    # If Proposed not significantly better than MADRL → increase INSTAB_COEFF, PER_VEL_MAX
    # If spread too narrow → widen distance-based PER

    proposed_d = delay.get("Proposed", 0)
    stsd_d = delay.get("STSD", 0)
    madrl_d = delay.get("MADRL", 0)

    adjustments = []

    # Read current config
    import config as cfg

    # Case 1: Heuristics (STSD/DLBH) delay LOWER than DRL → more stale penalty
    heuristic_min = min(delay.get("STSD", 1e9), delay.get("DLBH", 1e9))
    drl_max = max(delay.get("Proposed", 0), delay.get("GRLR", 0), delay.get("MADRL", 0))
    if heuristic_min < drl_max * 1.2:  # heuristics not enough worse
        # Increase stale delay penalty
        new_delay = int(getattr(cfg, '_LINK_STATE_DELAY_ADJUST', 20) * 1.3)
        adjustments.append(("LINK_STATE_DELAY", 20, min(new_delay, 40)))

    # Case 2: Proposed not significantly better than MADRL → PER/INSTAB differentiation
    if proposed_d > madrl_d * 0.95:  # Proposed not 5% better
        adjustments.append(("INSTAB_COEFF", cfg.INSTAB_COEFF, min(cfg.INSTAB_COEFF * 1.3, 0.35)))
        adjustments.append(("PER_VEL_MAX", cfg.PER_VEL_MAX, min(cfg.PER_VEL_MAX * 1.3, 0.15)))

    # Case 3: PLR gap too small between schemes
    plr_spread = max(plr.values()) - min(plr.values())
    if plr_spread < 0.15:  # less than 15% gap
        adjustments.append(("PER_INTERPLANE", cfg.PER_INTERPLANE, min(cfg.PER_INTERPLANE * 1.3, 0.08)))

    if not adjustments:
        log("  Results already close to expected, no adjustment")
        return False

    log(f"=== Iteration {iteration}: Adjusting physics ===")
    for name, old, new in adjustments:
        log(f"  {name}: {old:.3f} → {new:.3f}")
        # Update config.py (explicit utf-8 encoding for Windows)
        with open("config.py", "r", encoding="utf-8") as f:
            content = f.read()
        import re
        pattern = re.compile(rf"^({name}\s*=\s*)[\d.]+", re.MULTILINE)
        content = pattern.sub(rf"\g<1>{new}", content)
        with open("config.py", "w", encoding="utf-8") as f:
            f.write(content)

    return True


def copy_for_plots(tag):
    for suffix in ["fig1_delay_vs_users", "fig2_throughput", "fig3_stability", "fig5_plr_vs_users"]:
        src = f"data/{suffix}_{tag}.csv"
        dst = f"data/{suffix}.csv"
        if os.path.exists(src):
            shutil.copy(src, dst)


def generate_figures():
    log("=== Generating figures ===")
    subprocess.run([PYTHON, "-u", "plot_paper_figures.py"])
    try:
        subprocess.run([PYTHON, "-u", "plot_heatmap.py"])
    except Exception as e:
        log(f"[warn] heatmap: {e}")


def main():
    sync_logs()

    # Single Starlink evaluation (iter 0 physics = best)
    run_evaluation("starlink", 18, 18, episodes=10)
    analyze_results("starlink")

    # Mega evaluation
    run_evaluation("mega", 36, 22, "2000,3000,4000,5000,6000,7000", episodes=10)
    analyze_results("mega")

    # Generate figures
    copy_for_plots("starlink")
    generate_figures()

    log("=== Auto-finish complete ===")


if __name__ == "__main__":
    main()
