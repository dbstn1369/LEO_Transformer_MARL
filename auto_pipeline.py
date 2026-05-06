"""
End-to-end automation pipeline for LEO Transformer MARL paper.

Pipeline:
  Phase 1. Re-train all 3 models in parallel (Proposed, MADRL, GRLR)
           using the new reward (distance-based direction + delivery bonus)
           and Manhattan-aware training fallback.
  Phase 2. Run starlink + mega evaluation (5 schemes including STSD/DLBH)
  Phase 3. Generate figures (convergence, performance, heatmap)
  Phase 4. Auto-fill LaTeX X% Y% placeholders with computed values
  Phase 5. Write RESULTS_SUMMARY.md

Run as:
  python auto_pipeline.py                # full pipeline (~3-4 hours)
  python auto_pipeline.py --skip-train   # eval+figures+latex only
  python auto_pipeline.py --episodes 5   # quick eval mode
"""
import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

# Force UTF-8 stdout (Windows cp949 chokes on em dash, etc.)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
PYTHON = "C:/Users/yoon/anaconda3/python.exe"
LOG_DIR = ROOT / "auto_logs"
LOG_DIR.mkdir(exist_ok=True)


def run_blocking(cmd, desc, log_name=None):
    print(f"\n{'=' * 70}\n[STEP] {desc}\n{'=' * 70}")
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    if log_name:
        log_path = LOG_DIR / log_name
        with open(log_path, "w", encoding="utf-8") as f:
            result = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        print(f"  log: {log_path}")
    else:
        result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] {desc} (exit={result.returncode}, {elapsed:.0f}s)")
        sys.exit(result.returncode)
    print(f"[OK]   {desc} ({elapsed:.0f}s)")


def run_parallel(cmds_with_descs):
    """Launch multiple subprocesses in parallel and wait for all."""
    print(f"\n{'=' * 70}")
    print(f"[PARALLEL] Launching {len(cmds_with_descs)} jobs")
    print(f"{'=' * 70}")
    procs = []
    for cmd, desc, log_name in cmds_with_descs:
        log_path = LOG_DIR / log_name
        log_file = open(log_path, "w", encoding="utf-8")
        print(f"  [LAUNCH] {desc}")
        print(f"           cmd: {' '.join(cmd)}")
        print(f"           log: {log_path}")
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((p, desc, log_file, time.time()))

    # Wait for all
    for p, desc, log_file, t0 in procs:
        p.wait()
        log_file.close()
        elapsed = time.time() - t0
        if p.returncode == 0:
            print(f"  [DONE]  {desc} ({elapsed:.0f}s)")
        else:
            print(f"  [FAIL]  {desc} (exit={p.returncode}, {elapsed:.0f}s)")
            sys.exit(p.returncode)


# ─────────────────────────────────────────────────────────────────────────
# Phase 1: Training
# ─────────────────────────────────────────────────────────────────────────

def step_train_all(episodes=600):
    """Train Proposed, MADRL, GRLR sequentially on GPU."""
    run_blocking([PYTHON, "train.py",       "--episodes", str(episodes), "--seed", "42", "--device", "cuda"],
                 f"Train Proposed ({episodes} ep, GPU)", "train_proposed.log")
    run_blocking([PYTHON, "train_madrl.py",  "--episodes", str(episodes), "--seed", "43", "--device", "cuda"],
                 f"Train MADRL ({episodes} ep, GPU)",     "train_madrl.log")
    run_blocking([PYTHON, "train_grlr.py",  "--episodes", str(episodes), "--seed", "44", "--device", "cuda"],
                 f"Train GRLR ({episodes} ep, GPU)",     "train_grlr.log")


# ─────────────────────────────────────────────────────────────────────────
# Phase 2: Evaluation
# ─────────────────────────────────────────────────────────────────────────

def step_eval_all(episodes=20):
    """Run starlink and mega evaluations in parallel."""
    cmds = [
        ([PYTHON, "evaluate.py",
          "--tag", "starlink",
          "--planes", "18", "--sats", "18",
          "--episodes", str(episodes)],
         f"Evaluate Starlink ({episodes} ep)", "eval_starlink.log"),
        ([PYTHON, "evaluate.py",
          "--tag", "mega",
          "--planes", "36", "--sats", "22",
          "--episodes", str(episodes),
          "--n_users", "2000,3000,4000,5000,6000,7000"],
         f"Evaluate Mega ({episodes} ep)",     "eval_mega.log"),
    ]
    run_parallel(cmds)


# ─────────────────────────────────────────────────────────────────────────
# Phase 3: Figures
# ─────────────────────────────────────────────────────────────────────────

def step_plot():
    run_blocking([PYTHON, "plot_paper_figures.py"],
                 "Plot performance figures",
                 "plot_perf.log")
    run_blocking([PYTHON, "plot_heatmap.py", "--tag", "starlink",
                  "--planes", "18", "--sats", "18"],
                 "Plot heatmap (starlink)",
                 "plot_heatmap.log")


# ─────────────────────────────────────────────────────────────────────────
# Phase 4: Statistics + LaTeX fill
# ─────────────────────────────────────────────────────────────────────────

def compute_improvements(tag="starlink", n_u_target=None):
    delay_csv = ROOT / "data" / f"fig1_delay_vs_users_{tag}.csv"
    plr_csv   = ROOT / "data" / f"fig5_plr_vs_users_{tag}.csv"
    if not delay_csv.exists() or not plr_csv.exists():
        return None
    df_d = pd.read_csv(delay_csv)
    df_p = pd.read_csv(plr_csv)
    if n_u_target is None:
        n_u_target = df_d["N_users"].max()

    def get(df, scheme, col):
        sub = df[(df["Scheme"] == scheme) & (df["N_users"] == n_u_target)]
        return float(sub[col].iloc[0]) if len(sub) else None

    schemes = ["Proposed", "GRLR", "MADRL", "DLBH", "STSD"]
    delays = {s.lower(): get(df_d, s, "Delay_ms") for s in schemes}
    plrs   = {s.lower(): get(df_p, s, "PLR")      for s in schemes}

    def pct(base, ours):
        if base is None or ours is None or base == 0:
            return None
        return (base - ours) / base * 100

    return {
        "n_u":   n_u_target,
        "delay": delays,
        "plr":   plrs,
        "delay_red": {f"vs_{t}": pct(delays[t], delays["proposed"])
                      for t in ["grlr", "madrl", "dlbh", "stsd"]},
        "plr_red":   {f"vs_{t}": pct(plrs[t],   plrs["proposed"])
                      for t in ["grlr", "madrl", "dlbh", "stsd"]},
    }


def fill_latex(stats):
    tex_path = ROOT / "paper_section5_updated.tex"
    if not tex_path.exists():
        print("[WARN] paper_section5_updated.tex not found")
        return
    text = tex_path.read_text(encoding="utf-8")

    if stats is None or stats["delay_red"]["vs_grlr"] is None:
        print("[WARN] No stats to fill")
        return

    x_pct  = f"{stats['delay_red']['vs_grlr']:.1f}\\%"
    y_pct  = f"{stats['delay_red']['vs_madrl']:.1f}\\%"
    x2_pct = f"{stats['plr_red']['vs_grlr']:.1f}\\%"
    y2_pct = f"{stats['plr_red']['vs_madrl']:.1f}\\%"

    replacements = [
        ("\\textbf{X\\%}",  f"\\textbf{{{x_pct}}}"),
        ("\\textbf{Y\\%}",  f"\\textbf{{{y_pct}}}"),
        ("\\textbf{X2\\%}", f"\\textbf{{{x2_pct}}}"),
        ("\\textbf{Y2\\%}", f"\\textbf{{{y2_pct}}}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)

    tex_path.write_text(text, encoding="utf-8")
    print(f"\n[OK] LaTeX placeholders filled:")
    print(f"     delay vs GRLR = {x_pct}")
    print(f"     delay vs MADRL = {y_pct}")
    print(f"     PLR   vs GRLR = {x2_pct}")
    print(f"     PLR   vs MADRL = {y2_pct}")


def write_summary(stats_starlink, stats_mega):
    out = ROOT / "RESULTS_SUMMARY.md"
    lines = ["# Final Evaluation Results\n"]
    for name, stats in [("Starlink (324 sats)", stats_starlink),
                        ("Mega (792 sats)",     stats_mega)]:
        if stats is None:
            lines.append(f"\n## {name}: NO DATA\n")
            continue
        lines.append(f"\n## {name} - at |U|={stats['n_u']}\n")
        lines.append("| Scheme | Delay (ms) | PLR (%) |")
        lines.append("|--------|-----------:|--------:|")
        for s in ["Proposed", "GRLR", "MADRL", "DLBH", "STSD"]:
            d = stats["delay"][s.lower()]
            p = stats["plr"][s.lower()]
            d_str = f"{d:.1f}" if d is not None else "-"
            p_str = f"{p*100:.2f}" if p is not None else "-"
            tag = "**" if s == "Proposed" else ""
            lines.append(f"| {tag}{s}{tag} | {tag}{d_str}{tag} | {tag}{p_str}{tag} |")

        lines.append("\n**Improvements (Proposed vs others):**")
        for tgt in ["grlr", "maac", "dlbh", "stsd"]:
            d_red = stats["delay_red"][f"vs_{tgt}"]
            p_red = stats["plr_red"][f"vs_{tgt}"]
            if d_red is not None and p_red is not None:
                lines.append(f"- vs {tgt.upper()}: delay {d_red:+.1f}%, PLR {p_red:+.1f}%")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] Summary written to {out}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip retraining (use existing checkpoints)")
    parser.add_argument("--skip-eval",  action="store_true",
                        help="Skip evaluation")
    parser.add_argument("--episodes",   type=int, default=20,
                        help="Eval episodes per N_u point")
    parser.add_argument("--train-episodes", type=int, default=600,
                        help="Training episodes per model")
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n{'#' * 70}")
    print(f"# LEO TRANSFORMER MARL - AUTO PIPELINE")
    print(f"#   skip_train={args.skip_train}  skip_eval={args.skip_eval}")
    print(f"#   train_eps={args.train_episodes}  eval_eps={args.episodes}")
    print(f"{'#' * 70}\n")

    if not args.skip_train:
        step_train_all(args.train_episodes)

    if not args.skip_eval:
        step_eval_all(args.episodes)

    step_plot()

    stats_s = compute_improvements("starlink", n_u_target=3000)
    stats_m = compute_improvements("mega",     n_u_target=7000)

    fill_latex(stats_s)
    write_summary(stats_s, stats_m)

    elapsed = time.time() - t0
    print(f"\n{'#' * 70}")
    print(f"# AUTO PIPELINE COMPLETE - total {elapsed/60:.1f} min")
    print(f"# Outputs:")
    print(f"#   - figures/             : all paper figures")
    print(f"#   - paper_section5_updated.tex : auto-filled with results")
    print(f"#   - RESULTS_SUMMARY.md   : results table")
    print(f"#   - auto_logs/           : per-step logs")
    print(f"{'#' * 70}\n")


if __name__ == "__main__":
    main()
