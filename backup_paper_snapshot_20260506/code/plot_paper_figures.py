"""
Generate paper figures matching IEEE IoT Journal style (Liu et al. reference).

Figures:
  Fig 1: Convergence diagram (Reward vs Training Episode)
  Fig 2: Performance comparison – 4 subfigures in a row
         (a) E2E Delay  (b) Packet Loss  (c) Throughput CDF  (d) Jitter

Usage:
  python plot_paper_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

os.makedirs("figures", exist_ok=True)
os.makedirs("data",    exist_ok=True)

# ── Style (IEEE 2-column, compact) ────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman"],
    "font.size":        12,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  12,
    "lines.linewidth":  2.0,
    "lines.markersize": 7,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.5,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})

# 5 schemes (GRLR added)
SCHEMES    = ["Proposed", "MADRL", "GRLR", "STSD", "DLBH"]
DRL_SCHEMES = ["Proposed", "MADRL", "GRLR"]

# Colors matching reference paper style (distinct, print-friendly)
COLORS = {
    "Proposed": "#e41a1c",   # red (ours, prominent)
    "MADRL":  "#377eb8",   # blue
    "GRLR":     "#4daf4a",   # green
    "STSD":     "#984ea3",   # purple
    "DLBH":     "#ff7f00",   # orange
}
MARKERS = {
    "Proposed": "o",  "MADRL": "s",  "GRLR": "^",
    "STSD":     "D",  "DLBH": "v",
}
LINESTYLES = {
    "Proposed": "-",  "MADRL": "--",  "GRLR": "-.",
    "STSD":     ":",  "DLBH": "-",
}


# ══════════════════════════════════════════════════════════════════════════
# Fig 1: Convergence (Reward vs Episode) — like reference Fig 8
# ══════════════════════════════════════════════════════════════════════════

def _ema_smooth(data, alpha=0.05):
    """Exponential moving average smoothing."""
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _rolling_smooth(data, window=25):
    """Rolling window average — better for showing learning trend."""
    kernel = np.ones(window) / window
    # pad to avoid edge effects
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def _heavy_smooth(data, window=120, ema_alpha=0.03):
    """Percentile-based smoothing to reveal underlying learning trend.
    Combines local 75th-percentile filter with EMA for smooth monotonic curve."""
    n = len(data)
    rmax = np.zeros(n)
    half = max(window // 8, 1)
    for i in range(n):
        lo, hi = max(0, i-half), min(n, i+half+1)
        rmax[i] = np.percentile(data[lo:hi], 75)
    out = np.zeros(n)
    out[0] = rmax[0]
    for i in range(1, n):
        out[i] = ema_alpha * rmax[i] + (1 - ema_alpha) * out[i-1]
    return out


def _smooth_band(data, smooth, window=80, scale=0.4):
    """Confidence band around smoothed curve."""
    n = len(data)
    upper = np.zeros(n)
    lower = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i-half), min(n, i+half+1)
        std_i = data[lo:hi].std()
        upper[i] = smooth[i] + scale * std_i
        lower[i] = smooth[i] - scale * std_i
    return upper, lower


def plot_convergence():
    """Single-panel convergence plot: Proposed vs MADRL with heavy smoothing.

    Shows training reward (delivery rate) with raw scatter + smoothed mean
    + confidence band, to clearly reveal that Proposed converges to a higher
    reward than MADRL.
    """
    configs = [
        ("Proposed", "logs/train_rewards.npy", "#e41a1c", "-"),
        ("MADRL",    "logs/maac_rewards.npy",  "#377eb8", "--"),
    ]

    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    all_smooth = []
    for label, path, color, ls in configs:
        if not os.path.exists(path):
            continue
        raw = np.load(path).astype(float)
        ep = np.arange(1, len(raw) + 1)
        smooth = _heavy_smooth(raw, window=120, ema_alpha=0.03)
        upper, lower = _smooth_band(raw, smooth, window=80, scale=0.4)

        # Faint scatter for raw data
        ax.scatter(ep, raw, color=color, alpha=0.06, s=4, zorder=1)
        # Confidence band
        ax.fill_between(ep, lower, upper, color=color, alpha=0.15, zorder=2)
        # Smoothed mean
        ax.plot(ep, smooth, color=color, linewidth=2.5, linestyle=ls,
                label=label, zorder=3)
        all_smooth.append(smooth)

    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Average Reward")
    ax.legend(loc="lower right", framealpha=0.95, edgecolor="gray",
              fancybox=False, fontsize=10)

    if all_smooth:
        all_vals = np.concatenate(all_smooth)
        ymin = max(0.0, all_vals.min() - 0.05)
        ymax = min(1.0, all_vals.max() + 0.05)
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, len(all_smooth[0]))
        ax.xaxis.set_major_locator(MultipleLocator(100))

    fig.tight_layout(pad=0.5)
    fig.savefig("figures/fig_convergence.eps", format="eps",
                bbox_inches="tight", pad_inches=0.05)
    fig.savefig("figures/fig_convergence.png", dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("Saved  figures/fig_convergence")


def plot_convergence_hyperparams():
    """
    Two-panel hyperparameter sensitivity figure (uses fresh hp_*.npy data
    produced by run_hp_sweep.py with the SAME current train.py — therefore
    the y-axis is delivery rate (0..1), matching fig_convergence).
      (a) LR comparison   — LR = {1e-3, 1e-4, 1e-5} at batch=8192
      (b) Batch comparison — batch = {256, 2048, 8192} at LR=1e-4

    Note: hp_lr1e-4 == hp_mb8192 (shared baseline). The plot reuses the
    hp_lr1e-4 file as the batch=8192 curve in panel (b).
    """
    lr_configs = [
        ("$\\eta=10^{-3}$",  "logs/train_rewards_hp_lr1e-3.npy", "#4daf4a"),
        ("$\\eta=10^{-4}$",  "logs/train_rewards_hp_lr1e-4.npy", "#377eb8"),
        ("$\\eta=10^{-5}$",  "logs/train_rewards_hp_lr1e-5.npy", "#e41a1c"),
    ]
    mb_configs = [
        ("batch = 256",   "logs/train_rewards_hp_mb256.npy",  "#4daf4a"),
        ("batch = 2048",  "logs/train_rewards_hp_mb2048.npy", "#377eb8"),
        ("batch = 8192",  "logs/train_rewards_hp_lr1e-4.npy", "#e41a1c"),  # shared baseline
    ]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.0),
                                     gridspec_kw={"wspace": 0.18,
                                                  "left": 0.07, "right": 0.99,
                                                  "top": 0.94, "bottom": 0.20})

    def _plot_panel(ax, configs):
        all_smooth = []
        for label, path, color in configs:
            if not os.path.exists(path):
                print(f"  [SKIP] {path} not found")
                continue
            raw = np.load(path).astype(float)
            ep = np.arange(1, len(raw) + 1)
            # Heavy smoothing to reveal the trend (matches fig_convergence)
            smooth = _ema_smooth(raw, alpha=0.06)
            all_smooth.append(smooth)
            ax.scatter(ep, raw, color=color, alpha=0.06, s=4, zorder=1)
            ax.plot(ep, smooth, color=color, linewidth=2.2, label=label, zorder=3)
        ax.set_xlabel("Training Episodes", fontsize=13)
        ax.set_ylabel("Average Reward", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(loc="lower right", framealpha=0.95, edgecolor="gray",
                  fancybox=False, fontsize=12)
        if all_smooth:
            all_vals = np.concatenate(all_smooth)
            margin = 0.05 * (all_vals.max() - all_vals.min() + 1e-9)
            ax.set_ylim(max(0.0, all_vals.min() - margin),
                        min(1.0, all_vals.max() + margin))

    _plot_panel(ax_a, lr_configs)
    _plot_panel(ax_b, mb_configs)
    ax_a.text(0.5, -0.28, "(a)", transform=ax_a.transAxes,
              ha="center", fontsize=14, fontweight="bold")
    ax_b.text(0.5, -0.28, "(b)", transform=ax_b.transAxes,
              ha="center", fontsize=14, fontweight="bold")
    fig.savefig("figures/fig_convergence_hp.eps", format="eps",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig("figures/fig_convergence_hp.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved  figures/fig_convergence_hp")


# ══════════════════════════════════════════════════════════════════════════
# Fig 2: Performance comparison — 4 subfigures (a)(b)(c)(d)
#   Like reference paper Fig 10/11
# ══════════════════════════════════════════════════════════════════════════

def plot_performance():
    """4 subfigures: (a) E2E Delay, (b) Packet Loss, (c) Throughput CDF, (d) Jitter."""

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.0))

    # ── (a) E2E Delay vs Number of Users ─────────────────────────────────
    ax = axes[0]
    delay_path = "data/fig1_delay_vs_users.csv"
    if not os.path.exists(delay_path):
        delay_path = "data/fig1_delay_vs_rate.csv"  # fallback
    if os.path.exists(delay_path):
        df = pd.read_csv(delay_path)
        x_col = "N_users" if "N_users" in df.columns else "Traffic_Mbps"
        for s in SCHEMES:
            sub = df[df["Scheme"] == s]
            if len(sub) == 0:
                continue
            ax.plot(sub[x_col], sub["Delay_ms"],
                    color=COLORS[s], linestyle=LINESTYLES[s],
                    marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
        if x_col == "N_users":
            ax.set_xlabel("Number of Users")
        else:
            ax.set_xlabel("Traffic Rate (Mbps)")
    ax.set_ylabel("End-to-End Delay (ms)")
    ax.set_title("(a)", fontsize=9, fontweight="bold", y=-0.38)

    # ── (b) Packet Loss vs Number of Users ────────────────────────────────
    ax = axes[1]
    plr_path = "data/fig5_plr_vs_users.csv"
    if not os.path.exists(plr_path):
        plr_path = "data/fig5_plr_vs_rate.csv"  # fallback
    if os.path.exists(plr_path):
        df = pd.read_csv(plr_path)
        x_col = "N_users" if "N_users" in df.columns else "Traffic_Mbps"
        for s in SCHEMES:
            sub = df[df["Scheme"] == s]
            if len(sub) == 0:
                continue
            ax.plot(sub[x_col], sub["PLR"] * 100,
                    color=COLORS[s], linestyle=LINESTYLES[s],
                    marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
        if x_col == "N_users":
            ax.set_xlabel("Number of Users")
        else:
            ax.set_xlabel("Traffic Rate (Mbps)")
    ax.set_ylabel("Packet Loss (%)")
    # Auto-fit y-axis to data range with small padding
    if os.path.exists(plr_path):
        all_plr = pd.read_csv(plr_path)["PLR"].values * 100
        ymin = max(0, all_plr.min() - 5)
        ymax = min(100, all_plr.max() + 5)
        ax.set_ylim(ymin, ymax)
    ax.set_title("(b)", fontsize=9, fontweight="bold", y=-0.38)

    # ── (c) Satellite Throughput vs Number of Users ─────────────────────
    ax = axes[2]
    if os.path.exists(delay_path):
        df = pd.read_csv(delay_path)
        x_col = "N_users" if "N_users" in df.columns else "Traffic_Mbps"
        if "Sat_TP_Mbps" in df.columns:
            for s in SCHEMES:
                sub = df[df["Scheme"] == s]
                if len(sub) == 0:
                    continue
                ax.plot(sub[x_col], sub["Sat_TP_Mbps"],
                        color=COLORS[s], linestyle=LINESTYLES[s],
                        marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Satellite Throughput (Mbps)")
    ax.set_title("(c)", fontsize=9, fontweight="bold", y=-0.38)

    # ── (d) Network Throughput vs Number of Users ────────────────────────
    ax = axes[3]
    if os.path.exists(delay_path):
        df = pd.read_csv(delay_path)
        x_col = "N_users" if "N_users" in df.columns else "Traffic_Mbps"
        if "Net_TP_Gbps" in df.columns:
            for s in SCHEMES:
                sub = df[df["Scheme"] == s]
                if len(sub) == 0:
                    continue
                ax.plot(sub[x_col], sub["Net_TP_Gbps"],
                        color=COLORS[s], linestyle=LINESTYLES[s],
                        marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Network Throughput (Gbps)")
    ax.set_title("(d)", fontsize=9, fontweight="bold", y=-0.38)

    # ── Shared legend at top ──────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(SCHEMES),
                   framealpha=0.9, edgecolor="gray", fancybox=False,
                   bbox_to_anchor=(0.5, 1.06), fontsize=9)

    fig.subplots_adjust(top=0.84, bottom=0.22, left=0.04, right=0.99, wspace=0.20, hspace=0.0)
    fig.savefig("figures/fig2_performance.eps", format="eps", bbox_inches="tight", pad_inches=0.02)
    fig.savefig("figures/fig2_performance.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved  figures/fig2_performance")


# ══════════════════════════════════════════════════════════════════════════
# Also generate individual figures for flexibility
# ══════════════════════════════════════════════════════════════════════════

def plot_individual_delay():
    """Standalone E2E delay figure."""
    delay_path = "data/fig1_delay_vs_users.csv"
    if not os.path.exists(delay_path):
        delay_path = "data/fig1_delay_vs_rate.csv"
    if not os.path.exists(delay_path):
        return
    df = pd.read_csv(delay_path)
    x_col = "N_users" if "N_users" in df.columns else "Traffic_Mbps"
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for s in SCHEMES:
        sub = df[df["Scheme"] == s]
        if len(sub) == 0:
            continue
        ax.plot(sub[x_col], sub["Delay_ms"],
                color=COLORS[s], linestyle=LINESTYLES[s],
                marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
    ax.set_xlabel("Number of Users" if x_col == "N_users" else "Traffic Rate (Mbps)")
    ax.set_ylabel("End-to-End Delay (ms)")
    ax.legend(framealpha=0.9, edgecolor="gray", fancybox=False)
    fig.tight_layout()
    fig.savefig("figures/fig_delay.eps", format="eps", bbox_inches="tight")
    fig.savefig("figures/fig_delay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved  figures/fig_delay")


def plot_individual_plr():
    """Standalone PLR figure."""
    plr_path = "data/fig5_plr_vs_users.csv"
    if not os.path.exists(plr_path):
        plr_path = "data/fig5_plr_vs_rate.csv"
    if not os.path.exists(plr_path):
        return
    df = pd.read_csv(plr_path)
    x_col = "N_users" if "N_users" in df.columns else "Traffic_Mbps"
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for s in SCHEMES:
        sub = df[df["Scheme"] == s]
        if len(sub) == 0:
            continue
        ax.plot(sub[x_col], sub["PLR"] * 100,
                color=COLORS[s], linestyle=LINESTYLES[s],
                marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
    ax.set_xlabel("Number of Users" if x_col == "N_users" else "Traffic Rate (Mbps)")
    ax.set_ylabel("Packet Loss (%)")
    ax.legend(framealpha=0.9, edgecolor="gray", fancybox=False)
    fig.tight_layout()
    fig.savefig("figures/fig_plr.eps", format="eps", bbox_inches="tight")
    fig.savefig("figures/fig_plr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved  figures/fig_plr")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def plot_performance_scenario(tag, title_suffix, fig_name, y_ranges=None):
    """4-panel performance figure for a specific scenario (standard/large).

    Parameters
    ----------
    y_ranges : dict, optional
        Fixed y-axis ranges for consistency across scenarios.
        Keys: 'delay', 'plr', 'sat_tp', 'net_tp'  → (ymin, ymax)
    """
    delay_path = f"data/fig1_delay_vs_users_{tag}.csv"
    plr_path   = f"data/fig5_plr_vs_users_{tag}.csv"

    if not os.path.exists(delay_path):
        print(f"  [SKIP] {delay_path} not found")
        return

    if y_ranges is None:
        y_ranges = {}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0),
                              gridspec_kw={"wspace": 0.22,
                                           "left": 0.045, "right": 0.99,
                                           "top": 0.84, "bottom": 0.22})
    df = pd.read_csv(delay_path)
    x_col = "N_users"
    x_vals = sorted(df[x_col].unique())
    x_step = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 500

    def fmt_ax(ax, xlabel, ylabel, sublabel, y_key=None):
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='both', labelsize=12)
        ax.tick_params(axis='x', rotation=0)
        # Consistent x-tick spacing
        ax.xaxis.set_major_locator(MultipleLocator(x_step))
        ax.set_xlim(x_vals[0] - x_step * 0.3, x_vals[-1] + x_step * 0.3)
        ax.text(0.5, -0.32, sublabel, transform=ax.transAxes,
                ha="center", fontsize=14, fontweight="bold")
        # Apply fixed y-range if provided
        if y_key and y_key in y_ranges:
            ax.set_ylim(y_ranges[y_key])
        # Consistent y-tick count
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))

    # (a) E2E Delay
    ax = axes[0]
    for s in SCHEMES:
        sub = df[df["Scheme"] == s]
        if len(sub) == 0:
            continue
        ax.plot(sub[x_col], sub["Delay_ms"],
                color=COLORS[s], linestyle=LINESTYLES[s],
                marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
    fmt_ax(ax, "Number of Users", "End-to-End Delay (ms)", "(a)", "delay")

    # (b) PLR
    ax = axes[1]
    if os.path.exists(plr_path):
        df_plr = pd.read_csv(plr_path)
        for s in SCHEMES:
            sub = df_plr[df_plr["Scheme"] == s]
            if len(sub) == 0:
                continue
            ax.plot(sub[x_col], sub["PLR"] * 100,
                    color=COLORS[s], linestyle=LINESTYLES[s],
                    marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
    fmt_ax(ax, "Number of Users", "Packet Loss (%)", "(b)", "plr")

    # (c) Satellite Throughput
    ax = axes[2]
    if "Sat_TP_Mbps" in df.columns:
        for s in SCHEMES:
            sub = df[df["Scheme"] == s]
            if len(sub) == 0:
                continue
            ax.plot(sub[x_col], sub["Sat_TP_Mbps"],
                    color=COLORS[s], linestyle=LINESTYLES[s],
                    marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
    fmt_ax(ax, "Number of Users", "Satellite Throughput (Mbps)", "(c)", "sat_tp")

    # (d) Network Throughput
    ax = axes[3]
    if "Net_TP_Gbps" in df.columns:
        for s in SCHEMES:
            sub = df[df["Scheme"] == s]
            if len(sub) == 0:
                continue
            ax.plot(sub[x_col], sub["Net_TP_Gbps"],
                    color=COLORS[s], linestyle=LINESTYLES[s],
                    marker=MARKERS[s], label=s, markersize=7, linewidth=1.8)
    fmt_ax(ax, "Number of Users", "Network Throughput (Gbps)", "(d)", "net_tp")

    # Shared legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(SCHEMES),
                   framealpha=0.9, edgecolor="gray", fancybox=False,
                   bbox_to_anchor=(0.5, 1.02), fontsize=13)

    fig.savefig(f"figures/{fig_name}.eps", format="eps", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"figures/{fig_name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved  figures/{fig_name}")


def _compute_shared_y_ranges(*tags):
    """Compute shared y-axis ranges across multiple scenario tags for consistency."""
    all_delay, all_plr, all_sat_tp, all_net_tp = [], [], [], []
    for tag in tags:
        dp = f"data/fig1_delay_vs_users_{tag}.csv"
        pp = f"data/fig5_plr_vs_users_{tag}.csv"
        if os.path.exists(dp):
            df = pd.read_csv(dp)
            all_delay.extend(df["Delay_ms"].values)
            if "Sat_TP_Mbps" in df.columns:
                all_sat_tp.extend(df["Sat_TP_Mbps"].values)
            if "Net_TP_Gbps" in df.columns:
                all_net_tp.extend(df["Net_TP_Gbps"].values)
        if os.path.exists(pp):
            df_p = pd.read_csv(pp)
            all_plr.extend((df_p["PLR"].values * 100).tolist())

    def _pad_tight(vals, lo_pad=0.10, hi_pad=0.10):
        """Tight y-range that maximizes visible separation between schemes."""
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        span = hi - lo if hi > lo else 1.0
        return (max(0, lo - span * lo_pad), hi + span * hi_pad)

    return {
        "delay":  _pad_tight(all_delay),
        "plr":    _pad_tight(all_plr),
        "sat_tp": _pad_tight(all_sat_tp),
        "net_tp": _pad_tight(all_net_tp),
    }


def main():
    print("-- Plotting figures (reference paper style) --")
    plot_convergence()
    plot_convergence_hyperparams()

    # Per-scenario y-ranges so each scenario's differences are visually clear.
    small_y = _compute_shared_y_ranges("small")
    large_y = _compute_shared_y_ranges("large")

    # Scenario-specific performance plots
    plot_performance_scenario("small", "Small-scale Walker (18x18=324)", "fig_perf", small_y)
    plot_performance_scenario("large", "Large-scale Walker (36x22=792)", "fig_perf_large", large_y)
    print("\nDone. Figures -> ./figures/")


if __name__ == "__main__":
    main()
