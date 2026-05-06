"""
Plot the bias-term ablation: Proposed vs Proposed-NoBias vs MADRL.

Reads:  data/ablation_delay_<tag>.csv, data/ablation_plr_<tag>.csv
Writes: figures/fig_ablation.png/.eps  (Image/fig_ablation.png in paper)

A 3-panel figure:
  (a) end-to-end delay
  (b) packet loss ratio
  (c) aggregate network throughput
"""

import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

os.makedirs("figures", exist_ok=True)

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

SCHEMES = ["Proposed", "Proposed-NoBias", "MADRL"]
COLORS = {
    "Proposed":        "#e41a1c",
    "Proposed-NoBias": "#984ea3",
    "MADRL":           "#377eb8",
}
MARKERS = {
    "Proposed":        "o",
    "Proposed-NoBias": "D",
    "MADRL":           "s",
}
LINESTYLES = {
    "Proposed":        "-",
    "Proposed-NoBias": "-.",
    "MADRL":           "--",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="small")
    args = parser.parse_args()

    df_d = pd.read_csv(f"data/ablation_delay_{args.tag}.csv")
    df_p = pd.read_csv(f"data/ablation_plr_{args.tag}.csv")

    # 4-panel layout matching fig_perf: same figsize, same panel structure.
    # The four metrics are (a) E2E delay, (b) PLR, (c) per-satellite throughput,
    # (d) aggregate network throughput.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.0),
                              gridspec_kw={"wspace": 0.22,
                                           "left": 0.045, "right": 0.99,
                                           "top": 0.84, "bottom": 0.22})

    # Number of satellites used for the small-scale eval (18 x 18)
    n_sats_eval = 18 * 18

    x_vals = sorted(df_d["N_users"].unique())
    x_step = x_vals[1] - x_vals[0] if len(x_vals) > 1 else 500

    def fmt(ax, ylabel, sub):
        ax.set_xlabel("Number of Users", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.set_major_locator(MultipleLocator(x_step))
        ax.set_xlim(x_vals[0] - x_step * 0.3, x_vals[-1] + x_step * 0.3)
        ax.text(0.5, -0.32, sub, transform=ax.transAxes,
                ha="center", fontsize=14, fontweight="bold")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))

    # Match marker / linewidth used in fig_perf (markersize=7, linewidth=1.8)

    # Derive per-satellite throughput from aggregate (Mbps per sat)
    df_d = df_d.copy()
    df_d["Sat_TP_Mbps"] = df_d["Net_TP_Gbps"] * 1000.0 / n_sats_eval

    # (a) Delay
    ax = axes[0]
    for s in SCHEMES:
        sub = df_d[df_d["Scheme"] == s]
        ax.plot(sub["N_users"], sub["Delay_ms"],
                color=COLORS[s], linestyle=LINESTYLES[s],
                marker=MARKERS[s], label=s, markersize=7, linewidth=2.0)
    fmt(ax, "End-to-End Delay (ms)", "(a)")

    # (b) PLR
    ax = axes[1]
    for s in SCHEMES:
        sub = df_p[df_p["Scheme"] == s]
        ax.plot(sub["N_users"], sub["PLR"] * 100,
                color=COLORS[s], linestyle=LINESTYLES[s],
                marker=MARKERS[s], label=s, markersize=7, linewidth=2.0)
    fmt(ax, "Packet Loss (%)", "(b)")

    # (c) Per-satellite throughput
    ax = axes[2]
    for s in SCHEMES:
        sub = df_d[df_d["Scheme"] == s]
        ax.plot(sub["N_users"], sub["Sat_TP_Mbps"],
                color=COLORS[s], linestyle=LINESTYLES[s],
                marker=MARKERS[s], label=s, markersize=7, linewidth=2.0)
    fmt(ax, "Satellite Throughput (Mbps)", "(c)")

    # (d) Aggregate network throughput
    ax = axes[3]
    for s in SCHEMES:
        sub = df_d[df_d["Scheme"] == s]
        ax.plot(sub["N_users"], sub["Net_TP_Gbps"],
                color=COLORS[s], linestyle=LINESTYLES[s],
                marker=MARKERS[s], label=s, markersize=7, linewidth=2.0)
    fmt(ax, "Network Throughput (Gbps)", "(d)")

    # Tight per-panel y-range to maximize separation
    for ax_, col, df, scale in [
        (axes[0], "Delay_ms",     df_d, 1.0),
        (axes[1], "PLR",          df_p, 100.0),
        (axes[2], "Sat_TP_Mbps",  df_d, 1.0),
        (axes[3], "Net_TP_Gbps",  df_d, 1.0),
    ]:
        v = df[col].values * scale
        lo, hi = v.min(), v.max()
        span = hi - lo if hi > lo else 1.0
        ax_.set_ylim(lo - 0.10 * span, hi + 0.12 * span)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               framealpha=0.9, edgecolor="gray", fancybox=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=13)

    fig.savefig("figures/fig_ablation.eps", format="eps",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig("figures/fig_ablation.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved figures/fig_ablation.eps / .png")


if __name__ == "__main__":
    main()
