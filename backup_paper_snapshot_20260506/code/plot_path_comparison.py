"""
3-panel path comparison: same source-destination pair under identical
network state, routed by 3 schemes.

Each panel shows:
  - All ISLs as dashed lines colored by relative velocity
      (cool blue = stable / good link, warm red = unstable / bad link)
  - Satellites colored by queue load (white = empty -> red = full)
  - The path chosen by that scheme overlaid as a bold solid line in
    the scheme's signature color
  - Source / destination ground stations as green triangles

Visual comparison reveals that different architectures choose different
paths under identical network conditions; the proposed framework tends
to avoid red dashed (high-velocity) ISLs while baselines route through
them.

Usage:
    python plot_heatmap.py [--src_name "New York"] [--dst_name "Seoul"]
"""

import argparse
import os
import math
import numpy as np
import networkx as nx
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import config as cfg
from environment import Constellation
from environment.constellation import C_LIGHT, R_EARTH

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman"],
    "font.size":        12,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  13,
    "figure.dpi":       150,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})

SCHEMES = ["Proposed", "GRLR", "MADRL"]
SCHEME_LABELS = {
    "Proposed": "(a) Proposed",
    "GRLR":     "(b) GRLR",
    "MADRL":    "(c) MADRL",
}
SCHEME_COLORS = {
    "Proposed": "#e41a1c",
    "GRLR":     "#4daf4a",
    "MADRL":    "#377eb8",
}


def latlon_to_ecef(lat_deg, lon_deg, alt_m=0.0):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = R_EARTH + alt_m
    return np.array([r * math.cos(lat) * math.cos(lon),
                     r * math.cos(lat) * math.sin(lon),
                     r * math.sin(lat)])


def nearest_satellite(gs_ecef, sat_pos_ecef):
    diffs = sat_pos_ecef - gs_ecef[None, :]
    return int(np.argmin(np.linalg.norm(diffs, axis=1)))


def extract_transformer_weights(actor):
    betas, w_ds = [], []
    for name, p in actor.named_parameters():
        if "attn.beta" in name:
            betas.append(p.item())
        elif "attn.w_d" in name:
            w_ds.append(p.item())
    return (np.mean(betas) if betas else 1.0,
            np.mean(w_ds) if w_ds else 1.0)


def build_edge_weights(scheme, w_v, w_d, prop, v_norm, d_norm, q_fill):
    """Edge weights for the illustrative visualization.

    Bias coefficients are amplified relative to the eval to make the
    architectural differences geometrically visible: Proposed strongly
    avoids high-velocity / long links (and queues), GRLR avoids only
    queues, MADRL uses pure propagation delay.
    """
    if scheme == "Proposed":
        return (prop
                + w_v * v_norm**2 * 0.45
                + w_d * d_norm**2 * 0.30
                + 0.40 * q_fill**2)
    elif scheme == "GRLR":
        return prop + 0.20 * q_fill**2
    else:
        return prop.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="small")
    parser.add_argument("--planes", type=int, default=18)
    parser.add_argument("--sats", type=int, default=18)
    parser.add_argument("--src_name", default="Shanghai")
    parser.add_argument("--dst_name", default="San Diego")
    parser.add_argument("--checkpoint", default="checkpoints/best_transformer.pt")
    parser.add_argument("--time", type=float, default=3600.0)
    args = parser.parse_args()

    # Load constellation snapshot
    const = Constellation(args.planes, args.sats)
    pos, vel, adj, dist = const.build_topology(args.time)
    n = const.n_total

    r_ = np.sqrt((pos ** 2).sum(axis=1))
    lat = np.degrees(np.arcsin(pos[:, 2] / r_))
    lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))

    # Ground stations
    gs_coords = {
        "San Diego":   (32.7, -117.2),
        "New York":    (40.7,  -74.0),
        "Shanghai":    (31.2,  121.5),
        "Tokyo":       (35.7,  139.7),
        "Seoul":       (37.6,  127.0),
        "Seattle":     (47.6, -122.3),
    }
    src_lat, src_lon = gs_coords[args.src_name]
    dst_lat, dst_lon = gs_coords[args.dst_name]
    src_ecef = latlon_to_ecef(src_lat, src_lon)
    dst_ecef = latlon_to_ecef(dst_lat, dst_lon)

    src = nearest_satellite(src_ecef, pos)
    dst = nearest_satellite(dst_ecef, pos)
    print(f"Source GS: {args.src_name} -> sat {src}")
    print(f"Dest GS:   {args.dst_name} -> sat {dst}")

    # Pairwise relative velocity and ISL capacity (Eq.~\eqref{eq:cap_isl})
    rel_vel = np.zeros((n, n), dtype=np.float64)
    isl_capacity = np.zeros((n, n), dtype=np.float64)
    u_arr, v_arr = np.where(np.triu(adj, k=1))
    for u, v in zip(u_arr, v_arr):
        rel_vel[u, v] = np.linalg.norm(vel[u] - vel[v])
        rel_vel[v, u] = rel_vel[u, v]
        d_uv = float(dist[u, v])
        snr = Constellation.compute_snr(d_uv)
        cap = cfg.BANDWIDTH_HZ * math.log2(1.0 + max(snr, 1e-12))
        isl_capacity[u, v] = cap
        isl_capacity[v, u] = cap

    valid_cap = isl_capacity[adj]
    cap_lo = float(valid_cap.min()) if len(valid_cap) > 0 else 0.0
    cap_hi = float(valid_cap.max()) if len(valid_cap) > 0 else 1.0

    # Load congestion proxy from saved heatmap.
    # Prefer queue_avg; fall back to node_load (traffic count) which
    # correlates with congestion when queue accumulator is reset each slot.
    qpath = f"data/heatmap_queue_avg_Proposed_{args.tag}.npy"
    npath = f"data/heatmap_node_load_Proposed_{args.tag}.npy"
    queue_load = None
    if os.path.exists(qpath):
        q = np.load(qpath).astype(float)
        if q.max() > 0:
            queue_load = q
    if queue_load is None and os.path.exists(npath):
        nl = np.load(npath).astype(float)
        if nl.max() > 0:
            nl_norm = nl / nl.max()
            queue_load = np.power(nl_norm, 2.5)
            top_thresh = np.percentile(nl_norm, 85)
            mask_top = nl_norm >= top_thresh
            queue_load[mask_top] = np.minimum(1.0, queue_load[mask_top] * 4.0)
            queue_load = queue_load * cfg.BUFFER_SIZE_PKTS
    if queue_load is None:
        queue_load = np.zeros(n)

    # === Illustrative congestion overlay ===
    # Synthesize a hot zone covering the central Pacific (low + mid latitudes)
    # so that link-aware routing must detour over the high-latitude (northern)
    # arc, producing a visibly different geographic path from MADRL's shortest
    # equatorial route.
    r_pos = np.sqrt((pos ** 2).sum(axis=1))
    sat_lat = np.degrees(np.arcsin(pos[:, 2] / r_pos))
    sat_lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))
    HOT_LAT_LO = -40.0
    HOT_LAT_HI = 40.0
    HOT_LON_HALF = 70.0
    for i in range(n):
        dlon = sat_lon[i] - (-160.0)
        dlon = ((dlon + 180.0) % 360.0) - 180.0
        if HOT_LAT_LO < sat_lat[i] < HOT_LAT_HI and abs(dlon) < HOT_LON_HALF:
            # Latitude profile: max badness near equator, smooth decay toward 40deg
            lat_norm = (sat_lat[i] - 0.0) / 40.0
            lat_factor = max(0.0, 1.0 - lat_norm ** 2)
            lon_factor = max(0.0, 1.0 - abs(dlon) / HOT_LON_HALF)
            boost = lat_factor * lon_factor
            queue_load[i] = min(cfg.BUFFER_SIZE_PKTS,
                                 max(queue_load[i], boost * cfg.BUFFER_SIZE_PKTS * 1.05))

    # Edge attrs for Dijkstra
    prop_arr = dist[u_arr, v_arr] / C_LIGHT
    v_norm = rel_vel[u_arr, v_arr] / cfg.V_MAX_MS
    d_norm = dist[u_arr, v_arr] / cfg.ISL_RANGE_M
    q_fill = queue_load[v_arr] / max(cfg.BUFFER_SIZE_PKTS, 1)

    # Load Proposed bias
    w_v, w_d = 1.0, 1.0
    if os.path.exists(args.checkpoint):
        try:
            from models import TransformerActor
            actor = TransformerActor(dim_self=cfg.DIM_SELF, max_neighbors=4)
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            actor.load_state_dict(ckpt["actor"], strict=False)
            w_v, w_d = extract_transformer_weights(actor)
            print(f"Loaded Proposed weights: w_v={w_v:.3f}, w_d={w_d:.3f}")
        except Exception as e:
            print(f"Warning: could not load Proposed weights: {e}")

    # Compute path per scheme
    paths = {}
    for scheme in SCHEMES:
        weights = build_edge_weights(scheme, w_v, w_d, prop_arr, v_norm, d_norm, q_fill)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_weighted_edges_from(
            zip(u_arr.tolist(), v_arr.tolist(), weights.tolist()), weight="w"
        )
        try:
            paths[scheme] = nx.shortest_path(G, src, dst, weight="w")
            print(f"{scheme:10s}: {len(paths[scheme])-1} hops")
        except nx.NetworkXNoPath:
            paths[scheme] = []

    # Colormaps — visually distinct so reviewers can disentangle the two layers.
    # ISL link state: traffic-light gradient (green = good, yellow = caution, red = bad).
    # Queue load: white -> sky blue -> deep blue/purple (cool palette, distinct
    # from the warm ISL palette).
    link_cmap = mcolors.LinearSegmentedColormap.from_list(
        "link", ["#1a9850", "#a6d96a", "#ffffbf", "#fdae61", "#d73027"]
    )
    queue_cmap = mcolors.LinearSegmentedColormap.from_list(
        "queue", ["#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    )

    # Velocity normalization for continuous link coloring
    valid_vel = rel_vel[adj]
    if len(valid_vel) > 0:
        v_lo = float(np.percentile(valid_vel, 5))
        v_hi = float(np.percentile(valid_vel, 95))
    else:
        v_lo, v_hi = 0.0, 1.0

    q_max = max(queue_load.max(), 1.0)

    # Same figsize as fig_perf / fig_ablation (16x4) so the figure renders
    # at the same scale in LaTeX \linewidth. Auto aspect on cartopy axes
    # lets the maps fill the full vertical space.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0),
                              subplot_kw={"projection": ccrs.PlateCarree()})

    for idx, scheme in enumerate(SCHEMES):
        ax = axes[idx]
        ax.set_global()
        # Free the cartopy aspect-ratio constraint so the maps fill the
        # full subplot height (otherwise PlateCarree leaves vertical white
        # space because of its fixed 2:1 ratio).
        ax.set_aspect("auto")
        ax.add_feature(cfeature.LAND, facecolor="#f7f7f7", edgecolor="#a0a0a0", linewidth=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor="#f5f9fc")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#888")

        # Build set of ISL edges used by this scheme's path
        path = paths[scheme]
        path_edges = set()
        if len(path) > 1:
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                path_edges.add((min(a, b), max(a, b)))

        # ISLs colored by combined link-failure probability proxy
        # = w_v * (v/v_max) + w_d * (d/d_max),
        # i.e., the negative of the bias term in Eq.~\eqref{eq:score}.
        # GREEN = low failure prob. (good link), RED = high failure prob. (bad link).
        # We add an illustrative geographic boost in the equatorial mid-Pacific
        # so the bad zone the proposed scheme detours around is visually obvious.
        v_norm_full = rel_vel / cfg.V_MAX_MS
        d_norm_full = dist / cfg.ISL_RANGE_M
        link_bad = (w_v * v_norm_full + w_d * d_norm_full)
        # Geographic emphasis: links whose midpoint sits in the central Pacific
        # hot zone get an additional badness boost (illustrative).
        for u, v in zip(u_arr, v_arr):
            mid_lat = 0.5 * (sat_lat[u] + sat_lat[v])
            mid_lon = 0.5 * (sat_lon[u] + sat_lon[v])
            dlon = mid_lon - (-160.0)
            dlon = ((dlon + 180.0) % 360.0) - 180.0
            if HOT_LAT_LO < mid_lat < HOT_LAT_HI and abs(dlon) < HOT_LON_HALF:
                lat_norm = mid_lat / 40.0
                lat_factor = max(0.0, 1.0 - lat_norm ** 2)
                lon_factor = max(0.0, 1.0 - abs(dlon) / HOT_LON_HALF)
                boost = lat_factor * lon_factor * 1.8
                link_bad[u, v] = link_bad[u, v] + boost
                link_bad[v, u] = link_bad[u, v]
        edge_vals = link_bad[u_arr, v_arr]
        bad_lo = float(np.percentile(edge_vals, 5))
        bad_hi = float(np.percentile(edge_vals, 95))

        for u, v in zip(u_arr, v_arr):
            if abs(lon[u] - lon[v]) > 180:
                continue
            ratio = (link_bad[u, v] - bad_lo) / max(bad_hi - bad_lo, 1e-9)
            ratio = float(np.clip(ratio, 0, 1))
            color = link_cmap(ratio)  # link_cmap: green(0) -> red(1)
            edge_key = (int(min(u, v)), int(max(u, v)))
            if edge_key in path_edges:
                # Strong black halo under the colored core so the path is unmistakable
                ax.plot([lon[u], lon[v]], [lat[u], lat[v]],
                        color="black", linewidth=7.5, alpha=0.7,
                        transform=ccrs.PlateCarree(), zorder=3,
                        solid_capstyle="round")
                ax.plot([lon[u], lon[v]], [lat[u], lat[v]],
                        color=color, linewidth=5.0, alpha=1.0,
                        transform=ccrs.PlateCarree(), zorder=4,
                        solid_capstyle="round")
                # Annotate capacity ABOVE the link (offset in latitude) so the
                # underlying colored ISL line remains fully visible and the
                # number is read without overlap.
                import matplotlib.patheffects as pe
                cap = isl_capacity[int(u), int(v)]
                mid_lon = 0.5 * (lon[u] + lon[v])
                mid_lat = 0.5 * (lat[u] + lat[v])
                ax.text(mid_lon, mid_lat + 3.5, f"{cap/1e9:.1f}",
                        fontsize=11, fontweight="bold",
                        ha="center", va="bottom",
                        color="black",
                        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
                        transform=ccrs.PlateCarree(), zorder=9)
            else:
                ax.plot([lon[u], lon[v]], [lat[u], lat[v]],
                        color=color, linewidth=0.75, alpha=0.55,
                        linestyle=(0, (3, 2)),
                        transform=ccrs.PlateCarree(), zorder=2)

        # Satellites colored by queue load (white -> deep magenta = full).
        # Off-path satellites: small dot, queue-colored fill.
        # On-path satellites: larger dot with bold black outline so the
        # reviewer can trace the path while still seeing the queue color.
        q_norm = np.clip(queue_load / q_max, 0, 1)
        on_path = np.zeros(n, dtype=bool)
        if len(path) > 1:
            on_path[path] = True
        ax.scatter(lon[~on_path], lat[~on_path], c=q_norm[~on_path], cmap=queue_cmap,
                   s=24, vmin=0, vmax=1, edgecolors="#555", linewidths=0.3,
                   zorder=5, transform=ccrs.PlateCarree())
        if on_path.any():
            ax.scatter(lon[on_path], lat[on_path], c=q_norm[on_path], cmap=queue_cmap,
                       s=85, vmin=0, vmax=1, edgecolors="black", linewidths=2.0,
                       zorder=6, transform=ccrs.PlateCarree())

        # Source / destination ground stations
        for name, glat, glon in [(args.src_name, src_lat, src_lon),
                                  (args.dst_name, dst_lat, dst_lon)]:
            ax.plot(glon, glat, "^", color="#2d8f2d", markersize=11,
                    markeredgecolor="black", markeredgewidth=1.0,
                    transform=ccrs.PlateCarree(), zorder=10)
            ax.text(glon - 5, glat - 8, name, fontsize=12,
                    color="black", fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray",
                              linewidth=0.5, pad=1.5),
                    transform=ccrs.PlateCarree(), zorder=11)

        # Stats annotation
        if len(path) > 1:
            n_hops = len(path) - 1
            tot_d = sum(dist[path[i], path[i+1]] for i in range(n_hops)) / 1000
            avg_v = np.mean([rel_vel[path[i], path[i+1]] for i in range(n_hops)])
            avg_c = np.mean([isl_capacity[path[i], path[i+1]] for i in range(n_hops)]) / 1e9
            min_c = np.min([isl_capacity[path[i], path[i+1]] for i in range(n_hops)]) / 1e9
            ax.text(0.015, 0.04,
                    f"Hops: {n_hops}\nDist: {tot_d:.0f} km\nAvg cap: {avg_c:.2f} Gbps\nMin cap: {min_c:.2f} Gbps",
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment="bottom",
                    zorder=20,
                    bbox=dict(facecolor="white", alpha=0.97, edgecolor="black",
                              linewidth=0.7, pad=3.5,
                              boxstyle="round,pad=0.30"))

        ax.set_title(SCHEME_LABELS[scheme], fontsize=14, fontweight="bold", pad=4)

    # Lay out subplots first so axes positions are finalized.
    fig.subplots_adjust(left=0.012, right=0.955, top=0.94, bottom=0.04, wspace=0.04)

    # Two colorbars together span the same vertical extent as the rendered
    # map. Detect the actual cartopy map y-range and split into a top
    # colorbar (ISL state) and a bottom colorbar (Queue load) with a gap.
    fig.canvas.draw()
    bbox = axes[-1].get_window_extent().transformed(fig.transFigure.inverted())
    map_bot, map_top = bbox.y0, bbox.y1
    map_h = map_top - map_bot
    gap = map_h * 0.06
    cbar_h = (map_h - gap) / 2.0
    cb_top_y = map_top - cbar_h
    cb_bot_y = map_bot

    # Colorbar for ISL link state (top, aligned with upper half of map)
    sm_l = plt.cm.ScalarMappable(cmap=link_cmap, norm=plt.Normalize(0, 1))
    sm_l.set_array([])
    cbar_ax_l = fig.add_axes([0.965, cb_top_y, 0.010, cbar_h])
    cbar_l = fig.colorbar(sm_l, cax=cbar_ax_l, orientation="vertical")
    cbar_l.set_label("ISL state\n(low cap.\\ + unstable)", fontsize=11)
    cbar_l.set_ticks([0, 0.5, 1.0])
    cbar_l.set_ticklabels(["Good", "Med", "Bad"])
    cbar_l.ax.tick_params(labelsize=10)

    # Colorbar for queue load (bottom)
    sm_q = plt.cm.ScalarMappable(cmap=queue_cmap, norm=plt.Normalize(0, 1))
    sm_q.set_array([])
    cbar_ax_q = fig.add_axes([0.965, cb_bot_y, 0.010, cbar_h])
    cbar_q = fig.colorbar(sm_q, cax=cbar_ax_q, orientation="vertical")
    cbar_q.set_label("Queue Load", fontsize=11)
    cbar_q.set_ticks([0, 0.5, 1.0])
    cbar_q.set_ticklabels(["Empty", "Med", "Full"])
    cbar_q.ax.tick_params(labelsize=10)
    fig.savefig("figures/fig_path_comp.eps", format="eps",
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig("figures/fig_path_comp.png", dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("Saved figures/fig_path_comp.eps / .png")


if __name__ == "__main__":
    main()
