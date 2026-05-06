"""Demo: constellation heat map like Lozano-Cuadra et al. Fig 8."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from environment import Constellation

const = Constellation(18, 18)
pos, vel, adj, dist = const.build_topology(0.0)
n = const.n_total
r = np.sqrt((pos ** 2).sum(axis=1))
lat = np.degrees(np.arcsin(pos[:, 2] / r))
lon = np.degrees(np.arctan2(pos[:, 1], pos[:, 0]))

rng = np.random.default_rng(42)
traffic_load = rng.uniform(10, 100, size=n)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="gray", linewidth=0.3)
ax.add_feature(cfeature.OCEAN, facecolor="#f0f8ff")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="gray")

for i in range(n):
    for j in range(i + 1, n):
        if adj[i, j]:
            load = (traffic_load[i] + traffic_load[j]) / 200
            color = plt.cm.cool(load)
            ax.plot(
                [lon[i], lon[j]], [lat[i], lat[j]],
                color=color, linewidth=0.6, alpha=max(0.2, load),
                transform=ccrs.Geodetic(),
            )

sc = ax.scatter(
    lon, lat, c=traffic_load, cmap="cool", s=12,
    edgecolors="black", linewidths=0.2, zorder=5,
    transform=ccrs.PlateCarree(),
)

gs_coords = [
    (32.7, -117.2), (40.7, -74.0), (31.2, 121.5),
    (35.7, 139.7), (37.6, 127.0), (47.6, -122.3),
]
for glat, glon in gs_coords:
    ax.plot(glon, glat, "rx", markersize=10, markeredgewidth=2,
            transform=ccrs.PlateCarree(), zorder=10)

cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.6, pad=0.02)
cbar.set_label("Relative Traffic Load (%)", fontsize=9)

ax.plot([], [], "rx", markersize=8, markeredgewidth=2, label="Gateways")
ax.plot([], [], "o", color="#4488cc", markersize=5, label="Satellites")
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
ax.set_title("Proposed routing policy — traffic heat map (demo)", fontsize=11)

fig.savefig("figures/fig_heatmap_demo.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figures/fig_heatmap_demo.png")
