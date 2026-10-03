#!/usr/bin/env python3
"""Draw docs/seminar/region_figure.png for the "発想" slide.

An input space split into the regions of 10 classes (nearest-seed partition).
The explained point sits near where the regions of 0, 6 and 9 meet; inside its
neighborhood only those three regions appear. Illustration only, not data.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, ConnectionPatch
import numpy as np

plt.rcParams["font.family"] = ["IPAGothic", "DejaVu Sans"]

SEEDS = {
    0: (0.34, 0.56), 6: (0.64, 0.60), 9: (0.50, 0.30),
    1: (0.12, 0.88), 2: (0.45, 0.90), 3: (0.86, 0.88), 4: (0.92, 0.42),
    5: (0.80, 0.10), 7: (0.18, 0.12), 8: (0.08, 0.48),
}
HIGHLIGHT = {0: "#2F78A8", 6: "#D9532B", 9: "#E3A21A"}
GRAYS = ["#C9CED8", "#D8DCE4", "#BFC5D0", "#E1E4EA", "#CDD2DB", "#D3D7E0", "#C4CAD4"]
INK = "#1D2433"

labels = np.array(list(SEEDS))
points = np.array([SEEDS[k] for k in labels])
n = 700
xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
grid = np.stack([xs.ravel(), ys.ravel()], axis=1)
nearest = np.argmin(((grid[:, None, :] - points[None]) ** 2).sum(-1), axis=1).reshape(n, n)

# Point where 0, 6 and 9 are equidistant, then nudge toward 6 (predicted 6, runner-up 0).
a, b, c = (np.array(SEEDS[k]) for k in (0, 6, 9))
lhs = np.array([2 * (b - a), 2 * (c - a)])
rhs = np.array([b @ b - a @ a, c @ c - a @ a])
junction = np.linalg.solve(lhs, rhs)
star = junction + np.array([0.025, 0.012])
radius = 0.085


def colors(highlight_all: bool) -> np.ndarray:
    rgb = np.zeros((len(labels), 3))
    gray_index = 0
    for i, k in enumerate(labels):
        if k in HIGHLIGHT:
            hexcolor = HIGHLIGHT[k]
        else:
            hexcolor = GRAYS[gray_index % len(GRAYS)]
            gray_index += 1
        rgb[i] = [int(hexcolor[j:j + 2], 16) / 255 for j in (1, 3, 5)]
        if not highlight_all and k in HIGHLIGHT:
            rgb[i] = 0.55 * rgb[i] + 0.45  # paler in the overview
    return rgb


fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 5.0), gridspec_kw={"width_ratios": [1, 1]})
for axis, zoom in ((left, False), (right, True)):
    axis.imshow(colors(zoom)[nearest], origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
    axis.contour(xs, ys, nearest, levels=np.arange(len(labels)) + 0.5, colors="white", linewidths=1.6)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#9AA3B2")

for i, k in enumerate(labels):
    x, y = SEEDS[k]
    left.text(x, y, str(k), ha="center", va="center", fontsize=17, color=INK)
left.plot(*star, marker="*", markersize=18, color=INK, markeredgecolor="white", markeredgewidth=1.2)
left.add_patch(Circle(star, radius, fill=False, edgecolor=INK, linewidth=2.2, linestyle="--"))
left.set_title("入力空間全体　10クラスの領域", fontsize=14, color=INK, pad=8)

half = 0.15
right.set_xlim(star[0] - half, star[0] + half)
right.set_ylim(star[1] - half, star[1] + half)
right.add_patch(Circle(star, radius, fill=False, edgecolor=INK, linewidth=2.6, linestyle="--"))
rng = np.random.default_rng(3)
cloud = star + rng.normal(scale=radius / 2.2, size=(60, 2))
right.scatter(cloud[:, 0], cloud[:, 1], s=14, color="white", edgecolor=INK, linewidth=0.6, zorder=3)
right.plot(*star, marker="*", markersize=26, color=INK, markeredgecolor="white", markeredgewidth=1.5, zorder=4)
for k, offset in ((0, (-0.09, 0.06)), (6, (0.09, 0.07)), (9, (0.0, -0.10))):
    right.text(star[0] + offset[0], star[1] + offset[1], str(k), ha="center", va="center",
               fontsize=24, color=INK if k == 9 else "white")
right.set_title("説明したい点のまわり　0・6・9の3つだけ", fontsize=14, color=INK, pad=8)

for corner in ((star[0] + radius * 0.7, star[1] + radius * 0.7), (star[0] + radius * 0.7, star[1] - radius * 0.7)):
    target = (0, 1) if corner[1] > star[1] else (0, 0)
    fig.add_artist(ConnectionPatch(xyA=corner, coordsA=left.transData, xyB=target, coordsB=right.transAxes,
                                   color="#9AA3B2", linewidth=1.0))
fig.tight_layout(w_pad=3)
out = Path(__file__).with_name("region_figure.png")
fig.savefig(out, dpi=200, facecolor="white")
print(out, star.round(3))
