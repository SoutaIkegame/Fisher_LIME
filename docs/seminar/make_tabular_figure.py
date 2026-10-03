#!/usr/bin/env python3
"""Draw the tabular-data LIME figures for the seminar deck (illustration only).

* tabular_space.png: a 2D feature space split by a black box into classes
  A (blue), B (red), C (green), with data points and the explained point.
* tabular_ovr_{A,B,C}.png: one-vs-rest views. The class is coloured, the rest
  is gray, and the local linear boundary (tangent line) of that class's
  probability is drawn through the explained point.
* tabular_coef.json: the local linear coefficients (gradient of each class
  probability at the explained point, scaled for display).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["IPAGothic", "DejaVu Sans"]
OUT = Path(__file__).resolve().parent
COLORS = {"A": "#2F78A8", "B": "#D9532B", "C": "#3A9D7A"}
GRAY = "#4A5568"
INK = "#1D2433"
STAR = np.array([0.47, 0.60])


def scores(x, y):
    """Black-box class scores; curved boundaries meeting near the star."""
    s_a = 6.0 * (x - 0.52) + 2.0 * (y - 0.55)
    s_b = 6.0 * (0.55 - y) - 5.0 * (x - 0.45) ** 2
    s_c = 6.0 * (0.42 - x) + 4.0 * (y - 0.45) ** 2
    return np.stack([s_a, s_b, s_c], axis=-1)


def probabilities(x, y):
    s = scores(x, y)
    e = np.exp(s - s.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


rng = np.random.default_rng(7)
centers = {"A": (0.80, 0.62), "B": (0.42, 0.30), "C": (0.20, 0.72)}
points = {k: np.clip(np.array(c) + rng.normal(scale=0.09, size=(9, 2)), 0.04, 0.96) for k, c in centers.items()}
for k in points:  # keep only points the black box assigns to their class
    p = probabilities(points[k][:, 0], points[k][:, 1])
    points[k] = points[k][np.argmax(p, axis=1) == "ABC".index(k)]

n = 400
xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
label = np.argmax(probabilities(xs, ys), axis=-1)


def frame(axis):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#9AA3B2")
        axis.spines[side].set_linewidth(1.5)


def boundaries(axis, color, width):
    axis.contour(xs, ys, label, levels=[0.5, 1.5], colors=color, linewidths=width)


def star(axis, size):
    axis.plot(*STAR, marker="*", markersize=size, color="#E3A21A", markeredgecolor="white", markeredgewidth=1.2, zorder=5)


# Overview
fig, axis = plt.subplots(figsize=(4.6, 4.2))
frame(axis)
boundaries(axis, "#6B7280", 2.0)
for k, pts in points.items():
    axis.scatter(pts[:, 0], pts[:, 1], s=150, color=COLORS[k], edgecolor="white", linewidth=1.0, zorder=3)
star(axis, 24)
axis.set_xlabel("特徴1", fontsize=13, color=INK)
axis.set_ylabel("特徴2", fontsize=13, color=INK)
fig.tight_layout()
fig.savefig(OUT / "tabular_space.png", dpi=200, facecolor="white")
plt.close(fig)

# One-vs-rest panels with the local linear boundary of each class
eps = 1e-4
coef = {}
for i, k in enumerate("ABC"):
    gx = (probabilities(STAR[0] + eps, STAR[1])[i] - probabilities(STAR[0] - eps, STAR[1])[i]) / (2 * eps)
    gy = (probabilities(STAR[0], STAR[1] + eps)[i] - probabilities(STAR[0], STAR[1] - eps)[i]) / (2 * eps)
    coef[k] = [float(gx), float(gy)]
    fig, axis = plt.subplots(figsize=(3.2, 2.9))
    frame(axis)
    boundaries(axis, "#C5CAD3", 1.4)
    for kk, pts in points.items():
        axis.scatter(pts[:, 0], pts[:, 1], s=60, color=COLORS[k] if kk == k else GRAY, edgecolor="white", linewidth=0.6, zorder=3)
    direction = np.array([-gy, gx]) / np.hypot(gx, gy)
    line = STAR + np.outer(np.linspace(-0.7, 0.7, 2), direction)
    axis.plot(line[:, 0], line[:, 1], color="#E3A21A", linewidth=2.2, zorder=4)
    star(axis, 16)
    fig.tight_layout()
    fig.savefig(OUT / f"tabular_ovr_{k}.png", dpi=200, facecolor="white")
    plt.close(fig)

scale = 0.6 / max(abs(v) for c in coef.values() for v in c)
display = {k: [round(v * scale, 2) for v in c] for k, c in coef.items()}
(OUT / "tabular_coef.json").write_text(json.dumps({"raw": coef, "display": display, "star_probs": probabilities(*STAR).round(2).tolist()}, ensure_ascii=False))
print(display, probabilities(*STAR).round(2))
