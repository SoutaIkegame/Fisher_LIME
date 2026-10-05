#!/usr/bin/env python3
"""Draw the class-side figures for the "説明をまとめる2つの方向" slide.

* class_maps_10.png: ten small per-class heatmaps (one per class, 0-9).
* class_axis_0to6.png: one heatmap for the axis "0から6への移動"
  (class-6 map minus class-0 map).
Values are (mean image of class k) - (mean image of all classes) from the
sklearn digits data, shown on the strokes of the explained digit, as in
make_ten_heatmaps.py. Illustration, not LIME output.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
from sklearn.datasets import load_digits

plt.rcParams["font.family"] = ["IPAGothic", "DejaVu Sans"]
OUT = Path(__file__).resolve().parent
POS, NEG, INK = np.array(to_rgb("#156082")), np.array(to_rgb("#E97132")), "#1D2433"

digits = load_digits()
images = digits.images / 16.0
overall = images.mean(axis=0)
digit = np.array(json.loads((OUT / "slide_data.json").read_text())["digit_image"])
strokes = digit > 0.1
maps = {k: images[digits.target == k].mean(axis=0) - overall for k in range(10)}


def tiles(axis, values):
    scale = np.abs(values[strokes]).max()
    for i in range(8):
        for j in range(8):
            if strokes[i, j]:
                v = values[i, j] / scale
                color = 1 - abs(v) * (1 - (POS if v >= 0 else NEG))
            else:
                color = (0.94, 0.94, 0.94)
            axis.add_patch(plt.Rectangle((j + 0.06, 7 - i + 0.06), 0.88, 0.88, color=color, linewidth=0))
    axis.set_xlim(0, 8)
    axis.set_ylim(0, 8)
    axis.set_aspect("equal")
    axis.axis("off")


fig = plt.figure(figsize=(2.6, 1.3))
for k in range(10):
    r, c = divmod(k, 5)
    axis = fig.add_axes([0.01 + c * 0.198, 0.5 - r * 0.5, 0.18, 0.37])
    tiles(axis, maps[k])
    axis.set_title(f"{k}", fontsize=8, color=INK, pad=1.5)
fig.savefig(OUT / "class_maps_10.png", dpi=300, facecolor="white")
plt.close(fig)

fig = plt.figure(figsize=(2.0, 2.0))
axis = fig.add_axes([0.02, 0.02, 0.96, 0.96])
tiles(axis, maps[6] - maps[0])
fig.savefig(OUT / "class_axis_0to6.png", dpi=300, facecolor="white")
print("ok")
