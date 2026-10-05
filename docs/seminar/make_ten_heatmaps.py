#!/usr/bin/env python3
"""Draw ten_heatmaps.png: one illustrative LIME-style heatmap per class (0-9).

For class k, each pixel's value is (mean image of class k) - (mean image of
all classes) from the sklearn digits data, shown only on the strokes of the
explained digit. Navy = raises the class probability, orange = lowers it
(the colours of the deck's theme). Illustration, not LIME output.
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

fig = plt.figure(figsize=(3.1, 3.0))
positions = [(r, c) for r in range(3) for c in range(3)] + [(3, 1)]
for k, (r, c) in enumerate(positions):
    axis = fig.add_axes([0.04 + c * 0.33, 0.77 - r * 0.25, 0.26, 0.2])
    diff = images[digits.target == k].mean(axis=0) - overall
    scale = np.abs(diff[strokes]).max()
    rgb = np.ones((8, 8, 3))
    for i in range(8):
        for j in range(8):
            if strokes[i, j]:
                v = diff[i, j] / scale
                base = POS if v >= 0 else NEG
                rgb[i, j] = 1 - abs(v) * (1 - base)
            else:
                rgb[i, j] = (0.93, 0.93, 0.93)
    axis.imshow(rgb, interpolation="nearest")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_title(f"クラス{k}", fontsize=7, color=INK, pad=1.5)
fig.savefig(OUT / "ten_heatmaps.png", dpi=300, facecolor="white")
print("ok")
