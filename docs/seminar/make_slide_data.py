#!/usr/bin/env python3
"""Write docs/seminar/slide_data.json for build_slides.js.

* digit_image: a real digits test image (true 6, MLP predicts 6 with 0 second).
* deviation: per-class share of the squared deviation from the explained
  point's output, for the first low-margin digits point with 3 moving classes.
"""

import json
from pathlib import Path
import sys
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.global_local_dimension_study import build_model  # noqa: E402
from fisher_lime.diagnostics import NeighborhoodSampler, deviation_from_reference  # noqa: E402

warnings.filterwarnings("ignore")
data = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, stratify=data.target, random_state=11
)
model = build_model("mlp", 11)
model.fit(x_train, y_train)
probabilities = model.predict_proba(x_test)
order = np.argsort(probabilities, axis=1)[:, ::-1]
image_index = next(i for i in range(len(y_test)) if order[i, 0] == 6 and order[i, 1] == 0)

scaler = model.named_steps["standardscaler"]
classifier = model.steps[-1][1]
sampler = NeighborhoodSampler("data", reference=scaler.transform(x_train))
rng = np.random.default_rng(0)
ordered = np.sort(probabilities, axis=1)
deviation = None
for index in np.argsort(ordered[:, -1] - ordered[:, -2])[:60]:
    target = scaler.transform(x_test[index : index + 1])[0]
    reference = classifier.predict_proba(target[None])[0]
    points, weights = sampler.sample(target, 600, 0.15, rng)
    perturbed = classifier.predict_proba(points)
    result = deviation_from_reference(perturbed, weights, reference)
    normalized = weights / weights.sum()
    energy = np.sum(normalized[:, None] * (perturbed - reference) ** 2, axis=0)
    if result.moving_classes == 3:
        deviation = {
            "index": int(index),
            "label": int(y_test[index]),
            "probs": reference.round(3).tolist(),
            "share": (energy / energy.sum()).round(4).tolist(),
            "moving": result.moving_classes,
            "axes": result.effective_dimension_95,
        }
        break

out = {
    "digit_image": (x_test[image_index] / 16).round(3).reshape(8, 8).tolist(),
    "digit_label": int(y_test[image_index]),
    "digit_probs": probabilities[image_index].round(3).tolist(),
    "deviation": deviation,
}
Path(__file__).with_name("slide_data.json").write_text(json.dumps(out))
print(deviation)
