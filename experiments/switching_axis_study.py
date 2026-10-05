#!/usr/bin/env python3
"""Which output axes capture the black box's prediction switches?

For each explained point the predicted class c0 can switch to another class
inside the neighborhood. This study compares how five ways of choosing the q
output axes reproduce those switches, using the same perturbations and the
same weighted ridge surrogate (so only the choice of axes differs):

* ``top_class``: contrasts among the q + 1 most probable classes at the point.
* ``moving_class``: contrasts among the q + 1 classes with the
  largest squared deviation from the point's output.
* ``flip_class``: contrasts among c0 and the q classes the prediction most
  often switches to on the fit perturbations (falls back to deviation order).
* ``pca``: weighted PCA of the neighborhood probabilities.
* ``soft_fisher``: Fisher discriminant in probability space with the class
  probabilities as soft memberships (regularization 1.0 as in the repo's
  comparison experiments).

Switches are measured on held-out perturbations: ``flip_recall`` is the
weighted share of perturbations where the black box's prediction differs from
c0 and the surrogate predicts the same new class; ``argmax_agreement`` is over
all perturbations. ``axis1_flip_pair`` checks whether the two classes with the
largest absolute loading on axis 1 are c0 and the most frequent switch target.
"""

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.compare_pca_lime import select_margin_strata
from experiments.global_local_dimension_study import build_model
from experiments.topclass_vs_pca_study import load_dataset, pca_projection
from fisher_lime.diagnostics import NeighborhoodSampler, weighted_output_r2
from fisher_lime.fisher_projection import fit_fisher_projection
from fisher_lime.output_projection import axis_complexity, class_contrast_projection
from fisher_lime.surrogate import fit_weighted_ridge, weighted_argmax_agreement

METHODS = ["top_class", "moving_class", "flip_class", "pca", "soft_fisher"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["letter", "yeast", "wine_quality_white", "digits"]
    )
    parser.add_argument("--models", nargs="+", default=["mlp", "rbf_svm"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--targets-per-margin", type=int, default=10)
    parser.add_argument("--fit-perturbations", type=int, default=600)
    parser.add_argument("--eval-perturbations", type=int, default=600)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--fisher-regularization", type=float, default=1.0)
    parser.add_argument(
        "--min-flip-rate",
        type=float,
        default=0.05,
        help="neighborhoods whose held-out switch rate is below this are "
        "summarized separately (there is little switching to explain)",
    )
    parser.add_argument(
        "--data-cache", type=Path, default=REPOSITORY_ROOT / "result" / "data_cache"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "result" / "switching_axis"
    )
    return parser.parse_args()


def weighted_counts(labels: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(labels, weights=weights, minlength=size)


def flip_recall(expected: np.ndarray, predicted: np.ndarray, weights: np.ndarray, c0: int) -> float:
    true_label = np.argmax(expected, axis=1)
    flipped = true_label != c0
    if weights[flipped].sum() <= 0:
        return np.nan
    hit = np.argmax(predicted, axis=1) == true_label
    return float(np.sum(weights[flipped] * hit[flipped]) / weights[flipped].sum())


def evaluate_configuration(dataset: str, model_name: str, seed: int, args) -> list[dict]:
    rng = np.random.default_rng(seed)
    x, y = load_dataset(dataset, args.data_cache)
    class_count = int(y.max() + 1)
    x_train, x_test, y_train, _ = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed
    )
    model = build_model(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(x_train, y_train)
    scaler = model.named_steps["standardscaler"]
    classifier = model.steps[-1][1]
    scaled_test = scaler.transform(x_test)
    sampler = NeighborhoodSampler("data", reference=scaler.transform(x_train))
    test_probabilities = model.predict_proba(x_test)
    selected = select_margin_strata(test_probabilities, args.targets_per_margin, rng)

    rows: list[dict] = []
    for margin_group, target_indices in selected.items():
        for target_index in target_indices:
            target = scaled_test[target_index]
            p0 = classifier.predict_proba(target[None, :])[0]
            c0 = int(np.argmax(p0))
            top_classes = np.argsort(p0)[::-1]
            for radius in args.radii:
                fit_x, fit_w = sampler.sample(target, args.fit_perturbations, radius, rng)
                eval_x, eval_w = sampler.sample(target, args.eval_perturbations, radius, rng)
                fit_p = classifier.predict_proba(fit_x)
                eval_p = classifier.predict_proba(eval_x)
                fit_f = (fit_x - target) / radius
                eval_f = (eval_x - target) / radius
                fit_n = fit_w / fit_w.sum()
                eval_n = eval_w / eval_w.sum()

                deviation = np.sum(fit_n[:, None] * (fit_p - p0) ** 2, axis=0)
                moving_classes = np.argsort(deviation)[::-1]
                fit_flips = weighted_counts(np.argmax(fit_p, axis=1), fit_n, class_count)
                fit_flips[c0] = 0.0
                others = [k for k in range(class_count) if k != c0]
                # switch targets first (by frequency), then by deviation
                others.sort(key=lambda k: (-fit_flips[k], -deviation[k]))
                flip_classes = np.array([c0] + others)
                eval_labels = np.argmax(eval_p, axis=1)
                eval_flip_rate = float(np.sum(eval_n[eval_labels != c0]))
                main_target = int(others[0]) if fit_flips[others[0]] > 0 else -1

                ordinary = fit_weighted_ridge(fit_f, fit_p, fit_w, args.ridge_alpha)
                ordinary_eval = ordinary.predict(eval_f)
                common = {
                    "dataset": dataset,
                    "classes": class_count,
                    "black_box": model_name,
                    "seed": seed,
                    "target_index": int(target_index),
                    "margin_group": margin_group,
                    "radius": radius,
                    "fit_flip_rate": float(fit_flips.sum()),
                    "eval_flip_rate": eval_flip_rate,
                    "switch_targets": int(np.sum(fit_flips > 0)),
                    "top2_is_flip_pair": bool(main_target == top_classes[1]),
                    "ordinary_r2": weighted_output_r2(eval_p, ordinary_eval, eval_w),
                    "ordinary_flip_recall": flip_recall(eval_p, ordinary_eval, eval_n, c0),
                    "ordinary_argmax_agreement": weighted_argmax_agreement(
                        eval_p, ordinary_eval, eval_w
                    ),
                }

                for dimension in args.dimensions:
                    if dimension > class_count - 1:
                        continue
                    projections = {
                        "top_class": class_contrast_projection(
                            fit_p, fit_w, top_classes[: dimension + 1]
                        ),
                        "moving_class": class_contrast_projection(
                            fit_p, fit_w, moving_classes[: dimension + 1]
                        ),
                        "flip_class": class_contrast_projection(
                            fit_p, fit_w, flip_classes[: dimension + 1]
                        ),
                        "pca": pca_projection(fit_p, fit_w, dimension),
                    }
                    try:
                        projections["soft_fisher"] = fit_fisher_projection(
                            fit_p, fit_w, fit_p / fit_p.sum(axis=1, keepdims=True),
                            dimension, args.fisher_regularization,
                        )
                    except ValueError:
                        projections["soft_fisher"] = None
                    for method in METHODS:
                        projection = projections[method]
                        row = {**common, "dimension": dimension, "method": method}
                        if projection is None or projection.n_components < dimension:
                            row["available"] = False
                            rows.append(row)
                            continue
                        scores = fit_weighted_ridge(
                            fit_f, projection.transform(fit_p), fit_w, args.ridge_alpha
                        )
                        surrogate_eval = projection.inverse_transform(scores.predict(eval_f))
                        axis1 = np.abs(projection.directions[:, 0])
                        axis1_pair = set(np.argsort(axis1)[-2:].tolist())
                        row.update(
                            {
                                "available": True,
                                "surrogate_r2": weighted_output_r2(
                                    eval_p, surrogate_eval, eval_w
                                ),
                                "argmax_agreement": weighted_argmax_agreement(
                                    eval_p, surrogate_eval, eval_w
                                ),
                                "flip_recall": flip_recall(eval_p, surrogate_eval, eval_n, c0),
                                "axis1_flip_pair": (
                                    bool(axis1_pair == {c0, main_target})
                                    if main_target >= 0
                                    else np.nan
                                ),
                                "axis_effective_classes": axis_complexity(
                                    projection.directions
                                )["effective_classes"],
                            }
                        )
                        rows.append(row)
    return rows


def summarize(frame: pd.DataFrame, min_flip_rate: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["switching"] = np.where(
        frame["eval_flip_rate"] >= min_flip_rate, "switching", "few_switches"
    )
    keys = ["dataset", "black_box", "radius", "switching", "dimension", "method"]
    return frame.groupby(keys, as_index=False).agg(
        neighborhoods=("target_index", "count"),
        available_rate=("available", "mean"),
        mean_eval_flip_rate=("eval_flip_rate", "mean"),
        mean_switch_targets=("switch_targets", "mean"),
        top2_is_flip_pair=("top2_is_flip_pair", "mean"),
        mean_surrogate_r2=("surrogate_r2", "mean"),
        mean_ordinary_r2=("ordinary_r2", "mean"),
        mean_flip_recall=("flip_recall", "mean"),
        mean_ordinary_flip_recall=("ordinary_flip_recall", "mean"),
        mean_argmax_agreement=("argmax_agreement", "mean"),
        axis1_flip_pair=("axis1_flip_pair", "mean"),
        mean_axis_effective_classes=("axis_effective_classes", "mean"),
    )


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    total = len(args.datasets) * len(args.models) * len(args.seeds)
    completed = 0
    for dataset in args.datasets:
        for model_name in args.models:
            for seed in args.seeds:
                rows.extend(evaluate_configuration(dataset, model_name, seed, args))
                completed += 1
                print(f"[{completed}/{total}] {dataset} / {model_name} / seed={seed}", flush=True)
    frame = pd.DataFrame(rows)
    frame["axis1_flip_pair"] = frame["axis1_flip_pair"].astype(float)
    summary = summarize(frame, args.min_flip_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "method_evaluations.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        view = summary[(summary["switching"] == "switching") & (summary["dimension"] == 1)]
        print(
            view[
                [
                    "dataset", "black_box", "radius", "method", "neighborhoods",
                    "mean_eval_flip_rate", "top2_is_flip_pair", "mean_surrogate_r2",
                    "mean_flip_recall", "mean_ordinary_flip_recall", "axis1_flip_pair",
                ]
            ].round(3).to_string(index=False)
        )
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
