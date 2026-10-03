#!/usr/bin/env python3
"""Compare "X1 vs X2" class-contrast LIME with PCA-compressed LIME on real data.

For each black box, explanation target and neighborhood, the output side is
compressed to q axes in three ways that share the same center (the weighted
probability mean), the same perturbations and the same weighted ridge
surrogate, so only the choice of output directions differs:

* ``top_class``: contrasts among the q + 1 classes with the highest
  probability at the explained point. For q = 1 this is (e_X1 - e_X2)/sqrt(2),
  i.e. a LIME that explains p_X1 - p_X2.
* ``moving_class``: contrasts among the q + 1 classes whose probability varies
  most in the neighborhood. It uses the neighborhood, so it separates "which
  classes" from "which direction among them".
* ``pca``: weighted PCA of the neighborhood probabilities.

Fidelity is measured in the full class-probability space on held-out
perturbations. PCA maximizes the captured output variance on the fit set, so
it is expected to win on compression; the question is by how much, and in
which neighborhoods (low margin, several competing classes, top classes that
do not move).

Tabular data sets come from the PMLB GitHub mirror and are cached locally.
"""

import argparse
from pathlib import Path
import sys
import urllib.request
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.compare_pca_lime import select_margin_strata
from experiments.global_local_dimension_study import build_model
from experiments.unified_local_evaluation import add_strata, hierarchical_bootstrap
from fisher_lime.diagnostics import (
    NeighborhoodSampler,
    class_activity,
    deviation_from_reference,
    weighted_output_r2,
)
from fisher_lime.local_dimension import analyze_local_probabilities, fit_weighted_pca
from fisher_lime.output_projection import (
    LinearOutputProjection,
    axis_complexity,
    class_contrast_projection,
)
from fisher_lime.surrogate import fit_weighted_ridge, weighted_argmax_agreement

PMLB_URL = (
    "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/"
    "datasets/{name}/{name}.tsv.gz"
)
DATASET_CHOICES = [
    "letter",
    "yeast",
    "wine_quality_white",
    "digits",
    "satimage",
    "pendigits",
    "vowel",
    "segmentation",
]
MODEL_CHOICES = ["mlp", "rbf_svm", "logistic_regression", "random_forest"]
METHODS = ["top_class", "moving_class", "pca"]
METHOD_STYLE = {
    "top_class": ("#eb6834", "Top-class contrast (X1 vs X2)"),
    "moving_class": ("#1baf7a", "Most-moving-class contrast"),
    "pca": ("#2a78d6", "PCA"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["letter", "yeast", "wine_quality_white", "digits"],
        choices=DATASET_CHOICES,
    )
    parser.add_argument(
        "--models", nargs="+", default=["mlp", "rbf_svm"], choices=MODEL_CHOICES
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--targets-per-margin", type=int, default=10)
    parser.add_argument("--fit-perturbations", type=int, default=600)
    parser.add_argument("--eval-perturbations", type=int, default=600)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument(
        "--neighborhood", default="data", choices=["isotropic", "data"]
    )
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument(
        "--low-variation",
        type=float,
        default=1e-4,
        help="total weighted output variance below which a neighborhood is "
        "treated as near-constant and summarized separately",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--data-cache",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "data_cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "topclass_vs_pca",
    )
    return parser.parse_args()


def load_dataset(name: str, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if name == "digits":
        data = load_digits()
        x, y = data.data, data.target
    else:
        path = cache / f"{name}.tsv.gz"
        if not path.exists():
            cache.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(PMLB_URL.format(name=name), path)
        frame = pd.read_csv(path, sep="\t")
        x = frame.drop(columns="target").to_numpy(dtype=float)
        y = frame["target"].to_numpy()
    # Drop classes too small for a stratified split; relabel to 0..K-1.
    labels, counts = np.unique(y, return_counts=True)
    keep = np.isin(y, labels[counts >= 10])
    _, y = np.unique(y[keep], return_inverse=True)
    return x[keep], y


def pca_projection(
    probabilities: np.ndarray, weights: np.ndarray, dimension: int
) -> LinearOutputProjection:
    pca = fit_weighted_pca(probabilities, weights, dimension)
    return LinearOutputProjection(mean=pca.mean, directions=pca.components.T)


def captured_share(
    probabilities: np.ndarray,
    weights: np.ndarray,
    projection: LinearOutputProjection,
) -> float:
    """Share of weighted output variance kept by the projection."""

    reconstructed = projection.inverse_transform(projection.transform(probabilities))
    return weighted_output_r2(probabilities, reconstructed, weights)


def evaluate_configuration(
    dataset: str,
    model_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    x, y = load_dataset(dataset, args.data_cache)
    class_count = int(y.max() + 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed
    )
    model = build_model(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(x_train, y_train)
    test_probabilities = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, np.argmax(test_probabilities, axis=1))
    scaler = model.named_steps["standardscaler"]
    classifier = model.steps[-1][1]
    scaled_train = scaler.transform(x_train)
    scaled_test = scaler.transform(x_test)
    sampler = NeighborhoodSampler(args.neighborhood, reference=scaled_train)
    selected = select_margin_strata(test_probabilities, args.targets_per_margin, rng)

    neighborhood_rows: list[dict] = []
    method_rows: list[dict] = []
    for margin_group, target_indices in selected.items():
        for target_index in target_indices:
            target = scaled_test[target_index]
            target_probability = classifier.predict_proba(target[None, :])[0]
            top_classes = np.argsort(target_probability)[::-1]
            target_margin = float(
                target_probability[top_classes[0]] - target_probability[top_classes[1]]
            )
            for radius in args.radii:
                fit_x, fit_w = sampler.sample(target, args.fit_perturbations, radius, rng)
                eval_x, eval_w = sampler.sample(
                    target, args.eval_perturbations, radius, rng
                )
                fit_p = classifier.predict_proba(fit_x)
                eval_p = classifier.predict_proba(eval_x)
                fit_f = (fit_x - target) / radius
                eval_f = (eval_x - target) / radius
                zero_f = np.zeros((1, target.size))

                analysis = analyze_local_probabilities(fit_p, fit_w)
                activity = class_activity(fit_p, fit_w)
                deviation = deviation_from_reference(fit_p, fit_w, target_probability)
                normalized = fit_w / fit_w.sum()
                class_variance = np.sum(
                    normalized[:, None]
                    * (fit_p - normalized @ fit_p) ** 2,
                    axis=0,
                )
                moving_classes = np.argsort(class_variance)[::-1]
                total_variance = class_variance.sum()
                ordinary = fit_weighted_ridge(fit_f, fit_p, fit_w, args.ridge_alpha)
                ordinary_eval = ordinary.predict(eval_f)
                ordinary_r2 = weighted_output_r2(eval_p, ordinary_eval, eval_w)

                common = {
                    "dataset": dataset,
                    "classes": class_count,
                    "black_box": model_name,
                    "seed": seed,
                    "target_index": int(target_index),
                    "margin_group": margin_group,
                    "target_margin": target_margin,
                    "radius": radius,
                    "low_variation": bool(
                        analysis.variation_energy < args.low_variation
                    ),
                    "moving_classes": activity.moving_classes,
                    "output_q95": analysis.effective_dimension_95,
                    "top2_is_moving2": bool(
                        set(top_classes[:2]) == set(moving_classes[:2])
                    ),
                }
                neighborhood_rows.append(
                    {
                        **common,
                        "variation_energy": analysis.variation_energy,
                        "moving_classes_from_target": deviation.moving_classes,
                        "axes_from_target_q95": deviation.effective_dimension_95,
                        "deviation_energy": deviation.deviation_energy,
                        "top1_probability": float(target_probability[top_classes[0]]),
                        "top3_probability": float(target_probability[top_classes[2]]),
                        "top2_variance_share": (
                            float(class_variance[top_classes[:2]].sum() / total_variance)
                            if total_variance > 0
                            else np.nan
                        ),
                        "pca_axis1_share": float(analysis.explained_variance_ratio[0]),
                        "ordinary_r2": ordinary_r2,
                    }
                )

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
                        "pca": pca_projection(fit_p, fit_w, dimension),
                    }
                    for method, projection in projections.items():
                        scores = fit_weighted_ridge(
                            fit_f, projection.transform(fit_p), fit_w, args.ridge_alpha
                        )
                        surrogate_eval = projection.inverse_transform(
                            scores.predict(eval_f)
                        )
                        surrogate_target = projection.inverse_transform(
                            scores.predict(zero_f)
                        )[0]
                        compression_eval = projection.inverse_transform(
                            projection.transform(eval_p)
                        )
                        complexity = axis_complexity(projection.directions)
                        surrogate_r2 = weighted_output_r2(eval_p, surrogate_eval, eval_w)
                        method_rows.append(
                            {
                                **common,
                                "dimension": dimension,
                                "method": method,
                                "fit_captured_share": captured_share(
                                    fit_p, fit_w, projection
                                ),
                                "compression_r2": weighted_output_r2(
                                    eval_p, compression_eval, eval_w
                                ),
                                "surrogate_r2": surrogate_r2,
                                "ordinary_r2": ordinary_r2,
                                "r2_loss": ordinary_r2 - surrogate_r2,
                                "surrogate_argmax_agreement": weighted_argmax_agreement(
                                    eval_p, surrogate_eval, eval_w
                                ),
                                "target_error": float(
                                    np.linalg.norm(surrogate_target - target_probability)
                                ),
                                "axis_effective_classes": complexity[
                                    "effective_classes"
                                ],
                                "axis_top2_class_mass": complexity["top2_class_mass"],
                            }
                        )

    metadata = {
        "dataset": dataset,
        "classes": class_count,
        "features": x.shape[1],
        "samples": x.shape[0],
        "black_box": model_name,
        "seed": seed,
        "test_accuracy": accuracy,
    }
    return neighborhood_rows, method_rows, metadata


def moving_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values, bins=[0, 2, 3, np.inf], labels=["<=2", "3", ">=4"], right=True
    ).astype(str)


def summarize_methods(methods: pd.DataFrame) -> pd.DataFrame:
    frame = add_strata(methods)
    keys = ["dataset", "black_box", "radius", "stratum", "dimension", "method"]
    return (
        frame.groupby(keys, as_index=False)
        .agg(
            neighborhoods=("target_index", "count"),
            mean_fit_captured_share=("fit_captured_share", "mean"),
            mean_compression_r2=("compression_r2", "mean"),
            mean_surrogate_r2=("surrogate_r2", "mean"),
            mean_ordinary_r2=("ordinary_r2", "mean"),
            mean_r2_loss=("r2_loss", "mean"),
            mean_argmax_agreement=("surrogate_argmax_agreement", "mean"),
            mean_axis_effective_classes=("axis_effective_classes", "mean"),
            mean_axis_top2_class_mass=("axis_top2_class_mass", "mean"),
        )
    )


def paired_differences(
    methods: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """PCA minus each contrast method, per neighborhood, with breakdowns."""

    rng = np.random.default_rng(20261003)
    index = [
        "dataset", "black_box", "seed", "target_index", "radius", "dimension",
        "margin_group", "moving_classes", "top2_is_moving2", "low_variation",
    ]
    wide = methods.pivot_table(
        index=index,
        columns="method",
        values=["compression_r2", "surrogate_r2"],
    )
    wide.columns = [f"{value}_{method}" for value, method in wide.columns]
    wide = wide.reset_index()
    for baseline in ("top_class", "moving_class"):
        for value in ("compression_r2", "surrogate_r2"):
            wide[f"{value}_pca_minus_{baseline}"] = (
                wide[f"{value}_pca"] - wide[f"{value}_{baseline}"]
            )
    wide["moving_bucket"] = moving_bucket(wide["moving_classes"])
    wide = wide[~wide["low_variation"]]

    breakdowns = {
        "all": [],
        "margin": ["margin_group"],
        "moving": ["moving_bucket"],
        "top2_is_moving2": ["top2_is_moving2"],
    }
    rows = []
    base = ["dataset", "black_box", "radius", "dimension"]
    for breakdown, extra in breakdowns.items():
        for values, group in wide.groupby(base + extra, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            row = dict(zip(base + extra, values))
            row["breakdown"] = breakdown
            row["neighborhoods"] = group.shape[0]
            for column in (
                "surrogate_r2_top_class",
                "surrogate_r2_moving_class",
                "surrogate_r2_pca",
            ):
                row[f"mean_{column}"] = group[column].mean()
            for baseline in ("top_class", "moving_class"):
                column = f"surrogate_r2_pca_minus_{baseline}"
                row[f"mean_{column}"] = group[column].mean()
                row[f"median_{column}"] = group[column].median()
                row[f"share_{column}_over_0.05"] = (group[column] > 0.05).mean()
                low, high = hierarchical_bootstrap(
                    group, column, rng, args.bootstrap_repeats
                )
                row[f"{column}_ci_low"] = low
                row[f"{column}_ci_high"] = high
                compression = f"compression_r2_pca_minus_{baseline}"
                row[f"mean_{compression}"] = group[compression].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def plot_methods(summary: pd.DataFrame, output_path: Path) -> None:
    varying = summary[summary["stratum"] == "varying"]
    panels = varying[["dataset", "black_box"]].drop_duplicates().to_numpy()
    if len(panels) == 0:
        return
    columns = min(4, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.3 * rows),
        squeeze=False, sharey=True, constrained_layout=True,
    )
    for axis in axes.ravel()[len(panels):]:
        axis.set_visible(False)
    radius_style = {0.15: "-", 0.4: "--"}
    for axis, (dataset, black_box) in zip(axes.ravel(), panels):
        subset = varying[
            (varying["dataset"] == dataset) & (varying["black_box"] == black_box)
        ]
        for radius, by_radius in subset.groupby("radius"):
            style = radius_style.get(radius, ":")
            for method in METHODS:
                group = by_radius[by_radius["method"] == method].sort_values(
                    "dimension"
                )
                color, label = METHOD_STYLE[method]
                axis.plot(
                    group["dimension"], group["mean_surrogate_r2"], style,
                    color=color, marker="o", markersize=5, linewidth=2,
                    label=f"{label}, r={radius:g}",
                )
            axis.axhline(
                by_radius["mean_ordinary_r2"].mean(), linestyle=style,
                color="#898781", linewidth=1, label=f"Ordinary LIME, r={radius:g}",
            )
        axis.set_title(f"{dataset} / {black_box}", fontsize=10)
        axis.set_xlabel("Output axes q")
        axis.set_xticks(sorted(subset["dimension"].unique()))
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Held-out R² in probability space")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7)
    plt.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    neighborhood_rows: list[dict] = []
    method_rows: list[dict] = []
    metadata: list[dict] = []
    total = len(args.datasets) * len(args.models) * len(args.seeds)
    completed = 0
    for dataset in args.datasets:
        for model_name in args.models:
            for seed in args.seeds:
                n_rows, m_rows, meta = evaluate_configuration(
                    dataset, model_name, seed, args
                )
                neighborhood_rows.extend(n_rows)
                method_rows.extend(m_rows)
                metadata.append(meta)
                completed += 1
                print(
                    f"[{completed}/{total}] {dataset} (K={meta['classes']}) / "
                    f"{model_name} / seed={seed}: "
                    f"accuracy={meta['test_accuracy']:.3f}",
                    flush=True,
                )

    neighborhoods = pd.DataFrame(neighborhood_rows)
    methods = pd.DataFrame(method_rows)
    summary = summarize_methods(methods)
    paired = paired_differences(methods, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    neighborhoods.to_csv(args.output_dir / "neighborhoods.csv", index=False)
    methods.to_csv(args.output_dir / "method_evaluations.csv", index=False)
    summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output_dir / "model_metadata.csv", index=False)
    plot_methods(summary, args.output_dir / "surrogate_r2_by_method.png")

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        view = paired[(paired["breakdown"] == "all") & (paired["dimension"] == 1)]
        print("\nq=1, varying neighborhoods: held-out surrogate R² and PCA gain")
        print(
            view[
                [
                    "dataset", "black_box", "radius", "neighborhoods",
                    "mean_surrogate_r2_top_class", "mean_surrogate_r2_moving_class",
                    "mean_surrogate_r2_pca", "mean_surrogate_r2_pca_minus_top_class",
                    "surrogate_r2_pca_minus_top_class_ci_low",
                    "surrogate_r2_pca_minus_top_class_ci_high",
                    "share_surrogate_r2_pca_minus_top_class_over_0.05",
                ]
            ].round(3).to_string(index=False)
        )
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
