#!/usr/bin/env python3
"""Measure PCA-LIME fidelity as the compressed output dimension changes."""

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.compare_pca_lime import generate_neighborhood, select_margin_strata
from experiments.global_local_dimension_study import build_model
from fisher_lime.local_dimension import analyze_local_probabilities, fit_weighted_pca
from fisher_lime.surrogate import (
    fit_weighted_ridge,
    weighted_argmax_agreement,
    weighted_output_rmse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, nargs="+", default=[20])
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "logistic_regression",
            "mlp",
            "rbf_svm",
            "decision_tree",
            "bagged_trees",
            "random_forest",
            "gradient_boosting",
        ],
        choices=[
            "logistic_regression",
            "mlp",
            "rbf_svm",
            "decision_tree",
            "bagged_trees",
            "random_forest",
            "gradient_boosting",
        ],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--samples", type=int, default=9000)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument("--targets-per-margin", type=int, default=6)
    parser.add_argument("--train-perturbations", type=int, default=600)
    parser.add_argument("--test-perturbations", type=int, default=600)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15, 19]
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "pca_fidelity_tradeoff",
    )
    return parser.parse_args()


def weighted_output_r2(
    expected: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> float:
    weights = np.asarray(weights, dtype=float)
    normalized = weights / weights.sum()
    expected = np.asarray(expected, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mean = np.sum(normalized[:, None] * expected, axis=0)
    residual = np.sum(normalized * np.sum((expected - predicted) ** 2, axis=1))
    total = np.sum(normalized * np.sum((expected - mean) ** 2, axis=1))
    if total <= np.finfo(float).eps:
        return 1.0 if residual <= np.finfo(float).eps else np.nan
    return float(1.0 - residual / total)


def bootstrap_mean_interval(
    values: np.ndarray, rng: np.random.Generator, repeats: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def evaluate_configuration(
    class_count: int,
    model_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    if class_count - 1 > args.features:
        raise ValueError("features must be at least classes - 1")
    rng = np.random.default_rng(seed)
    informative = min(args.features - 2, max(12, class_count + 4))
    x, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=informative,
        n_redundant=args.features - informative,
        n_classes=class_count,
        n_clusters_per_class=1,
        class_sep=1.4,
        flip_y=0.02,
        random_state=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed
    )
    model = build_model(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    test_probabilities = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, np.argmax(test_probabilities, axis=1))
    scaler = model.named_steps["standardscaler"]
    classifier = model.steps[-1][1]
    scaled_test = scaler.transform(x_test)
    selected = select_margin_strata(
        test_probabilities, args.targets_per_margin, rng
    )
    fixed_dimensions = sorted(
        {dimension for dimension in args.dimensions if 1 <= dimension < class_count}
    )
    rows: list[dict] = []

    for margin_group, target_indices in selected.items():
        for target_index in target_indices:
            target = scaled_test[target_index]
            sorted_target_probability = np.sort(test_probabilities[target_index])
            target_margin = float(
                sorted_target_probability[-1] - sorted_target_probability[-2]
            )
            for radius in args.radii:
                train_x, train_weights = generate_neighborhood(
                    target, args.train_perturbations, radius, rng
                )
                evaluation_x, evaluation_weights = generate_neighborhood(
                    target, args.test_perturbations, radius, rng
                )
                train_probabilities = classifier.predict_proba(train_x)
                evaluation_probabilities = classifier.predict_proba(evaluation_x)
                train_features = (train_x - target) / radius
                evaluation_features = (evaluation_x - target) / radius

                ordinary = fit_weighted_ridge(
                    train_features,
                    train_probabilities,
                    train_weights,
                    args.ridge_alpha,
                )
                ordinary_prediction = ordinary.predict(evaluation_features)
                ordinary_rmse = weighted_output_rmse(
                    evaluation_probabilities,
                    ordinary_prediction,
                    evaluation_weights,
                )
                ordinary_r2 = weighted_output_r2(
                    evaluation_probabilities,
                    ordinary_prediction,
                    evaluation_weights,
                )
                ordinary_agreement = weighted_argmax_agreement(
                    evaluation_probabilities,
                    ordinary_prediction,
                    evaluation_weights,
                )
                analysis = analyze_local_probabilities(
                    train_probabilities, train_weights
                )
                adaptive_dimension = max(analysis.effective_dimension_95, 1)
                dimensions = sorted(set(fixed_dimensions + [adaptive_dimension]))

                common = {
                    "classes": class_count,
                    "features": args.features,
                    "black_box": model_name,
                    "seed": seed,
                    "target_index": int(target_index),
                    "margin_group": margin_group,
                    "target_margin": target_margin,
                    "radius": radius,
                    "adaptive_dimension_95": adaptive_dimension,
                    "local_variation_energy": analysis.variation_energy,
                    "ordinary_rmse": ordinary_rmse,
                    "ordinary_r2": ordinary_r2,
                    "ordinary_argmax_agreement": ordinary_agreement,
                }
                for dimension in dimensions:
                    pca = fit_weighted_pca(
                        train_probabilities, train_weights, dimension
                    )
                    score_model = fit_weighted_ridge(
                        train_features,
                        pca.transform(train_probabilities),
                        train_weights,
                        args.ridge_alpha,
                    )
                    oracle_prediction = pca.inverse_transform(
                        pca.transform(evaluation_probabilities)
                    )
                    surrogate_prediction = pca.inverse_transform(
                        score_model.predict(evaluation_features)
                    )
                    coefficient_count = dimension * (args.features + class_count)
                    ordinary_coefficient_count = args.features * class_count
                    surrogate_rmse = weighted_output_rmse(
                        evaluation_probabilities,
                        surrogate_prediction,
                        evaluation_weights,
                    )
                    surrogate_r2 = weighted_output_r2(
                        evaluation_probabilities,
                        surrogate_prediction,
                        evaluation_weights,
                    )
                    rows.append(
                        {
                            **common,
                            "dimension": dimension,
                            "fixed_dimension": dimension in fixed_dimensions,
                            "adaptive_95": dimension == adaptive_dimension,
                            "oracle_rmse": weighted_output_rmse(
                                evaluation_probabilities,
                                oracle_prediction,
                                evaluation_weights,
                            ),
                            "oracle_r2": weighted_output_r2(
                                evaluation_probabilities,
                                oracle_prediction,
                                evaluation_weights,
                            ),
                            "oracle_argmax_agreement": weighted_argmax_agreement(
                                evaluation_probabilities,
                                oracle_prediction,
                                evaluation_weights,
                            ),
                            "surrogate_rmse": surrogate_rmse,
                            "surrogate_r2": surrogate_r2,
                            "surrogate_argmax_agreement": weighted_argmax_agreement(
                                evaluation_probabilities,
                                surrogate_prediction,
                                evaluation_weights,
                            ),
                            "rmse_increase": surrogate_rmse - ordinary_rmse,
                            "r2_change": surrogate_r2 - ordinary_r2,
                            "coefficient_count": coefficient_count,
                            "coefficient_reduction": (
                                1.0
                                - coefficient_count / ordinary_coefficient_count
                            ),
                        }
                    )

    metadata = {
        "classes": class_count,
        "black_box": model_name,
        "seed": seed,
        "samples": args.samples,
        "features": args.features,
        "informative_features": informative,
        "test_accuracy": accuracy,
    }
    return rows, metadata


def summarize(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(20260922)
    rows = []
    group_columns = ["classes", "black_box", "radius", "dimension"]
    for keys, group in frame.groupby(group_columns, sort=True):
        rmse_low, rmse_high = bootstrap_mean_interval(
            group["rmse_increase"].to_numpy(), rng, args.bootstrap_repeats
        )
        r2_low, r2_high = bootstrap_mean_interval(
            group["r2_change"].to_numpy(), rng, args.bootstrap_repeats
        )
        mean_ordinary_rmse = group["ordinary_rmse"].mean()
        mean_rmse_increase = group["rmse_increase"].mean()
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "neighborhood_count": group.shape[0],
                "mean_adaptive_dimension_95": group[
                    "adaptive_dimension_95"
                ].mean(),
                "mean_oracle_rmse": group["oracle_rmse"].mean(),
                "mean_oracle_r2": group["oracle_r2"].mean(),
                "mean_oracle_argmax_agreement": group[
                    "oracle_argmax_agreement"
                ].mean(),
                "mean_surrogate_rmse": group["surrogate_rmse"].mean(),
                "mean_ordinary_rmse": mean_ordinary_rmse,
                "mean_rmse_increase": mean_rmse_increase,
                "relative_rmse_increase": (
                    mean_rmse_increase / mean_ordinary_rmse
                    if mean_ordinary_rmse > 0
                    else np.nan
                ),
                "rmse_increase_ci_low": rmse_low,
                "rmse_increase_ci_high": rmse_high,
                "mean_surrogate_r2": group["surrogate_r2"].mean(),
                "mean_ordinary_r2": group["ordinary_r2"].mean(),
                "mean_r2_change": group["r2_change"].mean(),
                "r2_change_ci_low": r2_low,
                "r2_change_ci_high": r2_high,
                "mean_surrogate_argmax_agreement": group[
                    "surrogate_argmax_agreement"
                ].mean(),
                "mean_ordinary_argmax_agreement": group[
                    "ordinary_argmax_agreement"
                ].mean(),
                "mean_argmax_change": (
                    group["surrogate_argmax_agreement"]
                    - group["ordinary_argmax_agreement"]
                ).mean(),
                "coefficient_reduction": group["coefficient_reduction"].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_tradeoff(summary: pd.DataFrame, output_path: Path) -> None:
    models = list(summary["black_box"].unique())
    radii = sorted(summary["radius"].unique())
    figure, axes = plt.subplots(
        len(models),
        len(radii),
        figsize=(5.2 * len(radii), 4.1 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, model_name in enumerate(models):
        for column, radius in enumerate(radii):
            axis = axes[row, column]
            group = summary[
                (summary["black_box"] == model_name)
                & (summary["radius"] == radius)
            ].sort_values("dimension")
            axis.errorbar(
                group["dimension"],
                group["mean_rmse_increase"],
                yerr=np.vstack(
                    [
                        group["mean_rmse_increase"]
                        - group["rmse_increase_ci_low"],
                        group["rmse_increase_ci_high"]
                        - group["mean_rmse_increase"],
                    ]
                ),
                marker="o",
                capsize=3,
            )
            axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(f"{model_name}, radius={radius:g}")
            axis.set_xlabel("PCA output axes")
            axis.set_ylabel("RMSE increase vs ordinary LIME")
            axis.grid(alpha=0.25)
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_model_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    model_order = [
        "logistic_regression",
        "mlp",
        "rbf_svm",
        "decision_tree",
        "bagged_trees",
        "random_forest",
        "gradient_boosting",
    ]
    labels = ["Logistic", "MLP", "RBF-SVM", "Tree", "Bagging", "RF", "Boosting"]
    dimensions = [5, 8, 10]
    radii = sorted(summary["radius"].unique())
    figure, axes = plt.subplots(
        1,
        len(radii),
        figsize=(7.2 * len(radii), 4.5),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    width = 0.24
    x = np.arange(len(model_order))
    for axis, radius in zip(axes.ravel(), radii):
        subset = summary[summary["radius"] == radius]
        for offset, dimension in zip((-1, 0, 1), dimensions):
            indexed = subset[subset["dimension"] == dimension].set_index("black_box")
            values = [
                100 * indexed.loc[model, "relative_rmse_increase"]
                if model in indexed.index
                else np.nan
                for model in model_order
            ]
            axis.bar(x + offset * width, values, width, label=f"q={dimension}")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(f"radius={radius:g}")
        axis.set_xlabel("Black-box model")
        axis.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("RMSE increase vs ordinary LIME (%)")
    axes.ravel()[-1].legend(title="PCA axes")
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = []
    metadata = []
    total = len(args.classes) * len(args.models) * len(args.seeds)
    completed = 0
    for class_count in args.classes:
        for model_name in args.models:
            for seed in args.seeds:
                configuration_rows, configuration_metadata = evaluate_configuration(
                    class_count, model_name, seed, args
                )
                rows.extend(configuration_rows)
                metadata.append(configuration_metadata)
                completed += 1
                print(
                    f"[{completed}/{total}] K={class_count} / {model_name} / seed={seed}: "
                    f"accuracy={configuration_metadata['test_accuracy']:.3f}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    summary = summarize(frame[frame["fixed_dimension"]], args)
    adaptive = frame[frame["adaptive_95"]].copy()
    adaptive_summary = (
        adaptive.groupby(["classes", "black_box", "radius"], as_index=False)
        .agg(
            neighborhood_count=("target_index", "count"),
            mean_dimension=("dimension", "mean"),
            mean_oracle_rmse=("oracle_rmse", "mean"),
            mean_surrogate_rmse=("surrogate_rmse", "mean"),
            mean_ordinary_rmse=("ordinary_rmse", "mean"),
            mean_rmse_increase=("rmse_increase", "mean"),
            mean_surrogate_r2=("surrogate_r2", "mean"),
            mean_ordinary_r2=("ordinary_r2", "mean"),
            mean_surrogate_argmax_agreement=(
                "surrogate_argmax_agreement",
                "mean",
            ),
            mean_ordinary_argmax_agreement=(
                "ordinary_argmax_agreement",
                "mean",
            ),
            mean_coefficient_reduction=("coefficient_reduction", "mean"),
        )
    )
    adaptive_summary["relative_rmse_increase"] = (
        adaptive_summary["mean_rmse_increase"]
        / adaptive_summary["mean_ordinary_rmse"]
    )
    adaptive_summary["mean_argmax_change"] = (
        adaptive_summary["mean_surrogate_argmax_agreement"]
        - adaptive_summary["mean_ordinary_argmax_agreement"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_evaluations.csv", index=False)
    summary.to_csv(args.output_dir / "dimension_summary.csv", index=False)
    adaptive_summary.to_csv(args.output_dir / "adaptive_95_summary.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output_dir / "model_metadata.csv", index=False)
    plot_tradeoff(summary, args.output_dir / "fidelity_tradeoff.png")
    plot_model_comparison(summary, args.output_dir / "model_comparison.png")

    print("\nFidelity by fixed dimension:")
    print(
        summary[
            [
                "classes",
                "black_box",
                "radius",
                "dimension",
                "mean_oracle_rmse",
                "mean_rmse_increase",
                "mean_surrogate_r2",
                "mean_surrogate_argmax_agreement",
                "coefficient_reduction",
            ]
        ].to_string(index=False)
    )
    print("\nAdaptive 95% dimension:")
    print(adaptive_summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
