#!/usr/bin/env python3
"""Compare global and local effective dimensions of multiclass BB outputs."""

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
from sklearn.ensemble import (
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fisher_lime.local_dimension import analyze_local_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "random_forest"],
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
    parser.add_argument("--evaluation-fraction", type=float, default=0.4)
    parser.add_argument("--targets-per-margin", type=int, default=10)
    parser.add_argument(
        "--neighborhood-fractions",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1, 0.25, 1.0],
    )
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "global_local_dimension",
    )
    return parser.parse_args()


def select_margin_strata(
    probabilities: np.ndarray,
    count_per_stratum: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    ordered = np.argsort(margins)
    selected = {}
    for name, candidates in zip(("low", "medium", "high"), np.array_split(ordered, 3)):
        if count_per_stratum > candidates.size:
            raise ValueError(
                f"requested {count_per_stratum} targets from only {candidates.size} candidates"
            )
        selected[name] = rng.choice(candidates, count_per_stratum, replace=False)
    return selected


def build_model(name: str, seed: int):
    if name == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
        )
    elif name == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            max_iter=350,
            early_stopping=True,
            random_state=seed,
        )
    elif name == "rbf_svm":
        classifier = SVC(
            C=2.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            random_state=seed,
        )
    elif name == "decision_tree":
        classifier = DecisionTreeClassifier(
            min_samples_leaf=2,
            random_state=seed,
        )
    elif name == "bagged_trees":
        classifier = BaggingClassifier(
            estimator=DecisionTreeClassifier(min_samples_leaf=2),
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    elif name == "gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            max_iter=150,
            l2_regularization=1e-3,
            early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    return make_pipeline(StandardScaler(), classifier)


def participation_ratio(explained_variance_ratio: np.ndarray) -> float:
    squared_sum = float(np.sum(np.asarray(explained_variance_ratio) ** 2))
    return 0.0 if squared_sum == 0.0 else 1.0 / squared_sum


def dimension_metrics(probabilities: np.ndarray) -> dict[str, float | int]:
    result = analyze_local_probabilities(
        probabilities, np.ones(probabilities.shape[0], dtype=float)
    )
    return {
        "q95": result.effective_dimension_95,
        "q99": result.effective_dimension_99,
        "participation_ratio": participation_ratio(result.explained_variance_ratio),
        "variation_energy": result.variation_energy,
        "hard_class_count": int(np.unique(np.argmax(probabilities, axis=1)).size),
    }


def evaluate_configuration(
    class_count: int,
    model_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    if class_count < 3:
        raise ValueError("at least three classes are required")
    if class_count - 1 > args.features:
        raise ValueError(
            "features must be at least classes - 1 so input dimension does not cap output rank"
        )

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
    x_train, x_eval, y_train, y_eval = train_test_split(
        x,
        y,
        test_size=args.evaluation_fraction,
        stratify=y,
        random_state=seed,
    )
    model = build_model(model_name, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_eval)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_eval, predictions)
    scaled_eval = model.named_steps["standardscaler"].transform(x_eval)
    global_metrics = dimension_metrics(probabilities)
    selected = select_margin_strata(
        probabilities, args.targets_per_margin, rng
    )
    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    rows: list[dict] = []
    for margin_group, target_indices in selected.items():
        for target_index in target_indices:
            squared_distances = np.sum(
                (scaled_eval - scaled_eval[target_index]) ** 2, axis=1
            )
            neighbor_order = np.argsort(squared_distances)
            for fraction in args.neighborhood_fractions:
                neighbor_count = min(
                    probabilities.shape[0],
                    max(class_count + 1, int(np.ceil(fraction * probabilities.shape[0]))),
                )
                if neighbor_count == probabilities.shape[0]:
                    local_metrics = global_metrics
                else:
                    local_indices = neighbor_order[:neighbor_count]
                    local_metrics = dimension_metrics(probabilities[local_indices])

                random_metrics = []
                if neighbor_count == probabilities.shape[0]:
                    random_metrics.append(global_metrics)
                else:
                    for _ in range(args.random_repeats):
                        random_indices = rng.choice(
                            probabilities.shape[0], neighbor_count, replace=False
                        )
                        random_metrics.append(
                            dimension_metrics(probabilities[random_indices])
                        )

                random_means = {
                    key: float(np.mean([metrics[key] for metrics in random_metrics]))
                    for key in random_metrics[0]
                }
                rows.append(
                    {
                        "classes": class_count,
                        "maximum_output_dimension": class_count - 1,
                        "black_box": model_name,
                        "seed": seed,
                        "target_index": int(target_index),
                        "margin_group": margin_group,
                        "target_margin": float(margins[target_index]),
                        "neighborhood_fraction": fraction,
                        "neighbor_count": neighbor_count,
                        **{f"local_{key}": value for key, value in local_metrics.items()},
                        **{f"random_{key}": value for key, value in random_means.items()},
                        **{f"global_{key}": value for key, value in global_metrics.items()},
                        "local_global_q95_ratio": (
                            local_metrics["q95"] / global_metrics["q95"]
                            if global_metrics["q95"] > 0
                            else np.nan
                        ),
                        "local_random_q95_difference": (
                            local_metrics["q95"] - random_means["q95"]
                        ),
                        "local_global_variation_ratio": (
                            local_metrics["variation_energy"]
                            / global_metrics["variation_energy"]
                            if global_metrics["variation_energy"] > 0
                            else np.nan
                        ),
                    }
                )

    metadata = {
        "classes": class_count,
        "black_box": model_name,
        "seed": seed,
        "samples": args.samples,
        "training_samples": x_train.shape[0],
        "evaluation_samples": x_eval.shape[0],
        "features": args.features,
        "informative_features": informative,
        "test_accuracy": accuracy,
        **{f"global_{key}": value for key, value in global_metrics.items()},
    }
    return rows, metadata


def bootstrap_mean_interval(
    values: np.ndarray, rng: np.random.Generator, repeats: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(20260922)
    rows = []
    group_columns = [
        "classes",
        "black_box",
        "margin_group",
        "neighborhood_fraction",
    ]
    for keys, group in frame.groupby(group_columns, sort=True):
        difference = group["local_random_q95_difference"].to_numpy()
        ci_low, ci_high = bootstrap_mean_interval(
            difference, rng, args.bootstrap_repeats
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "target_count": group.shape[0],
                "mean_neighbor_count": group["neighbor_count"].mean(),
                "mean_global_q95": group["global_q95"].mean(),
                "mean_local_q95": group["local_q95"].mean(),
                "mean_random_q95": group["random_q95"].mean(),
                "mean_local_global_q95_ratio": group[
                    "local_global_q95_ratio"
                ].mean(),
                "mean_local_random_q95_difference": difference.mean(),
                "difference_ci_low": ci_low,
                "difference_ci_high": ci_high,
                "mean_local_participation_ratio": group[
                    "local_participation_ratio"
                ].mean(),
                "mean_random_participation_ratio": group[
                    "random_participation_ratio"
                ].mean(),
                "mean_local_global_variation_ratio": group[
                    "local_global_variation_ratio"
                ].mean(),
                "mean_local_hard_class_count": group[
                    "local_hard_class_count"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_dimension_curves(summary: pd.DataFrame, output_path: Path) -> None:
    class_counts = sorted(summary["classes"].unique())
    models = list(summary["black_box"].unique())
    figure, axes = plt.subplots(
        len(models), len(class_counts),
        figsize=(4.4 * len(class_counts), 3.8 * len(models)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    colors = {"low": "tab:red", "medium": "tab:orange", "high": "tab:blue"}
    for row, model_name in enumerate(models):
        for column, class_count in enumerate(class_counts):
            axis = axes[row, column]
            subset = summary[
                (summary["black_box"] == model_name)
                & (summary["classes"] == class_count)
            ]
            for margin_group, group in subset.groupby("margin_group"):
                group = group.sort_values("neighborhood_fraction")
                axis.plot(
                    100 * group["neighborhood_fraction"],
                    group["mean_local_global_q95_ratio"],
                    marker="o",
                    label=margin_group,
                    color=colors[margin_group],
                )
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(f"{model_name}, K={class_count}")
            axis.grid(alpha=0.25)
            if row == len(models) - 1:
                axis.set_xlabel("Nearest evaluation data used (%)")
            if column == 0:
                axis.set_ylabel("Local q95 / global q95")
    axes[0, -1].legend(title="Prediction margin")
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_local_random(summary: pd.DataFrame, output_path: Path) -> None:
    plot_frame = (
        summary.groupby(
            ["classes", "black_box", "neighborhood_fraction"], as_index=False
        )
        .agg(
            mean_local_q95=("mean_local_q95", "mean"),
            mean_random_q95=("mean_random_q95", "mean"),
        )
    )
    class_counts = sorted(plot_frame["classes"].unique())
    models = list(plot_frame["black_box"].unique())
    figure, axes = plt.subplots(
        len(models), len(class_counts),
        figsize=(4.4 * len(class_counts), 4.2 * len(models)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row, model_name in enumerate(models):
        for column, class_count in enumerate(class_counts):
            axis = axes[row, column]
            group = plot_frame[
                (plot_frame["black_box"] == model_name)
                & (plot_frame["classes"] == class_count)
            ].sort_values("neighborhood_fraction")
            x = 100 * group["neighborhood_fraction"]
            axis.plot(x, group["mean_local_q95"], marker="o", label="nearest")
            axis.plot(x, group["mean_random_q95"], marker="s", label="random")
            axis.set_title(f"{model_name}, K={class_count}")
            axis.grid(alpha=0.25)
            if row == len(models) - 1:
                axis.set_xlabel("Evaluation data used (%)")
            if column == 0:
                axis.set_ylabel("Effective output dimension q95")
    axes[0, -1].legend()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not 0 < args.evaluation_fraction < 1:
        raise ValueError("evaluation-fraction must be between 0 and 1")
    if any(not 0 < fraction <= 1 for fraction in args.neighborhood_fractions):
        raise ValueError("neighborhood fractions must be between 0 and 1")
    if args.random_repeats < 1:
        raise ValueError("random-repeats must be positive")

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
                    f"accuracy={configuration_metadata['test_accuracy']:.3f}, "
                    f"global_q95={configuration_metadata['global_q95']}",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    summary = summarize(frame, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_evaluations.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output_dir / "model_metadata.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_dimension_curves(summary, args.output_dir / "local_global_dimension_curve.png")
    plot_local_random(summary, args.output_dir / "local_vs_random_dimension.png")

    compact = (
        summary.groupby(
            ["classes", "black_box", "neighborhood_fraction"], as_index=False
        )
        .agg(
            global_q95=("mean_global_q95", "mean"),
            local_q95=("mean_local_q95", "mean"),
            random_q95=("mean_random_q95", "mean"),
            local_global_ratio=("mean_local_global_q95_ratio", "mean"),
            local_random_difference=("mean_local_random_q95_difference", "mean"),
            variation_ratio=("mean_local_global_variation_ratio", "mean"),
        )
    )
    compact.to_csv(args.output_dir / "compact_summary.csv", index=False)
    print("\nLocal versus global dimension summary:")
    print(compact.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
