#!/usr/bin/env python3
"""Test whether multiclass black-box outputs are locally low-dimensional."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from fisher_lime.local_dimension import analyze_local_probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--targets-per-margin", type=int, default=30)
    parser.add_argument("--perturbations", type=int, default=1000)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4, 0.8])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "local_dimension",
    )
    return parser.parse_args()


def select_margin_strata(
    probabilities: np.ndarray,
    count_per_stratum: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    sorted_probabilities = np.sort(probabilities, axis=1)
    margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    order = np.argsort(margins)
    thirds = np.array_split(order, 3)
    names = ("low", "medium", "high")
    selected: dict[str, np.ndarray] = {}
    for name, candidates in zip(names, thirds):
        if count_per_stratum > candidates.size:
            raise ValueError(
                f"requested {count_per_stratum} targets from a stratum with "
                f"only {candidates.size} candidates"
            )
        selected[name] = rng.choice(candidates, size=count_per_stratum, replace=False)
    return selected


def plot_grouped_bars(
    summary: pd.DataFrame,
    value_column: str,
    ylabel: str,
    output_path: Path,
) -> None:
    pivot = summary.pivot(index="radius", columns="margin_group", values=value_column)
    pivot = pivot.reindex(columns=["low", "medium", "high"])
    ax = pivot.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_xlabel("Perturbation radius (standardized feature space)")
    ax.set_ylabel(ylabel)
    ax.legend(title="Prediction margin")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.classes < 3:
        raise ValueError("this multiclass experiment requires at least three classes")
    if args.features < args.classes:
        raise ValueError("features must be at least as large as classes")

    rng = np.random.default_rng(args.seed)
    x, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=max(args.classes, int(args.features * 0.7)),
        n_redundant=args.features - max(args.classes, int(args.features * 0.7)),
        n_classes=args.classes,
        n_clusters_per_class=1,
        class_sep=1.4,
        flip_y=0.02,
        random_state=args.seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=args.seed
    )
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            max_iter=400,
            early_stopping=True,
            random_state=args.seed,
        ),
    )
    model.fit(x_train, y_train)
    test_probabilities = model.predict_proba(x_test)
    test_accuracy = accuracy_score(y_test, np.argmax(test_probabilities, axis=1))

    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["mlpclassifier"]
    x_test_scaled = scaler.transform(x_test)
    selected = select_margin_strata(
        test_probabilities, args.targets_per_margin, rng
    )

    local_rows: list[dict[str, float | int | str]] = []
    curve_rows: list[dict[str, float | int | str]] = []

    for margin_group, indices in selected.items():
        for target_index in indices:
            target = x_test_scaled[target_index]
            target_probability = test_probabilities[target_index]
            sorted_probability = np.sort(target_probability)
            target_margin = float(sorted_probability[-1] - sorted_probability[-2])

            for radius in args.radii:
                perturbations = target + rng.normal(
                    loc=0.0,
                    scale=radius,
                    size=(args.perturbations, args.features),
                )
                squared_distance = np.sum(
                    ((perturbations - target) / radius) ** 2, axis=1
                )
                kernel_width = np.sqrt(args.features) * 0.75
                weights = np.exp(-squared_distance / (2.0 * kernel_width**2))
                local_probabilities = classifier.predict_proba(perturbations)
                result = analyze_local_probabilities(local_probabilities, weights)

                local_rows.append(
                    {
                        "target_index": int(target_index),
                        "margin_group": margin_group,
                        "target_margin": target_margin,
                        "radius": radius,
                        "effective_dimension_95": result.effective_dimension_95,
                        "effective_dimension_99": result.effective_dimension_99,
                        "numerical_rank": result.numerical_rank,
                        "variation_energy": result.variation_energy,
                        "hard_class_count": int(
                            np.unique(np.argmax(local_probabilities, axis=1)).size
                        ),
                        "top1_probability": float(target_probability.max()),
                    }
                )
                for dimension, (ratio, rmse, agreement) in enumerate(
                    zip(
                        result.explained_variance_ratio,
                        result.reconstruction_rmse,
                        result.argmax_agreement,
                    ),
                    start=1,
                ):
                    curve_rows.append(
                        {
                            "target_index": int(target_index),
                            "margin_group": margin_group,
                            "target_margin": target_margin,
                            "radius": radius,
                            "dimension": dimension,
                            "explained_variance_ratio": ratio,
                            "cumulative_explained_variance": float(
                                np.sum(result.explained_variance_ratio[:dimension])
                            ),
                            "reconstruction_rmse": rmse,
                            "argmax_agreement": agreement,
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_frame = pd.DataFrame(local_rows)
    curve_frame = pd.DataFrame(curve_rows)
    summary = (
        local_frame.groupby(["margin_group", "radius"], as_index=False)
        .agg(
            target_count=("target_index", "count"),
            mean_target_margin=("target_margin", "mean"),
            mean_effective_dimension_95=("effective_dimension_95", "mean"),
            median_effective_dimension_95=("effective_dimension_95", "median"),
            mean_effective_dimension_99=("effective_dimension_99", "mean"),
            mean_variation_energy=("variation_energy", "mean"),
            mean_hard_class_count=("hard_class_count", "mean"),
        )
        .sort_values(["radius", "margin_group"])
    )

    local_frame.to_csv(args.output_dir / "local_metrics.csv", index=False)
    curve_frame.to_csv(args.output_dir / "reconstruction_curves.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    plot_grouped_bars(
        summary,
        "mean_effective_dimension_95",
        "Mean effective output dimension (95%)",
        args.output_dir / "effective_dimension.png",
    )
    plot_grouped_bars(
        summary,
        "mean_variation_energy",
        "Mean weighted output variation",
        args.output_dir / "variation_energy.png",
    )

    metadata = pd.DataFrame(
        [
            {
                "seed": args.seed,
                "samples": args.samples,
                "features": args.features,
                "classes": args.classes,
                "targets_per_margin": args.targets_per_margin,
                "perturbations": args.perturbations,
                "radii": ",".join(map(str, args.radii)),
                "test_accuracy": test_accuracy,
            }
        ]
    )
    metadata.to_csv(args.output_dir / "metadata.csv", index=False)

    print(f"Black-box test accuracy: {test_accuracy:.3f}")
    print(summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()

