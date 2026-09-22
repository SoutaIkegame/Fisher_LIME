#!/usr/bin/env python3
"""Compare PCA, hard Fisher, and soft Fisher output compression for LIME."""

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.compare_pca_lime import (
    generate_neighborhood,
    make_black_box,
    select_margin_strata,
)
from fisher_lime.fisher_projection import (
    fit_fisher_projection,
    hard_memberships,
)
from fisher_lime.local_dimension import (
    analyze_local_probabilities,
    fit_weighted_pca,
)
from fisher_lime.surrogate import (
    fit_weighted_ridge,
    weighted_argmax_agreement,
    weighted_output_rmse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--targets-per-margin", type=int, default=15)
    parser.add_argument("--train-perturbations", type=int, default=1000)
    parser.add_argument("--test-perturbations", type=int, default=1000)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4, 0.8])
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--fisher-regularization", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "fisher_lime",
    )
    return parser.parse_args()


def projection_metrics(
    method: str,
    projection,
    train_features: np.ndarray,
    train_probabilities: np.ndarray,
    train_weights: np.ndarray,
    test_features: np.ndarray,
    test_probabilities: np.ndarray,
    test_weights: np.ndarray,
    ridge_alpha: float,
    input_feature_count: int,
    class_count: int,
) -> dict[str, float | int | str]:
    train_scores = projection.transform(train_probabilities)
    surrogate = fit_weighted_ridge(
        train_features, train_scores, train_weights, ridge_alpha
    )
    prediction = projection.inverse_transform(surrogate.predict(test_features))
    oracle = projection.inverse_transform(projection.transform(test_probabilities))
    dimension = train_scores.shape[1]
    return {
        "method": method,
        "dimension": dimension,
        "oracle_rmse": weighted_output_rmse(
            test_probabilities, oracle, test_weights
        ),
        "oracle_argmax_agreement": weighted_argmax_agreement(
            test_probabilities, oracle, test_weights
        ),
        "surrogate_rmse": weighted_output_rmse(
            test_probabilities, prediction, test_weights
        ),
        "surrogate_argmax_agreement": weighted_argmax_agreement(
            test_probabilities, prediction, test_weights
        ),
        "coefficient_count": dimension * (input_feature_count + class_count),
    }


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(args.seed)
    model, x_test, test_probabilities, accuracy = make_black_box(args)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["mlpclassifier"]
    x_test_scaled = scaler.transform(x_test)
    selected = select_margin_strata(test_probabilities, args.targets_per_margin, rng)
    rows: list[dict[str, float | int | str | bool]] = []

    for margin_group, indices in selected.items():
        for target_index in indices:
            target = x_test_scaled[target_index]
            sorted_probability = np.sort(test_probabilities[target_index])
            target_margin = float(sorted_probability[-1] - sorted_probability[-2])
            for radius in args.radii:
                train_x, train_weights = generate_neighborhood(
                    target, args.train_perturbations, radius, rng
                )
                test_x, test_weights = generate_neighborhood(
                    target, args.test_perturbations, radius, rng
                )
                train_p = classifier.predict_proba(train_x)
                test_p = classifier.predict_proba(test_x)
                train_features = (train_x - target) / radius
                test_features = (test_x - target) / radius
                hard_labels = np.argmax(train_p, axis=1)
                local_class_count = np.unique(hard_labels).size

                ordinary = fit_weighted_ridge(
                    train_features, train_p, train_weights, args.ridge_alpha
                )
                ordinary_prediction = ordinary.predict(test_features)
                common = {
                    "target_index": int(target_index),
                    "margin_group": margin_group,
                    "target_margin": target_margin,
                    "radius": radius,
                    "local_hard_class_count": local_class_count,
                }
                rows.append(
                    {
                        **common,
                        "method": "ordinary_lime",
                        "dimension": args.classes,
                        "available": True,
                        "oracle_rmse": 0.0,
                        "oracle_argmax_agreement": 1.0,
                        "surrogate_rmse": weighted_output_rmse(
                            test_p, ordinary_prediction, test_weights
                        ),
                        "surrogate_argmax_agreement": weighted_argmax_agreement(
                            test_p, ordinary_prediction, test_weights
                        ),
                        "coefficient_count": args.classes * args.features,
                    }
                )

                analysis = analyze_local_probabilities(train_p, train_weights)
                requested_dimension = max(analysis.effective_dimension_95, 1)
                pca = fit_weighted_pca(
                    train_p, train_weights, requested_dimension
                )
                rows.append(
                    {
                        **common,
                        "available": True,
                        **projection_metrics(
                            "pca_lime",
                            pca,
                            train_features,
                            train_p,
                            train_weights,
                            test_features,
                            test_p,
                            test_weights,
                            args.ridge_alpha,
                            args.features,
                            args.classes,
                        ),
                    }
                )

                memberships_by_method = {
                    "hard_fisher_lime": hard_memberships(
                        hard_labels, args.classes
                    ),
                    "soft_fisher_lime": train_p,
                }
                for method, memberships in memberships_by_method.items():
                    try:
                        projection = fit_fisher_projection(
                            train_p,
                            train_weights,
                            memberships,
                            requested_dimension,
                            regularization=args.fisher_regularization,
                        )
                        metrics = projection_metrics(
                            method,
                            projection,
                            train_features,
                            train_p,
                            train_weights,
                            test_features,
                            test_p,
                            test_weights,
                            args.ridge_alpha,
                            args.features,
                            args.classes,
                        )
                        rows.append({**common, "available": True, **metrics})
                    except ValueError:
                        rows.append(
                            {
                                **common,
                                "method": method,
                                "dimension": 0,
                                "available": False,
                                "oracle_rmse": np.nan,
                                "oracle_argmax_agreement": np.nan,
                                "surrogate_rmse": np.nan,
                                "surrogate_argmax_agreement": np.nan,
                                "coefficient_count": 0,
                            }
                        )
    return pd.DataFrame(rows), accuracy


def plot_results(summary: pd.DataFrame, output_path: Path) -> None:
    methods = ["pca_lime", "hard_fisher_lime", "soft_fisher_lime"]
    labels = ["PCA", "hard Fisher", "soft Fisher"]
    radii = sorted(summary["radius"].unique())
    figure, axes = plt.subplots(
        1, len(radii), figsize=(4.8 * len(radii), 4), sharey=True, squeeze=False
    )
    for axis, radius in zip(axes.ravel(), radii):
        subset = summary[summary["radius"] == radius]
        x = np.arange(3)
        width = 0.24
        for offset, margin_group in zip((-1, 0, 1), ("low", "medium", "high")):
            group = subset.set_index(["margin_group", "method"])
            values = [
                group.loc[(margin_group, method), "mean_surrogate_rmse"]
                if (margin_group, method) in group.index
                else np.nan
                for method in methods
            ]
            axis.bar(x + offset * width, values, width, label=margin_group)
        axis.set_xticks(x, labels, rotation=15)
        axis.set_title(f"radius={radius:g}")
        axis.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("Held-out weighted RMSE")
    axes.ravel()[-1].legend(title="Margin group")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    frame, accuracy = evaluate(args)
    ordinary = frame[frame["method"] == "ordinary_lime"].set_index(
        ["target_index", "radius"]
    )
    compressed = frame[frame["method"] != "ordinary_lime"].copy()
    compressed = compressed.join(
        ordinary[["surrogate_rmse", "surrogate_argmax_agreement"]].rename(
            columns={
                "surrogate_rmse": "ordinary_rmse",
                "surrogate_argmax_agreement": "ordinary_argmax_agreement",
            }
        ),
        on=["target_index", "radius"],
    )
    compressed["rmse_increase"] = (
        compressed["surrogate_rmse"] - compressed["ordinary_rmse"]
    )
    compressed["coefficient_reduction"] = 1.0 - compressed["coefficient_count"] / (
        args.classes * args.features
    )
    summary = (
        compressed.groupby(["margin_group", "radius", "method"], as_index=False)
        .agg(
            attempted=("target_index", "count"),
            available_rate=("available", "mean"),
            mean_dimension=("dimension", lambda x: x[x > 0].mean()),
            mean_oracle_rmse=("oracle_rmse", "mean"),
            mean_surrogate_rmse=("surrogate_rmse", "mean"),
            mean_ordinary_rmse=("ordinary_rmse", "mean"),
            mean_rmse_increase=("rmse_increase", "mean"),
            mean_argmax_agreement=("surrogate_argmax_agreement", "mean"),
            mean_ordinary_argmax_agreement=(
                "ordinary_argmax_agreement", "mean"
            ),
            mean_coefficient_reduction=("coefficient_reduction", lambda x: x[x < 1].mean()),
        )
        .sort_values(["radius", "margin_group", "method"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_evaluations.csv", index=False)
    compressed.to_csv(args.output_dir / "compressed_comparisons.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(
        [{**vars(args), "output_dir": str(args.output_dir), "test_accuracy": accuracy}]
    ).to_csv(args.output_dir / "metadata.csv", index=False)
    plot_results(summary, args.output_dir / "compression_comparison.png")
    print(f"Black-box test accuracy: {accuracy:.3f}")
    print(summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
