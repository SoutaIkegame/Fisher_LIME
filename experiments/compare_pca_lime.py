#!/usr/bin/env python3
"""Compare ordinary multiclass LIME with PCA-compressed output LIME."""

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
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "pca_lime",
    )
    return parser.parse_args()


def make_black_box(args: argparse.Namespace):
    informative = max(args.classes, int(args.features * 0.7))
    x, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=informative,
        n_redundant=args.features - informative,
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
    probabilities = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, np.argmax(probabilities, axis=1))
    return model, x_test, probabilities, accuracy


def select_margin_strata(
    probabilities: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    ordered_probabilities = np.sort(probabilities, axis=1)
    margins = ordered_probabilities[:, -1] - ordered_probabilities[:, -2]
    ordered_indices = np.argsort(margins)
    return {
        name: rng.choice(indices, size=count, replace=False)
        for name, indices in zip(
            ("low", "medium", "high"), np.array_split(ordered_indices, 3)
        )
    }


def generate_neighborhood(
    target: np.ndarray,
    count: int,
    radius: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_offsets = rng.normal(size=(count, target.size))
    points = target + radius * normalized_offsets
    squared_distance = np.sum(normalized_offsets**2, axis=1)
    kernel_width = np.sqrt(target.size) * 0.75
    weights = np.exp(-squared_distance / (2.0 * kernel_width**2))
    return points, weights


def evaluate(args: argparse.Namespace) -> tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(args.seed)
    model, x_test, test_probabilities, accuracy = make_black_box(args)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["mlpclassifier"]
    x_test_scaled = scaler.transform(x_test)
    selected = select_margin_strata(test_probabilities, args.targets_per_margin, rng)
    dimensions = sorted(set(args.dimensions))
    max_dimension = args.classes - 1
    if dimensions[0] < 1 or dimensions[-1] > max_dimension:
        raise ValueError(f"dimensions must be between 1 and {max_dimension}")

    rows: list[dict[str, float | int | str]] = []
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

                ordinary_model = fit_weighted_ridge(
                    train_features, train_p, train_weights, args.ridge_alpha
                )
                ordinary_prediction = ordinary_model.predict(test_features)
                ordinary_rmse = weighted_output_rmse(
                    test_p, ordinary_prediction, test_weights
                )
                ordinary_agreement = weighted_argmax_agreement(
                    test_p, ordinary_prediction, test_weights
                )
                rows.append(
                    {
                        "target_index": int(target_index),
                        "margin_group": margin_group,
                        "target_margin": target_margin,
                        "radius": radius,
                        "method": "ordinary_lime",
                        "dimension": args.classes,
                        "adaptive_95": False,
                        "oracle_rmse": 0.0,
                        "oracle_argmax_agreement": 1.0,
                        "surrogate_rmse": ordinary_rmse,
                        "surrogate_argmax_agreement": ordinary_agreement,
                        "coefficient_count": args.classes * args.features,
                    }
                )

                analysis = analyze_local_probabilities(train_p, train_weights)
                evaluated_dimensions = dimensions + [analysis.effective_dimension_95]
                for dimension in sorted(set(evaluated_dimensions)):
                    if dimension == 0:
                        continue
                    pca = fit_weighted_pca(train_p, train_weights, dimension)
                    train_scores = pca.transform(train_p)
                    score_model = fit_weighted_ridge(
                        train_features,
                        train_scores,
                        train_weights,
                        args.ridge_alpha,
                    )
                    predicted_scores = score_model.predict(test_features)
                    prediction = pca.inverse_transform(predicted_scores)
                    oracle_reconstruction = pca.inverse_transform(pca.transform(test_p))
                    rows.append(
                        {
                            "target_index": int(target_index),
                            "margin_group": margin_group,
                            "target_margin": target_margin,
                            "radius": radius,
                            "method": "pca_lime",
                            "dimension": dimension,
                            "adaptive_95": dimension
                            == analysis.effective_dimension_95,
                            "oracle_rmse": weighted_output_rmse(
                                test_p, oracle_reconstruction, test_weights
                            ),
                            "oracle_argmax_agreement": weighted_argmax_agreement(
                                test_p, oracle_reconstruction, test_weights
                            ),
                            "surrogate_rmse": weighted_output_rmse(
                                test_p, prediction, test_weights
                            ),
                            "surrogate_argmax_agreement": weighted_argmax_agreement(
                                test_p, prediction, test_weights
                            ),
                            # Input coefficients plus class loadings needed to read axes.
                            "coefficient_count": dimension
                            * (args.features + args.classes),
                        }
                    )
    return pd.DataFrame(rows), accuracy


def make_plot(summary: pd.DataFrame, output_path: Path) -> None:
    radii = sorted(summary["radius"].unique())
    figure, axes = plt.subplots(
        1, len(radii), figsize=(4.5 * len(radii), 4), sharey=True, squeeze=False
    )
    axes = axes.ravel()
    for axis, radius in zip(axes, radii):
        subset = summary[summary["radius"] == radius]
        for margin_group in ("low", "medium", "high"):
            group = subset[subset["margin_group"] == margin_group]
            (line,) = axis.plot(
                group["dimension"],
                group["mean_surrogate_rmse"],
                marker="o",
                label=margin_group,
            )
            baseline = group["mean_ordinary_rmse"].mean()
            axis.axhline(
                baseline,
                color=line.get_color(),
                linestyle="--",
                alpha=0.65,
            )
        axis.set_title(f"radius={radius:g}")
        axis.set_xlabel("PCA output dimension")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Held-out weighted RMSE")
    axes[-1].legend(title="PCA-LIME (dashed: ordinary)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    frame, accuracy = evaluate(args)
    ordinary = frame[frame["method"] == "ordinary_lime"]
    pca = frame[frame["method"] == "pca_lime"]
    ordinary_lookup = ordinary.set_index(["target_index", "radius"])[
        ["surrogate_rmse", "surrogate_argmax_agreement"]
    ].rename(
        columns={
            "surrogate_rmse": "ordinary_rmse",
            "surrogate_argmax_agreement": "ordinary_argmax_agreement",
        }
    )
    pca = pca.join(ordinary_lookup, on=["target_index", "radius"])
    pca["rmse_increase"] = pca["surrogate_rmse"] - pca["ordinary_rmse"]
    pca["coefficient_reduction"] = 1.0 - (
        pca["coefficient_count"] / (args.classes * args.features)
    )

    summary = (
        pca.groupby(["margin_group", "radius", "dimension"], as_index=False)
        .agg(
            target_count=("target_index", "count"),
            mean_oracle_rmse=("oracle_rmse", "mean"),
            mean_surrogate_rmse=("surrogate_rmse", "mean"),
            mean_ordinary_rmse=("ordinary_rmse", "mean"),
            mean_rmse_increase=("rmse_increase", "mean"),
            mean_pca_argmax_agreement=("surrogate_argmax_agreement", "mean"),
            mean_ordinary_argmax_agreement=(
                "ordinary_argmax_agreement",
                "mean",
            ),
            coefficient_reduction=("coefficient_reduction", "mean"),
        )
        .sort_values(["radius", "margin_group", "dimension"])
    )
    summary = summary[summary["dimension"].isin(args.dimensions)].reset_index(
        drop=True
    )
    adaptive = (
        pca[pca["adaptive_95"]]
        .groupby(["margin_group", "radius"], as_index=False)
        .agg(
            target_count=("target_index", "count"),
            mean_dimension=("dimension", "mean"),
            mean_oracle_rmse=("oracle_rmse", "mean"),
            mean_surrogate_rmse=("surrogate_rmse", "mean"),
            mean_ordinary_rmse=("ordinary_rmse", "mean"),
            mean_rmse_increase=("rmse_increase", "mean"),
            mean_pca_argmax_agreement=("surrogate_argmax_agreement", "mean"),
            mean_ordinary_argmax_agreement=(
                "ordinary_argmax_agreement",
                "mean",
            ),
            mean_coefficient_reduction=("coefficient_reduction", "mean"),
        )
        .sort_values(["radius", "margin_group"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_evaluations.csv", index=False)
    pca.to_csv(args.output_dir / "pca_comparisons.csv", index=False)
    summary.to_csv(args.output_dir / "dimension_summary.csv", index=False)
    adaptive.to_csv(args.output_dir / "adaptive_95_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": args.seed,
                "samples": args.samples,
                "features": args.features,
                "classes": args.classes,
                "targets_per_margin": args.targets_per_margin,
                "train_perturbations": args.train_perturbations,
                "test_perturbations": args.test_perturbations,
                "radii": ",".join(map(str, args.radii)),
                "dimensions": ",".join(map(str, args.dimensions)),
                "ridge_alpha": args.ridge_alpha,
                "test_accuracy": accuracy,
            }
        ]
    ).to_csv(args.output_dir / "metadata.csv", index=False)
    make_plot(summary, args.output_dir / "fidelity_by_dimension.png")

    print(f"Black-box test accuracy: {accuracy:.3f}")
    print("Adaptive 95% PCA-LIME versus ordinary LIME:")
    print(adaptive.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
