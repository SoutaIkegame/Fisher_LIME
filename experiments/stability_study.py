#!/usr/bin/env python3
"""Measure explanation stability across repeated local perturbation samples."""

import argparse
from itertools import combinations
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.compare_pca_lime import generate_neighborhood, select_margin_strata
from experiments.generalization_study import build_model, load_dataset
from fisher_lime.fisher_projection import (
    fit_fisher_projection,
    hard_memberships,
)
from fisher_lime.local_dimension import (
    analyze_local_probabilities,
    fit_weighted_pca,
)
from fisher_lime.surrogate import fit_weighted_ridge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["synthetic10", "iris", "wine", "digits"],
        choices=["synthetic10", "iris", "wine", "digits"],
    )
    parser.add_argument(
        "--models", nargs="+", default=["mlp", "random_forest"],
        choices=["mlp", "random_forest"]
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--targets-per-margin", type=int, default=2)
    parser.add_argument("--perturbations", type=int, default=400)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--fisher-regularization", type=float, default=1.0)
    parser.add_argument("--top-features", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "stability",
    )
    return parser.parse_args()


def orthogonal_projector(directions: np.ndarray) -> np.ndarray:
    basis, _ = np.linalg.qr(np.asarray(directions))
    return basis @ basis.T


def axis_complexity(directions: np.ndarray) -> tuple[float, float]:
    """Return effective class count and mass held by the two largest loadings."""

    absolute = np.abs(np.asarray(directions))
    normalized = absolute / np.maximum(absolute.sum(axis=0, keepdims=True), 1e-15)
    entropy = -np.sum(normalized * np.log(np.maximum(normalized, 1e-15)), axis=0)
    effective_count = np.exp(entropy)
    top_two_mass = np.sort(normalized, axis=0)[-2:].sum(axis=0)
    return float(effective_count.mean()), float(top_two_mass.mean())


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = left.ravel()
    right = right.ravel()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= 1e-15:
        return float(np.linalg.norm(left - right) <= 1e-15)
    return float(np.dot(left, right) / denominator)


def subspace_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = max(int(round(np.trace(left))), 1)
    right_rank = max(int(round(np.trace(right))), 1)
    return float(np.trace(left @ right) / np.sqrt(left_rank * right_rank))


def jaccard(left: set[int], right: set[int]) -> float:
    return len(left & right) / len(left | right)


def build_artifact(
    method: str,
    projection,
    train_features: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    ridge_alpha: float,
    top_feature_count: int,
) -> dict:
    scores = projection.transform(probabilities)
    surrogate = fit_weighted_ridge(
        train_features, scores, weights, ridge_alpha
    )
    if method == "pca_lime":
        decoder = projection.components
        directions = projection.components.T
    else:
        decoder = projection.decoder_coefficients
        directions = projection.directions
    effect = surrogate.coefficients @ decoder
    importance = np.linalg.norm(effect, axis=1)
    top_features = set(np.argsort(importance)[-top_feature_count:].tolist())
    effective_classes, top_two_mass = axis_complexity(directions)
    return {
        "dimension": scores.shape[1],
        "effect": effect,
        "top_features": top_features,
        "projector": orthogonal_projector(directions),
        "effective_classes_per_axis": effective_classes,
        "top_two_class_mass": top_two_mass,
    }


def summarize_artifacts(artifacts: list[dict]) -> dict[str, float]:
    if len(artifacts) < 2:
        return {
            "pair_count": 0,
            "coefficient_cosine": np.nan,
            "top_feature_jaccard": np.nan,
            "subspace_similarity": np.nan,
        }
    coefficient_values = []
    feature_values = []
    subspace_values = []
    for left, right in combinations(artifacts, 2):
        coefficient_values.append(cosine_similarity(left["effect"], right["effect"]))
        feature_values.append(jaccard(left["top_features"], right["top_features"]))
        if left["projector"] is not None and right["projector"] is not None:
            subspace_values.append(
                subspace_similarity(left["projector"], right["projector"])
            )
    return {
        "pair_count": len(coefficient_values),
        "coefficient_cosine": float(np.mean(coefficient_values)),
        "top_feature_jaccard": float(np.mean(feature_values)),
        "subspace_similarity": float(np.mean(subspace_values))
        if subspace_values
        else np.nan,
    }


def main() -> None:
    args = parse_args()
    repeat_rows = []
    target_rows = []
    model_rows = []
    configuration_count = len(args.datasets) * len(args.models)
    completed = 0

    for dataset_name in args.datasets:
        for model_name in args.models:
            rng = np.random.default_rng(args.seed)
            x, y = load_dataset(dataset_name, args.seed)
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.3, stratify=y, random_state=args.seed
            )
            model = build_model(model_name, args.seed, x_train.shape[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, y_train)
            test_probabilities = model.predict_proba(x_test)
            accuracy = accuracy_score(y_test, np.argmax(test_probabilities, axis=1))
            scaler = model.steps[0][1]
            classifier = model.steps[-1][1]
            x_test_scaled = scaler.transform(x_test)
            feature_count = x_test_scaled.shape[1]
            class_count = test_probabilities.shape[1]
            top_feature_count = min(args.top_features, feature_count)
            selected = select_margin_strata(
                test_probabilities, args.targets_per_margin, rng
            )

            for margin_group, indices in selected.items():
                for target_index in indices:
                    target = x_test_scaled[target_index]
                    for radius in args.radii:
                        artifacts = {
                            "ordinary_lime": [],
                            "pca_lime": [],
                            "hard_fisher_lime": [],
                            "soft_fisher_lime": [],
                        }
                        for repetition in range(args.repetitions):
                            points, weights = generate_neighborhood(
                                target, args.perturbations, radius, rng
                            )
                            probabilities = classifier.predict_proba(points)
                            features = (points - target) / radius
                            hard_labels = np.argmax(probabilities, axis=1)
                            ordinary = fit_weighted_ridge(
                                features, probabilities, weights, args.ridge_alpha
                            )
                            importance = np.linalg.norm(ordinary.coefficients, axis=1)
                            artifacts["ordinary_lime"].append(
                                {
                                    "dimension": class_count,
                                    "effect": ordinary.coefficients,
                                    "top_features": set(
                                        np.argsort(importance)[-top_feature_count:].tolist()
                                    ),
                                    "projector": None,
                                    "effective_classes_per_axis": np.nan,
                                    "top_two_class_mass": np.nan,
                                }
                            )

                            analysis = analyze_local_probabilities(probabilities, weights)
                            dimension = max(analysis.effective_dimension_95, 1)
                            projections = {
                                "pca_lime": fit_weighted_pca(
                                    probabilities, weights, dimension
                                )
                            }
                            memberships = {
                                "hard_fisher_lime": hard_memberships(
                                    hard_labels, class_count
                                ),
                                "soft_fisher_lime": probabilities,
                            }
                            for method, membership in memberships.items():
                                try:
                                    projections[method] = fit_fisher_projection(
                                        probabilities,
                                        weights,
                                        membership,
                                        dimension,
                                        regularization=args.fisher_regularization,
                                    )
                                except ValueError:
                                    pass
                            for method, projection in projections.items():
                                artifacts[method].append(
                                    build_artifact(
                                        method,
                                        projection,
                                        features,
                                        probabilities,
                                        weights,
                                        args.ridge_alpha,
                                        top_feature_count,
                                    )
                                )

                            for method in artifacts:
                                latest = artifacts[method][-1] if artifacts[method] else None
                                was_available = (
                                    method == "ordinary_lime"
                                    or method == "pca_lime"
                                    or method in projections
                                )
                                repeat_rows.append(
                                    {
                                        "dataset": dataset_name,
                                        "black_box": model_name,
                                        "target_index": int(target_index),
                                        "margin_group": margin_group,
                                        "radius": radius,
                                        "repetition": repetition,
                                        "method": method,
                                        "available": was_available,
                                        "dimension": latest["dimension"]
                                        if was_available and latest is not None
                                        else 0,
                                        "effective_classes_per_axis": latest[
                                            "effective_classes_per_axis"
                                        ]
                                        if was_available and latest is not None
                                        else np.nan,
                                        "top_two_class_mass": latest[
                                            "top_two_class_mass"
                                        ]
                                        if was_available and latest is not None
                                        else np.nan,
                                    }
                                )

                        for method, method_artifacts in artifacts.items():
                            stability = summarize_artifacts(method_artifacts)
                            target_rows.append(
                                {
                                    "dataset": dataset_name,
                                    "black_box": model_name,
                                    "target_index": int(target_index),
                                    "margin_group": margin_group,
                                    "radius": radius,
                                    "method": method,
                                    "availability_rate": len(method_artifacts)
                                    / args.repetitions,
                                    "mean_dimension": np.mean(
                                        [a["dimension"] for a in method_artifacts]
                                    )
                                    if method_artifacts
                                    else np.nan,
                                    "dimension_std": np.std(
                                        [a["dimension"] for a in method_artifacts]
                                    )
                                    if method_artifacts
                                    else np.nan,
                                    "mean_effective_classes_per_axis": np.nanmean(
                                        [
                                            a["effective_classes_per_axis"]
                                            for a in method_artifacts
                                        ]
                                    )
                                    if method != "ordinary_lime" and method_artifacts
                                    else np.nan,
                                    "mean_top_two_class_mass": np.nanmean(
                                        [a["top_two_class_mass"] for a in method_artifacts]
                                    )
                                    if method != "ordinary_lime" and method_artifacts
                                    else np.nan,
                                    **stability,
                                }
                            )

            model_rows.append(
                {
                    "dataset": dataset_name,
                    "black_box": model_name,
                    "seed": args.seed,
                    "test_accuracy": accuracy,
                }
            )
            completed += 1
            print(
                f"[{completed}/{configuration_count}] {dataset_name} / {model_name} "
                f"accuracy={accuracy:.3f}",
                flush=True,
            )

    repeat_frame = pd.DataFrame(repeat_rows)
    target_frame = pd.DataFrame(target_rows)
    summary = (
        target_frame.groupby("method", as_index=False)
        .agg(
            target_neighborhoods=("target_index", "count"),
            mean_availability=("availability_rate", "mean"),
            mean_dimension=("mean_dimension", "mean"),
            mean_dimension_std=("dimension_std", "mean"),
            mean_coefficient_cosine=("coefficient_cosine", "mean"),
            mean_top_feature_jaccard=("top_feature_jaccard", "mean"),
            mean_subspace_similarity=("subspace_similarity", "mean"),
            mean_effective_classes_per_axis=(
                "mean_effective_classes_per_axis", "mean"
            ),
            mean_top_two_class_mass=("mean_top_two_class_mass", "mean"),
        )
        .sort_values("method")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    repeat_frame.to_csv(args.output_dir / "repeat_measurements.csv", index=False)
    target_frame.to_csv(args.output_dir / "target_stability.csv", index=False)
    summary.to_csv(args.output_dir / "overall_summary.csv", index=False)
    pd.DataFrame(model_rows).to_csv(args.output_dir / "model_metadata.csv", index=False)

    plot = summary.set_index("method").loc[
        ["ordinary_lime", "pca_lime", "hard_fisher_lime", "soft_fisher_lime"]
    ]
    axis = plot[["mean_coefficient_cosine", "mean_top_feature_jaccard"]].plot(
        kind="bar", figsize=(9, 5), rot=15
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Mean pairwise stability")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.output_dir / "stability_comparison.png", dpi=180)
    plt.close()

    print("\nOverall stability:")
    print(summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
