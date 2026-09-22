#!/usr/bin/env python3
"""Measure how Fisher within-scatter regularization affects local explanations."""

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.compare_fisher_lime import projection_metrics
from experiments.compare_pca_lime import generate_neighborhood, select_margin_strata
from experiments.generalization_study import build_model, load_dataset
from experiments.stability_study import axis_complexity
from fisher_lime.fisher_projection import (
    fit_fisher_projection,
    hard_memberships,
)
from fisher_lime.local_dimension import analyze_local_probabilities
from fisher_lime.surrogate import (
    fit_weighted_ridge,
    weighted_argmax_agreement,
    weighted_output_rmse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", default=["synthetic10", "iris", "wine", "digits"]
    )
    parser.add_argument(
        "--models", nargs="+", default=["mlp", "random_forest"]
    )
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--targets-per-margin", type=int, default=2)
    parser.add_argument("--train-perturbations", type=int, default=400)
    parser.add_argument("--test-perturbations", type=int, default=400)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument(
        "--regularizations",
        type=float,
        nargs="+",
        default=[1e-6, 1e-4, 1e-2, 1e-1, 1.0],
    )
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "fisher_regularization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    total = len(args.datasets) * len(args.models)
    completed = 0
    for dataset_name in args.datasets:
        for model_name in args.models:
            rng = np.random.default_rng(args.seed)
            x, y = load_dataset(dataset_name, args.seed)
            x_train, x_test, y_train, _ = train_test_split(
                x, y, test_size=0.3, stratify=y, random_state=args.seed
            )
            model = build_model(model_name, args.seed, x_train.shape[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x_train, y_train)
            scaler = model.steps[0][1]
            classifier = model.steps[-1][1]
            x_test_scaled = scaler.transform(x_test)
            test_probabilities = classifier.predict_proba(x_test_scaled)
            class_count = test_probabilities.shape[1]
            feature_count = x_test_scaled.shape[1]
            selected = select_margin_strata(
                test_probabilities, args.targets_per_margin, rng
            )

            for margin_group, indices in selected.items():
                for target_index in indices:
                    target = x_test_scaled[target_index]
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
                        analysis = analyze_local_probabilities(train_p, train_weights)
                        dimension = max(analysis.effective_dimension_95, 1)
                        ordinary = fit_weighted_ridge(
                            train_features, train_p, train_weights, args.ridge_alpha
                        )
                        ordinary_prediction = ordinary.predict(test_features)
                        ordinary_rmse = weighted_output_rmse(
                            test_p, ordinary_prediction, test_weights
                        )
                        ordinary_agreement = weighted_argmax_agreement(
                            test_p, ordinary_prediction, test_weights
                        )

                        memberships = {
                            "hard_fisher_lime": hard_memberships(
                                hard_labels, class_count
                            ),
                            "soft_fisher_lime": train_p,
                        }
                        for method, membership in memberships.items():
                            for regularization in args.regularizations:
                                common = {
                                    "dataset": dataset_name,
                                    "black_box": model_name,
                                    "target_index": int(target_index),
                                    "margin_group": margin_group,
                                    "radius": radius,
                                    "method": method,
                                    "regularization": regularization,
                                    "ordinary_rmse": ordinary_rmse,
                                    "ordinary_argmax_agreement": ordinary_agreement,
                                }
                                try:
                                    projection = fit_fisher_projection(
                                        train_p,
                                        train_weights,
                                        membership,
                                        dimension,
                                        regularization=regularization,
                                    )
                                except ValueError:
                                    rows.append(
                                        {
                                            **common,
                                            "available": False,
                                            "dimension": 0,
                                            "oracle_rmse": np.nan,
                                            "surrogate_rmse": np.nan,
                                            "surrogate_argmax_agreement": np.nan,
                                            "effective_classes_per_axis": np.nan,
                                            "top_two_class_mass": np.nan,
                                        }
                                    )
                                    continue
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
                                    feature_count,
                                    class_count,
                                )
                                effective_classes, top_two_mass = axis_complexity(
                                    projection.directions
                                )
                                rows.append(
                                    {
                                        **common,
                                        **metrics,
                                        "available": True,
                                        "effective_classes_per_axis": effective_classes,
                                        "top_two_class_mass": top_two_mass,
                                    }
                                )
            completed += 1
            print(
                f"[{completed}/{total}] {dataset_name} / {model_name}", flush=True
            )

    frame = pd.DataFrame(rows)
    frame["rmse_increase"] = frame["surrogate_rmse"] - frame["ordinary_rmse"]
    frame["argmax_change"] = (
        frame["surrogate_argmax_agreement"]
        - frame["ordinary_argmax_agreement"]
    )
    summary = (
        frame.groupby(["method", "regularization"], as_index=False)
        .agg(
            attempted=("target_index", "count"),
            available_rate=("available", "mean"),
            mean_dimension=("dimension", lambda x: x[x > 0].mean()),
            mean_rmse_increase=("rmse_increase", "mean"),
            mean_argmax_change=("argmax_change", "mean"),
            mean_effective_classes_per_axis=(
                "effective_classes_per_axis", "mean"
            ),
            mean_top_two_class_mass=("top_two_class_mass", "mean"),
        )
        .sort_values(["method", "regularization"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_evaluations.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, group in summary.groupby("method"):
        axes[0].plot(
            group["regularization"], group["mean_rmse_increase"], marker="o", label=method
        )
        axes[1].plot(
            group["regularization"], group["mean_top_two_class_mass"], marker="o", label=method
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
        axis.set_xlabel("Within-scatter regularization")
    axes[0].set_ylabel("RMSE increase vs ordinary LIME")
    axes[1].set_ylabel("Top-two class loading mass")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "regularization_tradeoff.png", dpi=180)
    plt.close(figure)
    print("\nRegularization summary:")
    print(summary.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()

