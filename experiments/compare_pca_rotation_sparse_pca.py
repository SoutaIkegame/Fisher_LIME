#!/usr/bin/env python3
"""Compare weighted PCA, rotated PCA, and simplex-aware Sparse PCA."""

import argparse
import json
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
from experiments.pca_fidelity_tradeoff_study import weighted_output_r2
from fisher_lime.local_dimension import analyze_local_probabilities, fit_weighted_pca
from fisher_lime.output_projection import (
    LinearOutputProjection,
    axis_complexity,
    fit_weighted_sparse_pca,
    rotate_weighted_pca,
)
from fisher_lime.surrogate import (
    fit_weighted_ridge,
    weighted_argmax_agreement,
    weighted_output_rmse,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, default=20)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument("--samples", type=int, default=9000)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "random_forest", "gradient_boosting"],
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--targets-per-margin", type=int, default=3)
    parser.add_argument("--fit-perturbations", type=int, default=600)
    parser.add_argument("--validation-perturbations", type=int, default=600)
    parser.add_argument("--test-perturbations", type=int, default=1000)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument(
        "--sparse-alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.1, 0.3, 1.0],
    )
    parser.add_argument("--sparse-loss-tolerance", type=float, default=0.02)
    parser.add_argument("--sparse-max-iter", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "pca_rotation_sparse_pca",
    )
    return parser.parse_args()


def output_r2_with_mean(
    expected: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    reference_mean: np.ndarray,
) -> float:
    normalized = np.asarray(weights, dtype=float) / np.sum(weights)
    residual = np.sum(
        normalized * np.sum((np.asarray(expected) - np.asarray(predicted)) ** 2, axis=1)
    )
    total = np.sum(
        normalized
        * np.sum((np.asarray(expected) - np.asarray(reference_mean)) ** 2, axis=1)
    )
    if total <= np.finfo(float).eps:
        return 1.0 if residual <= np.finfo(float).eps else np.nan
    return float(1.0 - residual / total)


def margin_mae(
    expected: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> float:
    expected_sorted = np.sort(np.asarray(expected), axis=1)
    predicted_sorted = np.sort(np.asarray(predicted), axis=1)
    expected_margin = expected_sorted[:, -1] - expected_sorted[:, -2]
    predicted_margin = predicted_sorted[:, -1] - predicted_sorted[:, -2]
    normalized = np.asarray(weights, dtype=float) / np.sum(weights)
    return float(np.sum(normalized * np.abs(expected_margin - predicted_margin)))


def probability_constraint_metrics(
    predicted: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    normalized = np.asarray(weights, dtype=float) / np.sum(weights)
    predicted = np.asarray(predicted, dtype=float)
    sum_error = np.sum(normalized * np.abs(predicted.sum(axis=1) - 1.0))
    negative_mass = np.sum(normalized * np.maximum(-predicted, 0.0).sum(axis=1))
    return float(sum_error), float(negative_mass)


def projection_directions(projection) -> np.ndarray:
    if hasattr(projection, "directions"):
        return projection.directions
    return projection.components.T


def evaluate_projection(
    method: str,
    projection,
    train_features: np.ndarray,
    train_probabilities: np.ndarray,
    train_weights: np.ndarray,
    test_features: np.ndarray,
    test_probabilities: np.ndarray,
    test_weights: np.ndarray,
    ridge_alpha: float,
) -> dict[str, float | int | str]:
    train_scores = projection.transform(train_probabilities)
    surrogate = fit_weighted_ridge(
        train_features, train_scores, train_weights, ridge_alpha
    )
    oracle = projection.inverse_transform(projection.transform(test_probabilities))
    prediction = projection.inverse_transform(surrogate.predict(test_features))
    oracle_sum_error, oracle_negative_mass = probability_constraint_metrics(
        oracle, test_weights
    )
    prediction_sum_error, prediction_negative_mass = probability_constraint_metrics(
        prediction, test_weights
    )
    class_metrics = axis_complexity(projection_directions(projection))
    feature_metrics = axis_complexity(surrogate.coefficients)
    return {
        "method": method,
        "dimension": train_scores.shape[1],
        "oracle_output_r2": output_r2_with_mean(
            test_probabilities, oracle, test_weights, projection.mean
        ),
        "oracle_rmse": weighted_output_rmse(
            test_probabilities, oracle, test_weights
        ),
        "oracle_argmax_agreement": weighted_argmax_agreement(
            test_probabilities, oracle, test_weights
        ),
        "oracle_top2_margin_mae": margin_mae(
            test_probabilities, oracle, test_weights
        ),
        "oracle_simplex_sum_error": oracle_sum_error,
        "oracle_negative_probability_mass": oracle_negative_mass,
        "surrogate_output_r2": output_r2_with_mean(
            test_probabilities, prediction, test_weights, projection.mean
        ),
        "surrogate_rmse": weighted_output_rmse(
            test_probabilities, prediction, test_weights
        ),
        "surrogate_argmax_agreement": weighted_argmax_agreement(
            test_probabilities, prediction, test_weights
        ),
        "surrogate_top2_margin_mae": margin_mae(
            test_probabilities, prediction, test_weights
        ),
        "surrogate_simplex_sum_error": prediction_sum_error,
        "surrogate_negative_probability_mass": prediction_negative_mass,
        **class_metrics,
        "effective_features": feature_metrics["effective_classes"],
        "top2_feature_mass": feature_metrics["top2_class_mass"],
        "active_features_5pct": feature_metrics["active_classes_5pct"],
    }


def sparse_validation_candidate(
    probabilities: np.ndarray,
    weights: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_weights: np.ndarray,
    dimension: int,
    alpha: float,
    args: argparse.Namespace,
    random_state: int,
) -> tuple[LinearOutputProjection, dict[str, float]]:
    sparse = fit_weighted_sparse_pca(
        probabilities,
        weights,
        dimension,
        alpha,
        max_iter=args.sparse_max_iter,
        random_state=random_state,
    )
    reconstruction = sparse.inverse_transform(sparse.transform(validation_probabilities))
    validation_r2 = output_r2_with_mean(
        validation_probabilities, reconstruction, validation_weights, sparse.mean
    )
    complexity = axis_complexity(sparse.directions)
    return sparse, {
        "alpha": alpha,
        "validation_r2": validation_r2,
        "validation_loss": 1.0 - validation_r2,
        **complexity,
    }


def choose_sparse_alpha(
    probabilities: np.ndarray,
    weights: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_weights: np.ndarray,
    dimension: int,
    pca_validation_loss: float,
    args: argparse.Namespace,
    random_state: int,
) -> tuple[float, list[dict[str, float]], bool]:
    candidates = []
    for index, alpha in enumerate(args.sparse_alphas):
        _, metrics = sparse_validation_candidate(
            probabilities,
            weights,
            validation_probabilities,
            validation_weights,
            dimension,
            alpha,
            args,
            random_state + index,
        )
        candidates.append(metrics)
    feasible = [
        row
        for row in candidates
        if row["validation_loss"]
        <= pca_validation_loss + args.sparse_loss_tolerance
    ]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                row["effective_classes"],
                row["validation_loss"],
                -row["alpha"],
            ),
        )
        meets_tolerance = True
    else:
        selected = min(candidates, key=lambda row: row["validation_loss"])
        meets_tolerance = False
    return float(selected["alpha"]), candidates, meets_tolerance


def evaluate_configuration(
    model_name: str, seed: int, args: argparse.Namespace
) -> tuple[list[dict], list[dict], dict]:
    if args.classes - 1 > args.features:
        raise ValueError("features must be at least classes - 1")
    rng = np.random.default_rng(seed)
    informative = min(args.features - 2, max(12, args.classes + 4))
    x, y = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=informative,
        n_redundant=args.features - informative,
        n_classes=args.classes,
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
    probabilities = model.predict_proba(x_test)
    accuracy = accuracy_score(y_test, np.argmax(probabilities, axis=1))
    scaler = model.steps[0][1]
    classifier = model.steps[-1][1]
    scaled_test = scaler.transform(x_test)
    selected = select_margin_strata(probabilities, args.targets_per_margin, rng)
    rows: list[dict] = []
    tuning_rows: list[dict] = []

    for margin_group, target_indices in selected.items():
        for target_index in target_indices:
            target = scaled_test[target_index]
            sorted_target = np.sort(probabilities[target_index])
            target_margin = float(sorted_target[-1] - sorted_target[-2])
            for radius in args.radii:
                fit_x, fit_weights = generate_neighborhood(
                    target, args.fit_perturbations, radius, rng
                )
                validation_x, validation_weights = generate_neighborhood(
                    target, args.validation_perturbations, radius, rng
                )
                test_x_local, test_weights = generate_neighborhood(
                    target, args.test_perturbations, radius, rng
                )
                fit_p = classifier.predict_proba(fit_x)
                validation_p = classifier.predict_proba(validation_x)
                test_p = classifier.predict_proba(test_x_local)
                analysis = analyze_local_probabilities(fit_p, fit_weights)
                dimension = max(analysis.effective_dimension_95, 1)
                fit_pca = fit_weighted_pca(fit_p, fit_weights, dimension)
                validation_pca = fit_pca.inverse_transform(
                    fit_pca.transform(validation_p)
                )
                pca_validation_r2 = output_r2_with_mean(
                    validation_p,
                    validation_pca,
                    validation_weights,
                    fit_pca.mean,
                )
                selected_alpha, candidates, meets_tolerance = choose_sparse_alpha(
                    fit_p,
                    fit_weights,
                    validation_p,
                    validation_weights,
                    dimension,
                    1.0 - pca_validation_r2,
                    args,
                    random_state=seed * 100000 + int(target_index) * 10,
                )
                common = {
                    "black_box": model_name,
                    "seed": seed,
                    "target_index": int(target_index),
                    "margin_group": margin_group,
                    "target_margin": target_margin,
                    "radius": radius,
                    "q_var95": dimension,
                    "variation_energy": analysis.variation_energy,
                    "selected_sparse_alpha": selected_alpha,
                    "sparse_meets_validation_tolerance": meets_tolerance,
                    "pca_validation_r2": pca_validation_r2,
                }
                for candidate in candidates:
                    tuning_rows.append({**common, **candidate})

                combined_x = np.vstack([fit_x, validation_x])
                combined_weights = np.concatenate([fit_weights, validation_weights])
                combined_p = np.vstack([fit_p, validation_p])
                combined_features = (combined_x - target) / radius
                test_features = (test_x_local - target) / radius

                pca = fit_weighted_pca(combined_p, combined_weights, dimension)
                rotated = rotate_weighted_pca(pca, combined_p, combined_weights)
                sparse = fit_weighted_sparse_pca(
                    combined_p,
                    combined_weights,
                    dimension,
                    selected_alpha,
                    max_iter=args.sparse_max_iter,
                    random_state=seed * 100000 + int(target_index),
                )
                projections = {
                    "pca": pca,
                    "rotated_pca": rotated,
                    "sparse_pca": sparse,
                }
                method_rows = {}
                for method, projection in projections.items():
                    result = evaluate_projection(
                        method,
                        projection,
                        combined_features,
                        combined_p,
                        combined_weights,
                        test_features,
                        test_p,
                        test_weights,
                        args.ridge_alpha,
                    )
                    method_rows[method] = result
                    rows.append({**common, **result})

                oracle_delta = abs(
                    method_rows["pca"]["oracle_rmse"]
                    - method_rows["rotated_pca"]["oracle_rmse"]
                )
                surrogate_delta = abs(
                    method_rows["pca"]["surrogate_rmse"]
                    - method_rows["rotated_pca"]["surrogate_rmse"]
                )
                if oracle_delta > 1e-10 or surrogate_delta > 1e-10:
                    raise RuntimeError(
                        "orthogonal rotation changed PCA fidelity: "
                        f"oracle={oracle_delta:.3e}, surrogate={surrogate_delta:.3e}"
                    )

    metadata = {
        "black_box": model_name,
        "seed": seed,
        "classes": args.classes,
        "features": args.features,
        "samples": args.samples,
        "test_accuracy": accuracy,
    }
    return rows, tuning_rows, metadata


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "dimension",
        "oracle_output_r2",
        "oracle_rmse",
        "oracle_argmax_agreement",
        "surrogate_output_r2",
        "surrogate_rmse",
        "surrogate_argmax_agreement",
        "effective_classes",
        "top2_class_mass",
        "active_classes_5pct",
        "effective_features",
        "surrogate_negative_probability_mass",
    ]
    return (
        frame.groupby(["black_box", "radius", "method"], as_index=False)[metrics]
        .mean()
        .sort_values(["black_box", "radius", "method"])
    )


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    methods = ["pca", "rotated_pca", "sparse_pca"]
    labels = ["PCA", "Rotated PCA", "Sparse PCA"]
    models = list(summary["black_box"].unique())
    radii = sorted(summary["radius"].unique())
    figure, axes = plt.subplots(
        len(models),
        len(radii),
        figsize=(5.3 * len(radii), 4.0 * len(models)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, model_name in enumerate(models):
        for column, radius in enumerate(radii):
            axis = axes[row, column]
            group = summary[
                (summary["black_box"] == model_name)
                & (summary["radius"] == radius)
            ].set_index("method")
            x = [group.loc[method, "effective_classes"] for method in methods]
            y = [group.loc[method, "surrogate_output_r2"] for method in methods]
            axis.scatter(x, y, s=70)
            for x_value, y_value, label in zip(x, y, labels):
                axis.annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points")
            axis.set_title(f"{model_name}, radius={radius:g}")
            axis.set_xlabel("Effective classes per axis (lower is simpler)")
            axis.set_ylabel("Surrogate output R2")
            axis.grid(alpha=0.25)
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if any(alpha <= 0 for alpha in args.sparse_alphas):
        raise ValueError("sparse alphas must be positive")
    rows = []
    tuning_rows = []
    metadata = []
    total = len(args.models) * len(args.seeds)
    completed = 0
    for model_name in args.models:
        for seed in args.seeds:
            result_rows, result_tuning, result_metadata = evaluate_configuration(
                model_name, seed, args
            )
            rows.extend(result_rows)
            tuning_rows.extend(result_tuning)
            metadata.append(result_metadata)
            completed += 1
            print(
                f"[{completed}/{total}] {model_name} / seed={seed}: "
                f"accuracy={result_metadata['test_accuracy']:.3f}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    tuning = pd.DataFrame(tuning_rows)
    summary = summarize(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_evaluations.csv", index=False)
    tuning.to_csv(args.output_dir / "sparse_tuning.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output_dir / "model_metadata.csv", index=False)
    plot_summary(summary, args.output_dir / "fidelity_vs_axis_complexity.png")
    settings = vars(args).copy()
    settings["output_dir"] = str(settings["output_dir"])
    (args.output_dir / "settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    print("\nMean results:")
    print(
        summary[
            [
                "black_box",
                "radius",
                "method",
                "dimension",
                "surrogate_output_r2",
                "surrogate_argmax_agreement",
                "effective_classes",
                "top2_class_mass",
            ]
        ].to_string(index=False)
    )
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
