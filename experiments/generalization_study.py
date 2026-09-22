#!/usr/bin/env python3
"""Run output-compression LIME across datasets, black boxes, and seeds."""

import argparse
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import (
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.compare_fisher_lime import projection_metrics
from experiments.compare_pca_lime import generate_neighborhood, select_margin_strata
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--targets-per-margin", type=int, default=4)
    parser.add_argument("--train-perturbations", type=int, default=400)
    parser.add_argument("--test-perturbations", type=int, default=400)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--fisher-regularization", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "generalization",
    )
    return parser.parse_args()


def load_dataset(name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "synthetic10":
        return make_classification(
            n_samples=4000,
            n_features=20,
            n_informative=14,
            n_redundant=6,
            n_classes=10,
            n_clusters_per_class=1,
            class_sep=1.4,
            flip_y=0.02,
            random_state=seed,
        )
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name == "digits":
        data = load_digits()
    else:
        raise ValueError(f"unknown dataset: {name}")
    return data.data.astype(float), data.target.astype(int)


def build_model(name: str, seed: int, training_sample_count: int):
    if name == "mlp":
        if training_sample_count < 1000:
            classifier = MLPClassifier(
                hidden_layer_sizes=(32,),
                solver="lbfgs",
                alpha=1e-3,
                max_iter=1000,
                random_state=seed,
            )
        else:
            classifier = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-3,
                max_iter=500,
                early_stopping=True,
                random_state=seed,
            )
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    return make_pipeline(StandardScaler(), classifier)


def evaluate_configuration(
    dataset_name: str,
    model_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    x, y = load_dataset(dataset_name, seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, stratify=y, random_state=seed
    )
    model = build_model(model_name, seed, x_train.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    test_probabilities = model.predict_proba(x_test)
    test_accuracy = accuracy_score(y_test, np.argmax(test_probabilities, axis=1))
    scaler = model.steps[0][1]
    classifier = model.steps[-1][1]
    x_test_scaled = scaler.transform(x_test)
    class_count = test_probabilities.shape[1]
    feature_count = x_test_scaled.shape[1]
    selected = select_margin_strata(test_probabilities, args.targets_per_margin, rng)
    rows: list[dict] = []

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
                common = {
                    "dataset": dataset_name,
                    "black_box": model_name,
                    "seed": seed,
                    "target_index": int(target_index),
                    "margin_group": margin_group,
                    "target_margin": target_margin,
                    "radius": radius,
                    "class_count": class_count,
                    "feature_count": feature_count,
                    "local_hard_class_count": np.unique(hard_labels).size,
                }

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
                rows.append(
                    {
                        **common,
                        "method": "ordinary_lime",
                        "dimension": class_count,
                        "available": True,
                        "oracle_rmse": 0.0,
                        "surrogate_rmse": ordinary_rmse,
                        "surrogate_argmax_agreement": ordinary_agreement,
                        "ordinary_rmse": ordinary_rmse,
                        "ordinary_argmax_agreement": ordinary_agreement,
                        "coefficient_count": class_count * feature_count,
                        "coefficient_reduction": 0.0,
                    }
                )

                analysis = analyze_local_probabilities(train_p, train_weights)
                requested_dimension = max(analysis.effective_dimension_95, 1)
                pca = fit_weighted_pca(train_p, train_weights, requested_dimension)
                projection_specs = [("pca_lime", pca)]
                memberships = {
                    "hard_fisher_lime": hard_memberships(hard_labels, class_count),
                    "soft_fisher_lime": train_p,
                }
                for method, membership in memberships.items():
                    try:
                        projection_specs.append(
                            (
                                method,
                                fit_fisher_projection(
                                    train_p,
                                    train_weights,
                                    membership,
                                    requested_dimension,
                                    regularization=args.fisher_regularization,
                                ),
                            )
                        )
                    except ValueError:
                        rows.append(
                            {
                                **common,
                                "method": method,
                                "dimension": 0,
                                "available": False,
                                "oracle_rmse": np.nan,
                                "surrogate_rmse": np.nan,
                                "surrogate_argmax_agreement": np.nan,
                                "ordinary_rmse": ordinary_rmse,
                                "ordinary_argmax_agreement": ordinary_agreement,
                                "coefficient_count": 0,
                                "coefficient_reduction": np.nan,
                            }
                        )

                for method, projection in projection_specs:
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
                    rows.append(
                        {
                            **common,
                            **metrics,
                            "available": True,
                            "ordinary_rmse": ordinary_rmse,
                            "ordinary_argmax_agreement": ordinary_agreement,
                            "coefficient_reduction": 1.0
                            - metrics["coefficient_count"]
                            / (class_count * feature_count),
                        }
                    )

    metadata = {
        "dataset": dataset_name,
        "black_box": model_name,
        "seed": seed,
        "samples": x.shape[0],
        "features": feature_count,
        "classes": class_count,
        "test_accuracy": test_accuracy,
    }
    return rows, metadata


def bootstrap_mean_interval(
    values: pd.Series,
    rng: np.random.Generator,
    repetitions: int = 2000,
) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if array.size == 0:
        return np.nan, np.nan
    indices = rng.integers(0, array.size, size=(repetitions, array.size))
    means = array[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(frame: pd.DataFrame, seed: int = 20260916) -> pd.DataFrame:
    compressed = frame[frame["method"] != "ordinary_lime"].copy()
    compressed["rmse_increase"] = (
        compressed["surrogate_rmse"] - compressed["ordinary_rmse"]
    )
    compressed["argmax_change"] = (
        compressed["surrogate_argmax_agreement"]
        - compressed["ordinary_argmax_agreement"]
    )
    rng = np.random.default_rng(seed)
    rows = []
    for method, group in compressed.groupby("method"):
        available = group[group["available"]]
        lower, upper = bootstrap_mean_interval(available["rmse_increase"], rng)
        rows.append(
            {
                "method": method,
                "attempted": group.shape[0],
                "available_rate": group["available"].mean(),
                "mean_dimension": available["dimension"].mean(),
                "mean_oracle_rmse": available["oracle_rmse"].mean(),
                "mean_surrogate_rmse": available["surrogate_rmse"].mean(),
                "mean_ordinary_rmse": available["ordinary_rmse"].mean(),
                "mean_rmse_increase": available["rmse_increase"].mean(),
                "rmse_increase_ci_low": lower,
                "rmse_increase_ci_high": upper,
                "mean_argmax_change": available["argmax_change"].mean(),
                "mean_coefficient_reduction": available[
                    "coefficient_reduction"
                ].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("method")


def plot_configuration_summary(summary: pd.DataFrame, output_path: Path) -> None:
    methods = ["pca_lime", "hard_fisher_lime", "soft_fisher_lime"]
    labels = ["PCA", "hard Fisher", "soft Fisher"]
    datasets = list(summary["dataset"].unique())
    figure, axes = plt.subplots(
        1, len(datasets), figsize=(4.3 * len(datasets), 4), sharey=True, squeeze=False
    )
    for axis, dataset in zip(axes.ravel(), datasets):
        subset = summary[summary["dataset"] == dataset]
        x = np.arange(len(methods))
        width = 0.35
        for offset, black_box in zip((-0.5, 0.5), ("mlp", "random_forest")):
            indexed = subset[subset["black_box"] == black_box].set_index("method")
            values = [
                indexed.loc[method, "mean_rmse_increase"]
                if method in indexed.index
                else np.nan
                for method in methods
            ]
            axis.bar(x + offset * width, values, width, label=black_box)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=18)
        axis.set_title(dataset)
        axis.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("RMSE increase vs ordinary LIME")
    axes.ravel()[-1].legend(title="Black box")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    all_rows = []
    metadata_rows = []
    total = len(args.datasets) * len(args.models) * len(args.seeds)
    completed = 0
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                rows, metadata = evaluate_configuration(dataset, model, seed, args)
                all_rows.extend(rows)
                metadata_rows.append(metadata)
                completed += 1
                print(
                    f"[{completed}/{total}] {dataset} / {model} / seed={seed} "
                    f"accuracy={metadata['test_accuracy']:.3f}",
                    flush=True,
                )

    frame = pd.DataFrame(all_rows)
    frame["rmse_increase"] = frame["surrogate_rmse"] - frame["ordinary_rmse"]
    frame["argmax_change"] = (
        frame["surrogate_argmax_agreement"]
        - frame["ordinary_argmax_agreement"]
    )
    overall = summarize(frame)
    configuration = (
        frame[frame["method"] != "ordinary_lime"]
        .groupby(["dataset", "black_box", "method"], as_index=False)
        .agg(
            attempted=("target_index", "count"),
            available_rate=("available", "mean"),
            mean_dimension=("dimension", lambda x: x[x > 0].mean()),
            mean_rmse_increase=("rmse_increase", "mean"),
            mean_argmax_change=("argmax_change", "mean"),
            mean_coefficient_reduction=("coefficient_reduction", "mean"),
        )
        .sort_values(["dataset", "black_box", "method"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "all_evaluations.csv", index=False)
    pd.DataFrame(metadata_rows).to_csv(args.output_dir / "model_metadata.csv", index=False)
    overall.to_csv(args.output_dir / "overall_summary.csv", index=False)
    configuration.to_csv(args.output_dir / "configuration_summary.csv", index=False)
    plot_configuration_summary(
        configuration, args.output_dir / "rmse_increase_by_configuration.png"
    )
    print("\nOverall paired comparison:")
    print(overall.to_string(index=False))
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
