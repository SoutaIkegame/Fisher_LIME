#!/usr/bin/env python3
"""Evaluate local output dimension and compressed-LIME fidelity together.

Every quantity is measured for the same black box, the same explanation
target and the same neighborhood distribution, so that "the local output is
low-dimensional" and "a compressed explanation is faithful" can be linked
target by target. For each neighborhood the script records

* output-side dimension: q95, participation ratio, variation energy;
* competing classes: argmax classes, classes whose probability reaches a
  threshold, and the variance-effective number of moving classes;
* linear-side dimension: effective rank of the ordinary LIME coefficient
  matrix (features x classes), which bounds what an output-compressed linear
  surrogate can retain;
* held-out fidelity of compression only, ordinary LIME and PCA-LIME, their
  paired difference, and the error at the explained point itself.

Neighborhoods can be isotropic Gaussian (the earlier experiments) or Gaussian
with the covariance of the standardized training data, which keeps offsets
in the span of the data when features are linearly redundant.
"""

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

from experiments.compare_pca_lime import select_margin_strata
from experiments.global_local_dimension_study import build_model
from fisher_lime.diagnostics import (
    NeighborhoodSampler,
    class_activity,
    coefficient_matrix_dimension,
    linear_energy_in_subspace,
    participation_ratio,
    weighted_output_r2,
)
from fisher_lime.local_dimension import analyze_local_probabilities, fit_weighted_pca
from fisher_lime.surrogate import (
    fit_weighted_ridge,
    weighted_argmax_agreement,
    weighted_output_rmse,
)

MODEL_CHOICES = [
    "logistic_regression",
    "mlp",
    "rbf_svm",
    "decision_tree",
    "bagged_trees",
    "random_forest",
    "gradient_boosting",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--models", nargs="+", default=["mlp", "random_forest"], choices=MODEL_CHOICES
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--samples", type=int, default=9000)
    parser.add_argument("--features", type=int, default=40)
    parser.add_argument(
        "--informative",
        type=int,
        default=24,
        help="informative features, fixed across class counts; 0 uses the "
        "earlier rule min(features - 2, max(12, classes + 4))",
    )
    parser.add_argument("--targets-per-margin", type=int, default=6)
    parser.add_argument("--fit-perturbations", type=int, default=600)
    parser.add_argument("--eval-perturbations", type=int, default=600)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.15, 0.4])
    parser.add_argument(
        "--neighborhoods",
        nargs="+",
        default=["isotropic", "data"],
        choices=["isotropic", "data"],
    )
    parser.add_argument("--dimensions", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--activity-threshold", type=float, default=0.05)
    parser.add_argument(
        "--low-variation",
        type=float,
        default=1e-4,
        help="total weighted output variance below which a neighborhood is "
        "treated as near-constant and summarized separately",
    )
    parser.add_argument(
        "--acceptable-r2",
        type=float,
        default=0.8,
        help="absolute held-out R² for an explanation to count as faithful",
    )
    parser.add_argument(
        "--max-r2-loss",
        type=float,
        default=0.02,
        help="largest acceptable held-out R² drop caused by compression",
    )
    parser.add_argument(
        "--compression-r2",
        type=float,
        default=0.95,
        help="held-out compression-only R² counted as preserving the output",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "result" / "unified_local_evaluation",
    )
    return parser.parse_args()


def make_data(class_count: int, seed: int, args: argparse.Namespace):
    if args.informative > 0:
        informative = args.informative
    else:
        informative = min(args.features - 2, max(12, class_count + 4))
    if informative > args.features:
        raise ValueError("informative features cannot exceed features")
    if class_count > 2**informative:
        raise ValueError("too few informative features for the class count")
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
    return x, y, informative


def dimension_rules(
    fixed: list[int], output_q95: int, linear_q95: int, class_count: int
) -> dict[int, list[str]]:
    """Map each evaluated dimension to the selection rules that chose it."""

    rules: dict[int, list[str]] = {}
    for dimension in fixed:
        if 1 <= dimension < class_count:
            rules.setdefault(dimension, []).append(f"fixed_{dimension}")
    for name, dimension in (("output_q95", output_q95), ("linear_q95", linear_q95)):
        dimension = int(min(max(dimension, 1), class_count - 1))
        rules.setdefault(dimension, []).append(name)
    return rules


def evaluate_configuration(
    class_count: int,
    model_name: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    x, y, informative = make_data(class_count, seed, args)
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
    scaled_train = scaler.transform(x_train)
    scaled_test = scaler.transform(x_test)
    samplers = {
        kind: NeighborhoodSampler(kind, reference=scaled_train)
        for kind in args.neighborhoods
    }
    data_rank = samplers[args.neighborhoods[0]].span.shape[1]
    selected = select_margin_strata(test_probabilities, args.targets_per_margin, rng)

    neighborhood_rows: list[dict] = []
    dimension_rows: list[dict] = []
    for margin_group, target_indices in selected.items():
        for target_index in target_indices:
            target = scaled_test[target_index]
            target_probability = classifier.predict_proba(target[None, :])[0]
            sorted_probability = np.sort(target_probability)
            target_margin = float(sorted_probability[-1] - sorted_probability[-2])
            for kind, sampler in samplers.items():
                for radius in args.radii:
                    fit_x, fit_w = sampler.sample(
                        target, args.fit_perturbations, radius, rng
                    )
                    eval_x, eval_w = sampler.sample(
                        target, args.eval_perturbations, radius, rng
                    )
                    fit_p = classifier.predict_proba(fit_x)
                    eval_p = classifier.predict_proba(eval_x)
                    fit_f = (fit_x - target) / radius
                    eval_f = (eval_x - target) / radius
                    zero_f = np.zeros((1, target.size))

                    analysis = analyze_local_probabilities(fit_p, fit_w)
                    activity = class_activity(fit_p, fit_w, args.activity_threshold)
                    ordinary = fit_weighted_ridge(fit_f, fit_p, fit_w, args.ridge_alpha)
                    linear = coefficient_matrix_dimension(ordinary.coefficients)
                    ordinary_eval = ordinary.predict(eval_f)
                    ordinary_r2 = weighted_output_r2(eval_p, ordinary_eval, eval_w)
                    ordinary_target_error = float(
                        np.linalg.norm(ordinary.predict(zero_f)[0] - target_probability)
                    )
                    output_q95 = analysis.effective_dimension_95
                    low_variation = analysis.variation_energy < args.low_variation

                    common = {
                        "classes": class_count,
                        "black_box": model_name,
                        "seed": seed,
                        "target_index": int(target_index),
                        "margin_group": margin_group,
                        "target_margin": target_margin,
                        "neighborhood": kind,
                        "radius": radius,
                        "low_variation": bool(low_variation),
                    }
                    neighborhood_rows.append(
                        {
                            **common,
                            "off_span_fraction": sampler.off_span_fraction(
                                fit_x - target
                            ),
                            "variation_energy": analysis.variation_energy,
                            "output_q95": output_q95,
                            "output_q99": analysis.effective_dimension_99,
                            "output_participation_ratio": participation_ratio(
                                analysis.explained_variance_ratio
                            ),
                            "argmax_classes": activity.argmax_classes,
                            "active_classes": activity.active_classes,
                            "variance_effective_classes": (
                                activity.variance_effective_classes
                            ),
                            "linear_q95": linear.effective_dimension_95,
                            "linear_participation_ratio": linear.participation_ratio,
                            "ordinary_r2": ordinary_r2,
                            "ordinary_rmse": weighted_output_rmse(
                                eval_p, ordinary_eval, eval_w
                            ),
                            "ordinary_target_error": ordinary_target_error,
                        }
                    )

                    rules = dimension_rules(
                        args.dimensions,
                        output_q95,
                        linear.effective_dimension_95,
                        class_count,
                    )
                    for dimension, rule_names in sorted(rules.items()):
                        pca = fit_weighted_pca(fit_p, fit_w, dimension)
                        compression_only = pca.inverse_transform(pca.transform(eval_p))
                        score_model = fit_weighted_ridge(
                            fit_f, pca.transform(fit_p), fit_w, args.ridge_alpha
                        )
                        compressed_eval = pca.inverse_transform(
                            score_model.predict(eval_f)
                        )
                        compressed_target = pca.inverse_transform(
                            score_model.predict(zero_f)
                        )[0]
                        compressed_r2 = weighted_output_r2(
                            eval_p, compressed_eval, eval_w
                        )
                        row = {
                            **common,
                            "dimension": dimension,
                            "output_q95": output_q95,
                            "linear_q95": linear.effective_dimension_95,
                            "active_classes": activity.active_classes,
                            "compression_r2": weighted_output_r2(
                                eval_p, compression_only, eval_w
                            ),
                            "ordinary_r2": ordinary_r2,
                            "compressed_r2": compressed_r2,
                            "r2_loss": ordinary_r2 - compressed_r2,
                            "linear_energy_retained": linear_energy_in_subspace(
                                ordinary.coefficients, pca.components
                            ),
                            "ordinary_argmax_agreement": weighted_argmax_agreement(
                                eval_p, ordinary_eval, eval_w
                            ),
                            "compressed_argmax_agreement": weighted_argmax_agreement(
                                eval_p, compressed_eval, eval_w
                            ),
                            "ordinary_target_error": ordinary_target_error,
                            "compressed_target_error": float(
                                np.linalg.norm(compressed_target - target_probability)
                            ),
                            "coefficient_count": dimension
                            * (target.size + class_count),
                            "ordinary_coefficient_count": target.size * class_count,
                        }
                        for rule in rule_names:
                            dimension_rows.append({**row, "rule": rule})

    metadata = {
        "classes": class_count,
        "black_box": model_name,
        "seed": seed,
        "features": args.features,
        "informative_features": informative,
        "standardized_data_rank": data_rank,
        "test_accuracy": accuracy,
    }
    return neighborhood_rows, dimension_rows, metadata


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    column: str,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    """Resample seeds, then targets within each drawn seed.

    Rows sharing a seed share the trained black box and data set, so seeds
    are the outer resampling unit. With few seeds this interval is still
    exploratory, but it no longer treats targets from one model as
    independent replications across models.
    """

    groups = [
        group[column].to_numpy(dtype=float)
        for _, group in frame.groupby("seed")
    ]
    groups = [values[np.isfinite(values)] for values in groups]
    groups = [values for values in groups if values.size]
    if not groups:
        return np.nan, np.nan
    means = np.empty(repeats)
    for repeat in range(repeats):
        drawn = rng.integers(0, len(groups), size=len(groups))
        samples = [
            groups[index][rng.integers(0, groups[index].size, groups[index].size)]
            for index in drawn
        ]
        means[repeat] = np.concatenate(samples).mean()
    return tuple(np.quantile(means, [0.025, 0.975]))


def add_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Duplicate rows into all / varying / near_constant strata."""

    parts = [frame.assign(stratum="all")]
    parts.append(frame[~frame["low_variation"]].assign(stratum="varying"))
    parts.append(frame[frame["low_variation"]].assign(stratum="near_constant"))
    return pd.concat(parts, ignore_index=True)


def summarize_mechanism(neighborhoods: pd.DataFrame) -> pd.DataFrame:
    frame = add_strata(neighborhoods)
    frame = frame.assign(
        active_minus_one=frame["active_classes"] - 1,
        q95_below_active=frame["output_q95"] < frame["active_classes"] - 1,
        q95_above_active=frame["output_q95"] > frame["active_classes"] - 1,
        linear_below_output=frame["linear_q95"] < frame["output_q95"],
    )
    return (
        frame.groupby(
            ["classes", "black_box", "neighborhood", "radius", "stratum"],
            as_index=False,
        )
        .agg(
            neighborhoods=("target_index", "count"),
            mean_off_span_fraction=("off_span_fraction", "mean"),
            mean_variation_energy=("variation_energy", "mean"),
            mean_argmax_classes=("argmax_classes", "mean"),
            mean_active_minus_one=("active_minus_one", "mean"),
            mean_variance_effective_classes=("variance_effective_classes", "mean"),
            mean_output_q95=("output_q95", "mean"),
            mean_output_pr=("output_participation_ratio", "mean"),
            mean_linear_q95=("linear_q95", "mean"),
            mean_linear_pr=("linear_participation_ratio", "mean"),
            share_q95_below_active=("q95_below_active", "mean"),
            share_q95_above_active=("q95_above_active", "mean"),
            share_linear_below_output=("linear_below_output", "mean"),
            mean_ordinary_r2=("ordinary_r2", "mean"),
        )
    )


def summarize_fidelity(
    dimensions: pd.DataFrame,
    args: argparse.Namespace,
    by_seed: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(20260925)
    frame = add_strata(dimensions)
    frame = frame.assign(
        ordinary_ok=frame["ordinary_r2"] >= args.acceptable_r2,
        compressed_ok=frame["compressed_r2"] >= args.acceptable_r2,
        loss_ok=frame["r2_loss"] <= args.max_r2_loss,
        compression_ok=frame["compression_r2"] >= args.compression_r2,
        target_error_increase=frame["compressed_target_error"]
        - frame["ordinary_target_error"],
        coefficient_reduction=1.0
        - frame["coefficient_count"] / frame["ordinary_coefficient_count"],
    )
    frame["faithful_and_small_loss"] = frame["compressed_ok"] & frame["loss_ok"]
    keys = ["classes", "black_box", "neighborhood", "radius", "stratum", "rule"]
    if by_seed:
        keys = keys + ["seed"]
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update(
            {
                "neighborhoods": group.shape[0],
                "mean_dimension": group["dimension"].mean(),
                "mean_compression_r2": group["compression_r2"].mean(),
                "mean_ordinary_r2": group["ordinary_r2"].mean(),
                "mean_compressed_r2": group["compressed_r2"].mean(),
                "mean_r2_loss": group["r2_loss"].mean(),
                "median_r2_loss": group["r2_loss"].median(),
                "mean_linear_energy_retained": group["linear_energy_retained"].mean(),
                "rate_compression_r2_ok": group["compression_ok"].mean(),
                "rate_ordinary_faithful": group["ordinary_ok"].mean(),
                "rate_compressed_faithful": group["compressed_ok"].mean(),
                "rate_loss_ok": group["loss_ok"].mean(),
                "rate_faithful_and_small_loss": group[
                    "faithful_and_small_loss"
                ].mean(),
                "mean_argmax_change": (
                    group["compressed_argmax_agreement"]
                    - group["ordinary_argmax_agreement"]
                ).mean(),
                "mean_ordinary_target_error": group["ordinary_target_error"].mean(),
                "mean_target_error_increase": group["target_error_increase"].mean(),
                "coefficient_reduction": group["coefficient_reduction"].mean(),
            }
        )
        if not by_seed:
            low, high = hierarchical_bootstrap(
                group, "r2_loss", rng, args.bootstrap_repeats
            )
            row["r2_loss_ci_low"] = low
            row["r2_loss_ci_high"] = high
            row["seeds"] = group["seed"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fidelity(summary: pd.DataFrame, output_path: Path) -> None:
    fixed = summary[
        summary["rule"].str.startswith("fixed_") & (summary["stratum"] == "varying")
    ]
    if fixed.empty:
        return
    panels = fixed[["classes", "black_box", "neighborhood"]].drop_duplicates()
    panels = panels.sort_values(["classes", "black_box", "neighborhood"])
    columns = 4
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.2 * columns, 3.4 * rows),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for axis in axes.ravel()[len(panels):]:
        axis.set_visible(False)
    styles = {0.15: "-", 0.4: "--"}
    for axis, (_, panel) in zip(axes.ravel(), panels.iterrows()):
        subset = fixed[
            (fixed["classes"] == panel["classes"])
            & (fixed["black_box"] == panel["black_box"])
            & (fixed["neighborhood"] == panel["neighborhood"])
        ]
        for radius, group in subset.groupby("radius"):
            group = group.sort_values("mean_dimension")
            style = styles.get(radius, ":")
            axis.plot(
                group["mean_dimension"], group["mean_compression_r2"],
                style, color="tab:gray", marker="o", label=f"compression r={radius:g}",
            )
            axis.plot(
                group["mean_dimension"], group["mean_compressed_r2"],
                style, color="tab:blue", marker="s", label=f"PCA-LIME r={radius:g}",
            )
            axis.axhline(
                group["mean_ordinary_r2"].mean(),
                linestyle=style, color="tab:red", linewidth=1,
                label=f"ordinary LIME r={radius:g}",
            )
        axis.set_title(
            f"K={panel['classes']}, {panel['black_box']}, {panel['neighborhood']}"
        )
        axis.set_xlabel("Output axes")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Held-out R² (varying neighborhoods)")
    axes[0, 0].legend(fontsize=7)
    plt.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.neighborhoods:
        raise ValueError("at least one neighborhood kind is required")
    neighborhood_rows: list[dict] = []
    dimension_rows: list[dict] = []
    metadata: list[dict] = []
    total = len(args.classes) * len(args.models) * len(args.seeds)
    completed = 0
    for class_count in args.classes:
        for model_name in args.models:
            for seed in args.seeds:
                n_rows, d_rows, meta = evaluate_configuration(
                    class_count, model_name, seed, args
                )
                neighborhood_rows.extend(n_rows)
                dimension_rows.extend(d_rows)
                metadata.append(meta)
                completed += 1
                print(
                    f"[{completed}/{total}] K={class_count} / {model_name} / "
                    f"seed={seed}: accuracy={meta['test_accuracy']:.3f}, "
                    f"data rank={meta['standardized_data_rank']}",
                    flush=True,
                )

    neighborhoods = pd.DataFrame(neighborhood_rows)
    dimensions = pd.DataFrame(dimension_rows)
    mechanism = summarize_mechanism(neighborhoods)
    fidelity = summarize_fidelity(dimensions, args, by_seed=False)
    fidelity_by_seed = summarize_fidelity(dimensions, args, by_seed=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    neighborhoods.to_csv(args.output_dir / "neighborhoods.csv", index=False)
    dimensions.to_csv(args.output_dir / "dimension_evaluations.csv", index=False)
    mechanism.to_csv(args.output_dir / "mechanism_summary.csv", index=False)
    fidelity.to_csv(args.output_dir / "fidelity_summary.csv", index=False)
    fidelity_by_seed.to_csv(args.output_dir / "fidelity_by_seed.csv", index=False)
    pd.DataFrame(metadata).to_csv(args.output_dir / "model_metadata.csv", index=False)
    plot_fidelity(fidelity, args.output_dir / "fidelity_by_dimension.png")

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\nMechanism (varying neighborhoods):")
        print(
            mechanism[mechanism["stratum"] == "varying"][
                [
                    "classes", "black_box", "neighborhood", "radius",
                    "neighborhoods", "mean_off_span_fraction",
                    "mean_active_minus_one", "mean_output_q95", "mean_output_pr",
                    "mean_linear_q95", "share_q95_below_active",
                ]
            ].round(3).to_string(index=False)
        )
        print("\nFidelity by selection rule (varying neighborhoods):")
        view = fidelity[
            (fidelity["stratum"] == "varying")
            & fidelity["rule"].isin(["fixed_3", "output_q95", "linear_q95"])
        ]
        print(
            view[
                [
                    "classes", "black_box", "neighborhood", "radius", "rule",
                    "mean_dimension", "mean_compression_r2", "mean_ordinary_r2",
                    "mean_compressed_r2", "mean_r2_loss", "r2_loss_ci_low",
                    "r2_loss_ci_high", "rate_faithful_and_small_loss",
                ]
            ].round(3).to_string(index=False)
        )
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
