"""Measure the intrinsic dimension of local classifier outputs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocalDimensionResult:
    """Summary and reconstruction curve for one local probability cloud."""

    effective_dimension_95: int
    effective_dimension_99: int
    numerical_rank: int
    variation_energy: float
    explained_variance_ratio: np.ndarray
    reconstruction_rmse: np.ndarray
    argmax_agreement: np.ndarray


@dataclass(frozen=True)
class WeightedPCAModel:
    """Weighted PCA encoder and linear decoder."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) @ self.components.T

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.mean + np.asarray(scores) @ self.components


def _validate_inputs(probabilities: np.ndarray, weights: np.ndarray) -> None:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (samples, classes)")
    if weights.ndim != 1 or weights.shape[0] != probabilities.shape[0]:
        raise ValueError("weights must have one value per probability row")
    if probabilities.shape[0] < 2 or probabilities.shape[1] < 2:
        raise ValueError("at least two samples and two classes are required")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(weights)):
        raise ValueError("probabilities and weights must be finite")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and have a positive sum")


def fit_weighted_pca(
    probabilities: np.ndarray,
    weights: np.ndarray,
    n_components: int,
) -> WeightedPCAModel:
    """Fit a PCA projection using normalized sample weights."""

    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    _validate_inputs(probabilities, weights)
    max_dimension = min(probabilities.shape[1] - 1, probabilities.shape[0] - 1)
    if not 1 <= n_components <= max_dimension:
        raise ValueError(f"n_components must be between 1 and {max_dimension}")

    normalized_weights = weights / weights.sum()
    mean = np.sum(normalized_weights[:, None] * probabilities, axis=0)
    centered = probabilities - mean
    weighted_centered = np.sqrt(normalized_weights[:, None]) * centered
    _, singular_values, vt = np.linalg.svd(weighted_centered, full_matrices=False)
    eigenvalues = singular_values**2
    total = eigenvalues.sum()
    if total <= np.finfo(float).eps:
        ratios = np.zeros(n_components, dtype=float)
    else:
        ratios = eigenvalues[:n_components] / total
    return WeightedPCAModel(
        mean=mean,
        components=vt[:n_components],
        explained_variance_ratio=ratios,
    )


def analyze_local_probabilities(
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> LocalDimensionResult:
    """Run weighted PCA and evaluate reconstructions in probability space.

    The PCA is centered with the weighted probability mean. Reconstruction
    metrics are reported for dimensions 1 through ``min(classes - 1,
    samples - 1)``. The probability-simplex constraint makes ``classes - 1``
    the largest meaningful centered dimension.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    _validate_inputs(probabilities, weights)

    normalized_weights = weights / weights.sum()
    mean = np.sum(normalized_weights[:, None] * probabilities, axis=0)
    centered = probabilities - mean
    weighted_centered = np.sqrt(normalized_weights[:, None]) * centered

    _, singular_values, vt = np.linalg.svd(weighted_centered, full_matrices=False)
    eigenvalues = singular_values**2
    total_variation = float(eigenvalues.sum())
    max_dimension = min(probabilities.shape[1] - 1, probabilities.shape[0] - 1)

    scale = singular_values[0] if singular_values.size else 0.0
    tolerance = np.finfo(float).eps * max(probabilities.shape) * scale
    numerical_rank = int(np.sum(singular_values > tolerance))

    if total_variation <= np.finfo(float).eps:
        total_variation = 0.0
        explained = np.zeros(max_dimension, dtype=float)
        effective_95 = 0
        effective_99 = 0
    else:
        explained = eigenvalues[:max_dimension] / total_variation
        cumulative = np.cumsum(explained)
        effective_95 = int(np.searchsorted(cumulative, 0.95) + 1)
        effective_99 = int(np.searchsorted(cumulative, 0.99) + 1)

    rmses = np.empty(max_dimension, dtype=float)
    agreements = np.empty(max_dimension, dtype=float)
    original_labels = np.argmax(probabilities, axis=1)

    for dimension in range(1, max_dimension + 1):
        basis = vt[:dimension].T
        reconstructed = mean + centered @ basis @ basis.T
        squared_error = np.sum((probabilities - reconstructed) ** 2, axis=1)
        rmses[dimension - 1] = np.sqrt(np.sum(normalized_weights * squared_error))
        reconstructed_labels = np.argmax(reconstructed, axis=1)
        agreements[dimension - 1] = np.sum(
            normalized_weights * (reconstructed_labels == original_labels)
        )

    return LocalDimensionResult(
        effective_dimension_95=effective_95,
        effective_dimension_99=effective_99,
        numerical_rank=min(numerical_rank, max_dimension),
        variation_energy=total_variation,
        explained_variance_ratio=explained,
        reconstruction_rmse=rmses,
        argmax_agreement=agreements,
    )
