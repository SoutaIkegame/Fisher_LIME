"""Hard and probability-weighted Fisher projections for local BB outputs."""

from dataclasses import dataclass

import numpy as np

from .surrogate import fit_weighted_ridge


@dataclass(frozen=True)
class FisherProjection:
    """Fisher encoder with a learned linear decoder into probability space."""

    mean: np.ndarray
    directions: np.ndarray
    eigenvalues: np.ndarray
    decoder_intercept: np.ndarray
    decoder_coefficients: np.ndarray

    @property
    def n_components(self) -> int:
        return self.directions.shape[1]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.mean) @ self.directions

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.decoder_intercept + np.asarray(scores) @ self.decoder_coefficients


def hard_memberships(labels: np.ndarray, class_count: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    memberships = np.zeros((labels.size, class_count), dtype=float)
    memberships[np.arange(labels.size), labels] = 1.0
    return memberships


def fit_fisher_projection(
    values: np.ndarray,
    weights: np.ndarray,
    memberships: np.ndarray,
    requested_components: int,
    regularization: float = 1e-6,
) -> FisherProjection:
    """Fit a Fisher subspace and a weighted least-squares probability decoder.

    ``memberships`` may be one-hot hard labels or soft class memberships. If
    fewer positive discriminant directions exist than requested, all available
    directions are returned. At least two membership groups must have mass.
    """

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    memberships = np.asarray(memberships, dtype=float)
    if values.ndim != 2 or memberships.ndim != 2:
        raise ValueError("values and memberships must be two-dimensional")
    if memberships.shape[0] != values.shape[0]:
        raise ValueError("memberships must have one row per value")
    if weights.shape != (values.shape[0],):
        raise ValueError("weights must have one value per row")
    if requested_components < 1:
        raise ValueError("requested_components must be positive")
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    if np.any(memberships < 0):
        raise ValueError("memberships must be non-negative")
    row_mass = memberships.sum(axis=1)
    if not np.allclose(row_mass, 1.0, atol=1e-8):
        raise ValueError("each membership row must sum to one")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and have positive mass")

    normalized_weights = weights / weights.sum()
    class_mass = np.sum(normalized_weights[:, None] * memberships, axis=0)
    active = class_mass > np.finfo(float).eps * 100
    if np.count_nonzero(active) < 2:
        raise ValueError("Fisher projection requires at least two active groups")
    memberships = memberships[:, active]
    class_mass = class_mass[active]

    mean = np.sum(normalized_weights[:, None] * values, axis=0)
    class_means = (
        (normalized_weights[:, None] * memberships).T @ values
    ) / class_mass[:, None]

    feature_count = values.shape[1]
    within = np.zeros((feature_count, feature_count), dtype=float)
    for class_index, class_mean in enumerate(class_means):
        centered = values - class_mean
        class_weights = normalized_weights * memberships[:, class_index]
        within += (centered * class_weights[:, None]).T @ centered

    between = np.zeros_like(within)
    for mass, class_mean in zip(class_mass, class_means):
        difference = class_mean - mean
        between += mass * np.outer(difference, difference)

    scale = np.trace(within) / feature_count
    ridge = regularization * (scale if scale > 0 else 1.0)
    within_regularized = within + ridge * np.eye(feature_count)
    within_values, within_vectors = np.linalg.eigh(within_regularized)
    inverse_root = (
        within_vectors * (1.0 / np.sqrt(np.maximum(within_values, ridge)))
    ) @ within_vectors.T
    whitened_between = inverse_root @ between @ inverse_root
    eigenvalues, whitened_vectors = np.linalg.eigh(whitened_between)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    directions = inverse_root @ whitened_vectors[:, order]

    tolerance = max(eigenvalues[0], 1.0) * 1e-10
    available = min(
        int(np.sum(eigenvalues > tolerance)),
        memberships.shape[1] - 1,
        values.shape[1] - 1,
    )
    if available < 1:
        raise ValueError("no positive Fisher direction is available")
    component_count = min(requested_components, available)
    directions = directions[:, :component_count]
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    eigenvalues = eigenvalues[:component_count]

    scores = (values - mean) @ directions
    decoder = fit_weighted_ridge(
        scores, values, normalized_weights, alpha=1e-10
    )
    return FisherProjection(
        mean=mean,
        directions=directions,
        eigenvalues=eigenvalues,
        decoder_intercept=decoder.intercept,
        decoder_coefficients=decoder.coefficients,
    )

