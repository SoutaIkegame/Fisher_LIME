"""Interpretable linear projections for local probability outputs."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import dict_learning

from .local_dimension import WeightedPCAModel


@dataclass(frozen=True)
class LinearOutputProjection:
    """Linear encoder and decoder around a weighted probability mean."""

    mean: np.ndarray
    directions: np.ndarray
    encoder_ridge: float = 0.0

    @property
    def n_components(self) -> int:
        return self.directions.shape[1]

    def transform(self, values: np.ndarray) -> np.ndarray:
        centered = np.asarray(values, dtype=float) - self.mean
        gram = self.directions.T @ self.directions
        if self.encoder_ridge > 0:
            gram = gram + self.encoder_ridge * np.eye(self.n_components)
        return centered @ self.directions @ np.linalg.pinv(gram)

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        return self.mean + np.asarray(scores, dtype=float) @ self.directions.T


def _validate_probabilities_and_weights(
    probabilities: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (samples, classes)")
    if weights.shape != (probabilities.shape[0],):
        raise ValueError("weights must have one value per probability row")
    if probabilities.shape[0] < 2 or probabilities.shape[1] < 2:
        raise ValueError("at least two samples and two classes are required")
    if not np.all(np.isfinite(probabilities)) or not np.all(np.isfinite(weights)):
        raise ValueError("probabilities and weights must be finite")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and have positive mass")
    return probabilities, weights


def varimax_rotation(
    directions: np.ndarray,
    gamma: float = 1.0,
    max_iter: int = 200,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return an orthogonal Varimax rotation matrix for column directions."""

    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] < 1:
        raise ValueError("directions must be a non-empty two-dimensional array")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between zero and one")
    rotation = np.eye(directions.shape[1])
    previous_objective = 0.0
    row_count = directions.shape[0]
    for _ in range(max_iter):
        rotated = directions @ rotation
        column_energy = np.sum(rotated**2, axis=0)
        target = rotated**3 - (gamma / row_count) * rotated @ np.diag(
            column_energy
        )
        left, singular_values, right = np.linalg.svd(
            directions.T @ target, full_matrices=False
        )
        rotation = left @ right
        objective = float(singular_values.sum())
        if previous_objective > 0 and objective - previous_objective <= (
            tolerance * previous_objective
        ):
            break
        previous_objective = objective
    return rotation


def rotate_weighted_pca(
    pca: WeightedPCAModel,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> LinearOutputProjection:
    """Apply an orthogonal Varimax rotation without changing the PCA subspace."""

    probabilities, weights = _validate_probabilities_and_weights(
        probabilities, weights
    )
    if probabilities.shape[1] != pca.components.shape[1]:
        raise ValueError("probability class count does not match PCA components")
    base_directions = pca.components.T
    rotation = varimax_rotation(base_directions)
    directions = base_directions @ rotation
    scores = (probabilities - pca.mean) @ directions
    normalized_weights = weights / weights.sum()
    score_mean = np.sum(normalized_weights[:, None] * scores, axis=0)
    score_variance = np.sum(
        normalized_weights[:, None] * (scores - score_mean) ** 2, axis=0
    )
    order = np.argsort(score_variance)[::-1]
    directions = directions[:, order]
    for index in range(directions.shape[1]):
        pivot = int(np.argmax(np.abs(directions[:, index])))
        if directions[pivot, index] < 0:
            directions[:, index] *= -1
    return LinearOutputProjection(mean=pca.mean.copy(), directions=directions)


def _balance_sparse_direction(direction: np.ndarray) -> np.ndarray:
    """Enforce a zero class sum while retaining the selected support."""

    direction = np.asarray(direction, dtype=float).copy()
    active = np.flatnonzero(np.abs(direction) > 1e-12)
    if active.size < 2:
        largest = np.argsort(np.abs(direction))[-2:]
        if np.all(np.abs(direction[largest]) <= 1e-12):
            largest = np.array([0, 1])
            direction[largest] = [1.0, -1.0]
        active = largest
    direction[active] -= direction[active].mean()
    direction[np.setdiff1d(np.arange(direction.size), active)] = 0.0
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        first, second = active[:2]
        direction[:] = 0.0
        direction[first], direction[second] = 1.0, -1.0
        norm = np.sqrt(2.0)
    return direction / norm


def fit_weighted_sparse_pca(
    probabilities: np.ndarray,
    weights: np.ndarray,
    n_components: int,
    alpha: float,
    encoder_ridge: float = 1e-6,
    max_iter: int = 300,
    tolerance: float = 1e-7,
    random_state: int | None = None,
    enforce_simplex_tangent: bool = True,
) -> LinearOutputProjection:
    """Fit sparse class directions to a weighted local probability cloud.

    ``dict_learning`` is applied to the transpose of the explicitly weighted,
    centered observations. This places the L1 penalty on class directions,
    matching Sparse PCA rather than sparse coding of observations.
    """

    probabilities, weights = _validate_probabilities_and_weights(
        probabilities, weights
    )
    max_dimension = min(probabilities.shape[1] - 1, probabilities.shape[0] - 1)
    if not 1 <= n_components <= max_dimension:
        raise ValueError(f"n_components must be between 1 and {max_dimension}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if encoder_ridge < 0:
        raise ValueError("encoder_ridge must be non-negative")

    normalized_weights = weights / weights.sum()
    mean = np.sum(normalized_weights[:, None] * probabilities, axis=0)
    centered = probabilities - mean
    weighted_centered = np.sqrt(
        probabilities.shape[0] * normalized_weights[:, None]
    ) * centered
    code, _, _ = dict_learning(
        weighted_centered.T,
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        tol=tolerance,
        method="cd",
        random_state=random_state,
    )
    directions = code
    if enforce_simplex_tangent:
        directions = np.column_stack(
            [_balance_sparse_direction(directions[:, j]) for j in range(n_components)]
        )
    else:
        norms = np.linalg.norm(directions, axis=0)
        norms[norms <= 1e-12] = 1.0
        directions = directions / norms
    for index in range(directions.shape[1]):
        pivot = int(np.argmax(np.abs(directions[:, index])))
        if directions[pivot, index] < 0:
            directions[:, index] *= -1
    return LinearOutputProjection(
        mean=mean,
        directions=directions,
        encoder_ridge=encoder_ridge,
    )


def axis_complexity(directions: np.ndarray) -> dict[str, float]:
    """Summarize how many classes materially contribute to each axis."""

    directions = np.asarray(directions, dtype=float)
    absolute = np.abs(directions)
    mass = absolute / np.maximum(absolute.sum(axis=0, keepdims=True), 1e-15)
    entropy = -np.sum(mass * np.log(np.maximum(mass, 1e-15)), axis=0)
    effective = np.exp(entropy)
    top_two = np.sort(mass, axis=0)[-2:].sum(axis=0)
    active = np.sum(mass >= 0.05, axis=0)
    return {
        "effective_classes": float(np.mean(effective)),
        "top2_class_mass": float(np.mean(top_two)),
        "active_classes_5pct": float(np.mean(active)),
    }
