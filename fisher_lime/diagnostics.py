"""Diagnostics that link local output dimension to local linear explanations."""

from dataclasses import dataclass

import numpy as np


def participation_ratio(explained_variance_ratio: np.ndarray) -> float:
    """Return 1 / sum(r_i^2), a threshold-free count of dominant directions."""

    squared_sum = float(np.sum(np.asarray(explained_variance_ratio) ** 2))
    return 0.0 if squared_sum == 0.0 else 1.0 / squared_sum


def weighted_output_r2(
    expected: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> float:
    """Weighted multi-output R² pooled over all classes."""

    weights = np.asarray(weights, dtype=float)
    normalized = weights / weights.sum()
    expected = np.asarray(expected, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mean = np.sum(normalized[:, None] * expected, axis=0)
    residual = np.sum(normalized * np.sum((expected - predicted) ** 2, axis=1))
    total = np.sum(normalized * np.sum((expected - mean) ** 2, axis=1))
    if total <= np.finfo(float).eps:
        return 1.0 if residual <= np.finfo(float).eps else np.nan
    return float(1.0 - residual / total)


@dataclass(frozen=True)
class ClassActivity:
    """How many classes take part in a local probability cloud."""

    argmax_classes: int
    active_classes: int
    variance_effective_classes: float


def class_activity(
    probabilities: np.ndarray,
    weights: np.ndarray,
    probability_threshold: float = 0.05,
) -> ClassActivity:
    """Count competing classes in several complementary ways.

    ``active_classes`` counts classes whose probability reaches
    ``probability_threshold`` somewhere in the neighborhood.
    ``variance_effective_classes`` is exp(entropy) of the per-class share of
    the weighted output variance, i.e. how many classes actually move.
    """

    probabilities = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    normalized = weights / weights.sum()
    mean = np.sum(normalized[:, None] * probabilities, axis=0)
    variance = np.sum(normalized[:, None] * (probabilities - mean) ** 2, axis=0)
    total = variance.sum()
    if total <= np.finfo(float).eps:
        effective = 0.0
    else:
        share = variance / total
        share = share[share > 0]
        effective = float(np.exp(-np.sum(share * np.log(share))))
    return ClassActivity(
        argmax_classes=int(np.unique(np.argmax(probabilities, axis=1)).size),
        active_classes=int(np.sum(probabilities.max(axis=0) >= probability_threshold)),
        variance_effective_classes=effective,
    )


@dataclass(frozen=True)
class MatrixDimension:
    """Effective rank summary of a feature-by-class coefficient matrix."""

    effective_dimension_95: int
    participation_ratio: float
    energy: float
    singular_values: np.ndarray


def coefficient_matrix_dimension(
    coefficients: np.ndarray, threshold: float = 0.95
) -> MatrixDimension:
    """Measure the effective rank of a local linear explanation.

    ``coefficients`` has shape (features, classes). This is the dimension of
    the part of the local output change that a linear surrogate can express,
    which is what an output-compressed linear surrogate can at best retain.
    """

    coefficients = np.asarray(coefficients, dtype=float)
    singular_values = np.linalg.svd(coefficients, compute_uv=False)
    energy_values = singular_values**2
    energy = float(energy_values.sum())
    if energy <= np.finfo(float).eps:
        return MatrixDimension(0, 0.0, 0.0, singular_values)
    ratios = energy_values / energy
    dimension = int(np.searchsorted(np.cumsum(ratios), threshold) + 1)
    return MatrixDimension(
        effective_dimension_95=min(dimension, ratios.size),
        participation_ratio=participation_ratio(ratios),
        energy=energy,
        singular_values=singular_values,
    )


def linear_energy_in_subspace(coefficients: np.ndarray, basis: np.ndarray) -> float:
    """Share of a coefficient matrix kept by projecting its class side.

    ``basis`` has orthonormal rows in class space (e.g. PCA components). With
    a shared-design multi-output ridge, the output-compressed surrogate equals
    the ordinary surrogate projected onto this basis, so this share is exactly
    the fraction of the ordinary explanation that compression retains.
    """

    coefficients = np.asarray(coefficients, dtype=float)
    basis = np.asarray(basis, dtype=float)
    total = float(np.sum(coefficients**2))
    if total <= np.finfo(float).eps:
        return 1.0
    projected = coefficients @ basis.T @ basis
    return float(np.sum(projected**2) / total)


class NeighborhoodSampler:
    """Gaussian LIME-style neighborhoods in a standardized feature space.

    ``kind="isotropic"`` perturbs every feature independently, as in the
    earlier experiments. ``kind="data"`` draws offsets with the covariance of
    the (standardized) training data, so linear dependencies among features
    are preserved and offsets stay in the span of the data. Both kinds have
    the same expected squared offset length when features are standardized.
    Weights use the same kernel on the whitened offset length in both cases.
    """

    def __init__(
        self,
        kind: str = "isotropic",
        reference: np.ndarray | None = None,
        relative_tolerance: float = 1e-8,
    ) -> None:
        if kind not in {"isotropic", "data"}:
            raise ValueError("kind must be 'isotropic' or 'data'")
        self.kind = kind
        self.factor = None
        self.span = None
        if reference is not None:
            reference = np.asarray(reference, dtype=float)
            covariance = np.cov(reference, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            keep = eigenvalues > relative_tolerance * eigenvalues.max()
            self.span = eigenvectors[:, keep]
            self.factor = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
        if kind == "data" and self.factor is None:
            raise ValueError("kind='data' requires reference data")

    def sample(
        self,
        target: np.ndarray,
        count: int,
        radius: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target, dtype=float)
        if self.kind == "isotropic":
            latent = rng.normal(size=(count, target.size))
            offsets = latent
        else:
            latent = rng.normal(size=(count, self.factor.shape[1]))
            offsets = latent @ self.factor.T
        points = target + radius * offsets
        squared_distance = np.sum(offsets**2, axis=1)
        kernel_width = np.sqrt(target.size) * 0.75
        weights = np.exp(-squared_distance / (2.0 * kernel_width**2))
        return points, weights

    def off_span_fraction(self, offsets: np.ndarray) -> float:
        """Share of squared offset length outside the reference data span."""

        if self.span is None:
            return np.nan
        offsets = np.asarray(offsets, dtype=float)
        total = float(np.sum(offsets**2))
        if total <= np.finfo(float).eps:
            return 0.0
        inside = offsets @ self.span
        return float(np.clip(1.0 - np.sum(inside**2) / total, 0.0, 1.0))
