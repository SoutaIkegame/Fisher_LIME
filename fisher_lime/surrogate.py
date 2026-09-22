"""Weighted local linear surrogates used in LIME-style experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WeightedLinearModel:
    intercept: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.intercept + np.asarray(features) @ self.coefficients


def fit_weighted_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    alpha: float = 1e-3,
) -> WeightedLinearModel:
    """Fit a multi-output weighted ridge model with an unpenalized intercept."""

    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be two-dimensional")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    if weights.shape != (features.shape[0],):
        raise ValueError("weights must have one value per row")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must be non-negative and have a positive sum")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    design = np.column_stack([np.ones(features.shape[0]), features])
    root_weights = np.sqrt(weights / weights.sum())[:, None]
    weighted_design = design * root_weights
    weighted_targets = targets * root_weights
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_targets,
    )
    return WeightedLinearModel(
        intercept=parameters[0],
        coefficients=parameters[1:],
    )


def weighted_output_rmse(
    expected: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    normalized = np.asarray(weights, dtype=float) / np.sum(weights)
    row_error = np.sum((np.asarray(expected) - np.asarray(predicted)) ** 2, axis=1)
    return float(np.sqrt(np.sum(normalized * row_error)))


def weighted_argmax_agreement(
    expected: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> float:
    normalized = np.asarray(weights, dtype=float) / np.sum(weights)
    matches = np.argmax(expected, axis=1) == np.argmax(predicted, axis=1)
    return float(np.sum(normalized * matches))

