"""Core utilities for Fisher-LIME experiments."""

from .local_dimension import (
    LocalDimensionResult,
    WeightedPCAModel,
    analyze_local_probabilities,
    fit_weighted_pca,
)
from .output_projection import (
    LinearOutputProjection,
    axis_complexity,
    fit_weighted_sparse_pca,
    rotate_weighted_pca,
    varimax_rotation,
)

__all__ = [
    "LocalDimensionResult",
    "WeightedPCAModel",
    "analyze_local_probabilities",
    "fit_weighted_pca",
    "LinearOutputProjection",
    "axis_complexity",
    "fit_weighted_sparse_pca",
    "rotate_weighted_pca",
    "varimax_rotation",
]
