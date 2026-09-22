import unittest

import numpy as np

from fisher_lime.local_dimension import (
    analyze_local_probabilities,
    fit_weighted_pca,
)
from fisher_lime.fisher_projection import (
    fit_fisher_projection,
    hard_memberships,
)
from fisher_lime.surrogate import fit_weighted_ridge
from fisher_lime.output_projection import (
    axis_complexity,
    fit_weighted_sparse_pca,
    rotate_weighted_pca,
)


class LocalDimensionTest(unittest.TestCase):
    def test_rank_one_probability_curve_has_one_effective_dimension(self) -> None:
        coordinate = np.linspace(-0.2, 0.2, 101)
        probabilities = np.column_stack(
            [0.5 + coordinate, 0.3 - coordinate, np.full_like(coordinate, 0.2)]
        )
        result = analyze_local_probabilities(probabilities, np.ones(coordinate.size))

        self.assertEqual(result.effective_dimension_95, 1)
        self.assertEqual(result.numerical_rank, 1)
        self.assertLess(result.reconstruction_rmse[0], 1e-12)

    def test_reconstruction_error_decreases_with_dimension(self) -> None:
        rng = np.random.default_rng(3)
        probabilities = rng.dirichlet(np.ones(5), size=200)
        result = analyze_local_probabilities(probabilities, np.ones(200))

        differences = np.diff(result.reconstruction_rmse)
        self.assertTrue(np.all(differences <= 1e-12))
        self.assertLess(result.reconstruction_rmse[-1], 1e-12)

    def test_constant_outputs_have_zero_effective_dimension(self) -> None:
        probabilities = np.tile([0.6, 0.3, 0.1], (20, 1))
        result = analyze_local_probabilities(probabilities, np.ones(20))

        self.assertEqual(result.effective_dimension_95, 0)
        self.assertEqual(result.variation_energy, 0.0)

    def test_full_pca_surrogate_matches_ordinary_probability_surrogate(self) -> None:
        rng = np.random.default_rng(8)
        features = rng.normal(size=(300, 4))
        logits = features @ rng.normal(size=(4, 5))
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        weights = np.exp(-np.sum(features**2, axis=1) / 8)

        ordinary = fit_weighted_ridge(features, probabilities, weights)
        pca = fit_weighted_pca(probabilities, weights, n_components=4)
        compressed = fit_weighted_ridge(
            features, pca.transform(probabilities), weights
        )
        reconstructed = pca.inverse_transform(compressed.predict(features))

        np.testing.assert_allclose(
            reconstructed, ordinary.predict(features), atol=1e-12
        )

    def test_hard_fisher_finds_two_directions_for_three_groups(self) -> None:
        rng = np.random.default_rng(12)
        labels = np.repeat(np.arange(3), 100)
        centers = np.array(
            [[0.75, 0.20, 0.05], [0.15, 0.75, 0.10], [0.10, 0.20, 0.70]]
        )
        probabilities = centers[labels] + rng.normal(scale=0.01, size=(300, 3))
        probabilities = np.clip(probabilities, 1e-5, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        projection = fit_fisher_projection(
            probabilities,
            np.ones(300),
            hard_memberships(labels, 3),
            requested_components=3,
        )

        self.assertEqual(projection.n_components, 2)
        self.assertEqual(projection.transform(probabilities).shape, (300, 2))
        self.assertGreater(projection.eigenvalues[0], 0)

    def test_hard_fisher_rejects_single_group(self) -> None:
        probabilities = np.tile([0.7, 0.2, 0.1], (20, 1))
        memberships = hard_memberships(np.zeros(20, dtype=int), 3)
        with self.assertRaises(ValueError):
            fit_fisher_projection(
                probabilities,
                np.ones(20),
                memberships,
                requested_components=1,
            )

    def test_varimax_rotation_preserves_pca_predictions(self) -> None:
        rng = np.random.default_rng(19)
        features = rng.normal(size=(240, 6))
        logits = features @ rng.normal(size=(6, 5))
        exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        weights = np.exp(-np.sum(features**2, axis=1) / 12)
        pca = fit_weighted_pca(probabilities, weights, 3)
        rotated = rotate_weighted_pca(pca, probabilities, weights)

        pca_oracle = pca.inverse_transform(pca.transform(probabilities))
        rotated_oracle = rotated.inverse_transform(rotated.transform(probabilities))
        np.testing.assert_allclose(rotated_oracle, pca_oracle, atol=1e-12)

        pca_surrogate = fit_weighted_ridge(
            features, pca.transform(probabilities), weights
        )
        rotated_surrogate = fit_weighted_ridge(
            features, rotated.transform(probabilities), weights
        )
        pca_prediction = pca.inverse_transform(pca_surrogate.predict(features))
        rotated_prediction = rotated.inverse_transform(
            rotated_surrogate.predict(features)
        )
        np.testing.assert_allclose(rotated_prediction, pca_prediction, atol=1e-12)

    def test_sparse_pca_directions_preserve_probability_sum(self) -> None:
        rng = np.random.default_rng(27)
        probabilities = rng.dirichlet(np.ones(6), size=180)
        weights = np.exp(-rng.uniform(size=180))
        sparse = fit_weighted_sparse_pca(
            probabilities,
            weights,
            n_components=3,
            alpha=0.1,
            max_iter=100,
            random_state=27,
        )

        np.testing.assert_allclose(sparse.directions.sum(axis=0), 0.0, atol=1e-12)
        reconstructed = sparse.inverse_transform(sparse.transform(probabilities))
        np.testing.assert_allclose(reconstructed.sum(axis=1), 1.0, atol=1e-10)
        complexity = axis_complexity(sparse.directions)
        self.assertGreaterEqual(complexity["effective_classes"], 2.0)
        self.assertLessEqual(complexity["effective_classes"], 6.0)


if __name__ == "__main__":
    unittest.main()
