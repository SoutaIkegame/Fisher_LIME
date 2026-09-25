import unittest

import numpy as np

from experiments.compare_pca_lime import generate_neighborhood
from fisher_lime.diagnostics import (
    NeighborhoodSampler,
    class_activity,
    coefficient_matrix_dimension,
    linear_energy_in_subspace,
    weighted_feature_covariance,
    weighted_output_r2,
)
from fisher_lime.local_dimension import fit_weighted_pca
from fisher_lime.surrogate import fit_weighted_ridge


def softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class DiagnosticsTest(unittest.TestCase):
    def test_rank_one_coefficients_have_one_linear_dimension(self) -> None:
        coefficients = np.outer([1.0, -2.0, 0.5], [1.0, -1.0, 0.0, 0.0])
        result = coefficient_matrix_dimension(coefficients)

        self.assertEqual(result.effective_dimension_95, 1)
        self.assertAlmostEqual(result.participation_ratio, 1.0)

    def test_zero_coefficients_have_zero_linear_dimension(self) -> None:
        result = coefficient_matrix_dimension(np.zeros((4, 3)))

        self.assertEqual(result.effective_dimension_95, 0)
        self.assertEqual(result.energy, 0.0)

    def test_class_activity_counts_only_moving_classes(self) -> None:
        coordinate = np.linspace(-0.2, 0.2, 51)
        probabilities = np.column_stack(
            [
                0.5 + coordinate,
                0.4 - coordinate,
                np.full_like(coordinate, 0.07),
                np.full_like(coordinate, 0.03),
            ]
        )
        activity = class_activity(probabilities, np.ones(51), 0.05)

        self.assertEqual(activity.argmax_classes, 2)
        self.assertEqual(activity.active_classes, 3)
        self.assertEqual(activity.moving_classes, 2)
        self.assertAlmostEqual(activity.variance_effective_classes, 2.0)

    def test_constant_high_classes_are_active_but_not_moving(self) -> None:
        # Two classes trade probability while three stay at 0.10: the output
        # is one-dimensional because only two classes move, not because a
        # group of five moves together.
        coordinate = np.linspace(-0.2, 0.2, 51)
        constant = np.full_like(coordinate, 0.10)
        probabilities = np.column_stack(
            [0.35 + coordinate, 0.35 - coordinate, constant, constant, constant]
        )
        activity = class_activity(probabilities, np.ones(51), 0.05)

        self.assertEqual(activity.active_classes, 5)
        self.assertEqual(activity.moving_classes, 2)

    def test_prediction_retention_differs_from_coefficient_retention(self) -> None:
        # Two equal coefficient rows mapping to different class directions.
        # When the first input barely varies, dropping its class direction
        # removes half the coefficient energy but almost no predicted change.
        coefficients = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        basis = np.array([[0.0, 1.0, -1.0]]) / np.sqrt(2.0)
        covariance = np.diag([0.01, 1.0])

        coefficient_share = linear_energy_in_subspace(coefficients, basis)
        prediction_share = linear_energy_in_subspace(coefficients, basis, covariance)

        self.assertLess(coefficient_share, 0.8)
        self.assertGreater(prediction_share, 0.99)
        self.assertEqual(
            coefficient_matrix_dimension(coefficients).effective_dimension_95, 2
        )
        self.assertEqual(
            coefficient_matrix_dimension(
                coefficients, feature_covariance=covariance
            ).effective_dimension_95,
            1,
        )

    def test_weighted_feature_covariance_matches_numpy(self) -> None:
        rng = np.random.default_rng(9)
        features = rng.normal(size=(200, 3))
        weights = rng.uniform(0.1, 1.0, size=200)
        expected = np.cov(features, rowvar=False, aweights=weights, bias=True)
        np.testing.assert_allclose(
            weighted_feature_covariance(features, weights), expected, atol=1e-12
        )

    def test_linear_energy_matches_compressed_surrogate(self) -> None:
        rng = np.random.default_rng(4)
        features = rng.normal(size=(400, 6))
        probabilities = softmax(features @ rng.normal(size=(6, 5)))
        weights = np.ones(400)
        ordinary = fit_weighted_ridge(features, probabilities, weights)
        pca = fit_weighted_pca(probabilities, weights, 2)
        compressed = fit_weighted_ridge(features, pca.transform(probabilities), weights)
        effect = compressed.coefficients @ pca.components

        projected = ordinary.coefficients @ pca.components.T @ pca.components
        np.testing.assert_allclose(effect, projected, atol=1e-12)
        share = linear_energy_in_subspace(ordinary.coefficients, pca.components)
        self.assertAlmostEqual(
            share,
            float(np.sum(effect**2) / np.sum(ordinary.coefficients**2)),
        )
        self.assertLessEqual(share, 1.0 + 1e-12)

    def test_isotropic_sampler_matches_earlier_neighborhoods(self) -> None:
        target = np.linspace(-1.0, 1.0, 7)
        sampler = NeighborhoodSampler("isotropic")
        points, weights = sampler.sample(
            target, 30, 0.4, np.random.default_rng(5)
        )
        expected_points, expected_weights = generate_neighborhood(
            target, 30, 0.4, np.random.default_rng(5)
        )

        np.testing.assert_allclose(points, expected_points)
        np.testing.assert_allclose(weights, expected_weights)

    def test_data_sampler_stays_in_span_of_redundant_features(self) -> None:
        rng = np.random.default_rng(6)
        base = rng.normal(size=(500, 3))
        reference = np.column_stack([base, base @ rng.normal(size=(3, 2))])
        reference = (reference - reference.mean(axis=0)) / reference.std(axis=0)
        data_sampler = NeighborhoodSampler("data", reference=reference)
        isotropic_sampler = NeighborhoodSampler("isotropic", reference=reference)
        target = reference[0]

        data_points, _ = data_sampler.sample(target, 300, 0.4, rng)
        isotropic_points, _ = isotropic_sampler.sample(target, 300, 0.4, rng)

        self.assertEqual(data_sampler.span.shape[1], 3)
        self.assertLess(data_sampler.off_span_fraction(data_points - target), 1e-10)
        self.assertGreater(
            isotropic_sampler.off_span_fraction(isotropic_points - target), 0.2
        )

    def test_data_sampler_requires_reference(self) -> None:
        with self.assertRaises(ValueError):
            NeighborhoodSampler("data")

    def test_weighted_r2_is_one_for_perfect_prediction(self) -> None:
        values = np.random.default_rng(7).dirichlet(np.ones(4), size=20)
        self.assertEqual(weighted_output_r2(values, values, np.ones(20)), 1.0)


if __name__ == "__main__":
    unittest.main()
