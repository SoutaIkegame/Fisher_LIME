import unittest

import numpy as np

from fisher_lime.local_dimension import fit_weighted_pca
from fisher_lime.output_projection import class_contrast_projection


class ClassContrastProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.weights = rng.uniform(0.5, 1.0, size=200)
        self.t = rng.normal(scale=0.1, size=200)

    def test_two_classes_give_scaled_probability_difference(self) -> None:
        probabilities = np.full((200, 4), 0.05)
        probabilities[:, 1] = 0.45 + self.t
        probabilities[:, 3] = 0.45 - self.t
        projection = class_contrast_projection(
            probabilities, self.weights, np.array([1, 3])
        )
        scores = projection.transform(probabilities)[:, 0]
        difference = probabilities[:, 1] - probabilities[:, 3]
        centered = difference - np.average(difference, weights=self.weights)

        np.testing.assert_allclose(scores, centered / np.sqrt(2.0), atol=1e-12)
        np.testing.assert_allclose(
            projection.inverse_transform(projection.transform(probabilities)),
            probabilities,
            atol=1e-12,
        )

    def test_directions_are_orthonormal_contrasts(self) -> None:
        probabilities = np.full((200, 6), 1.0 / 6.0)
        projection = class_contrast_projection(
            probabilities, self.weights, np.array([4, 0, 2])
        )
        directions = projection.directions

        np.testing.assert_allclose(directions.T @ directions, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(directions.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(directions[[1, 3, 5]], 0.0, atol=1e-12)

    def test_matches_pca_when_only_those_classes_move(self) -> None:
        probabilities = np.full((200, 5), 0.1)
        probabilities[:, 0] = 0.4 + self.t
        probabilities[:, 2] = 0.3 - self.t
        contrast = class_contrast_projection(
            probabilities, self.weights, np.array([0, 2])
        )
        pca = fit_weighted_pca(probabilities, self.weights, 1)

        self.assertAlmostEqual(
            abs(float(contrast.directions[:, 0] @ pca.components[0])), 1.0, places=10
        )

    def test_misses_moving_class_outside_the_chosen_pair(self) -> None:
        # Class 0 is high but constant; classes 1 and 2 carry all variation.
        probabilities = np.full((200, 4), 0.05)
        probabilities[:, 0] = 0.5
        probabilities[:, 1] = 0.3 + self.t
        probabilities[:, 2] = 0.1 - self.t
        contrast = class_contrast_projection(
            probabilities, self.weights, np.array([0, 1])
        )
        reconstructed = contrast.inverse_transform(contrast.transform(probabilities))
        residual = np.sum((probabilities - reconstructed) ** 2)
        total = np.sum((probabilities - contrast.mean) ** 2)

        self.assertAlmostEqual(residual / total, 0.75, places=10)

    def test_rejects_invalid_classes(self) -> None:
        probabilities = np.full((10, 3), 1.0 / 3.0)
        weights = np.ones(10)
        with self.assertRaises(ValueError):
            class_contrast_projection(probabilities, weights, np.array([1]))
        with self.assertRaises(ValueError):
            class_contrast_projection(probabilities, weights, np.array([1, 1]))
        with self.assertRaises(ValueError):
            class_contrast_projection(probabilities, weights, np.array([0, 3]))


if __name__ == "__main__":
    unittest.main()
