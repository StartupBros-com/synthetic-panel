"""Exercise SSR temperature scaling without calling an embedding provider."""

import math
import unittest
from unittest.mock import patch

from synthetic_panel.scoring import calculate_ssr_score


class ScoringTemperatureTests(unittest.TestCase):
    def score(self, cosines, temperature):
        anchors = {i: str(i) for i in range(1, 6)}
        vectors = [[c, math.sqrt(1 - c * c)] for c in cosines]
        with (
            patch("synthetic_panel.scoring.get_embedding", return_value=[1.0, 0.0]),
            patch("synthetic_panel.scoring.get_embeddings_batch", return_value=vectors),
        ):
            score, distribution, method = calculate_ssr_score(
                "Synthetic response", {"set_1": anchors, "set_2": anchors},
                temperature=temperature,
            )
        self.assertEqual(method, "ssr")
        self.assertTrue(all(math.isfinite(p) and p >= 0 for p in distribution))
        self.assertAlmostEqual(sum(distribution), 1.0, delta=1e-12)
        return score, distribution

    def test_tiny_positive_temperatures_sharpen_the_actual_score(self):
        evidence = [0.0, 0.2, 0.4, 0.6, 0.8]
        default_score, default_distribution = self.score(evidence, 1.0)
        self.assertAlmostEqual(default_score, 4.0, delta=1e-12)
        for actual, expected in zip(default_distribution, [0.0, 0.1, 0.2, 0.3, 0.4]):
            self.assertAlmostEqual(actual, expected, delta=1e-12)

        for temperature in (0.001, 5e-324):
            with self.subTest(temperature=temperature):
                score, distribution = self.score(evidence, temperature)
                self.assertEqual(distribution[0], 0.0)
                self.assertAlmostEqual(distribution[4], 1.0, delta=1e-12)
                self.assertAlmostEqual(score, 5.0, delta=1e-12)

    def test_tiny_temperature_keeps_max_ties_and_true_no_signal(self):
        for temperature in (0.001, 5e-324):
            with self.subTest(temperature=temperature):
                score, distribution = self.score([0.0, 0.2, 0.8, 0.4, 0.8], temperature)
                self.assertEqual(distribution[0], 0.0)
                self.assertEqual(distribution[2], distribution[4])
                self.assertAlmostEqual(distribution[2], 0.5, delta=1e-12)
                self.assertAlmostEqual(score, 4.0, delta=1e-12)

                score, distribution = self.score([0.0] * 5, temperature)
                self.assertEqual(distribution, [0.2] * 5)
                self.assertAlmostEqual(score, 3.0, delta=1e-12)
