"""Tests for moving_lisa.lisa (Local Moran's I and Getis-Ord Local G).

Reference values for the 4-unit line are hand-computed (see the module
docstring) and were verified against esda 2.10.0 / libpysal 4.15.1.
The esda cross-checks are skipped when esda / libpysal are not
installed; they use continuous random data, for which the simulated
distributions have no ties.
"""

import importlib.util
import unittest

import numpy as np

from moving_lisa.lisa import _p_values, local_g, local_moran

HAS_ESDA = importlib.util.find_spec("esda") is not None and (
    importlib.util.find_spec("libpysal") is not None
)

# 4-unit line, x = [0, 0, 10, 10], rook weights, row-standardized.
LINE_X = np.array([0.0, 0.0, 10.0, 10.0])
LINE_W = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.0],
    ]
)


def queen_weights_6x6():
    """Row-standardized queen weights for a 6x6 lattice of unit squares."""
    n_side = 6
    n = n_side * n_side
    w = np.zeros((n, n))
    for r in range(n_side):
        for c in range(n_side):
            i = r * n_side + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n_side and 0 <= cc < n_side:
                        w[i, rr * n_side + cc] = 1.0
    return w / w.sum(axis=1, keepdims=True)


class TestLocalMoranLine(unittest.TestCase):
    def test_statistic_and_quadrants(self):
        # esda scaling: I = (n-1) * z * lag / sum(z^2); the textbook
        # population-moment version would be [1, 0, 0, 1].
        res = local_moran(LINE_X, LINE_W, permutations=999, random_state=0)
        np.testing.assert_allclose(
            res.statistic, [0.75, 0.0, 0.0, 0.75], atol=1e-12
        )
        # PySAL quadrants: 1 HH, 2 LH, 3 LL, 4 HL (zero lag is low).
        np.testing.assert_array_equal(res.quadrant, [3, 3, 4, 1])
        # No unit is significant at alpha = 0.05 (p ~ 1/3 each).
        self.assertEqual(list(res.cluster), ["NS"] * 4)
        self.assertTrue(np.all((res.p_value > 0.2) & (res.p_value < 0.5)))

    def test_constant_values(self):
        # Degenerate input: no crash, statistic is zero.
        res = local_moran(np.ones(4), LINE_W, permutations=99, random_state=0)
        np.testing.assert_allclose(res.statistic, 0.0, atol=1e-12)

    def test_validation(self):
        with self.assertRaises(ValueError):
            local_moran([1.0, 2.0], LINE_W[:2, :2])
        with self.assertRaises(ValueError):
            local_moran(LINE_X, LINE_W, permutations=0)


class TestLocalGLine(unittest.TestCase):
    def test_statistic_and_z(self):
        # Row-standardized G (a binary-weight hand calculation would
        # give [0, 0.5, 1, 1]; row-standardization is the difference).
        res = local_g(LINE_X, LINE_W, permutations=999, random_state=0)
        np.testing.assert_allclose(
            res.statistic, [0.0, 0.25, 0.5, 1.0], atol=1e-12
        )
        np.testing.assert_allclose(
            res.z,
            [-1.41421356, -0.35355339, 0.35355339, 1.41421356],
            atol=1e-8,
        )
        # Units 0 and 1 sit at the minimum of their simulated
        # distributions, so the min-tail rule gives p = 0.001 (the
        # tie/boundary behaviour of the (M + 1) / (R + 1) rule); the
        # middle units are not significant.
        np.testing.assert_allclose(
            res.p_value, [0.001, 0.001, 0.328, 0.302], atol=0.02
        )
        self.assertEqual(list(res.cluster), ["LL", "LL", "NS", "NS"])

    def test_constant_values(self):
        res = local_g(np.ones(4), LINE_W, permutations=99, random_state=0)
        self.assertTrue(np.all(np.isfinite(res.statistic)))
        self.assertEqual(list(res.cluster), ["NS"] * 4)

    def test_validation(self):
        with self.assertRaises(ValueError):
            local_g([1.0, 2.0], LINE_W[:2, :2])
        with self.assertRaises(ValueError):
            local_g(LINE_X, LINE_W, permutations=0)


class TestPValues(unittest.TestCase):
    def test_rules(self):
        # 99 simulations uniformly spread over [0, 1]; obs = 0.5 sits
        # exactly on a simulation value (index 49 = 49/98 = 0.5).
        sim = np.linspace(0.0, 1.0, 99).reshape(-1, 1)
        obs = np.array([0.5])
        # 50 sims >= 0.5 (ties count), 50 sims <= 0.5.
        self.assertAlmostEqual(
            _p_values(obs, sim, "greater")[0], 51 / 100
        )
        self.assertAlmostEqual(
            _p_values(obs, sim, "lesser")[0], 51 / 100
        )
        # Two-sided (min tail): min(50, 99 - 50) = 49.
        self.assertAlmostEqual(
            _p_values(obs, sim, "two-sided")[0], 50 / 100
        )

    def test_ties_count(self):
        # All simulations equal the observed value: the tie-tolerant
        # rule counts them, so "greater" is 1.0 and "two-sided" is the
        # minimum tail (0 here -> (0 + 1) / (R + 1)).
        sim = np.full((99, 1), 0.5)
        obs = np.array([0.5])
        self.assertAlmostEqual(
            _p_values(obs, sim, "greater")[0], 1.0
        )
        self.assertAlmostEqual(
            _p_values(obs, sim, "two-sided")[0], 1 / 100
        )

    def test_bad_alternative(self):
        with self.assertRaises(ValueError):
            _p_values(np.array([0.5]), np.zeros((9, 1)), "one-sided")


@unittest.skipUnless(HAS_ESDA, "esda / libpysal not installed")
class TestAgainstEsda(unittest.TestCase):
    """Cross-check against esda 2.10.0 on continuous random data."""

    def setUp(self):
        from libpysal.weights import W

        self.w = queen_weights_6x6()
        n = 36
        neighbors = {
            i: list(np.flatnonzero(self.w[i] > 0)) for i in range(n)
        }
        self.W = W(neighbors)
        # Continuous values: the simulated distributions have no ties,
        # so p-values agree up to Monte Carlo noise.
        self.y = np.random.default_rng(42).normal(0.0, 1.0, n)

    def test_moran(self):
        from esda import Moran_Local

        mine = local_moran(self.y, self.w, permutations=999, random_state=0)
        ref = Moran_Local(self.y, self.W, permutations=999, seed=0)
        np.testing.assert_allclose(mine.statistic, ref.Is, atol=1e-12)
        np.testing.assert_array_equal(mine.quadrant, ref.q)
        np.testing.assert_allclose(
            mine.p_value, ref.p_sim, atol=0.08
        )
        # Significance decisions agree except possibly on borderline
        # units (p near 0.05, where Monte Carlo noise flips the call).
        diff = np.flatnonzero(
            (mine.p_value < 0.05) != (ref.p_sim < 0.05)
        )
        self.assertLessEqual(len(diff), 2)

    def test_g(self):
        from esda import G_Local

        mine = local_g(self.y, self.w, permutations=999, random_state=0)
        ref = G_Local(self.y, self.W, transform="R", permutations=999, seed=0)
        np.testing.assert_allclose(mine.statistic, ref.Gs, atol=1e-12)
        np.testing.assert_allclose(mine.z, ref.Zs, atol=1e-12)
        np.testing.assert_allclose(
            mine.p_value, ref.p_sim, atol=0.08
        )
        diff = np.flatnonzero(
            (mine.p_value < 0.05) != (ref.p_sim < 0.05)
        )
        self.assertLessEqual(len(diff), 2)


if __name__ == "__main__":
    unittest.main()
