"""Tests for moving_lisa.movers (phase 2 union, phase 3 contiguity,
phase 4 types).  Overlap proportions are Monte Carlo values measured
with a fixed seed (n_samples = 5000, rng = default_rng(0)); the
tolerance covers RNG-implementation drift, not a change of method.
"""

import unittest

import numpy as np

from moving_lisa.data import load_toy
from moving_lisa.geometry import queen_weights_from_polygons
from moving_lisa.lisa import local_g
from moving_lisa.movers import (
    APPEARANCE,
    CONVERGENCE,
    DIVERGENCE,
    PERSISTENT,
    TERMINATION,
    Mover,
    assign_types,
    build_movers,
    contiguity_matrix,
    overlap_proportion,
)

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def grid(n_side):
    """Unit squares of an n_side x n_side lattice, plus queen weights."""
    polys = []
    for r in range(n_side):
        for c in range(n_side):
            polys.append(
                np.array(
                    [[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]],
                    dtype=float,
                )
            )
    n = len(polys)
    neighbors = queen_weights_from_polygons(polys)
    w = np.zeros((n, n))
    for i, js in neighbors.items():
        for j in js:
            w[i, j] = 1.0
    return polys, w / w.sum(axis=1, keepdims=True)


def toy_movers():
    """The toy movers at t=0 and t=1 (5-unit cross, Local G HH)."""
    toy = load_toy()
    out = []
    for t in (0, 1):
        res = local_g(toy["y"][:, t], toy["w"], permutations=999, random_state=0)
        out.append(
            build_movers(res.cluster == "HH", toy["polygons"], toy["w"])[0]
        )
    return out


def _mover(mid, time):
    return Mover(
        id=mid,
        time=time,
        members=(mid,),
        polygons=[SQUARE],
        centroid=(0.5, 0.5),
        area=1.0,
        std_distance=1.0,
    )


class TestBuildMovers(unittest.TestCase):
    def test_connected(self):
        polys, w = grid(3)
        # Opposite corners are not connected: two movers.
        mask = np.zeros(9, dtype=bool)
        mask[[0, 8]] = True
        movers = build_movers(mask, polys, w)
        self.assertEqual(len(movers), 2)
        self.assertEqual({m.members for m in movers}, {(0,), (8,)})

    def test_all_connected(self):
        polys, w = grid(3)
        movers = build_movers(np.ones(9, dtype=bool), polys, w)
        self.assertEqual(len(movers), 1)
        self.assertEqual(movers[0].members, tuple(range(9)))
        self.assertAlmostEqual(movers[0].area, 9.0)
        self.assertAlmostEqual(movers[0].centroid[0], 1.5)
        self.assertAlmostEqual(movers[0].centroid[1], 1.5)

    def test_empty(self):
        polys, w = grid(3)
        self.assertEqual(build_movers(np.zeros(9, dtype=bool), polys, w), [])

    def test_toy_mover(self):
        m0, _ = toy_movers()
        self.assertEqual(m0.members, (7, 12, 13, 14, 19))
        self.assertAlmostEqual(m0.area, 5.0)
        self.assertAlmostEqual(m0.centroid[0], 1.5)
        self.assertAlmostEqual(m0.centroid[1], 2.5)
        self.assertAlmostEqual(m0.std_distance, 1.1402, places=3)


class TestOverlapProportion(unittest.TestCase):
    def test_degenerate_ancestor(self):
        # Zero standard distance: the distribution collapses to the
        # ancestor's centroid, so the proportion is 0 or 1.
        ancestor = Mover(
            id=0, time=0, members=(0,), polygons=[SQUARE],
            centroid=(0.5, 0.5), area=1.0, std_distance=0.0,
        )
        inside = Mover(
            id=0, time=1, members=(0,), polygons=[SQUARE],
            centroid=(0.5, 0.5), area=1.0, std_distance=1.0,
        )
        far = np.array([[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]])
        outside = Mover(
            id=1, time=1, members=(1,), polygons=[far],
            centroid=(2.5, 0.5), area=1.0, std_distance=1.0,
        )
        rng = np.random.default_rng(0)
        self.assertEqual(
            overlap_proportion(ancestor, inside, 100, 2.0, rng), 1.0
        )
        self.assertEqual(
            overlap_proportion(ancestor, outside, 100, 2.0, rng), 0.0
        )

    def test_regression_values(self):
        m0, m1 = toy_movers()
        # (phi, identical, t0 -> t1), measured with n_samples = 5000
        # and a fresh default_rng(0) per call.
        cases = [
            (0.5, 0.651, 0.457),
            (1.0, 0.433, 0.338),
            (1.1, 0.405, 0.321),
            (1.5, 0.323, 0.267),
            (2.0, 0.253, 0.215),
        ]
        for phi, identical, forward in cases:
            a = overlap_proportion(
                m0, m0, 5000, phi, np.random.default_rng(0)
            )
            b = overlap_proportion(
                m0, m1, 5000, phi, np.random.default_rng(0)
            )
            self.assertAlmostEqual(a, identical, delta=0.02)
            self.assertAlmostEqual(b, forward, delta=0.02)
            # A wider spread (larger phi) can only lower the overlap.
            self.assertLess(b, a)

    def test_contiguity_matrix(self):
        m0, m1 = toy_movers()
        # At phi = 1.0 the t0 -> t1 overlap (0.338) clears pi = 0.3;
        # at phi = 2.0 it (0.215) does not.
        mat = contiguity_matrix(
            [m0], [m1], 0.3, 1.0, 5000, np.random.default_rng(0)
        )
        self.assertTrue(mat[0, 0])
        mat = contiguity_matrix(
            [m0], [m1], 0.3, 2.0, 5000, np.random.default_rng(0)
        )
        self.assertFalse(mat[0, 0])


class TestAssignTypes(unittest.TestCase):
    def test_appearance(self):
        prev = [_mover(0, 0)]
        curr = [_mover(0, 1)]
        types, ancestors, terminated = assign_types(
            prev, curr, np.array([[False]])
        )
        self.assertEqual(types, [APPEARANCE])
        self.assertEqual(ancestors, [()])
        self.assertEqual(terminated, prev)

    def test_persistent(self):
        prev = [_mover(0, 0)]
        curr = [_mover(0, 1)]
        types, ancestors, terminated = assign_types(
            prev, curr, np.array([[True]])
        )
        self.assertEqual(types, [PERSISTENT])
        self.assertEqual(ancestors, [((0, 0),)])
        self.assertEqual(terminated, [])

    def test_convergence(self):
        prev = [_mover(0, 0), _mover(1, 0)]
        curr = [_mover(0, 1)]
        types, ancestors, terminated = assign_types(
            prev, curr, np.array([[True], [True]])
        )
        self.assertEqual(types, [CONVERGENCE])
        self.assertEqual(ancestors, [((0, 0), (0, 1))])
        self.assertEqual(terminated, [])

    def test_divergence(self):
        prev = [_mover(0, 0)]
        curr = [_mover(0, 1), _mover(1, 1)]
        types, ancestors, terminated = assign_types(
            prev, curr, np.array([[True, True]])
        )
        self.assertEqual(types, [DIVERGENCE, DIVERGENCE])
        self.assertEqual(ancestors, [((0, 0),), ((0, 0),)])
        self.assertEqual(terminated, [])

    def test_mixed(self):
        prev = [_mover(0, 0), _mover(1, 0)]
        curr = [_mover(0, 1), _mover(1, 1)]
        # Ancestor 0 splits into two current movers (divergence);
        # ancestor 1 has no descendants (termination).
        types, ancestors, terminated = assign_types(
            prev, curr, np.array([[True, True], [False, False]])
        )
        self.assertEqual(types, [DIVERGENCE, DIVERGENCE])
        self.assertEqual(terminated, [prev[1]])
        # Two separate persistent pairs.
        types, _, terminated = assign_types(
            prev, curr, np.array([[True, False], [False, True]])
        )
        self.assertEqual(types, [PERSISTENT, PERSISTENT])
        self.assertEqual(terminated, [])


if __name__ == "__main__":
    unittest.main()
