"""Tests for moving_lisa.core (the full 5-phase Moving LISA run).

The toy expectations were produced by running the library itself with
a fixed seed (random_state = 0); they pin the end-to-end behaviour,
including the mover members, the lineage and the anchor drift.
"""

import importlib.util
import unittest

import numpy as np

from moving_lisa import MovingLISA, load_toy

HAS_PANDAS = importlib.util.find_spec("pandas") is not None

# Chapter 3 application parameters (dissertation Ch. 3.3.2): the toy
# mover links at these values.
TOY_KWARGS = dict(pi=0.2, phi=1.1, permutations=999, random_state=0)


class TestToyRun(unittest.TestCase):
    def _check_persistent_run(self, res):
        self.assertEqual(
            res.counts_,
            {
                "appearance": 1,
                "convergence": 0,
                "divergence": 0,
                "persistent": 2,
                "termination": 0,
            },
        )
        self.assertEqual(len(res.trajectories_), 1)
        traj = res.trajectories_[0]
        self.assertEqual(traj["lineage_id"], 0)
        anchors = traj["anchors"]
        self.assertEqual(len(anchors), 3)
        # Anchors drift one cell right per step, y stays at 2.5.
        self.assertEqual(
            [(a[1], a[2]) for a in anchors],
            [(1.5, 2.5), (2.5, 2.5), (3.5, 2.5)],
        )
        self.assertEqual(
            [a[3] for a in anchors],
            ["appearance", "persistent", "persistent"],
        )
        self.assertEqual(
            [row["members"] for row in res.table_],
            [
                (7, 12, 13, 14, 19),
                (8, 13, 14, 15, 20),
                (9, 14, 15, 16, 21),
            ],
        )
        self.assertEqual(res.terminations_, [])

    def test_g(self):
        toy = load_toy()
        res = MovingLISA(
            toy["y"], toy["polygons"], w=toy["w"],
            statistic="g", **TOY_KWARGS,
        )
        self._check_persistent_run(res)

    def test_moran(self):
        toy = load_toy()
        res = MovingLISA(
            toy["y"], toy["polygons"], w=toy["w"],
            statistic="moran", **TOY_KWARGS,
        )
        self._check_persistent_run(res)

    def test_ch2_defaults_no_persistence(self):
        # At the dissertation Ch. 2 parameters (pi = 0.3, phi = 2.0)
        # the toy mover does not link: the overlap (0.215) is below pi.
        # This is the documented parameter sensitivity, not a bug.
        toy = load_toy()
        for statistic in ("g", "moran"):
            res = MovingLISA(
                toy["y"], toy["polygons"], w=toy["w"],
                statistic=statistic, pi=0.3, phi=2.0,
                permutations=999, random_state=0,
            )
            self.assertEqual(res.counts_["appearance"], 3)
            self.assertEqual(res.counts_["persistent"], 0)
            self.assertEqual(res.counts_["termination"], 2)
            self.assertEqual(
                [row["type"] for row in res.table_],
                ["appearance"] * 3,
            )

    def test_single_time_step(self):
        toy = load_toy()
        res = MovingLISA(
            toy["y"][:, :1], toy["polygons"], w=toy["w"],
            statistic="g", **TOY_KWARGS,
        )
        self.assertEqual(len(res.table_), 1)
        self.assertEqual(res.table_[0]["type"], "appearance")
        self.assertEqual(res.counts_["appearance"], 1)

    def test_no_clusters(self):
        # Constant values: no significant clusters, no movers.
        toy = load_toy()
        res = MovingLISA(
            np.ones((36, 3)), toy["polygons"], w=toy["w"],
            statistic="g", **TOY_KWARGS,
        )
        self.assertEqual(res.table_, [])
        self.assertEqual(res.trajectories_, [])
        self.assertEqual(res.counts_["appearance"], 0)

    def test_reproducible(self):
        toy = load_toy()
        a = MovingLISA(
            toy["y"], toy["polygons"], w=toy["w"],
            statistic="g", **TOY_KWARGS,
        )
        b = MovingLISA(
            toy["y"], toy["polygons"], w=toy["w"],
            statistic="g", **TOY_KWARGS,
        )
        self.assertEqual(a.table_, b.table_)
        np.testing.assert_array_equal(a.lisa_p_, b.lisa_p_)

    def test_queen_weights_built_when_none(self):
        # w=None builds queen-contiguity weights from the geometries.
        toy = load_toy()
        res = MovingLISA(
            toy["y"], toy["polygons"], statistic="g", **TOY_KWARGS,
        )
        self._check_persistent_run(res)


class TestLoadToy(unittest.TestCase):
    def test_shapes(self):
        toy = load_toy()
        self.assertEqual(toy["y"].shape, (36, 3))
        self.assertEqual(len(toy["polygons"]), 36)
        self.assertEqual(toy["w"].shape, (36, 36))
        np.testing.assert_array_equal(toy["times"], [0, 1, 2])
        # The hotspot (value 10) is a 3x3 block moving right.
        for t, c0 in enumerate([0, 1, 2]):
            hot = {r * 6 + c for r in (1, 2, 3) for c in (c0, c0 + 1, c0 + 2)}
            self.assertEqual(set(np.flatnonzero(toy["y"][:, t] == 10)), hot)
            self.assertEqual(set(np.flatnonzero(toy["y"][:, t] == 1)), set(range(36)) - hot)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.toy = load_toy()

    def _run(self, **kwargs):
        return MovingLISA(
            self.toy["y"], self.toy["polygons"], w=self.toy["w"], **kwargs
        )

    def test_bad_statistic(self):
        with self.assertRaises(ValueError):
            self._run(statistic="geary")

    def test_bad_alpha(self):
        with self.assertRaises(ValueError):
            self._run(alpha=0.0)
        with self.assertRaises(ValueError):
            self._run(alpha=1.0)

    def test_bad_pi(self):
        with self.assertRaises(ValueError):
            self._run(pi=-0.1)
        with self.assertRaises(ValueError):
            self._run(pi=1.1)

    def test_bad_phi(self):
        with self.assertRaises(ValueError):
            self._run(phi=0.0)

    def test_bad_cluster_types(self):
        with self.assertRaises(ValueError):
            self._run(statistic="g", cluster_types=("HL",))
        with self.assertRaises(ValueError):
            self._run(statistic="moran", cluster_types=("XX",))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            MovingLISA(
                self.toy["y"][:, 0], self.toy["polygons"], w=self.toy["w"]
            )
        with self.assertRaises(ValueError):
            MovingLISA(
                self.toy["y"], self.toy["polygons"][:10], w=self.toy["w"]
            )

    def test_summary_and_repr(self):
        res = self._run(statistic="g", **TOY_KWARGS)
        self.assertEqual(res.summary(), res.counts_)
        self.assertIn("MovingLISA", repr(res))
        self.assertIn("statistic='g'", repr(res))


@unittest.skipUnless(HAS_PANDAS, "pandas not installed")
class TestDataFrame(unittest.TestCase):
    def test_to_dataframe(self):
        toy = load_toy()
        res = MovingLISA(
            toy["y"], toy["polygons"], w=toy["w"],
            statistic="g", **TOY_KWARGS,
        )
        df = res.to_dataframe()
        self.assertEqual(len(df), 3)
        self.assertEqual(
            list(df.columns),
            [
                "time", "mover_id", "lineage_id", "type", "n_units",
                "members", "centroid_x", "centroid_y", "ancestors",
                "parent_lineages",
            ],
        )
        self.assertEqual(list(df["type"]), ["appearance", "persistent", "persistent"])
        self.assertEqual(list(df["centroid_x"]), [1.5, 2.5, 3.5])


if __name__ == "__main__":
    unittest.main()
