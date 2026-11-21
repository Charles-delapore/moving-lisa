"""Tests for moving_lisa.geometry (Bourke centroid, standard distance,
point-in-polygon, queen weights).  Reference values are hand-computed.
"""

import unittest

import numpy as np

from moving_lisa.geometry import (
    as_polygons,
    close_ring,
    multipolygon_area_centroid,
    points_in_multipolygon,
    points_in_polygon,
    polygon_area_centroid,
    queen_weights_from_polygons,
    standard_distance,
)

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
TRIANGLE = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])


def grid(n_side):
    """Unit squares of an n_side x n_side lattice (row-major)."""
    polys = []
    for r in range(n_side):
        for c in range(n_side):
            polys.append(
                np.array(
                    [[c, r], [c + 1, r], [c + 1, r + 1], [c, r + 1]],
                    dtype=float,
                )
            )
    return polys


class TestCentroid(unittest.TestCase):
    def test_unit_square(self):
        area, cx, cy = polygon_area_centroid(SQUARE)
        self.assertAlmostEqual(area, 1.0)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)

    def test_right_triangle(self):
        # Legs of length 2: area 2, centroid at (2/3, 2/3).
        area, cx, cy = polygon_area_centroid(TRIANGLE)
        self.assertAlmostEqual(area, 2.0)
        self.assertAlmostEqual(cx, 2.0 / 3.0)
        self.assertAlmostEqual(cy, 2.0 / 3.0)

    def test_clockwise_ring(self):
        # Signed area is negative for a clockwise ring; centroid same.
        area, cx, cy = polygon_area_centroid(SQUARE[::-1])
        self.assertAlmostEqual(area, -1.0)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)

    def test_closed_ring_input(self):
        ring = np.vstack([SQUARE, SQUARE[0]])
        area, cx, cy = polygon_area_centroid(ring)
        self.assertAlmostEqual(area, 1.0)
        self.assertAlmostEqual(cx, 0.5)
        self.assertAlmostEqual(cy, 0.5)

    def test_multipolygon(self):
        left = SQUARE
        right = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
        area, cx, cy = multipolygon_area_centroid([left, right])
        self.assertAlmostEqual(area, 2.0)
        self.assertAlmostEqual(cx, 1.0)
        self.assertAlmostEqual(cy, 0.5)

    def test_close_ring(self):
        closed = close_ring(SQUARE)
        self.assertEqual(len(closed), 5)
        self.assertTrue(np.array_equal(closed[0], closed[-1]))
        # Already closed: unchanged.
        self.assertTrue(np.array_equal(close_ring(closed), closed))
        with self.assertRaises(ValueError):
            close_ring(np.array([[0.0, 0.0], [1.0, 0.0]]))


class TestStandardDistance(unittest.TestCase):
    def test_unit_square(self):
        # All four vertices are sqrt(0.5) from the centroid (0.5, 0.5).
        sd = standard_distance([SQUARE], (0.5, 0.5))
        self.assertAlmostEqual(sd, np.sqrt(0.5))

    def test_two_squares(self):
        # 8 vertices about the centroid (1, 0.5): four at distance
        # sqrt(1.25) (the outer corners) and four at 0.5 (the shared
        # edge).  Mean squared distance = (4*1.25 + 4*0.25) / 8 = 0.75.
        right = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
        sd = standard_distance([SQUARE, right], (1.0, 0.5))
        self.assertAlmostEqual(sd, np.sqrt(0.75))


class TestPointInPolygon(unittest.TestCase):
    def test_square(self):
        pts = np.array(
            [
                [0.5, 0.5],  # inside
                [1.5, 0.5],  # outside (right)
                [0.5, -0.1],  # outside (below)
                [2.0, 2.0],  # outside (far)
                [0.0, 0.5],  # on edge
                [0.0, 0.0],  # on vertex
                [1.0, 1.0],  # on vertex
            ]
        )
        inside = points_in_polygon(pts, SQUARE)
        self.assertEqual(
            list(inside), [True, False, False, False, True, True, True]
        )

    def test_triangle(self):
        # The hypotenuse is x + y = 2: (1.5, 0.5) lies on it (boundary
        # counts as inside); (1.0, 1.5) is outside it.
        pts = np.array([[0.5, 0.5], [1.5, 0.5], [1.0, 1.5], [0.1, 0.1]])
        inside = points_in_polygon(pts, TRIANGLE)
        self.assertEqual(list(inside), [True, True, False, True])

    def test_multipolygon(self):
        right = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
        pts = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
        inside = points_in_multipolygon(pts, [SQUARE, right])
        self.assertEqual(list(inside), [True, True, False])


class TestQueenWeights(unittest.TestCase):
    def test_2x2_grid(self):
        # Every unit touches every other unit (edges and corners).
        neighbors = queen_weights_from_polygons(grid(2))
        for i in range(4):
            self.assertEqual(sorted(neighbors[i]), [j for j in range(4) if j != i])

    def test_3x3_grid(self):
        neighbors = queen_weights_from_polygons(grid(3))
        # Centre (4) has 8 neighbours; corner (0) has 3.
        self.assertEqual(sorted(neighbors[4]), [0, 1, 2, 3, 5, 6, 7, 8])
        self.assertEqual(sorted(neighbors[0]), [1, 3, 4])
        # Edge-centre (1) has 5 neighbours.
        self.assertEqual(sorted(neighbors[1]), [0, 2, 3, 4, 5])

    def test_as_polygons_passthrough(self):
        polys = grid(2)
        self.assertEqual(len(as_polygons(polys)), 4)
        for p in as_polygons(polys):
            self.assertEqual(p.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()
