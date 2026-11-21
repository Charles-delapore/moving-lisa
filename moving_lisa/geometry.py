"""Geometry helpers for Moving LISA.

The Moving LISA procedure (Anderson & Yang 2025, dissertation Ch. 2 and
Appendix A) works with vector-polygon observational units.  This module
implements the geometric primitives the procedure needs using only NumPy,
so the core library has no shapely / geopandas dependency:

* polygon centroid by the Bourke (1988) formula (dissertation Eq. 6),
* standard distance of a (multi-)polygon,
* point-in-polygon tests (ray casting) for the contiguity overlay,
* queen-contiguity weights built from shared polygon vertices.

A "mover" is represented as a multi-polygon: the list of its member unit
polygons.  For non-overlapping lattice units (the dissertation's setting:
ZCTAs, grid cells) this representation is exact for the centroid
(area-weighted mean of member centroids equals the dissolved-polygon
centroid) and for point-in-polygon tests.  The standard distance is
computed from all member-polygon vertices; for a dissolved polygon this
is an approximation, because shared-boundary vertices are interior to the
dissolved shape.  The difference is small and the scaling factor phi is
user-tunable; the approximation is documented in the README.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "as_polygons",
    "close_ring",
    "multipolygon_area_centroid",
    "points_in_multipolygon",
    "points_in_polygon",
    "polygon_area_centroid",
    "queen_weights_from_polygons",
    "standard_distance",
]


def close_ring(poly):
    """Return a polygon ring with the first vertex repeated at the end."""
    poly = np.asarray(poly, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("a polygon must be an (k, 2) array of vertices")
    if len(poly) < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def polygon_area_centroid(poly):
    """Signed area and centroid of a polygon (Bourke 1988; Eq. 6).

    Returns ``(area, cx, cy)`` where ``area`` is the signed shoelace area.
    For a degenerate (zero-area) polygon the centroid falls back to the
    mean of the vertices.
    """
    ring = close_ring(poly)
    x = ring[:, 0]
    y = ring[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area2 = cross.sum()  # twice the signed area
    area = 0.5 * area2
    if area2 == 0.0:
        return 0.0, float(x[:-1].mean()), float(y[:-1].mean())
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6.0 * area)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6.0 * area)
    return float(area), float(cx), float(cy)


def multipolygon_area_centroid(polys):
    """Area and area-weighted centroid of a list of polygons.

    For non-overlapping member polygons this equals the centroid of the
    dissolved (union) polygon.
    """
    total_area = 0.0
    cx_sum = 0.0
    cy_sum = 0.0
    for poly in polys:
        area, cx, cy = polygon_area_centroid(poly)
        w = abs(area)
        total_area += w
        cx_sum += w * cx
        cy_sum += w * cy
    if total_area == 0.0:
        # Degenerate: fall back to the mean of all vertices.
        verts = np.vstack([np.asarray(p, dtype=float) for p in polys])
        return 0.0, float(verts[:, 0].mean()), float(verts[:, 1].mean())
    return total_area, cx_sum / total_area, cy_sum / total_area


def standard_distance(polys, centroid):
    """Standard distance of a (multi-)polygon about its centroid.

    The root-mean-square distance of all member-polygon vertices from the
    centroid.  See the module docstring for the dissolved-polygon caveat.
    """
    cx, cy = centroid
    verts = np.vstack(
        [close_ring(p)[:-1] for p in polys]
    )
    d2 = (verts[:, 0] - cx) ** 2 + (verts[:, 1] - cy) ** 2
    return float(np.sqrt(d2.mean()))


def points_in_polygon(points, poly):
    """Ray-casting point-in-polygon test.

    Parameters
    ----------
    points : (m, 2) array
    poly : (k, 2) array of vertices

    Returns
    -------
    inside : (m,) boolean array.  Points exactly on the boundary count
        as inside.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    ring = close_ring(poly)
    x = ring[:, 0]
    y = ring[:, 1]
    px = points[:, 0]
    py = points[:, 1]
    # Points on the boundary count as inside (kept separate from the
    # ray casting below, whose parity toggle must not undo it).
    on_boundary = np.zeros(len(points), dtype=bool)
    for i in range(len(ring) - 1):
        x1, y1 = x[i], y[i]
        x2, y2 = x[i + 1], y[i + 1]
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        dot = (px - x1) * (px - x2) + (py - y1) * (py - y2)
        on_boundary |= (np.abs(cross) <= 1e-12) & (dot <= 1e-12)
    # Ray casting: count crossings of a ray to +infinity in x.
    # (The division is guarded: when yj == yi the edge cannot straddle
    # the horizontal line through the point, so the term is unused.)
    inside = np.zeros(len(points), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(ring) - 1):
            xi, yi = x[i], y[i]
            xj, yj = x[i + 1], y[i + 1]
            # Edge straddles the horizontal line through the point.
            cond = ((yi > py) != (yj > py)) & (
                px < (xj - xi) * (py - yi) / (yj - yi) + xi
            )
            inside ^= cond
    return inside | on_boundary


def points_in_multipolygon(points, polys):
    """True if each point lies in any of the given polygons."""
    inside = np.zeros(len(np.atleast_2d(points)), dtype=bool)
    for poly in polys:
        inside |= points_in_polygon(points, poly)
    return inside


def _as_polygon(obj):
    """Normalise one geometry to an (k, 2) vertex array.

    Accepts plain (k, 2) arrays, shapely polygons (via ``__geo_interface__``
    or ``.exterior.coords``) and anything exposing ``exterior.coords``.
    """
    if hasattr(obj, "__geo_interface__"):
        geo = obj.__geo_interface__
        if geo.get("type") == "Polygon":
            return np.asarray(geo["coordinates"][0], dtype=float)
        raise ValueError(f"unsupported geometry type: {geo.get('type')}")
    if hasattr(obj, "exterior") and hasattr(obj.exterior, "coords"):
        return np.asarray(obj.exterior.coords, dtype=float)
    return np.asarray(obj, dtype=float)


def as_polygons(geometries):
    """Normalise a collection of geometries to a list of (k, 2) arrays.

    Accepts a list of polygons, a GeoDataFrame (uses its ``geometry``
    column), a GeoSeries, or any sequence of shapely geometries.
    """
    if hasattr(geometries, "geometry"):
        geometries = geometries.geometry
    if hasattr(geometries, "__iter__") and not isinstance(geometries, np.ndarray):
        try:
            items = list(geometries)
        except TypeError:
            items = [geometries]
    else:
        items = list(geometries)
    return [_as_polygon(g) for g in items]


def queen_weights_from_polygons(polys, decimals=9):
    """Queen-contiguity neighbour lists from shared polygon vertices.

    Two units are neighbours if they share at least one vertex (after
    rounding coordinates to ``decimals`` places, which guards against
    float noise in coordinates read from files).

    Returns
    -------
    neighbors : dict {unit_index: [neighbour indices, ...]}
    """
    vertex_sets = []
    for poly in polys:
        ring = close_ring(poly)
        verts = {
            (round(float(a), decimals), round(float(b), decimals))
            for a, b in ring[:-1]
        }
        vertex_sets.append(verts)
    n = len(polys)
    neighbors = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if vertex_sets[i] & vertex_sets[j]:
                neighbors[i].append(j)
                neighbors[j].append(i)
    return neighbors
