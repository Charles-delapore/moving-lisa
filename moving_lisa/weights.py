"""Spatial weights handling for Moving LISA.

Weights are always row-standardized internally (the esda default), with a
zero diagonal.  Accepted inputs:

* a dense ``(n, n)`` array,
* a SciPy sparse matrix,
* a ``libpysal.weights.W`` object (duck-typed via ``.sparse`` or
  ``.neighbors`` / ``.weights``; libpysal is an optional dependency),
* a neighbour dict ``{i: [j, ...]}`` (binary) or ``{i: {j: w_ij}}``.

If no weights are supplied, :func:`queen_weights_from_polygons` builds
queen-contiguity weights from the unit geometries (the dissertation's
demonstration uses queen contiguity).
"""

from __future__ import annotations

import numpy as np

from .geometry import queen_weights_from_polygons

__all__ = ["as_weights_matrix", "row_standardize"]


def row_standardize(w):
    """Row-standardize a dense weights matrix (zero rows stay zero)."""
    w = np.asarray(w, dtype=float).copy()
    np.fill_diagonal(w, 0.0)
    w[w < 0] = 0.0
    row_sum = w.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(w, row_sum, out=np.zeros_like(w), where=row_sum > 0)
    return out


def _from_sparse(mat, n):
    return np.asarray(mat.toarray(), dtype=float)


def _from_libpysal(w, n):
    if hasattr(w, "sparse") and w.sparse is not None:
        return np.asarray(w.sparse.toarray(), dtype=float)
    neighbors = getattr(w, "neighbors", None)
    weights = getattr(w, "weights", None)
    if neighbors is not None and weights is not None:
        mat = np.zeros((n, n))
        for i, nbrs in neighbors.items():
            ws = weights[i]
            for j, wij in zip(nbrs, ws):
                mat[i, j] = wij
        return mat
    raise ValueError("unrecognised libpysal weights object")


def as_weights_matrix(w, n, geometries=None):
    """Normalise weights input to a row-standardized dense ``(n, n)`` array.

    Parameters
    ----------
    w : array, sparse matrix, libpysal W, neighbour dict, or None
        If None, queen-contiguity weights are built from ``geometries``.
    n : int
        Number of observational units.
    geometries : sequence of polygons, optional
        Required when ``w`` is None.
    """
    if w is None:
        if geometries is None:
            raise ValueError(
                "weights must be supplied, or geometries given so queen "
                "contiguity can be built"
            )
        neighbors = queen_weights_from_polygons(geometries)
        w = neighbors
    if isinstance(w, dict):
        mat = np.zeros((n, n))
        for i, nbrs in w.items():
            if isinstance(nbrs, dict):
                for j, wij in nbrs.items():
                    mat[i, j] = wij
            else:
                for j in nbrs:
                    mat[i, j] = 1.0
        return row_standardize(mat)
    if hasattr(w, "toarray") and not isinstance(w, np.ndarray):
        # SciPy sparse matrix (libpysal W also has .sparse, handled below).
        return row_standardize(_from_sparse(w, n))
    if hasattr(w, "sparse") or (
        hasattr(w, "neighbors") and hasattr(w, "weights")
    ):
        return row_standardize(_from_libpysal(w, n))
    arr = np.asarray(w, dtype=float)
    if arr.shape != (n, n):
        raise ValueError(f"weights matrix must be ({n}, {n}), got {arr.shape}")
    return row_standardize(arr)
