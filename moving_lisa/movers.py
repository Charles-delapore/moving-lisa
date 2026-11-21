"""Movers: geometric union, contiguity overlay and lineage tracking.

This module implements phases 2-4 of the Moving LISA procedure
(Anderson & Yang 2025, dissertation Ch. 2 and Appendix A):

* Phase 2 (geometric union): significant clustering geographies at one
  time step are grouped into movers.  Two clustering units belong to the
  same mover when they are connected through the spatial weights matrix
  (for queen-contiguity weights this is exactly "sharing a boundary or a
  vertex", i.e. the geometric union followed by dissolving).
* Phase 3 (contiguity, Eq. 4): a mover ``A`` at ``t - 1`` is contiguous
  with a mover ``B`` at ``t`` when, for samples drawn from an uncorrelated
  bivariate normal distribution centred on ``A``'s centroid with
  variance ``std_distance(A)**2 * phi`` (Eq. 4 and Appendix A lines
  19-21; the prose under Eq. 4 describes the standard deviation as
  ``std_distance * phi`` instead, which contradicts the equation and the
  pseudocode -- the equation/pseudocode form is used here), the
  proportion of sampled points falling inside ``B``'s geometry is at
  least ``pi``.
* Phase 4 (contiguity types, Eq. 5): appearance / convergence /
  divergence / persistent / termination, following the prose definitions
  in section 2.3.  One point of the pseudocode needs interpretation:
  its ``count(C) > 1 and count_unique(C) == 1 -> DIVERGENCE`` branch
  assumes the ancestor list ``C`` can contain repeat identifiers, which
  cannot happen for a per-mover ancestor list.  It is implemented here
  via the ancestor's descendant count (the prose definition): a current
  mover with a single ancestor is a "divergence" when that ancestor is
  contiguous with more than one current mover, and "persistent"
  otherwise.  The two formulations coincide when ``C`` is read as the
  multiset of ancestor identifiers over all current movers sharing that
  ancestor.

A mover's geometry is the list of its member unit polygons (a
multi-polygon); see :mod:`moving_lisa.geometry` for the exactness and
approximation notes of that representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import (
    multipolygon_area_centroid,
    points_in_multipolygon,
    standard_distance,
)

__all__ = [
    "Mover",
    "assign_types",
    "build_movers",
    "contiguity_matrix",
    "overlap_proportion",
]

#: Contiguity type labels (dissertation Eq. 5).
APPEARANCE = "appearance"
CONVERGENCE = "convergence"
DIVERGENCE = "divergence"
PERSISTENT = "persistent"
TERMINATION = "termination"


@dataclass
class Mover:
    """One aggregate mover at one time step."""

    id: int  # unique within the time step
    time: int
    members: tuple  # unit indices forming the mover
    polygons: list  # member unit polygons, (k, 2) vertex arrays
    centroid: tuple  # (x, y), Bourke (1988) centroid, Eq. 6
    area: float
    std_distance: float
    lineage_id: int = -1
    type: str = APPEARANCE
    ancestors: tuple = ()  # ((time, mover_id), ...) at t - 1
    parent_lineages: tuple = ()  # lineage ids of the ancestors


def build_movers(cluster_mask, polygons, w):
    """Group clustering units into movers (phase 2).

    Parameters
    ----------
    cluster_mask : (n,) boolean array, True for clustering units
    polygons : list of (k, 2) vertex arrays, one per unit
    w : (n, n) weights matrix; two clustering units are in the same
        mover when connected through positive weights

    Returns
    -------
    movers : list of :class:`Mover` (without lineage/type set yet)
    """
    n = len(cluster_mask)
    idx = np.flatnonzero(cluster_mask)
    movers = []
    if len(idx) == 0:
        return movers
    # Connected components of the induced subgraph (BFS).
    remaining = set(int(i) for i in idx)
    components = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        comp = set()
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in np.flatnonzero(w[u] > 0):
                v = int(v)
                if v in remaining and v not in comp:
                    stack.append(v)
        components.append(sorted(comp))
        remaining -= comp
    for mid, members in enumerate(components):
        member_polys = [polygons[m] for m in members]
        area, cx, cy = multipolygon_area_centroid(member_polys)
        sd = standard_distance(member_polys, (cx, cy))
        movers.append(
            Mover(
                id=mid,
                time=-1,
                members=tuple(members),
                polygons=member_polys,
                centroid=(cx, cy),
                area=area,
                std_distance=sd,
            )
        )
    return movers


def overlap_proportion(ancestor, mover, n_samples, phi, rng):
    """Proportion of bivariate-normal samples from ``ancestor`` inside ``mover``.

    Samples are drawn from an uncorrelated bivariate normal distribution
    centred on the ancestor's centroid with variance
    ``std_distance(ancestor)**2 * phi`` (Eq. 4).  Returns the fraction of
    the ``n_samples`` points falling inside the mover's geometry.
    """
    cx, cy = ancestor.centroid
    sd = ancestor.std_distance
    sigma = sd * np.sqrt(phi) if sd > 0 else 0.0
    if sigma <= 0.0:
        # Degenerate ancestor: the distribution collapses to its centroid.
        inside = points_in_multipolygon(
            np.array([[cx, cy]]), mover.polygons
        )
        return float(inside[0])
    points = rng.normal(loc=[cx, cy], scale=sigma, size=(n_samples, 2))
    inside = points_in_multipolygon(points, mover.polygons)
    return float(inside.mean())


def contiguity_matrix(prev_movers, curr_movers, pi, phi, n_samples, rng):
    """Boolean (n_prev, n_curr) contiguity matrix (phase 3).

    Entry ``[a, b]`` is True when the proportion of bivariate-normal
    samples centred on ``prev_movers[a]`` falling inside
    ``curr_movers[b]`` is at least ``pi``.
    """
    mat = np.zeros((len(prev_movers), len(curr_movers)), dtype=bool)
    for a, ancestor in enumerate(prev_movers):
        for b, mover in enumerate(curr_movers):
            prop = overlap_proportion(ancestor, mover, n_samples, phi, rng)
            mat[a, b] = prop >= pi
    return mat


def assign_types(prev_movers, curr_movers, contiguous):
    """Assign contiguity types (phase 4).

    Parameters
    ----------
    prev_movers, curr_movers : lists of :class:`Mover`
    contiguous : (n_prev, n_curr) boolean array from
        :func:`contiguity_matrix`

    Returns
    -------
    types : list of str, one per current mover
    ancestors : list of tuple, ancestor ``(time, id)`` pairs per mover
    terminated : list of :class:`Mover` from ``prev_movers`` with no
        descendants at the current step
    """
    n_prev = len(prev_movers)
    n_curr = len(curr_movers)
    descendants = contiguous.sum(axis=1)  # per ancestor
    types = []
    ancestors = []
    for b in range(n_curr):
        anc = tuple(
            (prev_movers[a].time, prev_movers[a].id)
            for a in range(n_prev)
            if contiguous[a, b]
        )
        ancestors.append(anc)
        if len(anc) == 0:
            types.append(APPEARANCE)
        elif len(anc) > 1:
            types.append(CONVERGENCE)
        else:
            a = next(
                a for a in range(n_prev) if contiguous[a, b]
            )
            types.append(
                DIVERGENCE if descendants[a] > 1 else PERSISTENT
            )
    terminated = [
        prev_movers[a] for a in range(n_prev) if descendants[a] == 0
    ]
    return types, ancestors, terminated
