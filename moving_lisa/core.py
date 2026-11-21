"""Moving LISA: local indicators of spatial association over time.

Implements the 5-phase procedure of Anderson & Yang (2025),
"Moving Local Indicators of Spatial Association for Time Geography",
Transactions in GIS, doi:10.1111/tgis.70158 (dissertation Ch. 2 and
Appendix A, "Moving LISA Pseudocode"):

1. **LISA**: a cross-sectional local indicator of spatial association is
   applied at each time step (Local Moran's I by default; Getis-Ord
   Local G as in the dissertation's demonstration).  Significant
   clustering geographies are those with ``p < alpha`` and a cluster
   label in ``cluster_types``.
2. **Geometric union**: clustering geographies at a time step are grouped
   into movers (connected through the spatial weights matrix, which for
   queen-contiguity weights is the geometric union with dissolved shared
   boundaries).  At the first time step movers proceed directly to
   phase 5.
3. **Contiguity**: a mover at ``t - 1`` is contiguous with a mover at
   ``t`` when samples from an uncorrelated bivariate normal distribution
   centred on the earlier mover's centroid, with variance
   ``std_distance**2 * phi``, fall inside the later mover's geometry
   with proportion at least ``pi`` (Eq. 4).
4. **Contiguity types**: appearance / convergence / divergence /
   persistent / termination (Eq. 5; see :mod:`moving_lisa.movers` for
   the interpretation of the pseudocode's count/count_unique test).
5. **Space-time anchors**: the geometric centroid of each active mover
   (Bourke 1988, Eq. 6) is appended to its trajectory.

The result is a set of mover trajectories ``R``: sequences of
space-time anchors describing the progression of each aggregate mover,
plus a table recording the ancestry and LISA results for each mover.

Lineage rules (the dissertation records contiguity types but does not
prescribe lineage inheritance, so the following is stated explicitly):
a persistent mover keeps its ancestor's lineage; a divergent mover and
a convergent mover each start a new lineage recording the ancestor
lineage(s) as parent(s); an appearing mover starts a new lineage; a
terminated mover's trajectory ends with the anchor from its last active
time step.
"""

from __future__ import annotations

import numpy as np

from .geometry import as_polygons
from .lisa import local_g, local_moran
from .movers import (
    APPEARANCE,
    CONVERGENCE,
    DIVERGENCE,
    PERSISTENT,
    TERMINATION,
    assign_types,
    build_movers,
    contiguity_matrix,
)
from .weights import as_weights_matrix

__all__ = ["MovingLISA"]

_MORAN_CLUSTERS = ("HH", "LL", "HL", "LH")
_G_CLUSTERS = ("HH", "LL")


class MovingLISA:
    """Moving Local Indicators of Spatial Association.

    Parameters
    ----------
    y : (n, T) array-like
        Values for ``n`` observational units over ``T`` time steps.
    geometries : sequence of polygons, GeoDataFrame, or GeoSeries
        One polygon per unit, each an ``(k, 2)`` array of vertices
        (shapely geometries are accepted too).
    w : weights, optional
        Dense ``(n, n)`` array, SciPy sparse matrix, libpysal ``W`` or
        neighbour dict.  If None, queen-contiguity weights are built
        from the geometries.  Weights are row-standardized internally.
    statistic : {"moran", "g"}, default "moran"
        The cross-sectional LISA statistic.  ``"g"`` is the Getis-Ord
        Local G used in the dissertation's demonstration (Gi, focal
        unit excluded).
    alpha : float, default 0.05
        Significance threshold for the LISA statistic.
    pi : float, default 0.3
        Contiguity threshold: the minimum proportion of bivariate-normal
        samples that must fall inside the later mover's geometry.
    phi : float, default 2.0
        Standard-distance multiplier for the contiguity distribution
        (variance ``std_distance**2 * phi``).
    permutations : int, default 999
        Conditional permutations for the LISA pseudo p-values.
    n_samples : int, default 1000
        Monte Carlo draws for each contiguity overlay.
    random_state : int or None
        Seed for reproducibility (separate streams for the LISA
        permutations and the contiguity draws).
    cluster_types : tuple of str, optional
        Cluster labels counted as clustering geographies.  Defaults:
        ``("HH", "LL")`` for Moran's I, ``("HH",)`` for Local G (the
        dissertation's demonstration selects high-high clusters only).
    star : bool, default False
        For ``statistic="g"`` only: use Gi* (focal unit included).
    alternative : {"two-sided", "greater", "lesser"}, default "two-sided"
        Alternative for the permutation p-values.  Use ``"greater"``
        with ``statistic="g"`` and ``cluster_types=("HH",)`` to
        reproduce the dissertation's one-tailed high-high test.

    Attributes
    ----------
    lisa_stats_, lisa_p_, lisa_z_, lisa_cluster_ : (T, n) arrays
        Per-time-step LISA statistics, pseudo p-values, z-scores and
        cluster labels.
    movers_ : list of lists of :class:`~moving_lisa.movers.Mover`
        Movers per time step.
    table_ : list of dict
        One row per (mover, time step): time, mover_id, lineage_id,
        type, n_units, members, centroid, ancestors, parent_lineages.
    trajectories_ : list of dict
        One entry per lineage: lineage_id, parent_lineages and the
        anchor sequence ``[(time, x, y, type, mover_id), ...]``.
    counts_ : dict
        Counts of movers by contiguity type over the whole run.
    terminations_ : list of dict
        Terminated movers: lineage_id and last active time step.
    """

    def __init__(
        self,
        y,
        geometries,
        w=None,
        *,
        statistic="moran",
        alpha=0.05,
        pi=0.3,
        phi=2.0,
        permutations=999,
        n_samples=1000,
        random_state=None,
        cluster_types=None,
        star=False,
        alternative="two-sided",
    ):
        if statistic not in ("moran", "g"):
            raise ValueError("statistic must be 'moran' or 'g'")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if not 0.0 <= pi <= 1.0:
            raise ValueError("pi must be in [0, 1]")
        if phi <= 0:
            raise ValueError("phi must be > 0")
        if permutations < 1:
            raise ValueError("permutations must be >= 1")
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        allowed = _MORAN_CLUSTERS if statistic == "moran" else _G_CLUSTERS
        if cluster_types is None:
            cluster_types = ("HH", "LL") if statistic == "moran" else ("HH",)
        if any(c not in allowed for c in cluster_types):
            raise ValueError(
                f"cluster_types for {statistic!r} must be a subset of {allowed}"
            )

        y = np.asarray(y, dtype=float)
        if y.ndim != 2:
            raise ValueError("y must be a 2D (n, T) array")
        n, n_steps = y.shape
        if n < 3:
            raise ValueError("Moving LISA needs at least 3 units")
        if n_steps < 1:
            raise ValueError("Moving LISA needs at least 1 time step")
        polygons = as_polygons(geometries)
        if len(polygons) != n:
            raise ValueError(
                f"got {len(polygons)} geometries for {n} units"
            )
        w_mat = as_weights_matrix(w, n, geometries=polygons)

        self.statistic = statistic
        self.alpha = alpha
        self.pi = pi
        self.phi = phi
        self.permutations = permutations
        self.n_samples = n_samples
        self.cluster_types = tuple(cluster_types)
        self.n_units_ = n
        self.n_steps_ = n_steps

        # Separate RNG streams for the LISA permutations and the
        # contiguity draws, so results are reproducible given a seed.
        seed_seq = np.random.SeedSequence(random_state)
        rng_lisa = np.random.default_rng(seed_seq.spawn(1)[0])
        rng_contig = np.random.default_rng(seed_seq.spawn(1)[0])

        # Phase 1: cross-sectional LISA at each time step.
        lisa_stats = np.zeros((n_steps, n))
        lisa_p = np.zeros((n_steps, n))
        lisa_z = np.zeros((n_steps, n))
        lisa_cluster = np.zeros((n_steps, n), dtype=object)
        for t in range(n_steps):
            if statistic == "moran":
                res = local_moran(
                    y[:, t], w_mat, permutations=permutations,
                    random_state=rng_lisa, alternative=alternative,
                    alpha=alpha,
                )
            else:
                res = local_g(
                    y[:, t], w_mat, permutations=permutations,
                    random_state=rng_lisa, alternative=alternative,
                    alpha=alpha, star=star,
                )
            lisa_stats[t] = res.statistic
            lisa_p[t] = res.p_value
            lisa_z[t] = res.z
            lisa_cluster[t] = res.cluster
        self.lisa_stats_ = lisa_stats
        self.lisa_p_ = lisa_p
        self.lisa_z_ = lisa_z
        self.lisa_cluster_ = lisa_cluster

        # Phases 2-5.
        movers_per_step = []
        table = []
        trajectories = {}
        counts = {
            APPEARANCE: 0,
            CONVERGENCE: 0,
            DIVERGENCE: 0,
            PERSISTENT: 0,
            TERMINATION: 0,
        }
        terminations = []
        next_lineage = 0
        prev_movers = []

        for t in range(n_steps):
            cluster_mask = np.array(
                [c in self.cluster_types for c in lisa_cluster[t]]
            )
            movers = build_movers(cluster_mask, polygons, w_mat)
            for m in movers:
                m.time = t
            if t == 0 or not prev_movers:
                # No comparable prior step: all appearances.
                for m in movers:
                    m.type = APPEARANCE
                    m.lineage_id = next_lineage
                    next_lineage += 1
                    counts[APPEARANCE] += 1
            else:
                contig = contiguity_matrix(
                    prev_movers, movers, pi, phi, n_samples, rng_contig
                )
                types, ancestors, terminated = assign_types(
                    prev_movers, movers, contig
                )
                # Lineage inheritance.
                descendants = contig.sum(axis=1)
                for b, m in enumerate(movers):
                    m.type = types[b]
                    m.ancestors = ancestors[b]
                    counts[types[b]] += 1
                    if types[b] == APPEARANCE:
                        m.lineage_id = next_lineage
                        next_lineage += 1
                    elif types[b] == PERSISTENT:
                        a = next(
                            a for a in range(len(prev_movers))
                            if contig[a, b]
                        )
                        m.lineage_id = prev_movers[a].lineage_id
                    else:  # CONVERGENCE or DIVERGENCE: new lineage
                        m.lineage_id = next_lineage
                        next_lineage += 1
                        m.parent_lineages = tuple(
                            prev_movers[a].lineage_id
                            for a in range(len(prev_movers))
                            if contig[a, b]
                        )
                for term in terminated:
                    counts[TERMINATION] += 1
                    terminations.append(
                        {
                            "lineage_id": term.lineage_id,
                            "last_time": term.time,
                            "mover_id": term.id,
                        }
                    )
            for m in movers:
                row = {
                    "time": t,
                    "mover_id": m.id,
                    "lineage_id": m.lineage_id,
                    "type": m.type,
                    "n_units": len(m.members),
                    "members": m.members,
                    "centroid_x": m.centroid[0],
                    "centroid_y": m.centroid[1],
                    "ancestors": m.ancestors,
                    "parent_lineages": m.parent_lineages,
                }
                table.append(row)
                trajectories.setdefault(
                    m.lineage_id,
                    {
                        "lineage_id": m.lineage_id,
                        "parent_lineages": m.parent_lineages,
                        "anchors": [],
                    },
                )["anchors"].append(
                    (t, m.centroid[0], m.centroid[1], m.type, m.id)
                )
            movers_per_step.append(movers)
            prev_movers = movers

        self.movers_ = movers_per_step
        self.table_ = table
        self.trajectories_ = [trajectories[k] for k in sorted(trajectories)]
        self.counts_ = counts
        self.terminations_ = terminations

    def summary(self):
        """Counts of movers by contiguity type over the whole run."""
        return dict(self.counts_)

    def to_dataframe(self):
        """The mover table as a pandas DataFrame (pandas required)."""
        import pandas as pd

        return pd.DataFrame(self.table_)

    def __repr__(self):
        return (
            f"MovingLISA(statistic={self.statistic!r}, "
            f"n_units={self.n_units_}, n_steps={self.n_steps_}, "
            f"alpha={self.alpha}, pi={self.pi}, phi={self.phi}, "
            f"trajectories={len(self.trajectories_)}, "
            f"counts={self.counts_})"
        )
