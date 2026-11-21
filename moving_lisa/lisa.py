"""Cross-sectional LISA statistics: Local Moran's I and Getis-Ord Local G.

The formulas follow esda 2.10 (PySAL) exactly, so results match
``esda.Moran_Local`` and ``esda.G_Local`` on the same data and weights:

* Local Moran (Anselin 1995, as implemented in esda): with
  ``z = (y - mean) / std`` (population standard deviation) and a
  row-standardized weights matrix,
  ``I_i = (n - 1) * z_i * lag_i / sum(z**2)`` where
  ``lag_i = sum_j w_ij z_j``.  This is Anselin's formula with the sample
  second moment, i.e. the textbook population-moment version times
  ``(n - 1) / n``.  Quadrants use the PySAL scheme: 1 = HH, 2 = LH,
  3 = LL, 4 = HL (a zero spatial lag counts as a low lag, as in esda).
* Getis-Ord Local G (Gi, excluding the focal unit; dissertation Eq. 2):
  ``G_i = sum_{j!=i} w_ij y_j / sum_{j!=i} y_j`` with row-standardized
  weights.  The analytical moments are esda's: with ``N = n - 1``,
  ``E[G_i] = cardinality_i / N`` and
  ``Var(G_i) = cardinality_i (N - cardinality_i) / (N - 1) / N**2
  * empirical_variance_i / empirical_mean_i**2``,
  where the empirical moments are taken over the non-focal values.

Significance comes from conditional randomization (Anselin 1995): for
each location ``i`` the values at all other locations are permuted and
the statistic recomputed.  Pseudo p-values use the ``(M + 1) / (R + 1)``
rule (esda ``significance.calculate_significance``):

* ``alternative="greater"``: ``M`` counts simulations ``>=`` observed;
* ``alternative="lesser"``: ``M`` counts simulations ``<=`` observed;
* ``alternative="two-sided"`` (default): ``M`` counts the smaller tail.
  This is esda's ``"directed"`` rule, which is also esda's default
  behaviour; esda's own ``"two-sided"`` uses a percentile construction
  instead.

The dissertation (Ch. 2) uses the same conditional-randomization
p-values and, for its demonstration, a one-tailed test for high-high
Local G clusters (``alternative="greater"`` with cluster selection
``("HH",)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LISAResult", "local_g", "local_moran"]


@dataclass
class LISAResult:
    """Result of a cross-sectional LISA computation for one time step."""

    statistic: np.ndarray
    p_value: np.ndarray
    z: np.ndarray
    cluster: np.ndarray  # "HH" / "LL" / "HL" / "LH" / "NS"
    quadrant: np.ndarray | None = None  # PySAL scheme, Moran only
    sim: np.ndarray | None = None  # (permutations, n) simulated statistics
    statistic_name: str = ""
    alpha: float = 0.05


def _p_values(obs, sim, alternative):
    """Pseudo p-values from a (permutations, n) simulation matrix.

    The comparison is tie-tolerant: simulated values within ``tol`` of
    the observed statistic count as equal to it.  This matters because
    statistics such as Local G take only a few discrete values, so the
    observed value often sits exactly on a mass point of the simulated
    distribution; without the tolerance, whether those tied simulations
    count as ">= observed" would depend on floating-point summation
    order (the observed statistic and the simulations are computed by
    different code paths).  ``tol`` is far below any meaningful
    difference, so continuous statistics are unaffected.
    """
    perms = sim.shape[0]
    tol = 1e-9 * np.maximum(1.0, np.abs(obs))
    if alternative == "greater":
        m = (sim >= obs - tol).sum(axis=0)
    elif alternative == "lesser":
        m = (sim <= obs + tol).sum(axis=0)
    elif alternative == "two-sided":
        larger = (sim >= obs - tol).sum(axis=0)
        m = np.minimum(larger, perms - larger)
    else:
        raise ValueError(
            f"alternative={alternative!r}, expected 'two-sided', "
            "'greater' or 'lesser'"
        )
    return (m + 1.0) / (perms + 1.0)


def _conditional_permutations(values, perms, rng):
    """Permuted values at the neighbours of one location.

    Returns a ``(perms, k)`` array: each row is a random permutation of
    ``values`` (the values at all locations except the focal one).
    """
    k = len(values)
    if k == 0:
        return np.zeros((perms, 0))
    draws = rng.random((perms, k))
    order = np.argsort(draws, axis=1)
    return values[order]


def local_moran(y, w, permutations=999, random_state=None,
                alternative="two-sided", alpha=0.05):
    """Local Moran's I for one cross-section (esda ``Moran_Local``).

    Parameters
    ----------
    y : (n,) array of values
    w : (n, n) row-standardized weights matrix (zero diagonal)
    permutations : number of conditional permutations
    random_state : int, Generator or None
    alternative : "two-sided" (default), "greater" or "lesser"
    alpha : significance threshold used for the cluster labels

    Returns
    -------
    LISAResult with ``statistic`` = local Moran's I, ``quadrant`` in the
    PySAL scheme (1 HH, 2 LH, 3 LL, 4 HL) and ``cluster`` labels.
    """
    y = np.asarray(y, dtype=float).flatten()
    w = np.asarray(w, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError("local Moran's I needs at least 3 observations")
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    rng = np.random.default_rng(random_state)

    z = y - y.mean()
    sd = y.std()
    z = z / sd if sd > 0 else np.zeros(n)
    den = float((z * z).sum())
    lag = w @ z
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = (n - 1) * z * lag / den if den > 0 else np.zeros(n)
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

    # Conditional permutations: for each i, permute z at j != i.
    sim = np.zeros((permutations, n))
    for i in range(n):
        others = np.delete(z, i)
        w_row = np.delete(w[i], i)
        perm = _conditional_permutations(others, permutations, rng)
        lag_sim = perm @ w_row
        sim[:, i] = (n - 1) * z[i] * lag_sim / den if den > 0 else 0.0
    p_value = _p_values(stat, sim, alternative)
    with np.errstate(invalid="ignore", divide="ignore"):
        ei_sim = sim.mean(axis=0)
        se_sim = sim.std(axis=0)
        z_sim = (stat - ei_sim) / np.where(se_sim > 0, se_sim, np.nan)
    z_sim = np.nan_to_num(z_sim, nan=0.0, posinf=0.0, neginf=0.0)

    # PySAL quadrants: 1 HH, 2 LH, 3 LL, 4 HL (zero lag counts as low).
    quadrant = np.zeros(n, dtype=int)
    quadrant[(z > 0) & (lag > 0)] = 1
    quadrant[(z < 0) & (lag > 0)] = 2
    quadrant[(z < 0) & (lag <= 0)] = 3
    quadrant[(z > 0) & (lag <= 0)] = 4
    label = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    cluster = np.array(
        [
            label[q] if (q != 0 and p_value[k] < alpha) else "NS"
            for k, q in enumerate(quadrant)
        ],
        dtype=object,
    )
    return LISAResult(
        statistic=stat,
        p_value=p_value,
        z=z_sim,
        cluster=cluster,
        quadrant=quadrant,
        sim=sim,
        statistic_name="moran",
        alpha=alpha,
    )


def local_g(y, w, permutations=999, random_state=None,
            alternative="two-sided", alpha=0.05, star=False):
    """Getis-Ord Local G for one cross-section (esda ``G_Local``).

    Parameters
    ----------
    y : (n,) array of values (intended for non-negative variables)
    w : (n, n) row-standardized weights matrix (zero diagonal for Gi)
    permutations : number of conditional permutations
    random_state : int, Generator or None
    alternative : "two-sided" (default), "greater" or "lesser"
    alpha : significance threshold used for the cluster labels
    star : if True, include the focal unit (Gi*); default False (Gi,
        dissertation Eq. 2).  Gi* needs a non-zero diagonal in ``w``.

    Returns
    -------
    LISAResult with ``statistic`` = G values, ``z`` = analytical
    z-scores (esda's moments) and ``cluster`` labels "HH" / "LL" / "NS".
    """
    y = np.asarray(y, dtype=float).flatten()
    w = np.asarray(w, dtype=float)
    n = len(y)
    if n < 3:
        raise ValueError("local G needs at least 3 observations")
    if permutations < 1:
        raise ValueError("permutations must be >= 1")
    rng = np.random.default_rng(random_state)

    remove_self = not star
    n_eff = n - (1 if remove_self else 0)
    y_sum = y.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = y_sum - y * remove_self
        stat = (w @ y) / denom
    stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

    # Analytical moments (esda G_Local.calc).
    with np.errstate(invalid="ignore", divide="ignore"):
        empirical_mean = (y_sum - y * remove_self) / n_eff
        mean_of_squares = ((y ** 2).sum() - (y ** 2) * remove_self) / n_eff
        empirical_variance = mean_of_squares - empirical_mean ** 2
        cardinality = w.sum(axis=1)
        expected = cardinality / n_eff
        expected_var = (
            cardinality * (n_eff - cardinality) / (n_eff - 1)
            * (1.0 / n_eff ** 2)
            * (empirical_variance / (empirical_mean ** 2))
        )
        z_scores = (stat - expected) / np.sqrt(expected_var)
    z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Conditional permutations: for each i, permute y at j != i.
    sim = np.zeros((permutations, n))
    for i in range(n):
        others = np.delete(y, i)
        w_row = np.delete(w[i], i)
        perm = _conditional_permutations(others, permutations, rng)
        numer_sim = perm @ w_row
        if star:
            numer_sim = numer_sim + w[i, i] * y[i]
        sim[:, i] = numer_sim / denom[i] if denom[i] != 0 else 0.0
    p_value = _p_values(stat, sim, alternative)

    cluster = np.array(
        [
            "HH" if (z_scores[k] > 0 and p_value[k] < alpha)
            else "LL" if (z_scores[k] < 0 and p_value[k] < alpha)
            else "NS"
            for k in range(n)
        ],
        dtype=object,
    )
    return LISAResult(
        statistic=stat,
        p_value=p_value,
        z=z_scores,
        cluster=cluster,
        quadrant=None,
        sim=sim,
        statistic_name="g_star" if star else "g",
        alpha=alpha,
    )
