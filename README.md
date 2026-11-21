# moving-lisa

**Moving Local Indicators of Spatial Association (Moving LISA)** — local
indicators of spatial association over time, following Anderson & Yang
(2025), *"Moving Local Indicators of Spatial Association for Time
Geography"*, Transactions in GIS, doi:[10.1111/tgis.70158](https://doi.org/10.1111/tgis.70158)
(dissertation Ch. 2 and Appendix A, "Moving LISA Pseudocode").

PySAL (`esda`) only provides *static* LISA. This library extends it in
time: given a series of cross-sections, it returns **trajectories of
clustering locations** — in the authors' words, "Given a series, LISA
returns trajectories of clustering locations."

Core dependencies are **NumPy and SciPy only**. libpysal, geopandas,
shapely and pandas are optional (duck-typed inputs).

## The method (5 phases)

1. **LISA** — a cross-sectional local indicator is applied at each time
   step: Local Moran's I (default) or Getis-Ord Local G (Gi, focal unit
   excluded), as in the dissertation's demonstration. Significance comes
   from conditional randomization (pseudo p-values, `(M + 1) / (R + 1)`).
2. **Geometric union** — significant clustering geographies at one time
   step are grouped into *movers* (connected through the spatial weights
   matrix; for queen-contiguity weights this is the geometric union with
   dissolved shared boundaries). At the first time step movers proceed
   directly to phase 5.
3. **Contiguity (Eq. 4)** — a mover `A` at `t − 1` is contiguous with a
   mover `B` at `t` when, for samples drawn from an uncorrelated
   bivariate normal distribution centred on `A`'s centroid with variance
   `std_distance(A)² · φ`, the proportion falling inside `B`'s geometry
   is at least `π`.
4. **Contiguity types (Eq. 5)** — *appearance* (no contiguous
   ancestors), *convergence* (one current mover, > 1 ancestors),
   *divergence* (one ancestor, > 1 current movers), *persistent* (one
   ancestor, one descendant, otherwise unchanged), *termination*
   (ancestor with no descendants; its trajectory ends at its last
   active step).
5. **Space-time anchors** — the geometric centroid of each active mover
   (Bourke 1988, Eq. 6) is appended to its trajectory.

## Quick start

```python
from moving_lisa import MovingLISA, load_toy

toy = load_toy()  # bundled 6x6 lattice, 3 time steps, runs offline
res = MovingLISA(
    toy["y"], toy["polygons"], w=toy["w"],
    statistic="g",        # Getis-Ord Local G (dissertation demo)
    alpha=0.05, pi=0.2, phi=1.1,
    permutations=999, random_state=0,
)
print(res.counts_)        # {'appearance': 1, 'persistent': 2, ...}
for row in res.table_:
    print(row["time"], row["type"], row["members"],
          (row["centroid_x"], row["centroid_y"]))
for traj in res.trajectories_:
    print(traj["lineage_id"], traj["anchors"])
```

See `examples/toy_example.py` for a runnable version.

## API

```python
MovingLISA(y, geometries, w=None, *, statistic="moran", alpha=0.05,
           pi=0.3, phi=2.0, permutations=999, n_samples=1000,
           random_state=None, cluster_types=None, star=False,
           alternative="two-sided")
```

| Parameter | Meaning | Default |
|---|---|---|
| `y` | `(n, T)` values array (units × time steps) | — |
| `geometries` | one polygon per unit: `(k, 2)` vertex arrays, shapely geometries, GeoDataFrame/GeoSeries | — |
| `w` | weights: dense array, SciPy sparse matrix, libpysal `W`, or neighbour dict. `None` builds queen-contiguity weights from the geometries. Row-standardized internally. | `None` |
| `statistic` | `"moran"` (Local Moran's I) or `"g"` (Getis-Ord Local G, Gi) | `"moran"` |
| `alpha` | significance threshold | `0.05` |
| `pi` | contiguity threshold (phase 3) | `0.3` |
| `phi` | standard-distance multiplier (phase 3) | `2.0` |
| `permutations` | conditional permutations for the LISA p-values | `999` |
| `n_samples` | Monte Carlo draws per contiguity overlay | `1000` |
| `random_state` | seed (separate streams for LISA and contiguity) | `None` |
| `cluster_types` | cluster labels counted as clustering geographies | `("HH","LL")` for Moran, `("HH",)` for G |
| `star` | for `"g"` only: use Gi\* (focal unit included) | `False` |
| `alternative` | `"two-sided"` (default), `"greater"`, `"lesser"` | `"two-sided"` |

Results: `lisa_stats_`, `lisa_p_`, `lisa_z_`, `lisa_cluster_` (`(T, n)`
arrays), `movers_`, `table_` (one row per mover per step), `trajectories_`
(anchor sequences per lineage), `counts_`, `terminations_`.
`summary()` returns the counts; `to_dataframe()` returns the mover table
(pandas required).

Standalone cross-sectional functions are also exported:
`local_moran(y, w, ...)` and `local_g(y, w, ...)` match `esda.Moran_Local`
and `esda.G_Local` (see *Verification* below).

## Parameters: what the dissertation uses, and what the toy uses

The dissertation's parameters are **data-dependent** — the authors select
them by sweeping (Ch. 2 sweeps `π` from 0.10 to 1.0 at `φ = 2`, `α = 0.05`;
Ch. 3 sweeps `π` and then `φ` and selects the inflection point):

| | π | φ | α | statistic |
|---|---|---|---|---|
| Ch. 2 demonstration (ZCTA data) | 0.3 | 2.0 | 0.05 | Local G, HH only |
| Ch. 3 application (vape shops) | 0.2 | 1.1 | 0.05 | Local G, HH only |
| This library's defaults | 0.3 | 2.0 | 0.05 | Moran's I |
| Bundled toy example | 0.2 | 1.1 | 0.05 | Local G, HH only |

**Parameter sensitivity (measured on the toy).** The overlap proportion
between the toy mover at `t` and at `t + 1` (5,000 samples, seed 0) is:

| φ | identical mover | t → t+1 |
|---|---|---|
| 0.5 | 0.651 | 0.457 |
| 1.0 | 0.433 | 0.338 |
| 1.1 | 0.405 | 0.321 |
| 1.5 | 0.323 | 0.267 |
| 2.0 | 0.253 | 0.215 |

At the Ch. 2 defaults (`π = 0.3`, `φ = 2.0`) the toy mover does **not**
link (0.215 < 0.3: three appearances, two terminations); at the Ch. 3
parameters (`π = 0.2`, `φ = 1.1`) it links persistently (0.321 > 0.2).
This is expected behaviour — the dissertation itself says `φ` "may need a
user parameter sweep" — and it is pinned by a test
(`test_ch2_defaults_no_persistence`). **Sweep `π` and `φ` for your own
data; do not assume the defaults link.**

The toy example (6×6 lattice, a 3×3 high-value block moving one cell
right per step) detects a 5-unit cross-shaped mover at each step and, at
`π = 0.2, φ = 1.1`, one persistent trajectory with anchors
`(1.5, 2.5) → (2.5, 2.5) → (3.5, 2.5)`.

## Method ambiguities found in the source (and how they are resolved)

The dissertation and its Appendix A pseudocode are the authority; where
they disagree or are silent, the choice is documented here and in the
code. Nothing was invented silently.

1. **Contiguity variance (Eq. 4 vs its prose).** Eq. 4 and Appendix A
   (lines 19–21) agree on variance `σ² = std_distance² · φ`
   (`σ = SD · √φ`). The prose under Eq. 4 instead describes the standard
   deviation as `SD · φ` (`σ² = SD² · φ²`). **This library follows the
   equation and the pseudocode.**
2. **Phase 4 divergence branch.** The pseudocode's
   `count(C) > 1 and count_unique(C) == 1 → DIVERGENCE` assumes the
   ancestor list `C` can contain repeat identifiers, which cannot happen
   for a per-mover ancestor list. **Implemented via the prose
   definition:** a current mover with a single ancestor is a divergence
   when that ancestor is contiguous with more than one current mover,
   persistent otherwise.
3. **Standard distance.** Computed from all member-polygon vertices (RMS
   distance to the centroid). For a dissolved mover this is an
   approximation, because shared-boundary vertices are interior to the
   dissolved shape; the difference is small and `φ` is user-tunable.
4. **Lineage rules.** The dissertation records contiguity types but does
   not prescribe lineage inheritance. This library: a persistent mover
   keeps its ancestor's lineage; divergent and convergent movers each
   start a new lineage recording the ancestor lineage(s) as parent(s);
   an appearing mover starts a new lineage; a terminated mover's
   trajectory ends at its last active step.
5. **"Time windows".** The procedure applies the LISA cross-sectionally
   at each time step (each step is a cross-section); no rolling-window
   aggregation exists in the published method, so none is implemented.
   A rolling window would be a future extension, not part of the method.
6. **No exact numeric fixtures exist in the paper** (the simulation is
   stochastic and no per-unit values are published). Tests therefore use
   hand-computed small cases plus an independent cross-check against
   esda (below).

## Verification

- **Hand-computed cases.** 4-unit line `x = [0, 0, 10, 10]`, rook
  weights: Local Moran's I `= [0.75, 0, 0, 0.75]` (esda scaling; the
  textbook population-moment version would be `[1, 0, 0, 1]`),
  quadrants `[3, 3, 4, 1]`; Local G `= [0, 0.25, 0.5, 1]`
  (row-standardized), z `= [-1.414, -0.354, 0.354, 1.414]`.
- **esda cross-check (esda 2.10.0, libpysal 4.15.1).** On continuous
  random data (6×6 lattice, queen weights): Local Moran's I and Local G
  statistics match esda to ~1e-16; G z-scores match to ~1e-16;
  pseudo p-values agree to ≤ 0.047 (Monte Carlo noise at 999 perms);
  significance decisions agree except possibly on borderline units
  (p near 0.05). Note: Moran's `z` is simulation-based and G's `z` is
  analytical — both matching esda's convention for each statistic.
- **Tie-tolerant pseudo p-values.** Statistics such as Local G take only
  a few discrete values, so the observed value often sits exactly on a
  mass point of the simulated distribution; without a tolerance, whether
  tied simulations count as "≥ observed" would depend on floating-point
  summation order (the observed statistic and the simulations are
  computed by different code paths). This library counts simulations
  within `1e-9` (relative) of the observed value as tied, per the
  standard `(M + 1) / (R + 1)` definition where `M` counts simulations
  equal to or more extreme than the observed value. Verified against the
  exact hypergeometric p-value for the toy's unit 6 (Local Moran):
  theory 0.0666, this library 0.0674, esda 0.007 — esda's value excludes
  the tied simulations because its observed statistic is 1 ULP above
  the mass point. The library's value is the correct one.
- **Full test suite:** 53 tests (unittest and pytest), all passing; see
  `tests/`.

## Installation

```bash
pip install moving-lisa        # (once published)
# or, from a checkout:
pip install .
```

Requires Python ≥ 3.10, NumPy, SciPy. Optional: `pandas` (for
`to_dataframe()`), `libpysal`/`geopandas`/`shapely` (weights and
geometry inputs), `esda` (only for the cross-check tests).

## Tests

```bash
python -m unittest discover -s tests -v
# or
pytest tests/ -q
```

## Reference

Anderson, J. & Yang, A. (2025). Moving Local Indicators of Spatial
Association for Time Geography. *Transactions in GIS*.
doi:10.1111/tgis.70158. Method: dissertation Ch. 2 (§2.3, §2.7 Appendix A
pseudocode); application: Ch. 3. Bourke, P. (1988). Calculating the area
and centroid of a polygon.
