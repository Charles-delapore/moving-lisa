"""Moving LISA toy example (runs offline, no downloads).

A 6x6 lattice of unit squares with a 3x3 high-value block that moves
one cell to the right at each of three time steps.  The Moving LISA
procedure should find one persistent mover whose space-time anchors
drift rightwards.  Contiguity uses the Chapter 3 application
parameters of Anderson & Yang (2025): pi = 0.2, phi = 1.1, alpha = 0.05.

Run from the repository root::

    python examples/toy_example.py

(Works without installing the package: the script adds the repository
root to ``sys.path``.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moving_lisa import MovingLISA, load_toy


def main():
    toy = load_toy()
    res = MovingLISA(
        toy["y"],
        toy["polygons"],
        w=toy["w"],
        statistic="g",  # Getis-Ord Local G, as in the dissertation demo
        alpha=0.05,
        pi=0.2,
        phi=1.1,
        permutations=999,
        random_state=0,
    )
    print(res)
    print()
    print("Mover table:")
    for row in res.table_:
        print(
            f"  t={row['time']} mover={row['mover_id']} "
            f"lineage={row['lineage_id']} {row['type']:<11} "
            f"n={row['n_units']} members={row['members']} "
            f"centroid=({row['centroid_x']:.1f}, {row['centroid_y']:.1f})"
        )
    print()
    print("Trajectories (space-time anchors):")
    for traj in res.trajectories_:
        print(f"  lineage {traj['lineage_id']}:")
        for time, x, y, kind, mover_id in traj["anchors"]:
            print(f"    t={time} ({x:.1f}, {y:.1f}) {kind}")
    print()
    print("Counts:", res.counts_)
    print("Terminations:", res.terminations_)


if __name__ == "__main__":
    main()
