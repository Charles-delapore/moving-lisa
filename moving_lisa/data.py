"""Bundled toy data: a small moving-hotspot example that runs offline.

``load_toy()`` returns a 6x6 lattice of unit-square polygons with three
time steps of values.  A 3x3 high-value block (the "hotspot") moves one
cell to the right at each step, so the Moving LISA procedure should find
one persistent mover whose space-time anchors drift rightwards.  No
downloads are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_toy"]


def load_toy():
    """Load the bundled toy dataset.

    Returns
    -------
    dict with keys:

    * ``y``: (36, 3) values array (units x time steps),
    * ``polygons``: list of 36 (4, 2) vertex arrays (unit squares),
    * ``w``: (36, 36) queen-contiguity weights, row-standardized,
    * ``times``: (3,) time-step labels.
    """
    path = Path(__file__).parent / "data" / "toy_grid.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "y": data["y"],
        "polygons": [np.asarray(p) for p in data["polygons"]],
        "w": data["w"],
        "times": data["times"],
    }
