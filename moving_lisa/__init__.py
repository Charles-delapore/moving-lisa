"""Moving LISA: local indicators of spatial association over time.

Anderson & Yang (2025), "Moving Local Indicators of Spatial Association
for Time Geography", Transactions in GIS, doi:10.1111/tgis.70158.
"""

from .core import MovingLISA
from .data import load_toy
from .lisa import local_g, local_moran

__version__ = "0.1.0"

__all__ = ["MovingLISA", "load_toy", "local_g", "local_moran", "__version__"]
