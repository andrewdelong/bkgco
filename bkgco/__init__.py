"""Python 3 bindings for GCoptimization (graph-cut multi-label optimization)
and for the layers underneath it: BKEnergy (binary energies) and BKGraph (max-flow).

    from bkgco import GCO
    with GCO.grid((h, w), num_labels) as g:
        g.set_data_cost(D).set_smooth_cost(V)
        g.expansion()
        labels = g.get_labeling().reshape(h, w)
"""

from ._core import (
    DTYPES,
    GCO,
    MAX_ENERGY_TERM,
    BKEnergy,
    BKGraph,
    max_energy_term,
)

__all__ = ["GCO", "BKEnergy", "BKGraph", "DTYPES", "MAX_ENERGY_TERM", "max_energy_term"]
__version__ = "0.1.0"
