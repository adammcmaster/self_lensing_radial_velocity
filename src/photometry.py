"""Discovery helpers for the input photometric simulations.

The pipeline no longer creates an intermediate, dense RV product.  Photometry
is used directly by stage 1 and its fitted posterior is the hand-off to the RV
stages.
"""

from pathlib import Path

import pandas

from utils import DATA_PATH


PHOTOMETRY_SIMULATION_PATH = DATA_PATH / "simulations/2.0/lensing"
PHOTOMETRY_METADATA_PATH = DATA_PATH / "photometry_prefits.gz"


def photometry_simulation_paths(path=PHOTOMETRY_SIMULATION_PATH):
    """Return source photometry simulations in deterministic order."""
    return sorted(Path(path).glob("*.fits.gz"))


def load_photometry_timeseries():
    """Load stage-1 photometry-prefit metadata when it is available."""
    if not PHOTOMETRY_METADATA_PATH.exists():
        return pandas.DataFrame()
    return pandas.read_pickle(PHOTOMETRY_METADATA_PATH)
