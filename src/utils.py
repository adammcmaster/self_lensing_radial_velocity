from pathlib import Path

from astropy import units
from astropy.time import Time
import numpy


DEFAULTS = {
    # We just fix impact param as it doesn't make enough difference to change it
    # "b": 0 * units.dimensionless_unscaled,
    # We assume a Sun-like star. This shouldn't really matter for RV calculations
    "Mcomp": 1 * units.M_sun,
    "Rcomp": 1 * units.R_sun,
    "Teff": 5500 * units.K,
    # Pick an arbitrary start and end time
    "rv_start_time": Time("2025-01-01"),
    "rv_end_time": Time("2030-01-01"),
    "lc_start_time": Time("2024-01-01"),
    "lc_end_time": Time("2025-01-01"),
    "peak_time": Time("2024-06-01"),
}


default_rv_times = numpy.linspace(
    DEFAULTS["rv_start_time"],
    DEFAULTS["rv_end_time"],
    int((DEFAULTS["rv_end_time"] - DEFAULTS["rv_start_time"]).value),
)

DATA_PATH = Path(__file__).parent / Path("../data/")
DATA_PATH.mkdir(exist_ok=True, parents=True)
