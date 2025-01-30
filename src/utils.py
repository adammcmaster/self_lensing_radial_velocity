from astropy import units
from astropy.time import Time

from selflensing.models import SelfLensingSystem


DEFAULTS = {
    # We just fix impact param as it doesn't make enough difference to change it
    "b": 0 * units.dimensionless_unscaled,
    # We assume a Sun-like star. This shouldn't really matter for RV calculations
    "Mcomp": 1 * units.M_sun,
    "Rcomp": 1 * units.R_sun,
    "Teff": 5500 * units.K,
    # Pick an arbitrary start and end time
    "rv_start_time": Time("2025-01-01"),
    "rv_end_time": Time("2026-01-01"),
    "lc_start_time": Time("2024-01-01"),
    "lc_end_time": Time("2025-01-01"),
    "peak_time": Time("2024-06-01"),
}

default_rv_times = list(
    range(int((DEFAULTS["rv_end_time"] - DEFAULTS["rv_start_time"]).value))
)
default_lc_times = list(
    range(int((DEFAULTS["lc_end_time"] - DEFAULTS["lc_start_time"]).value))
)

PROC_LIMIT = 100
