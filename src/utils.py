from astropy import units
from astropy.time import Time


DEFAULTS = {
    # We just fix impact param as it doesn't make enough difference to change it
    "b": 0 * units.dimensionless_unscaled,
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


def generate_time_range(start_time, end_time):
    """
    Generate a list of astropy Time objects, one per day, from start_time to end_time (inclusive).
    """
    n_days = int((end_time - start_time).to("day").value) + 1
    return Time([start_time + i * units.day for i in range(n_days)])


default_rv_times = generate_time_range(
    DEFAULTS["rv_start_time"], DEFAULTS["rv_end_time"]
)
default_lc_times = generate_time_range(
    DEFAULTS["lc_start_time"], DEFAULTS["lc_end_time"]
)
