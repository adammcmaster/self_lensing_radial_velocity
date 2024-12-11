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
    "start_time": Time("2025-01-01"),
    "end_time": Time("2026-01-01"),
}

default_times = list(range(int((DEFAULTS["end_time"] - DEFAULTS["start_time"]).value)))


def gen_sls(a, e, periapsis_phase, macc):
    return SelfLensingSystem(
        a=a * units.au,
        b=DEFAULTS["b"],
        Macc=macc * units.M_sun,
        e=e * units.dimensionless_unscaled,
        periapsis_phase=periapsis_phase * units.rad,
        Mcomp=DEFAULTS["Mcomp"],
        Rcomp=DEFAULTS["Rcomp"],
        Teff=DEFAULTS["Teff"],
    )
