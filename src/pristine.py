import numpy

from astropy.time import Time
from astropy import units

from pathlib import Path

from self_lensing import SelfLensingSystem

DATA_PATH = Path("../data/pristine_timeseries")
DATA_PATH.mkdir(exist_ok=True, parents=True)

PARAMS = {
    "a_sample": numpy.linspace(0.1, 200, 1000),
    "b_sample": range(0, 10),
    "macc_sample": numpy.linspace(0.1, 20, 2000),
    # We assume a Sun-like star. This shouldn't really matter for RV calculations
    "Mcomp": 1 * units.M_sun,
    "Rcomp": 1 * units.R_sun,
    "Teff": 5500 * units.K,
    # Pick an arbitrary start and end time
    "start_time": Time("2025-01-01"),
    "end_time": Time("2026-01-01"),
}

PARAMS["total"] = (
    len(PARAMS["a_sample"]) * len(PARAMS["b_sample"]) * len(PARAMS["macc_sample"]),
)


def gen_sls(a, b, macc):
    return SelfLensingSystem(
        a=a * units.au,
        b=b * units.dimensionless_unscaled,
        Macc=macc * units.M_sun,
        Mcomp=PARAMS["Mcomp"],
        Rcomp=PARAMS["Rcomp"],
        Teff=PARAMS["Teff"],
    )
