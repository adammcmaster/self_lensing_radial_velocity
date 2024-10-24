import numpy
import pandas

from astropy.time import Time
from astropy import units

from pathlib import Path

from self_lensing import SelfLensingSystem, LensingEvent

DATA_PATH = Path(__file__).parent / Path("../data/pristine_timeseries")
DATA_PATH.mkdir(exist_ok=True, parents=True)

TIMESERIES_TABLE_PATH = DATA_PATH / "pristine_timeseries.parquet.gz"

PARAMS = {
    "a_sample": numpy.concatenate(
        [
            numpy.linspace(0.1, 5, 10),
            numpy.linspace(5, 50, 10),
            numpy.linspace(50, 200, 10),
        ]
    ),
    "macc_sample": numpy.linspace(0.1, 20, 10),
    "e_sample": numpy.linspace(0, 0.9, 10),
    "periapsis_phase_sample": numpy.linspace(0, 2 * numpy.pi, 10),
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

PARAMS["total"] = (
    len(PARAMS["a_sample"])
    * len(PARAMS["e_sample"])
    * len(PARAMS["periapsis_phase_sample"])
    * len(PARAMS["macc_sample"])
)


def gen_sls(a, e, periapsis_phase, macc):
    return SelfLensingSystem(
        a=a * units.au,
        b=PARAMS["b"],
        Macc=macc * units.M_sun,
        e=e * units.dimensionless_unscaled,
        periapsis_phase=periapsis_phase * units.rad,
        Mcomp=PARAMS["Mcomp"],
        Rcomp=PARAMS["Rcomp"],
        Teff=PARAMS["Teff"],
    )


def calc_rv(a, e, periapsis_phase, macc):
    sls = gen_sls(a, e, periapsis_phase, macc)
    le = LensingEvent(sls, PARAMS["start_time"])

    rv = le.rv_range(PARAMS["start_time"], PARAMS["end_time"])
    return {
        "a": a,
        "e": e,
        "periapsis_phase": periapsis_phase,
        "Macc": macc,
        "Mcomp": PARAMS["Mcomp"].value,
        "Rcomp": PARAMS["Rcomp"].value,
        "b": PARAMS["b"].value,
        "rv": rv.value.tolist(),
    }


def load_pristine_timeseries():
    return pandas.read_parquet(TIMESERIES_TABLE_PATH)
