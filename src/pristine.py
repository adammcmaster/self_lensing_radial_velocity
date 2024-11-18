import numpy
import pandas

from astropy.time import Time
from astropy import units

from pathlib import Path

from self_lensing import SelfLensingSystem, LensingEvent, gen_sls, DEFAULTS

DATA_PATH = Path(__file__).parent / Path("../data/pristine_timeseries")
DATA_PATH.mkdir(exist_ok=True, parents=True)

TIMESERIES_TABLE_PATH = DATA_PATH / "pristine_timeseries.parquet"
LOMBSCARGLE_TABLE_PATH = DATA_PATH / "pristine_lombscargle.parquet"

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
}

PARAMS["total"] = (
    len(PARAMS["a_sample"])
    * len(PARAMS["e_sample"])
    * len(PARAMS["periapsis_phase_sample"])
    * len(PARAMS["macc_sample"])
)


def calc_rv(a, e, periapsis_phase, macc):
    sls = gen_sls(a, e, periapsis_phase, macc)
    le = LensingEvent(sls, DEFAULTS["start_time"])

    rv = le.rv_range(DEFAULTS["start_time"], DEFAULTS["end_time"])
    return {
        "a": a,
        "e": e,
        "periapsis_phase": periapsis_phase,
        "Macc": macc,
        "Porb": sls.porb.value,
        "Mcomp": DEFAULTS["Mcomp"].value,
        "Rcomp": DEFAULTS["Rcomp"].value,
        "b": DEFAULTS["b"].value,
        "rv": rv.value.tolist(),
    }


def load_pristine_timeseries():
    return pandas.read_parquet(TIMESERIES_TABLE_PATH)


def load_pristine_lombscargle():
    return pandas.read_parquet(LOMBSCARGLE_TABLE_PATH)
