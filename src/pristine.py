import numpy
import pandas

from astropy import units
from pathlib import Path

from selflensing.generators import RVGenerator, LightcurveGenerator
from selflensing.models import SelfLensingSystem

from utils import DEFAULTS

DATA_PATH = Path(__file__).parent / Path("../data/")
DATA_PATH.mkdir(exist_ok=True, parents=True)

TIMESERIES_TABLE_PATH = DATA_PATH / "pristine_timeseries.parquet"
LOMBSCARGLE_TABLE_PATH = DATA_PATH / "pristine_rv_lombscargle.parquet"

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
    sls = SelfLensingSystem(
        a=a * units.au,
        b=DEFAULTS["b"],
        Macc=macc * units.M_sun,
        e=e * units.dimensionless_unscaled,
        periapsis_phase=periapsis_phase * units.rad,
        Mcomp=DEFAULTS["Mcomp"],
        Rcomp=DEFAULTS["Rcomp"],
        Teff=DEFAULTS["Teff"],
    )
    if sls.r_roche <= sls.Rcomp:
        return
    rvg = RVGenerator(
        sls,
        start_time=DEFAULTS["rv_start_time"],
        end_time=DEFAULTS["rv_end_time"],
        peak_time=DEFAULTS["peak_time"],
    )
    lcg = LightcurveGenerator(
        sls,
        start_time=DEFAULTS["lc_start_time"],
        end_time=DEFAULTS["lc_end_time"],
        peak_time=DEFAULTS["peak_time"],
    )

    return {
        "a": a,
        "e": e,
        "periapsis_phase": periapsis_phase,
        "Macc": macc,
        "Porb": sls.porb.value,
        "Mcomp": DEFAULTS["Mcomp"].value,
        "Rcomp": DEFAULTS["Rcomp"].value,
        "b": DEFAULTS["b"].value,
        "rv": rvg.rv.value.tolist(),
        "rv_times": list(rvg.times.jd),
        "lc": list(lcg.flux_array),
        "lc_times": list(lcg.times.jd),
        "peak_time": DEFAULTS["peak_time"].jd,
    }


def load_pristine_timeseries():
    return pandas.read_parquet(TIMESERIES_TABLE_PATH)


def load_pristine_lombscargle():
    return pandas.read_parquet(LOMBSCARGLE_TABLE_PATH)
