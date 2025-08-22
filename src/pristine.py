import numpy
import pandas

from astropy import units
from astropy.time import Time
from pathlib import Path

from tqdm.contrib.concurrent import process_map

from selflensing.generators import RVGenerator, LightcurveGenerator
from selflensing.models import SelfLensingSystem

from utils import DEFAULTS, default_rv_times

DATA_PATH = Path(__file__).parent / Path("../data/")
DATA_PATH.mkdir(exist_ok=True, parents=True)

TIMESERIES_TABLE_PATH = DATA_PATH / "pristine_timeseries.gz"

PARAMS = {
    "a_sample": numpy.logspace(-1, 2, 50),
    # "a_sample": numpy.concatenate(
    #     [
    #         numpy.linspace(0.1, 1, 15),
    #         numpy.linspace(1, 10, 15),
    #         numpy.linspace(10, 100, 15),
    #     ]
    # ),
    "b_sample": numpy.linspace(0, 10, 50),
    "macc_sample": numpy.linspace(5, 30, 25),
    "e_sample": numpy.array([0]),  # numpy.linspace(0, 0.9, 5),
    "periapsis_phase_sample": numpy.array([0]),  # numpy.linspace(0, 2 * numpy.pi, 5),
}

PARAMS["total"] = (
    len(PARAMS["a_sample"])
    * len(PARAMS["b_sample"])
    * len(PARAMS["e_sample"])
    * len(PARAMS["periapsis_phase_sample"])
    * len(PARAMS["macc_sample"])
)


def calc_rv(params):
    a, b, e, periapsis_phase, macc = params
    sls = SelfLensingSystem(
        a=a * units.au,
        b=b * units.dimensionless_unscaled,
        Macc=macc * units.M_sun,
        e=e * units.dimensionless_unscaled,
        periapsis_phase=periapsis_phase * units.rad,
        Mcomp=DEFAULTS["Mcomp"],
        Rcomp=DEFAULTS["Rcomp"],
        Teff=DEFAULTS["Teff"],
    )
    if numpy.isnan(sls.tau_eff):
        # b is too large to cause lensing
        return None

    if sls.r_roche <= sls.Rcomp:
        return
    rvg = RVGenerator(
        sls,
        times=default_rv_times,
        peak_time=DEFAULTS["peak_time"],
    )

    # Generate a light curve with good cadence based on the crossing time
    lc_times = numpy.arange(
        DEFAULTS["lc_start_time"], DEFAULTS["lc_end_time"], 1 * units.day
    )
    # lc_step = sls.porb / 4
    # if lc_step > 5 * units.day:
    #     lc_step = 5 * units.day
    # lc_times = numpy.arange(DEFAULTS["lc_start_time"], DEFAULTS["lc_end_time"], lc_step)

    # lensing_len is the length of the high res portion
    # lensing_step is the time between points in the high res portion
    MAX_LENSING_LEN = 14 * units.day
    MIN_LENSING_LEN = 7 * units.day

    lensing_len = sls.tau_eff * 4
    if lensing_len > MAX_LENSING_LEN:
        lensing_len = MAX_LENSING_LEN
    elif lensing_len < MIN_LENSING_LEN:
        lensing_len = MIN_LENSING_LEN
    lensing_step = sls.tau_eff / 4

    # High res around the main peak
    lc_times = numpy.concat(
        [
            lc_times,
            numpy.arange(
                DEFAULTS["peak_time"] - lensing_len / 2,
                DEFAULTS["peak_time"] + lensing_len / 2,
                lensing_step,
            ),
        ]
    )
    # # High res around the previous peak
    # if DEFAULTS["peak_time"] - sls.porb > DEFAULTS["lc_start_time"]:
    #     lc_times = numpy.concat(
    #         [
    #             lc_times,
    #             numpy.arange(
    #                 DEFAULTS["peak_time"] - sls.porb - lensing_len / 2,
    #                 DEFAULTS["peak_time"] - sls.porb + lensing_len / 2,
    #                 lensing_step,
    #             ),
    #         ]
    #     )
    # # High res around the next peak
    # if DEFAULTS["peak_time"] + sls.porb < DEFAULTS["lc_end_time"]:
    #     lc_times = numpy.concat(
    #         [
    #             lc_times,
    #             numpy.arange(
    #                 DEFAULTS["peak_time"] + sls.porb - lensing_len / 2,
    #                 DEFAULTS["peak_time"] + sls.porb + lensing_len / 2,
    #                 lensing_step,
    #             ),
    #         ]
    #     )

    lc_times = Time(lc_times)

    lcg = LightcurveGenerator(
        sls,
        times=lc_times,
        peak_time=DEFAULTS["peak_time"],
    )
    return {
        "a": a,
        "e": e,
        "periapsis_phase": periapsis_phase,
        "Macc": macc,
        "Porb": sls.porb.value,
        "tau_eff": sls.tau_eff.value,
        "Mcomp": DEFAULTS["Mcomp"].value,
        "Rcomp": DEFAULTS["Rcomp"].value,
        "b": b,
        "rv": rvg.rv.value.tolist(),
        "rv_times": list(rvg.times.jd),
        "lc": list(lcg.flux_array.value),
        "lc_times": list(lcg.times.jd),
        "peak_time": DEFAULTS["peak_time"].jd,
    }


def generate_pristine_timeseries(desc=None):
    params = []
    for a in PARAMS["a_sample"]:
        for b in PARAMS["b_sample"]:
            for e in PARAMS["e_sample"]:
                for periapsis_phase in PARAMS["periapsis_phase_sample"]:
                    for macc in PARAMS["macc_sample"]:
                        params.append((a, b, e, periapsis_phase, macc))

    results = process_map(calc_rv, params, desc=desc)
    results = [r for r in results if r is not None]
    return pandas.DataFrame(results)


def load_pristine_timeseries():
    return pandas.read_pickle(TIMESERIES_TABLE_PATH)
