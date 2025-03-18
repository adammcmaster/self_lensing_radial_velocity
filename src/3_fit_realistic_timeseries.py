"""
This takes the pristine RV timeseries, samples them at various observing cadences and then adds noise.
"""

import numpy
import pandas

from astropy import units
from astropy.time import Time
from tqdm import tqdm

from pathlib import Path

from selflensing.fitting import self_lensing_system_from_fit

from realistic import load_realistic_timeseries
from spectrograph import Spectrograph
from utils import DEFAULTS

import warnings

warnings.filterwarnings("ignore")

DATA_PATH = Path(__file__).parent / Path("../data/")
DATA_PATH.mkdir(exist_ok=True, parents=True)

TIMESERIES_TABLE_PATH = DATA_PATH / "fitted_timeseries.parquet"

if __name__ == "__main__":
    rts = load_realistic_timeseries()
    results = []

    total = len(rts)

    with tqdm(total=total, desc="Total") as pbar:
        for i, ts in rts.iterrows():
            sls, sampler = self_lensing_system_from_fit(
                rv_times=Time(ts["rv_times"], format="jd"),
                observed_rv=ts["rv"] * units.m / units.s,
                observed_rv_err=ts["rv_err"] * units.m / units.s,
                lc_times=Time(ts["lc_times"], format="jd"),
                observed_lc=ts["lc"] * units.dimensionless_unscaled,
                observed_lc_err=1 * units.dimensionless_unscaled,
                peak_time=Time(ts["peak_time"], format="jd"),
                Mcomp=DEFAULTS["Mcomp"],
                Rcomp=DEFAULTS["Rcomp"],
                Teff=DEFAULTS["Teff"],
                nsteps=2000,
                nwalkers=50,
                burn=10,
            )
            new_ts = pandas.Series(ts)
            new_ts["fitted_a"] = sls.a
            new_ts["fitted_Macc"] = sls.Macc
            new_ts["fitted_e"] = sls.e
            new_ts["fitted_periapsis_phase"] = sls.periapsis_phase
            new_ts["fitted_porb"] = sls.porb
            results.append(new_ts)
            pbar.update(1)

    results = pandas.DataFrame(results)
    results.reset_index(inplace=True, drop=True)
    results.to_parquet(TIMESERIES_TABLE_PATH)
