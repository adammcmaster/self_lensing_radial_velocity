"""
This fits a sample of the realistic timeseries and records the fitted parameters.
"""

import pandas

from astropy import units
from astropy.time import Time
from tqdm import tqdm

from multiprocessing import Pool

from selflensing.fitting import self_lensing_system_from_fit

from realistic import load_realistic_timeseries
from utils import DEFAULTS
from fitting import TIMESERIES_TABLE_PATH

import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    rts = load_realistic_timeseries()
    rts = rts.sample(200)
    results = []

    total = len(rts)

    with tqdm(total=total, desc="Total") as pbar:
        with Pool() as pool:
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
                    nsteps=1000,
                    nwalkers=20,
                    burn=300,
                    pool=pool,
                    progress=False,
                )
                new_ts = pandas.Series(ts)
                new_ts["realistic_idx"] = i
                new_ts["fitted_a"] = sls.a.value
                new_ts["fitted_Macc"] = sls.Macc.value
                new_ts["fitted_e"] = sls.e.value
                new_ts["fitted_periapsis_phase"] = sls.periapsis_phase.value
                new_ts["fitted_porb"] = sls.porb.value
                results.append(new_ts)
                pbar.update(1)

    results = pandas.DataFrame(results)
    results.reset_index(inplace=True, drop=True)
    results.to_parquet(TIMESERIES_TABLE_PATH)
