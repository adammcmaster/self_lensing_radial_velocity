"""
This fits a sample of the realistic timeseries and records the fitted parameters.
"""

import warnings
import os

import numpy
import pandas

from astropy import units
from astropy.time import Time

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from selflensing.fitting import self_lensing_system_from_fit

from fitting import TIMESERIES_TABLE_PATH, plot_fit, plot_emcee_diagnostics
from realistic import load_realistic_timeseries
from utils import DEFAULTS, DATA_PATH


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="erfa.core")


def fit(a):
    i, ts = a
    sls, sampler = self_lensing_system_from_fit(
        rv_times=Time(ts["rv_times"], format="jd"),
        observed_rv=ts["rv"] * units.m / units.s,
        observed_rv_err=ts["rv_err"] * units.m / units.s,
        lc_times=Time(ts["lc_times"], format="jd"),
        observed_lc=ts["lc"] * units.dimensionless_unscaled,
        observed_lc_err=ts["lc_err"] * units.dimensionless_unscaled,
        peak_time=Time(ts["peak_time"], format="jd"),
        Mcomp=DEFAULTS["Mcomp"],
        Rcomp=DEFAULTS["Rcomp"],
        Teff=DEFAULTS["Teff"],
        tau_eff=ts["tau_eff"] * units.day,
        false_alarm_probability=0.5,
        nsteps=2000,
        nwalkers=20,
        burn=500,
        pool=False,
        progress=False,
    )
    new_ts = pandas.Series(ts)
    new_ts["realistic_idx"] = i
    new_ts["fitted_a"] = sls.a.value
    new_ts["fitted_b"] = sls.b.value
    new_ts["fitted_Macc"] = sls.Macc.value
    new_ts["fitted_e"] = sls.e.value
    new_ts["fitted_periapsis_phase"] = sls.periapsis_phase.value
    new_ts["fitted_porb"] = sls.porb.value
    new_ts["fitted_tau_eff"] = sls.tau_eff.value
    return new_ts, sampler


FIG_PATH = DATA_PATH / "figures"
FIG_PATH.mkdir(exist_ok=True, parents=True)


def save_plots(a):
    ts, sampler = a

    out_path = (
        FIG_PATH
        / f"a1e{int(numpy.floor(numpy.log10(ts['a'])))}_b{int(ts['b'])}_Macc{int(ts['Macc'])}"
        / str(ts["realistic_idx"])
    )
    out_path.mkdir(exist_ok=True, parents=True)

    fit_fig = plot_fit(ts)
    params = ["a", "b", "Macc", "e", "periapsis_phase"]
    walker_fig, corner_fig = plot_emcee_diagnostics(
        sampler,
        param_names=params,
        truths=ts[params],
        axes_scale=["log", "linear", "linear", "linear", "linear"],
    )

    fit_fig.savefig(out_path / "fit.png")
    walker_fig.savefig(out_path / "walkers.png")
    corner_fig.savefig(out_path / "corner.png")


if __name__ == "__main__":
    rts = load_realistic_timeseries()
    rts = rts.sample(5000)

    chunk_size = 50
    chunks = [rts.iloc[i : i + chunk_size] for i in range(0, len(rts), chunk_size)]

    results = []

    for chunk in tqdm(chunks, desc="All chunks"):
        new_results = process_map(
            fit, chunk.iterrows(), total=len(chunk), desc="Current chunk"
        )
        results += [r[0] for r in new_results]
        pandas.DataFrame(results).reset_index(drop=True).to_pickle(
            f"{TIMESERIES_TABLE_PATH}.new.gz"
        )
        os.replace(f"{TIMESERIES_TABLE_PATH}.new.gz", TIMESERIES_TABLE_PATH)
        process_map(save_plots, new_results, desc="Saving plots")
