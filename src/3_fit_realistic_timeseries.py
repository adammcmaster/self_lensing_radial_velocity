"""
This fits a sample of the realistic timeseries and records the fitted parameters.
"""

import warnings
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy
import pandas

from astropy import units
from astropy.time import Time

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from selflensing.fitting import self_lensing_system_from_fit
from selflensing.models import SelfLensingSystem

from fitting import TIMESERIES_TABLE_PATH, plot_fit, plot_emcee_diagnostics
from realistic import load_realistic_timeseries, load_realistic_timeseries_curve
from utils import DEFAULTS, DATA_PATH


warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="erfa.core")

FIT_MAX_WORKERS = int(os.environ.get("SLRV_FIT_MAX_WORKERS", 8))
FIT_SAMPLE_SIZE = int(os.environ.get("SLRV_FIT_SAMPLE_SIZE", 5000))
FIT_CHUNK_SIZE = int(os.environ.get("SLRV_FIT_CHUNK_SIZE", 2 * FIT_MAX_WORKERS))
PLOT_MAX_WORKERS = int(os.environ.get("SLRV_PLOT_MAX_WORKERS", 8))


def truth_self_lensing_system(ts):
    return SelfLensingSystem(
        a=ts["a"] * units.au,
        i=ts["i"] * units.rad,
        Macc=ts["Macc"] * units.M_sun,
        e=ts["e"] * units.dimensionless_unscaled,
        periapsis_phase=ts["periapsis_phase"] * units.rad,
        Mcomp=ts["Mcomp"] * units.M_sun,
        Rcomp=ts["Rcomp"] * units.R_sun,
        Teff=ts["Teff"] * units.K,
    )


def fit(a):
    i, ts = a
    curve = load_realistic_timeseries_curve(ts)
    truth_sls = truth_self_lensing_system(ts)
    sls, sampler = self_lensing_system_from_fit(
        rv_times=Time(curve["rv_times"], format="jd"),
        observed_rv=curve["rv"] * units.m / units.s,
        observed_rv_err=curve["rv_err"] * units.m / units.s,
        lc_times=Time(curve["lc_times"], format="jd"),
        observed_lc=curve["lc"] * units.dimensionless_unscaled,
        observed_lc_err=curve["lc_err"] * units.dimensionless_unscaled,
        peak_time=Time(ts["peak_time"], format="jd"),
        Mcomp=DEFAULTS["Mcomp"],
        Rcomp=DEFAULTS["Rcomp"],
        Teff=DEFAULTS["Teff"],
        # tau_eff=ts["tau_eff"] * units.day,
        false_alarm_probability=0.5,
        nsteps=2000,
        nwalkers=20,
        burn=500,
        pool=False,
        progress=False,
    )
    new_ts = pandas.Series(ts)
    new_ts["realistic_idx"] = i
    new_ts["b_max"] = truth_sls.b_max.value
    new_ts["fitted_a"] = sls.a.value
    new_ts["fitted_b"] = sls.b.value
    new_ts["fitted_b_max"] = sls.b_max.value
    new_ts["fitted_i"] = sls.i.value
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
    curve = load_realistic_timeseries_curve(ts)
    plot_ts = pandas.Series(ts)
    plot_ts["rv_times"] = curve["rv_times"]
    plot_ts["rv"] = curve["rv"]
    plot_ts["rv_err"] = curve["rv_err"]
    plot_ts["lc_times"] = curve["lc_times"]
    plot_ts["lc"] = curve["lc"]
    plot_ts["lc_err"] = curve["lc_err"]
    if "b_max" not in plot_ts or pandas.isna(plot_ts["b_max"]):
        plot_ts["b_max"] = truth_self_lensing_system(plot_ts).b_max.value

    out_path = (
        FIG_PATH
        / f"a1e{int(numpy.floor(numpy.log10(ts['a'])))}_b{int(ts['b'])}_Macc{int(ts['Macc'])}"
        / str(ts["realistic_idx"])
    )
    out_path.mkdir(exist_ok=True, parents=True)

    fit_fig = plot_fit(plot_ts)
    params = ["a", "i", "Macc", "e", "periapsis_phase"]
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
    sample_size = min(FIT_SAMPLE_SIZE, len(rts))
    rts = rts.sample(sample_size)

    chunks = [
        rts.iloc[i : i + FIT_CHUNK_SIZE] for i in range(0, len(rts), FIT_CHUNK_SIZE)
    ]

    results = []

    for chunk in tqdm(chunks, desc="All chunks"):
        new_results = process_map(
            fit,
            chunk.iterrows(),
            total=len(chunk),
            desc="Current chunk",
            max_workers=FIT_MAX_WORKERS,
        )
        results += [r[0] for r in new_results]
        pandas.DataFrame(results).reset_index(drop=True).to_pickle(
            f"{TIMESERIES_TABLE_PATH}.new.gz"
        )
        os.replace(f"{TIMESERIES_TABLE_PATH}.new.gz", TIMESERIES_TABLE_PATH)
        process_map(
            save_plots,
            new_results,
            desc="Saving plots",
            max_workers=PLOT_MAX_WORKERS,
        )
