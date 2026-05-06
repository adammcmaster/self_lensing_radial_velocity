"""
This fits a sample of the realistic timeseries and records the fitted parameters.
"""

import logging
import multiprocessing
import warnings
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy
import pandas

from astropy import units
from astropy.time import Time

from tqdm import tqdm

from selflensing.fitting import self_lensing_system_from_fit
from selflensing.models import SelfLensingSystem

from fitting import TIMESERIES_TABLE_PATH, plot_fit, plot_emcee_diagnostics
from realistic import load_realistic_timeseries, load_realistic_timeseries_curve
from utils import DEFAULTS, DATA_PATH, configure_logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="erfa.core")

FIT_MAX_WORKERS = int(os.environ.get("SLRV_FIT_MAX_WORKERS", 8))
FIT_SAMPLE_SIZE = int(os.environ.get("SLRV_FIT_SAMPLE_SIZE", 5000))
FIT_CHUNK_SIZE = int(os.environ.get("SLRV_FIT_CHUNK_SIZE", 2 * FIT_MAX_WORKERS))
PLOT_MAX_WORKERS = int(os.environ.get("SLRV_PLOT_MAX_WORKERS", 8))
MP_START_METHOD = os.environ.get("SLRV_MP_START_METHOD", "spawn")
FIT_NSTEPS = int(os.environ.get("SLRV_FIT_NSTEPS", 2000))
FIT_NWALKERS = int(os.environ.get("SLRV_FIT_NWALKERS", 20))
FIT_BURN = int(os.environ.get("SLRV_FIT_BURN", 500))
FIT_USE_LC = os.environ.get("SLRV_FIT_USE_LC", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def process_map(fn, iterable, total=None, desc=None, max_workers=None, chunksize=1):
    """Run a tqdm-wrapped process map with an explicit multiprocessing context."""
    items = list(iterable)
    if total is None:
        total = len(items)
    if not items:
        return []

    context = multiprocessing.get_context(MP_START_METHOD)
    logger.debug(
        "Starting process map desc=%s items=%d max_workers=%s chunksize=%d mp_start_method=%s",
        desc,
        len(items),
        max_workers,
        chunksize,
        MP_START_METHOD,
    )
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        return list(
            tqdm(
                executor.map(fn, items, chunksize=chunksize),
                total=total,
                desc=desc,
            )
        )


def load_fitted_timeseries():
    if not TIMESERIES_TABLE_PATH.exists():
        logger.debug("Fitted timeseries table does not exist at %s", TIMESERIES_TABLE_PATH)
        return pandas.DataFrame()
    table = pandas.read_pickle(TIMESERIES_TABLE_PATH)
    logger.debug("Loaded %d fitted timeseries rows from %s", len(table), TIMESERIES_TABLE_PATH)
    return table


def fitted_key_sets(results):
    if results.empty:
        logger.debug("No fitted keys because fitted results table is empty")
        return set(), set()
    realistic_ids = set()
    realistic_indices = set()
    if "realistic_id" in results.columns:
        realistic_ids = set(results["realistic_id"].dropna().astype(str))
    if "realistic_idx" in results.columns:
        if "realistic_id" in results.columns:
            legacy_results = results[results["realistic_id"].isna()]
        else:
            legacy_results = results
        realistic_indices = set(legacy_results["realistic_idx"].dropna().astype(str))
    logger.debug(
        "Prepared fitted key sets with %d ids and %d legacy indices",
        len(realistic_ids),
        len(realistic_indices),
    )
    return realistic_ids, realistic_indices


def is_already_fitted(row, realistic_ids, realistic_indices):
    if "realistic_id" in row and not pandas.isna(row["realistic_id"]):
        if str(row["realistic_id"]) in realistic_ids:
            logger.debug("Skipping already fitted realistic_id=%s", row["realistic_id"])
            return True
    already_fitted = str(row.name) in realistic_indices
    if already_fitted:
        logger.debug("Skipping already fitted legacy realistic_idx=%s", row.name)
    return already_fitted


def result_checkpoint_key(row):
    if "realistic_id" in row and not pandas.isna(row["realistic_id"]):
        return f"id:{row['realistic_id']}"
    if "realistic_idx" in row and not pandas.isna(row["realistic_idx"]):
        return f"idx:{row['realistic_idx']}"
    return f"row:{row.name}"


def save_results(results):
    if not results:
        logger.debug("No fit results to save")
        return
    logger.debug("Saving %d fit result rows to %s", len(results), TIMESERIES_TABLE_PATH)
    table = pandas.DataFrame(results).reset_index(drop=True)
    if "realistic_id" in table.columns or "realistic_idx" in table.columns:
        table["_fit_key"] = table.apply(result_checkpoint_key, axis=1)
        table.drop_duplicates(subset="_fit_key", keep="last", inplace=True)
        table.drop(columns="_fit_key", inplace=True)
    table.reset_index(drop=True).to_pickle(f"{TIMESERIES_TABLE_PATH}.new.gz")
    os.replace(f"{TIMESERIES_TABLE_PATH}.new.gz", TIMESERIES_TABLE_PATH)
    logger.debug("Saved fit results atomically to %s", TIMESERIES_TABLE_PATH)


def truth_self_lensing_system(ts):
    logger.debug(
        "Building truth SelfLensingSystem for realistic_idx=%s realistic_id=%s",
        ts.get("realistic_idx", getattr(ts, "name", None)),
        ts.get("realistic_id", None),
    )
    sls = SelfLensingSystem(
        a=ts["a"] * units.au,
        i=ts["i"] * units.rad,
        Macc=ts["Macc"] * units.M_sun,
        e=ts["e"] * units.dimensionless_unscaled,
        periapsis_phase=ts["periapsis_phase"] * units.rad,
        Mcomp=ts["Mcomp"] * units.M_sun,
        Rcomp=ts["Rcomp"] * units.R_sun,
        Teff=ts["Teff"] * units.K,
    )
    logger.debug(
        "Built truth SelfLensingSystem for realistic_idx=%s realistic_id=%s",
        ts.get("realistic_idx", getattr(ts, "name", None)),
        ts.get("realistic_id", None),
    )
    return sls


def fit(a):
    i, ts = a
    logger.debug(
        "Starting fit for row index=%s realistic_id=%s path=%s",
        i,
        ts.get("realistic_id", None),
        ts.get("realistic_path", None),
    )
    curve = load_realistic_timeseries_curve(ts)
    truth_sls = truth_self_lensing_system(ts)
    logger.debug(
        "Fit inputs row index=%s: %d RV samples, %d LC samples",
        i,
        len(curve["rv"]),
        len(curve["lc"]),
    )
    logger.debug("Calling self_lensing_system_from_fit for row index=%s", i)
    sls, sampler = self_lensing_system_from_fit(
        rv_times=Time(curve["rv_times"], format="jd"),
        observed_rv=curve["rv"] * units.m / units.s,
        observed_rv_err=curve["rv_err"] * units.m / units.s,
        lc_times=Time(curve["lc_times"], format="jd") if FIT_USE_LC else None,
        observed_lc=(
            curve["lc"] * units.dimensionless_unscaled if FIT_USE_LC else None
        ),
        observed_lc_err=(
            curve["lc_err"] * units.dimensionless_unscaled if FIT_USE_LC else None
        ),
        peak_time=Time(ts["peak_time"], format="jd"),
        Mcomp=DEFAULTS["Mcomp"],
        Rcomp=DEFAULTS["Rcomp"],
        Teff=DEFAULTS["Teff"],
        tau_eff=ts["tau_eff"] * units.day,
        false_alarm_probability=0.5,
        nsteps=FIT_NSTEPS,
        nwalkers=FIT_NWALKERS,
        burn=FIT_BURN,
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
    logger.debug(
        "Finished fit for row index=%s: fitted_a=%.6g fitted_Macc=%.6g fitted_e=%.6g",
        i,
        new_ts["fitted_a"],
        new_ts["fitted_Macc"],
        new_ts["fitted_e"],
    )
    return new_ts, sampler


FIG_PATH = DATA_PATH / "figures"
FIG_PATH.mkdir(exist_ok=True, parents=True)


def save_plots(a):
    ts, sampler = a
    logger.debug(
        "Saving plots for realistic_idx=%s realistic_id=%s",
        ts.get("realistic_idx", None),
        ts.get("realistic_id", None),
    )
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
    logger.debug("Writing fit diagnostic plots to %s", out_path)

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
    logger.debug("Saved fit diagnostic plots to %s", out_path)


if __name__ == "__main__":
    configure_logging()
    logger.debug(
        "Starting fitting script with FIT_MAX_WORKERS=%d FIT_SAMPLE_SIZE=%d FIT_CHUNK_SIZE=%d PLOT_MAX_WORKERS=%d",
        FIT_MAX_WORKERS,
        FIT_SAMPLE_SIZE,
        FIT_CHUNK_SIZE,
        PLOT_MAX_WORKERS,
    )
    logger.debug(
        "MCMC settings: FIT_NSTEPS=%d FIT_NWALKERS=%d FIT_BURN=%d MP_START_METHOD=%s",
        FIT_NSTEPS,
        FIT_NWALKERS,
        FIT_BURN,
        MP_START_METHOD,
    )
    logger.debug("Fit likelihood settings: FIT_USE_LC=%s", FIT_USE_LC)
    rts = load_realistic_timeseries()
    existing_results = load_fitted_timeseries()
    fitted_ids, fitted_indices = fitted_key_sets(existing_results)
    logger.debug(
        "Loaded %d realistic rows and %d existing fitted rows",
        len(rts),
        len(existing_results),
    )

    if fitted_ids or fitted_indices:
        rts = rts[
            ~rts.apply(
                lambda row: is_already_fitted(row, fitted_ids, fitted_indices),
                axis=1,
            )
        ].copy()
        logger.debug("Filtered realistic rows down to %d unfitted rows", len(rts))

    sample_size = min(FIT_SAMPLE_SIZE, len(rts))
    if sample_size == 0:
        logger.debug("No realistic rows left to fit; exiting")
        raise SystemExit

    rts = rts.sample(sample_size)
    logger.debug("Sampled %d realistic rows for fitting", len(rts))

    chunks = [
        rts.iloc[i : i + FIT_CHUNK_SIZE] for i in range(0, len(rts), FIT_CHUNK_SIZE)
    ]
    logger.debug("Split sampled rows into %d chunks", len(chunks))

    results = (
        [row for _, row in existing_results.iterrows()]
        if not existing_results.empty
        else []
    )

    for chunk in tqdm(chunks, desc="All chunks"):
        logger.debug("Fitting chunk with %d rows", len(chunk))
        new_results = process_map(
            fit,
            chunk.iterrows(),
            total=len(chunk),
            desc="Current chunk",
            max_workers=FIT_MAX_WORKERS,
        )
        results += [r[0] for r in new_results]
        logger.debug("Chunk produced %d fit results; cumulative results=%d", len(new_results), len(results))
        save_results(results)
        process_map(
            save_plots,
            new_results,
            desc="Saving plots",
            max_workers=PLOT_MAX_WORKERS,
        )
    logger.debug("Finished fitting script")
