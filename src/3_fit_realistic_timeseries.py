"""
This fits a sample of the realistic timeseries and records the fitted parameters.
"""

import logging
import multiprocessing
import warnings
import os
import argparse
import shutil
import inspect
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    os.environ.get("SLRV_JAX_PREALLOCATE", "false"),
)
if "SLRV_JAX_PLATFORMS" in os.environ:
    os.environ.setdefault("JAX_PLATFORMS", os.environ["SLRV_JAX_PLATFORMS"])

import jax

JAX_DEVICES = jax.devices()

import numpy
import pandas

from astropy import units
from astropy.time import Time
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot

from tqdm import tqdm

from jax_compat import install_quax_jax_stage_patch

install_quax_jax_stage_patch()

from selflensing.fitting import self_lensing_system_from_fit_blackjax

try:
    from selflensing.fitting import self_lensing_systems_from_batched_fit_blackjax
except ImportError:
    self_lensing_systems_from_batched_fit_blackjax = None

from selflensing.models import DEFAULT_LIMITS, SelfLensingSystem

from fitting import TIMESERIES_TABLE_PATH, plot_fit, plot_emcee_diagnostics
from realistic import (
    load_realistic_timeseries,
    load_realistic_timeseries_curve,
    load_realistic_timeseries_curves,
)
from utils import DATA_PATH, configure_logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="erfa.core")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit realistic timeseries and record fitted parameters."
    )
    parser.add_argument(
        "--clear-results",
        action="store_true",
        help="Remove saved fit results and generated fit figures before running.",
    )
    return parser.parse_args()


def jax_device_summary():
    return ", ".join(
        f"{device.platform}:{device.device_kind}" for device in JAX_DEVICES
    )


def default_fit_max_workers():
    if any(device.platform == "gpu" for device in JAX_DEVICES):
        return 2
    return 4


def default_fit_device():
    if any(device.platform == "gpu" for device in JAX_DEVICES):
        return "gpu:0"
    return "auto"


FIT_MAX_WORKERS = int(os.environ.get("SLRV_FIT_MAX_WORKERS", default_fit_max_workers()))
FIT_SAMPLE_SIZE = int(os.environ.get("SLRV_FIT_SAMPLE_SIZE", 5000))
FIT_BATCH_FILE = os.environ.get("SLRV_FIT_BATCH_FILE")
PLOT_MAX_WORKERS = int(os.environ.get("SLRV_PLOT_MAX_WORKERS", FIT_MAX_WORKERS))
SAVE_FIT_PLOTS = os.environ.get("SLRV_SAVE_FIT_PLOTS", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
MP_START_METHOD = os.environ.get("SLRV_MP_START_METHOD", "spawn")
FIT_NSTEPS = int(os.environ.get("SLRV_FIT_NSTEPS", 2000))
FIT_NWALKERS = int(os.environ.get("SLRV_FIT_NWALKERS", 1024))
FIT_BURN = int(os.environ.get("SLRV_FIT_BURN", 200))
FIT_PROGRESS_CHUNK_SIZE = int(os.environ.get("SLRV_FIT_PROGRESS_CHUNK_SIZE", 100))
FIT_LC_BINS = 100
FIT_E_MIN = os.environ.get("SLRV_FIT_E_MIN")
FIT_E_MAX = os.environ.get("SLRV_FIT_E_MAX")
FIT_USE_LC = os.environ.get("SLRV_FIT_USE_LC", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_DEVICE = os.environ.get("SLRV_FIT_DEVICE", default_fit_device())
FIT_LENSING_METHOD = os.environ.get("SLRV_FIT_LENSING_METHOD")
FIT_MCMC_PROGRESS = os.environ.get("SLRV_FIT_MCMC_PROGRESS", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_MCMC_DIAGNOSTICS = os.environ.get("SLRV_FIT_MCMC_DIAGNOSTICS", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_BATCHED_JOINT = os.environ.get("SLRV_FIT_BATCHED_JOINT", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
REQUIRE_GPU = os.environ.get("SLRV_REQUIRE_GPU", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def configure_fit_parameter_limits():
    if FIT_E_MIN is None and FIT_E_MAX is None:
        logger.debug("Using upstream fit eccentricity limits: %s", DEFAULT_LIMITS["e"])
        return

    e_min = DEFAULT_LIMITS["e"][0] if FIT_E_MIN is None else float(FIT_E_MIN)
    e_max = DEFAULT_LIMITS["e"][1] if FIT_E_MAX is None else float(FIT_E_MAX)
    DEFAULT_LIMITS["e"] = (e_min, e_max)
    logger.debug("Configured fit eccentricity limits to [%s, %s]", e_min, e_max)


configure_fit_parameter_limits()

BLACKJAX_SUPPORTS_RETURN_SAMPLER = (
    "return_sampler"
    in inspect.signature(self_lensing_system_from_fit_blackjax).parameters
)
BLACKJAX_SUPPORTS_RETURN_WALKER_STATE = (
    "return_walker_state"
    in inspect.signature(self_lensing_system_from_fit_blackjax).parameters
)
BLACKJAX_SUPPORTS_BATCHED_FIT = self_lensing_systems_from_batched_fit_blackjax is not None
BLACKJAX_UNSUPPORTED_KWARGS_WARNED = set()


def blackjax_sampler_kwargs(return_sampler, return_walker_state=True):
    kwargs = {}
    if BLACKJAX_SUPPORTS_RETURN_SAMPLER:
        kwargs["return_sampler"] = return_sampler
    elif (
        not return_sampler
        and "return_sampler" not in BLACKJAX_UNSUPPORTED_KWARGS_WARNED
    ):
        BLACKJAX_UNSUPPORTED_KWARGS_WARNED.add("return_sampler")
        logger.warning(
            "Installed selflensing BlackJAX fitter does not support return_sampler=False; "
            "full chains will be copied back after each fit"
        )
    if BLACKJAX_SUPPORTS_RETURN_WALKER_STATE:
        kwargs["return_walker_state"] = return_walker_state
    elif (
        not return_walker_state
        and "return_walker_state" not in BLACKJAX_UNSUPPORTED_KWARGS_WARNED
    ):
        BLACKJAX_UNSUPPORTED_KWARGS_WARNED.add("return_walker_state")
        logger.warning(
            "Installed selflensing BlackJAX fitter does not support "
            "return_walker_state=False; final walker coordinates will be copied "
            "back after each fit"
        )
    return kwargs


def process_map(fn, iterable, total=None, desc=None, max_workers=None, chunksize=1):
    """Run a tqdm-wrapped process map with an explicit multiprocessing context."""
    items = list(iterable)
    if total is None:
        total = len(items)
    if not items:
        return []
    if max_workers == 1:
        logger.info(
            "Starting serial map desc=%s items=%d because max_workers=1",
            desc,
            len(items),
        )
        return [fn(item) for item in tqdm(items, total=total, desc=desc)]

    context = multiprocessing.get_context(MP_START_METHOD)
    logger.info(
        "Starting process map desc=%s items=%d max_workers=%s chunksize=%d mp_start_method=%s",
        desc,
        len(items),
        max_workers,
        chunksize,
        MP_START_METHOD,
    )
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        results = list(
            tqdm(
                executor.map(fn, items, chunksize=chunksize),
                total=total,
                desc=desc,
            )
        )
    logger.info("Finished process map desc=%s items=%d", desc, len(items))
    return results


def return_sampler_from_fit():
    return FIT_MCMC_DIAGNOSTICS and FIT_MAX_WORKERS == 1 and PLOT_MAX_WORKERS == 1


def plot_fit_in_worker():
    return not return_sampler_from_fit()


def load_fitted_timeseries():
    if not TIMESERIES_TABLE_PATH.exists():
        logger.info(
            "Fitted timeseries table does not exist at %s", TIMESERIES_TABLE_PATH
        )
        return pandas.DataFrame()
    table = pandas.read_pickle(TIMESERIES_TABLE_PATH)
    logger.info(
        "Loaded %d fitted timeseries rows from %s", len(table), TIMESERIES_TABLE_PATH
    )
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


def load_batch_realistic_ids(path):
    batch = pandas.read_csv(path)
    if "realistic_id" not in batch.columns:
        raise ValueError(f"Batch file {path} must contain a realistic_id column")
    realistic_ids = set(batch["realistic_id"].dropna().astype(str))
    logger.debug("Loaded %d realistic IDs from batch file %s", len(realistic_ids), path)
    return realistic_ids


def filter_to_batch(table, path):
    realistic_ids = load_batch_realistic_ids(path)
    if not realistic_ids:
        return table.iloc[0:0].copy()
    return table[table["realistic_id"].astype(str).isin(realistic_ids)].copy()


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
    logger.info("Saving %d fit result rows to %s", len(results), TIMESERIES_TABLE_PATH)
    table = pandas.DataFrame(results).reset_index(drop=True)
    if "realistic_id" in table.columns or "realistic_idx" in table.columns:
        table["_fit_key"] = table.apply(result_checkpoint_key, axis=1)
        table.drop_duplicates(subset="_fit_key", keep="last", inplace=True)
        table.drop(columns="_fit_key", inplace=True)
    table.reset_index(drop=True).to_pickle(f"{TIMESERIES_TABLE_PATH}.new.gz")
    os.replace(f"{TIMESERIES_TABLE_PATH}.new.gz", TIMESERIES_TABLE_PATH)
    logger.info("Saved fit results atomically to %s", TIMESERIES_TABLE_PATH)


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


def lightcurve_fit_window(curve, ts):
    peak_time = float(ts["peak_time"])
    tau_eff = float(ts["tau_eff"])
    if not numpy.isfinite(tau_eff) or tau_eff <= 0:
        raise ValueError(
            f"Cannot clip light curve for realistic_id={ts.get('realistic_id', None)} "
            f"because tau_eff={tau_eff!r}"
        )

    lc_times = numpy.asarray(curve["lc_times"], dtype=float)
    logger.debug(
        "Cutting light curve fit window for realistic_id=%s: total photometric points=%d",
        ts.get("realistic_id", None),
        len(lc_times),
    )
    mask = (lc_times >= peak_time - tau_eff) & (lc_times <= peak_time + tau_eff)
    lc = numpy.asarray(curve["lc"], dtype=float)
    lc_err = numpy.asarray(curve["lc_err"], dtype=float)

    # If lc_err is all NaNs, calculate from data
    if numpy.all(~numpy.isfinite(lc_err)):
        logger.debug(
            "lc_err is all NaNs for realistic_id=%s, calculating from data",
            ts.get("realistic_id", None),
        )
        lc_err = numpy.full_like(lc, numpy.std(lc[numpy.isfinite(lc)]))
        logger.debug(
            "Calculated lc_err for realistic_id=%s: %s",
            ts.get("realistic_id", None),
            lc_err,
        )

    mask &= numpy.isfinite(lc_times) & numpy.isfinite(lc) & numpy.isfinite(lc_err)
    mask &= lc_err > 0

    logger.debug(
        "Mask for realistic_id=%s: %s (True count=%d)",
        ts.get("realistic_id", None),
        mask,
        numpy.sum(mask),
    )
    logger.debug(
        "lc_err for realistic_id=%s: %s",
        ts.get("realistic_id", None),
        lc_err,
    )
    if not numpy.any(mask):
        raise ValueError(
            f"Light-curve fit window is empty for realistic_id={ts.get('realistic_id', None)} "
            f"peak_time={peak_time!r} tau_eff={tau_eff!r} lc_times range={lc_times[0]} to {lc_times[-1]}"
        )

    lc_times = lc_times[mask]
    lc = lc[mask]
    lc_err = lc_err[mask]
    logger.debug(
        "After quality filtering for realistic_id=%s: %d photometric points remain",
        ts.get("realistic_id", None),
        len(lc_times),
    )
    logger.debug(
        "Masked lc_err for realistic_id=%s: %s",
        ts.get("realistic_id", None),
        lc_err,
    )
    bin_edges = numpy.linspace(
        peak_time - tau_eff, peak_time + tau_eff, FIT_LC_BINS + 1
    )
    bin_indices = numpy.searchsorted(bin_edges, lc_times, side="right") - 1
    bin_indices = numpy.clip(bin_indices, 0, FIT_LC_BINS - 1)

    binned_lc_times = []
    binned_lc = []
    binned_lc_err = []
    for bin_index in range(FIT_LC_BINS):
        bin_mask = bin_indices == bin_index
        if not numpy.any(bin_mask):
            continue

        weights = 1.0 / numpy.square(lc_err[bin_mask])
        weight_sum = numpy.sum(weights)
        binned_lc_times.append(numpy.average(lc_times[bin_mask], weights=weights))
        binned_lc.append(numpy.average(lc[bin_mask], weights=weights))
        binned_lc_err.append(numpy.sqrt(1.0 / weight_sum))

    return {
        "lc_times": numpy.asarray(binned_lc_times, dtype=float),
        "lc": numpy.asarray(binned_lc, dtype=float),
        "lc_err": numpy.asarray(binned_lc_err, dtype=float),
    }


def unpack_fit_result(result):
    if len(result) == 2:
        sls, sampler = result
        return sls, sampler, None
    sls, sampler, walker_state = result
    return sls, sampler, walker_state


def walker_state_coords(walker_state):
    if walker_state is None:
        return None
    return numpy.asarray(getattr(walker_state, "coords", walker_state), dtype=float)


def short_id(value, length=8):
    if value is None or pandas.isna(value):
        return "unknown"
    return str(value)[:length]


def fitted_timeseries_row(i, ts, truth_sls, sls):
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
    return new_ts


def fit_photometry_prefit(i, ts, curve):
    fit_lc = lightcurve_fit_window(curve, ts)
    mcmc_desc = f"MCMC prefit {short_id(ts.get('realistic_id', None))}"
    logger.info(
        "Starting photometry-only prefit for realistic_id=%s with %d binned LC samples",
        ts.get("realistic_id", None),
        len(fit_lc["lc"]),
    )
    sls, _, walker_state = unpack_fit_result(
        self_lensing_system_from_fit_blackjax(
            lc_times=Time(fit_lc["lc_times"], format="jd"),
            observed_lc=fit_lc["lc"] * units.dimensionless_unscaled,
            observed_lc_err=fit_lc["lc_err"] * units.dimensionless_unscaled,
            peak_time=Time(ts["peak_time"], format="jd"),
            Mcomp=ts["Mcomp"] * units.M_sun,
            Rcomp=ts["Rcomp"] * units.R_sun,
            Teff=ts["Teff"] * units.K,
            false_alarm_probability=0.5,
            nsteps=FIT_NSTEPS,
            nwalkers=FIT_NWALKERS,
            burn=FIT_BURN,
            progress=FIT_MCMC_PROGRESS,
            progress_desc=mcmc_desc,
            progress_chunk_size=FIT_PROGRESS_CHUNK_SIZE,
            lensing_method=FIT_LENSING_METHOD,
            center_lc_model=True,
            device=FIT_DEVICE,
            prefer_gpu=FIT_DEVICE == "auto",
            **blackjax_sampler_kwargs(return_sampler=False),
        )
    )
    logger.info(
        "Finished photometry-only prefit for realistic_id=%s walker_state=%s",
        ts.get("realistic_id", None),
        walker_state is not None,
    )
    prefit_ts = fitted_timeseries_row(i, ts, truth_self_lensing_system(ts), sls)
    return prefit_ts, walker_state_coords(walker_state)


def fit(a):
    fit_result = run_fit(a)
    return materialize_fit_result(fit_result)


def run_fit(a):
    if len(a) == 5:
        i, ts, curve, initial_walker_state, mcmc_desc = a
    elif len(a) == 4:
        i, ts, curve, initial_walker_state = a
        mcmc_desc = f"MCMC joint {short_id(ts.get('realistic_id', None))}"
    elif len(a) == 3:
        i, ts, curve = a
        initial_walker_state = None
        mcmc_desc = f"MCMC joint {short_id(ts.get('realistic_id', None))}"
    else:
        i, ts = a
        curve = load_realistic_timeseries_curve(ts)
        initial_walker_state = None
        mcmc_desc = f"MCMC joint {short_id(ts.get('realistic_id', None))}"
    logger.info(
        "Starting fit for row index=%s realistic_id=%s path=%s",
        i,
        ts.get("realistic_id", None),
        ts.get("realistic_path", None),
    )
    truth_sls = truth_self_lensing_system(ts)
    fit_lc = lightcurve_fit_window(curve, ts) if FIT_USE_LC else None
    logger.info(
        "Fit inputs row index=%s: %d RV samples, %d LC samples%s",
        i,
        len(curve["rv"]),
        len(curve["lc"]),
        f" binned to {len(fit_lc['lc'])}" if fit_lc is not None else "",
    )
    if initial_walker_state is not None:
        initial_walker_state = numpy.array(initial_walker_state, copy=True)
    logger.info(
        "Running MCMC fit for row index=%s realistic_id=%s",
        i,
        ts.get("realistic_id", None),
    )
    fit_start = time.perf_counter()
    sls, sampler, _ = unpack_fit_result(
        self_lensing_system_from_fit_blackjax(
            rv_times=Time(curve["rv_times"], format="jd"),
            observed_rv=curve["rv"] * units.m / units.s,
            observed_rv_err=curve["rv_err"] * units.m / units.s,
            lc_times=Time(fit_lc["lc_times"], format="jd") if FIT_USE_LC else None,
            observed_lc=(
                fit_lc["lc"] * units.dimensionless_unscaled if FIT_USE_LC else None
            ),
            observed_lc_err=(
                fit_lc["lc_err"] * units.dimensionless_unscaled if FIT_USE_LC else None
            ),
            peak_time=Time(ts["peak_time"], format="jd"),
            Mcomp=ts["Mcomp"] * units.M_sun,
            Rcomp=ts["Rcomp"] * units.R_sun,
            Teff=ts["Teff"] * units.K,
            false_alarm_probability=0.5,
            nsteps=FIT_NSTEPS,
            nwalkers=FIT_NWALKERS,
            burn=FIT_BURN,
            progress=FIT_MCMC_PROGRESS,
            progress_desc=mcmc_desc,
            progress_chunk_size=FIT_PROGRESS_CHUNK_SIZE,
            lensing_method=FIT_LENSING_METHOD,
            center_lc_model=True,
            device=FIT_DEVICE,
            prefer_gpu=FIT_DEVICE == "auto",
            initial_walker_state=initial_walker_state,
            **blackjax_sampler_kwargs(
                return_sampler=return_sampler_from_fit(),
                return_walker_state=False,
            ),
        )
    )
    logger.info(
        "MCMC fit returned for row index=%s realistic_id=%s in %.2f s",
        i,
        ts.get("realistic_id", None),
        time.perf_counter() - fit_start,
    )
    return i, ts, truth_sls, sls, sampler, curve


def materialize_fit_result(fit_result):
    i, ts, truth_sls, sls, sampler, curve = fit_result
    materialize_start = time.perf_counter()
    new_ts = fitted_timeseries_row(i, ts, truth_sls, sls)
    logger.info(
        "Materialized fit result for row index=%s in %.2f s: "
        "fitted_a=%.6g fitted_Macc=%.6g fitted_e=%.6g",
        i,
        time.perf_counter() - materialize_start,
        new_ts["fitted_a"],
        new_ts["fitted_Macc"],
        new_ts["fitted_e"],
    )
    return new_ts, sampler, curve


def finish_pending_fit(pending_fit):
    if pending_fit is None:
        return None
    return pending_fit.result()


def can_use_batched_joint_fit():
    return (
        FIT_BATCHED_JOINT
        and BLACKJAX_SUPPORTS_BATCHED_FIT
        and FIT_USE_LC
        and not return_sampler_from_fit()
    )


def padded_series(series, fill_value=0.0, fill_from_first=False):
    max_len = max(len(values) for values in series)
    data = numpy.full((len(series), max_len), fill_value, dtype=float)
    mask = numpy.zeros((len(series), max_len), dtype=bool)
    for index, values in enumerate(series):
        values = numpy.asarray(values, dtype=float)
        if fill_from_first and len(values) > 0:
            data[index, :] = values[0]
        data[index, : len(values)] = values
        mask[index, : len(values)] = True
    return data, mask


def batched_joint_fit_inputs(rows, curves):
    rv_times = []
    rv = []
    rv_err = []
    lc_times = []
    lc = []
    lc_err = []
    for _, ts in rows:
        curve = curves[str(ts["realistic_id"])]
        rv_times.append(curve["rv_times"])
        rv.append(curve["rv"])
        rv_err.append(curve["rv_err"])
        if FIT_USE_LC:
            fit_lc = lightcurve_fit_window(curve, ts)
            lc_times.append(fit_lc["lc_times"])
            lc.append(fit_lc["lc"])
            lc_err.append(fit_lc["lc_err"])

    rv_times, rv_mask = padded_series(rv_times, fill_from_first=True)
    rv, _ = padded_series(rv)
    rv_err, _ = padded_series(rv_err, fill_value=1.0)
    inputs = {
        "rv_times": rv_times,
        "observed_rv": rv,
        "observed_rv_err": rv_err,
        "rv_mask": rv_mask,
    }
    if FIT_USE_LC:
        lc_times, lc_mask = padded_series(lc_times, fill_from_first=True)
        lc, _ = padded_series(lc)
        lc_err, _ = padded_series(lc_err, fill_value=1.0)
        inputs.update(
            {
                "lc_times": lc_times,
                "observed_lc": lc,
                "observed_lc_err": lc_err,
                "lc_mask": lc_mask,
            }
        )
    return inputs


def fit_joint_batch(rows, curves, initial_walker_state):
    start = time.perf_counter()
    inputs = batched_joint_fit_inputs(rows, curves)
    row_table = pandas.DataFrame([ts for _, ts in rows])
    logger.info(
        "Running batched joint MCMC for %d rows: max_rv=%d%s",
        len(rows),
        inputs["rv_times"].shape[1],
        (
            f" max_lc={inputs['lc_times'].shape[1]}"
            if FIT_USE_LC and "lc_times" in inputs
            else ""
        ),
    )
    systems, sampler, _ = self_lensing_systems_from_batched_fit_blackjax(
        rv_times=Time(inputs["rv_times"], format="jd"),
        observed_rv=inputs["observed_rv"] * units.m / units.s,
        observed_rv_err=inputs["observed_rv_err"] * units.m / units.s,
        rv_mask=inputs["rv_mask"],
        lc_times=(
            Time(inputs["lc_times"], format="jd")
            if FIT_USE_LC and "lc_times" in inputs
            else None
        ),
        observed_lc=(
            inputs["observed_lc"] * units.dimensionless_unscaled
            if FIT_USE_LC and "observed_lc" in inputs
            else None
        ),
        observed_lc_err=(
            inputs["observed_lc_err"] * units.dimensionless_unscaled
            if FIT_USE_LC and "observed_lc_err" in inputs
            else None
        ),
        lc_mask=inputs.get("lc_mask"),
        peak_time=Time(row_table["peak_time"].to_numpy(dtype=float), format="jd"),
        Mcomp=row_table["Mcomp"].to_numpy(dtype=float) * units.M_sun,
        Rcomp=row_table["Rcomp"].to_numpy(dtype=float) * units.R_sun,
        Teff=row_table["Teff"].to_numpy(dtype=float) * units.K,
        false_alarm_probability=0.5,
        nsteps=FIT_NSTEPS,
        nwalkers=FIT_NWALKERS,
        burn=FIT_BURN,
        progress=FIT_MCMC_PROGRESS,
        progress_desc=f"MCMC batched joint {short_id(rows[0][1].get('pristine_id', None))}",
        progress_chunk_size=FIT_PROGRESS_CHUNK_SIZE,
        lensing_method=FIT_LENSING_METHOD,
        center_lc_model=True,
        device=FIT_DEVICE,
        prefer_gpu=FIT_DEVICE == "auto",
        initial_walker_state=initial_walker_state,
        return_sampler=False,
        return_walker_state=False,
    )
    logger.info(
        "Batched joint MCMC returned %d systems in %.2f s",
        len(systems),
        time.perf_counter() - start,
    )
    if sampler is not None:
        logger.warning("Ignoring unexpected batched sampler result")
    results = []
    materialize_start = time.perf_counter()
    for (i, ts), sls in zip(rows, systems):
        curve = curves[str(ts["realistic_id"])]
        new_ts = fitted_timeseries_row(i, ts, truth_self_lensing_system(ts), sls)
        results.append((new_ts, None, curve))
    logger.info(
        "Materialized %d batched fit results in %.2f s",
        len(results),
        time.perf_counter() - materialize_start,
    )
    return results


def fit_realistic_file(rows):
    rows = list(rows)
    if not rows:
        return []

    path = rows[0][1]["realistic_path"]
    logger.info("Fitting %d realistic rows from %s", len(rows), path)
    logger.info("Loading grouped realistic curves from %s", path)
    curves = load_realistic_timeseries_curves(rows)
    logger.info("Loaded %d grouped realistic curves from %s", len(curves), path)
    initial_walker_state = None
    prefit_result = None
    if FIT_USE_LC:
        prefit_i, prefit_ts = rows[0]
        prefit_curve = curves[str(prefit_ts["realistic_id"])]
        logger.info(
            "Prefitting photometry once for %d-row realistic group from %s using row index=%s",
            len(rows),
            path,
            prefit_i,
        )
        prefit_ts, initial_walker_state = fit_photometry_prefit(
            prefit_i, prefit_ts, prefit_curve
        )
        prefit_result = (prefit_ts, None, prefit_curve)
    results = []
    if can_use_batched_joint_fit():
        results = fit_joint_batch(rows, curves, initial_walker_state)
    else:
        if FIT_BATCHED_JOINT and not BLACKJAX_SUPPORTS_BATCHED_FIT:
            logger.warning(
                "Installed selflensing does not support batched BlackJAX fits; "
                "falling back to per-row joint fits"
            )
        elif FIT_BATCHED_JOINT and return_sampler_from_fit():
            logger.info(
                "Falling back to per-row joint fits because sampler diagnostics "
                "require per-fit sampler results"
            )
        elif FIT_BATCHED_JOINT and not FIT_USE_LC:
            logger.info(
                "Falling back to per-row joint fits because batched joint fits "
                "currently require FIT_USE_LC=True"
            )
        file_progress_desc = (
            f"RV joint fits {short_id(rows[0][1].get('pristine_id', Path(path).stem))}"
        )
        row_iterator = tqdm(
            rows,
            total=len(rows),
            desc=file_progress_desc,
            leave=False,
            dynamic_ncols=True,
        )
        pending_fit = None
        with ThreadPoolExecutor(max_workers=1) as row_executor:
            for row_number, (i, ts) in enumerate(row_iterator, start=1):
                logger.info(
                    "Starting joint fit %d/%d for realistic_id=%s",
                    row_number,
                    len(rows),
                    ts.get("realistic_id", None),
                )
                curve = curves[str(ts["realistic_id"])]
                mcmc_desc = (
                    f"MCMC joint {row_number}/{len(rows)} "
                    f"{short_id(ts.get('realistic_id', None))}"
                )
                fit_result = run_fit((i, ts, curve, initial_walker_state, mcmc_desc))
                finished_fit = finish_pending_fit(pending_fit)
                if finished_fit is not None:
                    results.append(finished_fit)
                pending_fit = row_executor.submit(materialize_fit_result, fit_result)
            finished_fit = finish_pending_fit(pending_fit)
            if finished_fit is not None:
                results.append(finished_fit)
    if SAVE_FIT_PLOTS:
        logger.info("Finished all joint fits for %s; writing diagnostic PDF", path)
        save_plots_pdf(results, prefit_result=prefit_result)
    else:
        logger.info("Finished all joint fits for %s; skipping diagnostic PDF", path)
    logger.info("Finished grouped fit for %s", path)
    if return_sampler_from_fit():
        return results
    return [(ts, None, None) for ts, _, _ in results]


def grouped_realistic_rows(table):
    groups = []
    for _, group in table.groupby("realistic_path", sort=True):
        groups.append(list(group.iterrows()))
    logger.debug(
        "Grouped %d realistic rows into %d FITS-file jobs", len(table), len(groups)
    )
    return groups


def sample_realistic_path_groups(table, sample_size):
    if table.empty:
        return table.copy()

    paths = pandas.Series(table["realistic_path"].drop_duplicates())
    group_sample_size = min(int(sample_size), len(paths))
    logger.info(
        "Sampling %d/%d realistic FITS-file groups for standalone fitting",
        group_sample_size,
        len(paths),
    )
    sampled_paths = set(paths.sample(group_sample_size))
    sampled = table[table["realistic_path"].isin(sampled_paths)].copy()
    logger.info(
        "Sampled %d realistic rows across %d FITS-file groups for fitting",
        len(sampled),
        group_sample_size,
    )
    return sampled


FIG_PATH = DATA_PATH / "figures"
FIG_PATH.mkdir(exist_ok=True, parents=True)


def clear_existing_results():
    paths = [
        TIMESERIES_TABLE_PATH,
        Path(f"{TIMESERIES_TABLE_PATH}.new.gz"),
    ]
    for path in paths:
        if path.exists():
            logger.info("Removing saved fit results at %s", path)
            path.unlink()

    if FIG_PATH.exists():
        logger.info("Removing generated fit figures at %s", FIG_PATH)
        shutil.rmtree(FIG_PATH)
    FIG_PATH.mkdir(exist_ok=True, parents=True)


def plot_label(ts):
    return (
        f"realistic_id={ts.get('realistic_id')} | "
        f"baseline={float(ts['baseline']):g} d | "
        f"cadence={float(ts['cadence']):g} d | "
        f"RV accuracy={float(ts['RV accuracy']):g} m/s"
    )


def add_pdf_page_title(fig, ts, label):
    fig.suptitle(f"{label}\n{plot_label(ts)}", fontsize=10)
    fig.subplots_adjust(top=0.9)


def figure_shard_keys(ts):
    return (
        f"a1e{int(numpy.floor(numpy.log10(ts['a'])))}",
        f"b{int(ts['b'])}",
        f"Macc{int(ts['Macc'])}",
    )


def pdf_path_for_results(results):
    ts = results[0][0]
    pristine_id = str(ts.get("pristine_id", "unknown"))
    a_key, b_key, macc_key = figure_shard_keys(ts)
    name = f"{a_key}_{b_key}_{macc_key}_{pristine_id}.pdf"
    return FIG_PATH / a_key / b_key / macc_key / name


def plot_ts_with_curve(ts, curve):
    plot_ts = pandas.Series(ts)
    plot_ts["rv_times"] = curve["rv_times"]
    plot_ts["rv"] = curve["rv"]
    plot_ts["rv_err"] = curve["rv_err"]
    plot_ts["lc_times"] = curve["lc_times"]
    plot_ts["lc"] = curve["lc"]
    plot_ts["lc_err"] = curve["lc_err"]
    if "b_max" not in plot_ts or pandas.isna(plot_ts["b_max"]):
        plot_ts["b_max"] = truth_self_lensing_system(plot_ts).b_max.value
    return plot_ts


def save_plots_pdf(results, prefit_result=None):
    if not results:
        return

    out_path = pdf_path_for_results(results)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    logger.info("Writing %d fit diagnostic plot sets to %s", len(results), out_path)
    params = ["a", "i", "Macc", "e", "periapsis_phase"]
    with PdfPages(out_path) as pdf:
        if prefit_result is not None:
            ts, _, curve = prefit_result
            if curve is None:
                curve = load_realistic_timeseries_curve(ts)
            logger.info(
                "Writing photometry prefit page for realistic_id=%s to %s",
                ts.get("realistic_id", None),
                out_path,
            )
            prefit_fig = plot_fit(plot_ts_with_curve(ts, curve))
            add_pdf_page_title(prefit_fig, ts, "Photometry Prefit")
            pdf.savefig(prefit_fig)
            pyplot.close(prefit_fig)

        for ts, sampler, curve in tqdm(results, desc="Plotting fits", unit="fit"):
            if curve is None:
                curve = load_realistic_timeseries_curve(ts)

            plot_ts = plot_ts_with_curve(ts, curve)
            fit_fig = plot_fit(plot_ts)
            add_pdf_page_title(fit_fig, ts, "Fit")
            pdf.savefig(fit_fig)
            pyplot.close(fit_fig)

            if sampler is None:
                logger.debug(
                    "Skipping MCMC diagnostic pages for realistic_id=%s because sampler is unavailable",
                    ts.get("realistic_id", None),
                )
                continue

            walker_fig, corner_fig = plot_emcee_diagnostics(
                sampler,
                param_names=params,
                truths=ts[params],
                axes_scale=["log", "linear", "linear", "linear", "linear"],
            )
            add_pdf_page_title(walker_fig, ts, "Walkers")
            add_pdf_page_title(corner_fig, ts, "Corner")
            pdf.savefig(walker_fig)
            pdf.savefig(corner_fig)
            pyplot.close(walker_fig)
            pyplot.close(corner_fig)
    logger.info("Saved fit diagnostic PDF to %s", out_path)


def save_plots(a):
    if len(a) == 3:
        ts, sampler, curve = a
        if curve is None:
            curve = load_realistic_timeseries_curve(ts)
    else:
        ts, sampler = a
        curve = load_realistic_timeseries_curve(ts)
    logger.debug(
        "Saving plots for realistic_idx=%s realistic_id=%s",
        ts.get("realistic_idx", None),
        ts.get("realistic_id", None),
    )
    save_plots_pdf([(ts, sampler, curve)])


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    logger.info("Starting realistic timeseries fitting script")
    if args.clear_results:
        clear_existing_results()
    logger.info(
        "Starting fitting script with FIT_MAX_WORKERS=%d FIT_SAMPLE_SIZE=%d "
        "PLOT_MAX_WORKERS=%d SAVE_FIT_PLOTS=%s",
        FIT_MAX_WORKERS,
        FIT_SAMPLE_SIZE,
        PLOT_MAX_WORKERS,
        SAVE_FIT_PLOTS,
    )
    logger.info("Fit batch file: %s", FIT_BATCH_FILE)
    logger.info(
        "MCMC settings: FIT_NSTEPS=%d FIT_NWALKERS=%d FIT_BURN=%d "
        "FIT_PROGRESS_CHUNK_SIZE=%d MP_START_METHOD=%s",
        FIT_NSTEPS,
        FIT_NWALKERS,
        FIT_BURN,
        FIT_PROGRESS_CHUNK_SIZE,
        MP_START_METHOD,
    )
    logger.info("Fit eccentricity limits: %s", DEFAULT_LIMITS["e"])
    logger.info(
        "Fit likelihood settings: FIT_USE_LC=%s "
        "FIT_MCMC_PROGRESS=%s FIT_MCMC_DIAGNOSTICS=%s sampler=blackjax",
        FIT_USE_LC,
        FIT_MCMC_PROGRESS,
        FIT_MCMC_DIAGNOSTICS,
    )
    logger.info("Fit device setting: FIT_DEVICE=%s", FIT_DEVICE)
    logger.info("Fit lensing method: FIT_LENSING_METHOD=%s", FIT_LENSING_METHOD)
    logger.info(
        "Batched joint fit setting: FIT_BATCHED_JOINT=%s supported=%s",
        FIT_BATCHED_JOINT,
        BLACKJAX_SUPPORTS_BATCHED_FIT,
    )
    logger.info(
        "MCMC diagnostic plot settings: return_sampler=%s plot_fit_in_worker=%s",
        return_sampler_from_fit(),
        plot_fit_in_worker(),
    )
    logger.info(
        "JAX backend: default=%s devices=[%s]",
        jax.default_backend(),
        jax_device_summary(),
    )
    if REQUIRE_GPU and not any(device.platform == "gpu" for device in JAX_DEVICES):
        raise RuntimeError(
            "SLRV_REQUIRE_GPU is set, but JAX does not see a GPU. "
            f"Available devices: [{jax_device_summary()}]"
        )
    logger.info("Loading realistic metadata")
    rts = load_realistic_timeseries()
    logger.info("Loaded %d realistic metadata rows", len(rts))
    if FIT_BATCH_FILE:
        # SLURM array tasks pass an explicit manifest of realistic_id values.
        # In that mode we must fit exactly those rows; FIT_SAMPLE_SIZE is only
        # for standalone exploratory runs.
        logger.info("Filtering realistic rows to batch manifest %s", FIT_BATCH_FILE)
        rts = filter_to_batch(rts, FIT_BATCH_FILE)
        logger.info("Filtered realistic rows down to %d batch rows", len(rts))

    logger.info("Loading existing fitted results from %s", TIMESERIES_TABLE_PATH)
    existing_results = load_fitted_timeseries()
    fitted_ids, fitted_indices = fitted_key_sets(existing_results)
    logger.info(
        "Loaded %d realistic rows and %d existing fitted rows",
        len(rts),
        len(existing_results),
    )

    if fitted_ids or fitted_indices:
        logger.info(
            "Filtering already fitted rows using %d ids and %d legacy indices",
            len(fitted_ids),
            len(fitted_indices),
        )
        rts = rts[
            ~rts.apply(
                lambda row: is_already_fitted(row, fitted_ids, fitted_indices),
                axis=1,
            )
        ].copy()
        logger.info("Filtered realistic rows down to %d unfitted rows", len(rts))

    if len(rts) == 0:
        logger.info("No realistic rows left to fit; exiting")
        raise SystemExit

    if FIT_BATCH_FILE:
        logger.info("Using all %d manifest rows for fitting", len(rts))
    else:
        rts = sample_realistic_path_groups(rts, FIT_SAMPLE_SIZE)

    jobs = grouped_realistic_rows(rts)
    logger.info(
        "Grouped %d rows into %d FITS-file jobs",
        len(rts),
        len(jobs),
    )

    results = (
        [row for _, row in existing_results.iterrows()]
        if not existing_results.empty
        else []
    )

    grouped_new_results = process_map(
        fit_realistic_file,
        jobs,
        total=len(jobs),
        desc="Realistic simulation fits",
        max_workers=FIT_MAX_WORKERS,
    )
    new_results = [result for group in grouped_new_results for result in group]
    results += [r[0] for r in new_results]
    logger.info(
        "Fit jobs produced %d fit results; cumulative results=%d",
        len(new_results),
        len(results),
    )
    save_results(results)
    if SAVE_FIT_PLOTS:
        logger.debug("Skipping parent-process plot saving; PDFs were saved per FITS job")
    else:
        logger.debug("Skipping parent-process plot saving; fit plots are disabled")
    logger.info("Finished fitting script")
