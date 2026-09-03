"""Fit RV-only simulations on an independent evenly spaced schedule."""

import argparse
import hashlib
import logging
import multiprocessing
import os
import re
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
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
if os.environ.get("SLRV_FIT_CPU_SIDECAR") == "1":
    # Checkpoint workers only merge and serialize pandas tables.  Keep them
    # from creating a CUDA context alongside the MCMC parent process.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif "SLRV_JAX_PLATFORMS" in os.environ:
    os.environ.setdefault("JAX_PLATFORMS", os.environ["SLRV_JAX_PLATFORMS"])

DEFAULT_JAX_COMPILATION_CACHE_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "jax_compilation_cache"
)
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR",
    os.environ.get(
        "SLRV_JAX_COMPILATION_CACHE_DIR",
        str(DEFAULT_JAX_COMPILATION_CACHE_DIR),
    ),
)
os.environ.setdefault(
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
    os.environ.get("SLRV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1"),
)
os.environ.setdefault(
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    os.environ.get("SLRV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0"),
)
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_MAX_SIZE",
    # Disable JAX's LRU eviction for the shared cache.  Its companion atime
    # files can otherwise be removed during concurrent cache access.
    os.environ.get("SLRV_JAX_COMPILATION_CACHE_MAX_SIZE", "-1"),
)
os.environ.setdefault(
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES",
    os.environ.get("SLRV_JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "all"),
)
os.environ.setdefault(
    "JAX_EXPLAIN_CACHE_MISSES",
    os.environ.get("SLRV_JAX_EXPLAIN_CACHE_MISSES", "false"),
)
Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

FIT_RESULTS_LOCK = None
if __name__ == "__main__":
    from fitting import acquire_fit_results_lock

    from fitting import RV_EVEN_SPACED_RESULTS_PATH
    FIT_RESULTS_LOCK = acquire_fit_results_lock(
        "RV evenly spaced fitting script", RV_EVEN_SPACED_RESULTS_PATH
    )

import jax

jax.config.update(
    "jax_compilation_cache_dir",
    os.environ["JAX_COMPILATION_CACHE_DIR"],
)
jax.config.update(
    "jax_persistent_cache_min_compile_time_secs",
    float(os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"]),
)
jax.config.update(
    "jax_persistent_cache_min_entry_size_bytes",
    int(os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"]),
)
jax.config.update(
    "jax_compilation_cache_max_size",
    int(os.environ["JAX_COMPILATION_CACHE_MAX_SIZE"]),
)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"],
)
jax.config.update(
    "jax_explain_cache_misses",
    os.environ["JAX_EXPLAIN_CACHE_MISSES"].lower() in ("1", "true", "yes", "on"),
)
import numpy
import pandas
from astropy import units
from astropy.time import Time
from tqdm import tqdm

from jax_compat import install_quax_jax_stage_patch

install_quax_jax_stage_patch()

from fitting import PREFIT_RESULTS_PATH, RV_EVEN_SPACED_RESULTS_PATH, SAMPLER_STATE_PATH
from rv_timeseries import load_rv_timeseries
from rv_only_no_prior import (
    self_lensing_system_from_rv_fit_no_peak_prior_blackjax,
    self_lensing_systems_from_batched_rv_fit_no_peak_prior_blackjax,
)
from schedule import evenly_spaced_rv_observations
from selflensing.models import DEFAULT_LIMITS
from selflensing.lightcurves import radial_velocity_m_per_s
from spectrograph import Spectrograph
from utils import configure_logging

logger = logging.getLogger(__name__)

RV_ONLY_PREFIX = "rvnoprior"
FIT_RESULT_COLUMNS = (
    "fitted_a",
    "fitted_b",
    "fitted_b_max",
    "fitted_i",
    "fitted_Macc",
    "fitted_e",
    "fitted_periapsis_phase",
    "fitted_porb",
    "fitted_tau_eff",
)
RV_ONLY_RESULT_COLUMNS = tuple(
    f"{RV_ONLY_PREFIX}_{column}" for column in FIT_RESULT_COLUMNS
) + (
    f"{RV_ONLY_PREFIX}_fitted_rv_phase",
    f"{RV_ONLY_PREFIX}_fitted_rv_reference_time",
    f"{RV_ONLY_PREFIX}_fitted_peak_time",
)
RV_ONLY_PARAMETER_NAMES = numpy.asarray(
    ["a", "i", "Macc", "e", "periapsis_phase", "rv_phase"],
    dtype=str,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit evenly spaced RV simulations with a free RV phase."
    )
    parser.add_argument(
        "--clear-results",
        action="store_true",
        help="Drop existing RV-only result columns before fitting.",
    )
    return parser.parse_args()


def optional_positive_int(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    if value in ("", "0", "all", "none", "false", "no"):
        return None
    value = int(value)
    if value < 1:
        return None
    return value


def default_fit_max_workers():
    # Every worker defaults to gpu:0.  More than one worker would therefore
    # contend for the same accelerator rather than add useful parallelism.
    return 1 if any(device.platform == "gpu" for device in jax.devices()) else 4


def default_fit_device():
    return "gpu:0" if any(device.platform == "gpu" for device in jax.devices()) else "auto"


FIT_MAX_WORKERS = int(os.environ.get("SLRV_FIT_MAX_WORKERS", default_fit_max_workers()))
FIT_SAMPLE_SIZE = optional_positive_int(os.environ.get("SLRV_FIT_SAMPLE_SIZE", "all"))
FIT_BATCH_FILE = os.environ.get("SLRV_FIT_BATCH_FILE")
CHECKPOINT_RESULTS = os.environ.get("SLRV_FIT_CHECKPOINT_RESULTS", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_CHECKPOINT_INTERVAL = int(os.environ.get("SLRV_FIT_CHECKPOINT_INTERVAL", 16384))
MP_START_METHOD = os.environ.get("SLRV_MP_START_METHOD", "spawn")
FIT_NSTEPS = int(os.environ.get("SLRV_FIT_NSTEPS", 1000))
FIT_NWALKERS = int(os.environ.get("SLRV_FIT_NWALKERS", 128))
FIT_BURN = int(os.environ.get("SLRV_FIT_BURN", 200))
# This bounds the compiled MCMC scan even when progress output is disabled.
# At cross-file batch sizes, the scan output is [chunk, fit, walker, parameter],
# so a short chunk avoids multi-tens-of-GiB XLA temporaries.
FIT_PROGRESS_CHUNK_SIZE = int(os.environ.get("SLRV_FIT_PROGRESS_CHUNK_SIZE", 10))
FIT_E_MIN = os.environ.get("SLRV_FIT_E_MIN")
FIT_E_MAX = os.environ.get("SLRV_FIT_E_MAX")
FIT_DEVICE = os.environ.get("SLRV_FIT_DEVICE", default_fit_device())
FIT_LENSING_METHOD = os.environ.get("SLRV_FIT_LENSING_METHOD")
FIT_MCMC_PROGRESS = os.environ.get("SLRV_FIT_MCMC_PROGRESS", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_BATCHED = os.environ.get(
    "SLRV_RV_EVEN_SPACED_FIT_BATCHED",
    os.environ.get("SLRV_RV_NO_PRIOR_FIT_BATCHED", "1"),
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# The previous implementation grouped by FITS file, limiting a GPU batch to
# 40 RV variants.  These synthetic even-grid curves need no FITS I/O, so use
# cross-file batches that give the accelerator materially more parallel work.
# 512 systems × 128 walkers is already 65,536 concurrent chains.  Larger
# batches make the compiled MCMC scan memory-bound rather than faster.
FIT_BATCH_SIZE = optional_positive_int(os.environ.get("SLRV_FIT_BATCH_SIZE", 512))
FIT_PROPOSAL_SCALE_FRACTION = float(
    os.environ.get("SLRV_FIT_PROPOSAL_SCALE_FRACTION", 0.005)
)
FIT_SAVE_SAMPLERS = os.environ.get("SLRV_FIT_SAVE_SAMPLERS", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_SAMPLER_WRITE_MAX_WORKERS = int(
    os.environ.get("SLRV_FIT_SAMPLER_WRITE_MAX_WORKERS", 16)
)
FIT_PRELOAD_FILE_POOL_SIZE = int(os.environ.get("SLRV_FIT_PRELOAD_FILE_POOL_SIZE", 16))
FIT_PRELOAD_MAX_WORKERS = int(
    os.environ.get(
        "SLRV_FIT_PRELOAD_MAX_WORKERS",
        max(1, min(FIT_PRELOAD_FILE_POOL_SIZE, 4)),
    )
)
REQUIRE_GPU = os.environ.get("SLRV_REQUIRE_GPU", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def configure_fit_parameter_limits():
    if FIT_E_MIN is None and FIT_E_MAX is None:
        return
    e_min = DEFAULT_LIMITS["e"][0] if FIT_E_MIN is None else float(FIT_E_MIN)
    e_max = DEFAULT_LIMITS["e"][1] if FIT_E_MAX is None else float(FIT_E_MAX)
    DEFAULT_LIMITS["e"] = (e_min, e_max)


configure_fit_parameter_limits()


def jax_device_summary():
    return ", ".join(
        f"{device.platform}:{device.device_kind}" for device in jax.devices()
    )


def safe_path_token(value):
    token = "unknown" if value is None or pandas.isna(value) else str(value)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", token)


def sampler_state_shard_keys(ts):
    a = float(ts["a"]) if "a" in ts and not pandas.isna(ts["a"]) else numpy.nan
    b = float(ts["b"]) if "b" in ts and not pandas.isna(ts["b"]) else numpy.nan
    macc = float(ts["Macc"]) if "Macc" in ts and not pandas.isna(ts["Macc"]) else numpy.nan
    a_key = f"a1e{int(numpy.floor(numpy.log10(a)))}" if a > 0 else "a_unknown"
    b_key = f"b{int(b)}" if numpy.isfinite(b) else "b_unknown"
    macc_key = f"Macc{int(macc)}" if numpy.isfinite(macc) else "Macc_unknown"
    return a_key, b_key, macc_key


def sampler_state_path_for_row(ts):
    if "rv_id" in ts and not pandas.isna(ts["rv_id"]):
        token = safe_path_token(ts["rv_id"])
    elif "rv_index" in ts and not pandas.isna(ts["rv_index"]):
        token = f"idx_{safe_path_token(ts['rv_index'])}"
    else:
        token = f"row_{safe_path_token(getattr(ts, 'name', None))}"
    system_id = safe_path_token(ts.get("system_id", "unknown"))
    a_key, b_key, macc_key = sampler_state_shard_keys(ts)
    return (
        SAMPLER_STATE_PATH
        / "rv_no_prior"
        / a_key
        / b_key
        / macc_key
        / system_id
        / f"{token}.npz"
    )


def sampler_arrays(sampler):
    if sampler is None:
        return None
    return {
        "chain": numpy.asarray(sampler.get_chain(), dtype=float),
        "log_prob": numpy.asarray(sampler.get_log_prob(), dtype=float),
        "acceptance": (
            numpy.asarray(sampler.acceptance, dtype=bool)
            if sampler.acceptance is not None
            else numpy.asarray([], dtype=bool)
        ),
    }


def walker_state_coords(walker_state):
    if walker_state is None:
        return None
    return numpy.asarray(getattr(walker_state, "coords", walker_state), dtype=float)


def sampler_state_payload(ts, sampler, walker_state):
    if not FIT_SAVE_SAMPLERS:
        return None
    if sampler is None:
        logger.warning(
            "Sampler saving is enabled, but no sampler was returned for rv_id=%s",
            ts.get("rv_id", None),
        )
        return None
    arrays = sampler_arrays(sampler)
    final_coords = walker_state_coords(walker_state)
    return {
        "path": sampler_state_path_for_row(ts),
        "chain": arrays["chain"],
        "log_prob": arrays["log_prob"],
        "acceptance": arrays["acceptance"],
        "final_walker_coords": (
            numpy.asarray(final_coords, dtype=float)
            if final_coords is not None
            else numpy.asarray([], dtype=float)
        ),
        "parameter_names": RV_ONLY_PARAMETER_NAMES,
        "burn": numpy.asarray(FIT_BURN, dtype=int),
        "nsteps": numpy.asarray(FIT_NSTEPS, dtype=int),
        "nwalkers": numpy.asarray(FIT_NWALKERS, dtype=int),
        "rv_id": numpy.asarray(str(ts.get("rv_id", ""))),
        "system_id": numpy.asarray(str(ts.get("system_id", ""))),
        "rv_index": numpy.asarray(
            str(ts.get("rv_index", getattr(ts, "name", "")))
        ),
    }


def write_sampler_state_payload(payload):
    path = payload["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy.savez_compressed(
        path,
        chain=payload["chain"],
        log_prob=payload["log_prob"],
        acceptance=payload["acceptance"],
        final_walker_coords=payload["final_walker_coords"],
        parameter_names=payload["parameter_names"],
        burn=payload["burn"],
        nsteps=payload["nsteps"],
        nwalkers=payload["nwalkers"],
        rv_id=payload["rv_id"],
        system_id=payload["system_id"],
        rv_index=payload["rv_index"],
    )
    return path


class SamplerStateWriter:
    def __init__(self):
        self.executor = None
        self.futures = []
        if FIT_SAVE_SAMPLERS:
            self.executor = ThreadPoolExecutor(
                max_workers=max(1, FIT_SAMPLER_WRITE_MAX_WORKERS)
            )

    def submit(self, ts, sampler, walker_state):
        payload = sampler_state_payload(ts, sampler, walker_state)
        if payload is None:
            return None
        path = payload["path"]
        if self.executor is None:
            write_sampler_state_payload(payload)
        else:
            self.futures.append(self.executor.submit(write_sampler_state_payload, payload))
        return path

    def close(self):
        for future in self.futures:
            future.result()
        if self.executor is not None:
            self.executor.shutdown()


ACTIVE_SAMPLER_WRITER = None


def shared_sampler_writer():
    return ACTIVE_SAMPLER_WRITER


def scalar_is_finite(value):
    try:
        array = numpy.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    if array.shape == ():
        return bool(numpy.isfinite(array))
    if array.size == 1:
        return bool(numpy.isfinite(array.reshape(-1)[0]))
    return bool(numpy.all(numpy.isfinite(array)))


def fit_columns_complete(row):
    return all(
        column in row and scalar_is_finite(row[column])
        for column in RV_ONLY_RESULT_COLUMNS
    )


def result_checkpoint_key(row):
    if "rv_id" in row and not pandas.isna(row["rv_id"]):
        return f"id:{row['rv_id']}"
    if "rv_index" in row and not pandas.isna(row["rv_index"]):
        return f"idx:{row['rv_index']}"
    return f"row:{row.name}"


def load_results():
    if RV_EVEN_SPACED_RESULTS_PATH.exists():
        table = pandas.read_pickle(RV_EVEN_SPACED_RESULTS_PATH)
    else:
        table = load_rv_timeseries()
        if not PREFIT_RESULTS_PATH.exists():
            raise FileNotFoundError(
                f"Photometry-prefit table is missing at {PREFIT_RESULTS_PATH}; "
                "it supplies the RV campaign start times."
            )
        prefits = pandas.read_pickle(PREFIT_RESULTS_PATH)
        required = {"simulation_id", "campaign_start"}
        missing = required.difference(prefits.columns)
        if missing:
            raise KeyError(f"Photometry-prefit table is missing columns: {sorted(missing)}")
        table = table.merge(
            prefits.loc[:, ["simulation_id", "campaign_start"]].rename(
                columns={"simulation_id": "system_id"}
            ),
            on="system_id",
            how="left",
            validate="many_to_one",
        )
        if table["campaign_start"].isna().any():
            raise KeyError("No campaign start time was found for one or more RV systems")
    logger.info(
        "Loaded %d RV evenly spaced result rows from %s",
        len(table),
        RV_EVEN_SPACED_RESULTS_PATH,
    )
    return table


def load_batch_rv_ids(path):
    batch = pandas.read_csv(path)
    if "rv_id" not in batch.columns:
        raise ValueError(f"Batch file {path} must contain a rv_id column")
    return set(batch["rv_id"].dropna().astype(str))


def filter_to_batch(table, path):
    rv_ids = load_batch_rv_ids(path)
    if not rv_ids:
        return table.iloc[0:0].copy()
    return table[table["rv_id"].astype(str).isin(rv_ids)].copy()


def rows_requiring_fit(table):
    selected = []
    complete_count = 0
    for index, row in table.iterrows():
        if fit_columns_complete(row):
            complete_count += 1
        else:
            selected.append((index, row))
    logger.info(
        "RV-only no-prior completion check selected %d row(s); %d already complete",
        len(selected),
        complete_count,
    )
    if not selected:
        return table.iloc[0:0].copy()
    return pandas.DataFrame([row for _, row in selected], index=[index for index, _ in selected])


def sample_rv_path_groups(table, sample_size):
    if table.empty:
        return table.copy()
    paths = pandas.Series(table["rv_path"].drop_duplicates())
    sampled_paths = set(paths.sample(min(int(sample_size), len(paths))))
    return table[table["rv_path"].isin(sampled_paths)].copy()


def grouped_rv_rows(table):
    return [list(group.iterrows()) for _, group in table.groupby("rv_path", sort=True)]


def gpu_fit_batches(table):
    """Pack independent RV variants across files into full GPU batches."""
    rows = list(table.sort_values(["rv_path", "rv_id"]).iterrows())
    if FIT_BATCH_SIZE is None:
        return [rows]
    return [rows[start : start + FIT_BATCH_SIZE] for start in range(0, len(rows), FIT_BATCH_SIZE)]


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


def evenly_spaced_curve(ts):
    """Generate the independent, deterministic evenly spaced RV realization.

    Targeted RV curves are retained for the joint-fit campaign.  This helper
    creates the additional RV-only experiment from the simulation truth at a
    separate regular grid, with a seed tied to ``rv_id`` so checkpointed
    reruns reproduce exactly the same noisy measurements.
    """
    if "campaign_start" not in ts:
        raise KeyError("RV metadata must include campaign_start for evenly spaced fits")
    times = evenly_spaced_rv_observations(
        ts["campaign_start"], ts["baseline"], ts["nobs"]
    )
    parameters = numpy.asarray(
        [ts[name] for name in ("a", "i", "Macc", "Mcomp", "Rcomp", "Teff", "e", "periapsis_phase")],
        dtype=float,
    )
    noiseless_rv = numpy.asarray(
        radial_velocity_m_per_s(times, parameters, float(ts["peak_time"])), dtype=float
    )
    rv_err = numpy.sqrt(
        Spectrograph.STELLAR_NOISE.to_value(units.m / units.s) ** 2
        + float(ts["RV accuracy"]) ** 2
    )
    seed = int.from_bytes(
        hashlib.sha256(str(ts["rv_id"]).encode("utf-8")).digest()[:8], "little"
    )
    rv = noiseless_rv + numpy.random.default_rng(seed).normal(0.0, rv_err, len(times))
    return {"rv_times": times, "rv": rv, "rv_err": numpy.full(len(times), rv_err)}


def batched_fit_inputs(rows, curves):
    rv_times = []
    rv = []
    rv_err = []
    for _, ts in rows:
        curve = curves[str(ts["rv_id"])]
        rv_times.append(curve["rv_times"])
        rv.append(curve["rv"])
        rv_err.append(curve["rv_err"])
    rv_times, rv_mask = padded_series(rv_times, fill_from_first=True)
    rv, _ = padded_series(rv)
    rv_err, _ = padded_series(rv_err, fill_value=1.0)
    return {
        "rv_times": rv_times,
        "observed_rv": rv,
        "observed_rv_err": rv_err,
        "rv_mask": rv_mask,
    }


def add_rv_no_prior_columns(row, fit_result):
    new_row = pandas.Series(row).copy()
    sls = fit_result.system
    new_row[f"{RV_ONLY_PREFIX}_fitted_a"] = sls.a.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_b"] = sls.b.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_b_max"] = sls.b_max.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_i"] = sls.i.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_Macc"] = sls.Macc.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_e"] = sls.e.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_periapsis_phase"] = sls.periapsis_phase.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_porb"] = sls.porb.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_tau_eff"] = sls.tau_eff.value
    new_row[f"{RV_ONLY_PREFIX}_fitted_rv_phase"] = fit_result.rv_phase
    new_row[f"{RV_ONLY_PREFIX}_fitted_rv_reference_time"] = (
        fit_result.rv_reference_time
    )
    new_row[f"{RV_ONLY_PREFIX}_fitted_peak_time"] = fit_result.rv_peak_time
    return new_row


def short_id(value, length=8):
    if value is None or pandas.isna(value):
        return "unknown"
    return str(value)[:length]


def run_rv_no_prior_fit(args):
    i, ts, curve = args
    logger.info(
        "Starting RV-only no-prior fit for row index=%s rv_id=%s",
        i,
        ts.get("rv_id", None),
    )
    fit_result, sampler, walker_state = (
        self_lensing_system_from_rv_fit_no_peak_prior_blackjax(
            rv_times=Time(curve["rv_times"], format="jd"),
            observed_rv=curve["rv"] * units.m / units.s,
            observed_rv_err=curve["rv_err"] * units.m / units.s,
            Mcomp=ts["Mcomp"] * units.M_sun,
            Rcomp=ts["Rcomp"] * units.R_sun,
            Teff=ts["Teff"] * units.K,
            nsteps=FIT_NSTEPS,
            nwalkers=FIT_NWALKERS,
            burn=FIT_BURN,
            progress=FIT_MCMC_PROGRESS,
            progress_desc=f"MCMC RV no prior {short_id(ts.get('rv_id', None))}",
            progress_chunk_size=FIT_PROGRESS_CHUNK_SIZE,
            lensing_method=FIT_LENSING_METHOD,
            device=FIT_DEVICE,
            prefer_gpu=FIT_DEVICE == "auto",
            proposal_scale_fraction=FIT_PROPOSAL_SCALE_FRACTION,
            return_sampler=FIT_SAVE_SAMPLERS,
            return_walker_state=FIT_SAVE_SAMPLERS,
        )
    )
    new_row = add_rv_no_prior_columns(ts, fit_result)
    sampler_writer = shared_sampler_writer()
    close_sampler_writer = False
    if sampler_writer is None:
        sampler_writer = SamplerStateWriter()
        close_sampler_writer = True
    sampler_state_path = sampler_writer.submit(new_row, sampler, walker_state)
    if close_sampler_writer:
        sampler_writer.close()
    if sampler_state_path is not None:
        new_row[f"{RV_ONLY_PREFIX}_fit_sampler_state_path"] = str(sampler_state_path)
    return new_row


def run_rv_no_prior_batch(rows, curves):
    start = time.perf_counter()
    inputs = batched_fit_inputs(rows, curves)
    row_table = pandas.DataFrame([ts for _, ts in rows])
    logger.info(
        "Running batched RV-only no-prior MCMC for %d rows: max_rv=%d",
        len(rows),
        inputs["rv_times"].shape[1],
    )
    fit_results, sampler, walker_state = (
        self_lensing_systems_from_batched_rv_fit_no_peak_prior_blackjax(
            rv_times=Time(inputs["rv_times"], format="jd"),
            observed_rv=inputs["observed_rv"] * units.m / units.s,
            observed_rv_err=inputs["observed_rv_err"] * units.m / units.s,
            rv_mask=inputs["rv_mask"],
            Mcomp=row_table["Mcomp"].to_numpy(dtype=float) * units.M_sun,
            Rcomp=row_table["Rcomp"].to_numpy(dtype=float) * units.R_sun,
            Teff=row_table["Teff"].to_numpy(dtype=float) * units.K,
            nsteps=FIT_NSTEPS,
            nwalkers=FIT_NWALKERS,
            burn=FIT_BURN,
            progress=FIT_MCMC_PROGRESS,
            progress_desc=f"MCMC batched RV no prior {short_id(rows[0][1].get('system_id', None))}",
            progress_chunk_size=FIT_PROGRESS_CHUNK_SIZE,
            lensing_method=FIT_LENSING_METHOD,
            device=FIT_DEVICE,
            prefer_gpu=FIT_DEVICE == "auto",
            proposal_scale_fraction=FIT_PROPOSAL_SCALE_FRACTION,
            return_sampler=False,
            return_walker_state=False,
        )
    )
    del sampler, walker_state
    logger.info(
        "Batched RV-only no-prior MCMC returned %d systems in %.2f s",
        len(fit_results),
        time.perf_counter() - start,
    )
    return [
        add_rv_no_prior_columns(row, fit_result)
        for (_, row), fit_result in zip(rows, fit_results)
    ]


def batched_row_chunks(rows):
    if FIT_BATCH_SIZE is None or FIT_BATCH_SIZE >= len(rows):
        yield rows
        return
    for start in range(0, len(rows), FIT_BATCH_SIZE):
        yield rows[start : start + FIT_BATCH_SIZE]


def fit_rv_file(rows):
    rows = list(rows)
    if not rows:
        return []
    path = rows[0][1]["rv_path"]
    logger.info(
        "Generating %d evenly spaced RV curves (starting with %s)", len(rows), path
    )
    curves = {str(ts["rv_id"]): evenly_spaced_curve(ts) for _, ts in rows}
    if FIT_BATCHED and not FIT_SAVE_SAMPLERS:
        results = []
        for batch_rows in batched_row_chunks(rows):
            results.extend(run_rv_no_prior_batch(batch_rows, curves))
        return results
    results = []
    for i, ts in tqdm(
        rows,
        total=len(rows),
        desc=f"RV no-prior fits {short_id(rows[0][1].get('system_id', Path(path).stem))}",
        leave=False,
        dynamic_ncols=True,
    ):
        curve = curves[str(ts["rv_id"])]
        results.append(run_rv_no_prior_fit((i, ts, curve)))
    return results


def warm_rv_file(rows):
    """Read a FITS file into the shared filesystem cache without serializing curves."""
    path = Path(rows[0][1]["rv_path"])
    logger.debug("Preloading rv FITS file into the filesystem cache: %s", path)
    with path.open("rb", buffering=0) as handle:
        while handle.read(1024 * 1024):
            pass
    return rows


def preloaded_rv_file_jobs(jobs):
    """Yield groups after a bounded, ordered look-ahead of file-cache warming."""
    jobs = iter(jobs)
    pool_size = max(0, FIT_PRELOAD_FILE_POOL_SIZE)
    if pool_size == 0:
        for rows in jobs:
            yield rows
        return

    def fill_pending(pending, executor):
        while len(pending) < pool_size:
            try:
                rows = next(jobs)
            except StopIteration:
                return
            pending.append((rows, executor.submit(warm_rv_file, rows)))

    with ThreadPoolExecutor(max_workers=max(1, FIT_PRELOAD_MAX_WORKERS)) as executor:
        pending = deque()
        fill_pending(pending, executor)
        while pending:
            rows, future = pending.popleft()
            future.result()
            fill_pending(pending, executor)
            yield rows


def preloaded_process_map_iter(jobs, total=None, desc=None, max_workers=None):
    """Fit groups while keeping a small pool of following groups preloaded."""
    if total is None:
        total = len(jobs)
    if not total:
        return
    if FIT_PRELOAD_FILE_POOL_SIZE <= 0:
        yield from process_map_iter(
            fit_rv_file,
            jobs,
            total=total,
            desc=desc,
            max_workers=max_workers,
        )
        return
    preloaded_jobs = preloaded_rv_file_jobs(jobs)
    if max_workers == 1:
        for rows in tqdm(preloaded_jobs, total=total, desc=desc):
            yield fit_rv_file(rows)
        return

    context = multiprocessing.get_context(MP_START_METHOD)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        futures = deque()
        with tqdm(total=total, desc=desc) as progress:
            for rows in preloaded_jobs:
                futures.append(executor.submit(fit_rv_file, rows))
                if len(futures) >= max_workers:
                    yield futures.popleft().result()
                    progress.update()
            while futures:
                yield futures.popleft().result()
                progress.update()


def process_map_iter(fn, iterable, total=None, desc=None, max_workers=None, chunksize=1):
    items = list(iterable)
    if total is None:
        total = len(items)
    if not items:
        return
    if max_workers == 1:
        for item in tqdm(items, total=total, desc=desc):
            yield fn(item)
        return
    context = multiprocessing.get_context(MP_START_METHOD)
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        yield from tqdm(
            executor.map(fn, items, chunksize=chunksize),
            total=total,
            desc=desc,
        )


def merge_rv_even_spaced_results(existing_results, new_rows):
    if not new_rows:
        return existing_results.copy()
    merged = existing_results.copy()
    if "rv_id" not in merged.columns:
        raise ValueError("Existing fitted results must contain rv_id")
    index_by_key = {
        result_checkpoint_key(row): index for index, row in merged.iterrows()
    }
    for row in new_rows:
        key = result_checkpoint_key(row)
        if key not in index_by_key:
            raise ValueError(f"Refusing to add new result row for unknown key {key}")
        target_index = index_by_key[key]
        for column, value in row.items():
            if column.startswith(f"{RV_ONLY_PREFIX}_"):
                merged.at[target_index, column] = value
    return merged


def save_results(table):
    RV_EVEN_SPACED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(f"{RV_EVEN_SPACED_RESULTS_PATH}.new.gz")
    table.reset_index(drop=True).to_pickle(temp_path)
    os.replace(temp_path, RV_EVEN_SPACED_RESULTS_PATH)
    logger.info("Saved RV evenly spaced results atomically to %s", RV_EVEN_SPACED_RESULTS_PATH)


def write_checkpoint(base_results, new_rows):
    """CPU-sidecar entry point for merging and atomically writing a snapshot."""
    save_results(merge_rv_even_spaced_results(base_results, new_rows))


class AsyncCheckpointWriter:
    def __init__(self, base_results, enabled=True):
        self.base_results = base_results
        self.enabled = enabled
        self.latest_rows = None
        self.pending_future = None
        self.executor = None
        self.previous_sidecar_flag = None
        if self.enabled:
            # Keep this flag set until shutdown: spawned workers import this
            # module before they execute ``write_checkpoint``.
            self.previous_sidecar_flag = os.environ.get("SLRV_FIT_CPU_SIDECAR")
            os.environ["SLRV_FIT_CPU_SIDECAR"] = "1"
            self.executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )

    def submit(self, new_rows):
        if not self.enabled:
            return
        self.latest_rows = list(new_rows)
        self._start_latest_if_idle()

    def _start_latest_if_idle(self):
        if self.pending_future is not None:
            if not self.pending_future.done():
                return
            self.pending_future.result()
            self.pending_future = None
        if self.latest_rows is not None:
            rows = self.latest_rows
            self.latest_rows = None
            self.pending_future = self.executor.submit(
                write_checkpoint, self.base_results, rows
            )

    def check_error(self):
        self._start_latest_if_idle()

    def close(self, final_rows=None):
        if not self.enabled:
            if final_rows is not None:
                write_checkpoint(self.base_results, final_rows)
            return
        if final_rows is not None:
            self.latest_rows = list(final_rows)
        # Finish the current snapshot, then write the newest cumulative one.
        if self.pending_future is not None:
            self.pending_future.result()
            self.pending_future = None
        self._start_latest_if_idle()
        if self.pending_future is not None:
            self.pending_future.result()
        self.executor.shutdown()
        if self.previous_sidecar_flag is None:
            del os.environ["SLRV_FIT_CPU_SIDECAR"]
        else:
            os.environ["SLRV_FIT_CPU_SIDECAR"] = self.previous_sidecar_flag


@dataclass
class FitProgress:
    new_rows: list
    new_result_count: int = 0
    last_checkpoint_result_count: int = 0


def record_group_results(group_results, progress, checkpoint_writer):
    checkpoint_writer.check_error()
    progress.new_rows.extend(group_results)
    progress.new_result_count += len(group_results)
    if (
        FIT_CHECKPOINT_INTERVAL > 0
        and progress.new_result_count - progress.last_checkpoint_result_count
        >= FIT_CHECKPOINT_INTERVAL
    ):
        checkpoint_writer.submit(progress.new_rows)
        progress.last_checkpoint_result_count = progress.new_result_count


def clear_rv_fit_columns(table):
    columns = [
        column
        for column in table.columns
        if column.startswith(("rvprior_", f"{RV_ONLY_PREFIX}_"))
    ]
    if columns:
        logger.info("Dropping %d existing RV-only result column(s)", len(columns))
        return table.drop(columns=columns)
    return table


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    logger.info("Starting RV evenly spaced fitting script")
    logger.info(
        "Settings: FIT_MAX_WORKERS=%d FIT_SAMPLE_SIZE=%s FIT_NSTEPS=%d "
        "FIT_NWALKERS=%d FIT_BURN=%d FIT_BATCH_SIZE=%s FIT_BATCHED=%s "
        "FIT_CHECKPOINT_INTERVAL=%d checkpoint_writer=cpu-sidecar",
        FIT_MAX_WORKERS,
        FIT_SAMPLE_SIZE if FIT_SAMPLE_SIZE is not None else "all",
        FIT_NSTEPS,
        FIT_NWALKERS,
        FIT_BURN,
        FIT_BATCH_SIZE if FIT_BATCH_SIZE is not None else "all",
        FIT_BATCHED,
        FIT_CHECKPOINT_INTERVAL,
    )
    logger.info("Fit eccentricity limits: %s", DEFAULT_LIMITS["e"])
    logger.info("Fit device setting: FIT_DEVICE=%s", FIT_DEVICE)
    logger.info("JAX backend: default=%s devices=[%s]", jax.default_backend(), jax_device_summary())
    logger.info(
        "JAX compilation cache: %s max_size=%s bytes xla_caches=%s",
        os.environ["JAX_COMPILATION_CACHE_DIR"],
        os.environ["JAX_COMPILATION_CACHE_MAX_SIZE"],
        os.environ["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"],
    )
    if REQUIRE_GPU and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError(
            "SLRV_REQUIRE_GPU is set, but JAX does not see a GPU. "
            f"Available devices: [{jax_device_summary()}]"
        )

    existing_results = load_results()
    if args.clear_results:
        existing_results = clear_rv_fit_columns(existing_results)
        save_results(existing_results)
    candidate_results = existing_results
    if FIT_BATCH_FILE:
        candidate_results = filter_to_batch(existing_results, FIT_BATCH_FILE)

    rows_to_fit = rows_requiring_fit(candidate_results)
    if len(rows_to_fit) == 0:
        logger.info("No rows left with missing RV-only fit columns; exiting")
        raise SystemExit
    if FIT_BATCH_FILE:
        logger.info("Using all %d incomplete manifest rows for fitting", len(rows_to_fit))
    elif FIT_SAMPLE_SIZE is None:
        logger.info("Using all %d incomplete rows for standalone fitting", len(rows_to_fit))
    else:
        rows_to_fit = sample_rv_path_groups(rows_to_fit, FIT_SAMPLE_SIZE)

    jobs = gpu_fit_batches(rows_to_fit)
    progress = FitProgress(new_rows=[])
    if FIT_MAX_WORKERS == 1:
        ACTIVE_SAMPLER_WRITER = SamplerStateWriter()
    checkpoint_writer = AsyncCheckpointWriter(
        base_results=existing_results,
        enabled=CHECKPOINT_RESULTS,
    )
    try:
        for group_results in process_map_iter(
            fit_rv_file,
            jobs,
            total=len(jobs),
            desc="RV-only evenly spaced fits",
            max_workers=FIT_MAX_WORKERS,
        ):
            record_group_results(group_results, progress, checkpoint_writer)
        if ACTIVE_SAMPLER_WRITER is not None:
            ACTIVE_SAMPLER_WRITER.close()
            ACTIVE_SAMPLER_WRITER = None
        checkpoint_writer.close(final_rows=progress.new_rows)
    except BaseException:
        if ACTIVE_SAMPLER_WRITER is not None:
            ACTIVE_SAMPLER_WRITER.close()
            ACTIVE_SAMPLER_WRITER = None
        checkpoint_writer.close(final_rows=progress.new_rows)
        raise
    logger.info(
        "Finished RV evenly spaced fitting script with %d new row update(s)",
        progress.new_result_count,
    )
