"""
This fits a sample of the realistic timeseries and records the fitted parameters.
"""

import logging
import multiprocessing
import warnings
import os
import argparse
import re
import shutil
import time
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
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
if "SLRV_JAX_PLATFORMS" in os.environ:
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
    os.environ.get("SLRV_JAX_COMPILATION_CACHE_MAX_SIZE", str(1024**3)),
)
Path(os.environ["JAX_COMPILATION_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

FIT_RESULTS_LOCK = None
if __name__ == "__main__":
    from fitting import acquire_fit_results_lock

    FIT_RESULTS_LOCK = acquire_fit_results_lock("realistic timeseries fitting script")

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

JAX_DEVICES = jax.devices()

import numpy
import pandas

from astropy import units
from astropy.time import Time
from tqdm import tqdm

from jax_compat import install_quax_jax_stage_patch

install_quax_jax_stage_patch()

from selflensing.fitting import (
    self_lensing_system_from_fit_blackjax,
    self_lensing_systems_from_batched_fit_blackjax,
)

from selflensing.models import DEFAULT_LIMITS, SelfLensingSystem

from fitting import SAMPLER_STATE_PATH, TIMESERIES_TABLE_PATH
from fit_materialization import (
    materialize_joint_batch_result as materialize_joint_batch_result_in_process,
    materialize_rv_only_batch_result as materialize_rv_only_batch_result_in_process,
)
from realistic import (
    load_realistic_timeseries,
    load_realistic_timeseries_curve,
    load_realistic_timeseries_curves,
)
from utils import configure_logging

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
        help="Remove saved fit results before running.",
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


FIT_MAX_WORKERS = int(os.environ.get("SLRV_FIT_MAX_WORKERS", default_fit_max_workers()))
FIT_SAMPLE_SIZE = optional_positive_int(os.environ.get("SLRV_FIT_SAMPLE_SIZE", "all"))
FIT_BATCH_FILE = os.environ.get("SLRV_FIT_BATCH_FILE")
CHECKPOINT_RESULTS = os.environ.get("SLRV_FIT_CHECKPOINT_RESULTS", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_CHECKPOINT_INTERVAL = int(os.environ.get("SLRV_FIT_CHECKPOINT_INTERVAL", 45))
MP_START_METHOD = os.environ.get("SLRV_MP_START_METHOD", "spawn")
FIT_NSTEPS = int(os.environ.get("SLRV_FIT_NSTEPS", 1000))
FIT_NWALKERS = int(os.environ.get("SLRV_FIT_NWALKERS", 128))
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
FIT_MCMC_PROGRESS = os.environ.get("SLRV_FIT_MCMC_PROGRESS", "0").lower() in (
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
FIT_BATCHED_RV_ONLY = os.environ.get("SLRV_FIT_BATCHED_RV_ONLY", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_SAVE_SAMPLERS = os.environ.get("SLRV_FIT_SAVE_SAMPLERS", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_MATERIALIZE_IN_PROCESS = os.environ.get(
    "SLRV_FIT_MATERIALIZE_IN_PROCESS",
    "1",
).lower() in (
    "1",
    "true",
    "yes",
    "on",
)
FIT_SAMPLER_WRITE_MAX_WORKERS = int(
    os.environ.get("SLRV_FIT_SAMPLER_WRITE_MAX_WORKERS", 16)
)
FIT_BATCH_SIZE = optional_positive_int(os.environ.get("SLRV_FIT_BATCH_SIZE", 50))
FIT_PROPOSAL_SCALE_FRACTION = float(
    os.environ.get("SLRV_FIT_PROPOSAL_SCALE_FRACTION", 0.005)
)
FIT_PREFIT_PROPOSAL_SCALE_FRACTION = float(
    os.environ.get("SLRV_FIT_PREFIT_PROPOSAL_SCALE_FRACTION", 0.02)
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


def blackjax_sampler_kwargs(
    return_sampler,
    return_walker_state=True,
    proposal_scale_fraction=FIT_PROPOSAL_SCALE_FRACTION,
):
    return {
        "return_sampler": return_sampler,
        "return_walker_state": return_walker_state,
        "proposal_scale_fraction": proposal_scale_fraction,
    }


FIT_PARAMETER_NAMES = numpy.asarray(
    ["a", "i", "Macc", "e", "periapsis_phase"], dtype=str
)
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
PREFIT_RESULT_COLUMNS = tuple(f"prefit_{column}" for column in FIT_RESULT_COLUMNS)
RV_ONLY_RESULT_COLUMNS = tuple(f"rvfit_{column}" for column in FIT_RESULT_COLUMNS)


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
    if "realistic_id" in ts and not pandas.isna(ts["realistic_id"]):
        token = safe_path_token(ts["realistic_id"])
    elif "realistic_idx" in ts and not pandas.isna(ts["realistic_idx"]):
        token = f"idx_{safe_path_token(ts['realistic_idx'])}"
    else:
        token = f"row_{safe_path_token(getattr(ts, 'name', None))}"
    pristine_id = safe_path_token(ts.get("pristine_id", "unknown"))
    a_key, b_key, macc_key = sampler_state_shard_keys(ts)
    return SAMPLER_STATE_PATH / a_key / b_key / macc_key / pristine_id / f"{token}.npz"


def prefit_sampler_state_path_for_row(ts):
    pristine_id = safe_path_token(ts.get("pristine_id", "unknown"))
    a_key, b_key, macc_key = sampler_state_shard_keys(ts)
    return SAMPLER_STATE_PATH / a_key / b_key / macc_key / pristine_id / "_prefit.npz"


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


def sampler_state_payload(ts, sampler, walker_state, prefit_sampler_state_path=None):
    if not FIT_SAVE_SAMPLERS:
        return None
    if sampler is None:
        logger.warning(
            "Sampler saving is enabled, but no sampler was returned for realistic_id=%s",
            ts.get("realistic_id", None),
        )
        return None

    joint_sampler_state = sampler_arrays(sampler)
    path = sampler_state_path_for_row(ts)
    final_coords = walker_state_coords(walker_state)
    payload = {
        "path": path,
        "chain": joint_sampler_state["chain"],
        "log_prob": joint_sampler_state["log_prob"],
        "acceptance": joint_sampler_state["acceptance"],
        "final_walker_coords": (
            numpy.asarray(final_coords, dtype=float)
            if final_coords is not None
            else numpy.asarray([], dtype=float)
        ),
        "parameter_names": FIT_PARAMETER_NAMES,
        "burn": numpy.asarray(FIT_BURN, dtype=int),
        "nsteps": numpy.asarray(FIT_NSTEPS, dtype=int),
        "nwalkers": numpy.asarray(FIT_NWALKERS, dtype=int),
        "realistic_id": numpy.asarray(str(ts.get("realistic_id", ""))),
        "pristine_id": numpy.asarray(str(ts.get("pristine_id", ""))),
        "realistic_idx": numpy.asarray(
            str(ts.get("realistic_idx", getattr(ts, "name", "")))
        ),
    }
    if prefit_sampler_state_path is not None:
        payload["prefit_sampler_state_path"] = numpy.asarray(
            str(prefit_sampler_state_path)
        )
    return payload


def prefit_sampler_state_payload(ts, prefit_sampler_state):
    if not FIT_SAVE_SAMPLERS or prefit_sampler_state is None:
        return None
    return {
        "path": prefit_sampler_state_path_for_row(ts),
        "chain": prefit_sampler_state["chain"],
        "log_prob": prefit_sampler_state["log_prob"],
        "acceptance": prefit_sampler_state["acceptance"],
        "parameter_names": FIT_PARAMETER_NAMES,
        "nsteps": numpy.asarray(prefit_sampler_state["chain"].shape[0], dtype=int),
        "pristine_id": numpy.asarray(str(ts.get("pristine_id", ""))),
    }


def write_sampler_state_payload(payload):
    path = payload["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Saving sampler state to %s",
        path,
    )
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
        realistic_id=payload["realistic_id"],
        pristine_id=payload["pristine_id"],
        realistic_idx=payload["realistic_idx"],
        **({"prefit_sampler_state_path": payload["prefit_sampler_state_path"]}
           if "prefit_sampler_state_path" in payload
           else {}),
    )
    return path


def write_prefit_sampler_state_payload(payload):
    path = payload["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving prefit sampler state to %s", path)
    numpy.savez_compressed(
        path,
        chain=payload["chain"],
        log_prob=payload["log_prob"],
        acceptance=payload["acceptance"],
        parameter_names=payload["parameter_names"],
        nsteps=payload["nsteps"],
        pristine_id=payload["pristine_id"],
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

    def submit(self, ts, sampler, walker_state, prefit_sampler_state_path=None):
        payload = sampler_state_payload(
            ts,
            sampler,
            walker_state,
            prefit_sampler_state_path=prefit_sampler_state_path,
        )
        if payload is None:
            return None
        path = payload["path"]
        if self.executor is None:
            write_sampler_state_payload(payload)
        else:
            logger.debug("Queueing sampler state write to %s", path)
            self.futures.append(self.executor.submit(write_sampler_state_payload, payload))
        return path

    def submit_prefit(self, ts, prefit_sampler_state):
        payload = prefit_sampler_state_payload(ts, prefit_sampler_state)
        if payload is None:
            return None
        path = payload["path"]
        if self.executor is None:
            write_prefit_sampler_state_payload(payload)
        else:
            logger.debug("Queueing prefit sampler state write to %s", path)
            self.futures.append(
                self.executor.submit(write_prefit_sampler_state_payload, payload)
            )
        return path

    def close(self):
        if self.futures:
            logger.info("Waiting for %d queued sampler state write(s)", len(self.futures))
        for future in self.futures:
            future.result()
        if self.executor is not None:
            self.executor.shutdown()


ACTIVE_SAMPLER_WRITER = None


def shared_sampler_writer():
    return ACTIVE_SAMPLER_WRITER


def process_map(fn, iterable, total=None, desc=None, max_workers=None, chunksize=1):
    """Run a tqdm-wrapped process map with an explicit multiprocessing context."""
    return list(
        process_map_iter(
            fn,
            iterable,
            total=total,
            desc=desc,
            max_workers=max_workers,
            chunksize=chunksize,
        )
    )


def process_map_iter(fn, iterable, total=None, desc=None, max_workers=None, chunksize=1):
    """Run a tqdm-wrapped process map and yield results as they become available."""
    items = list(iterable)
    if total is None:
        total = len(items)
    if not items:
        return
    if max_workers == 1:
        logger.info(
            "Starting serial map desc=%s items=%d because max_workers=1",
            desc,
            len(items),
        )
        for item in tqdm(items, total=total, desc=desc):
            yield fn(item)
        return

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
        yield from tqdm(
            executor.map(fn, items, chunksize=chunksize),
            total=total,
            desc=desc,
        )
    logger.info("Finished process map desc=%s items=%d", desc, len(items))


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


def fit_columns_complete(row, columns):
    return all(column in row and scalar_is_finite(row[column]) for column in columns)


def row_needs_prefit(row):
    return FIT_USE_LC and not fit_columns_complete(row, PREFIT_RESULT_COLUMNS)


def row_needs_joint_fit(row):
    return not fit_columns_complete(row, FIT_RESULT_COLUMNS)


def row_needs_rv_only_fit(row):
    return not fit_columns_complete(row, RV_ONLY_RESULT_COLUMNS)


def row_needs_any_fit(row):
    return row_needs_prefit(row) or row_needs_joint_fit(row) or row_needs_rv_only_fit(row)


def existing_result_map(results):
    mapped = {}
    if results.empty:
        return mapped
    for _, row in results.iterrows():
        mapped[result_checkpoint_key(row)] = row
    logger.debug("Prepared %d existing result rows by checkpoint key", len(mapped))
    return mapped


def merge_existing_result(row, existing):
    if existing is None:
        return row
    merged = pandas.Series(row).copy()
    for column, value in existing.items():
        merged[column] = value
    return merged


def rows_requiring_fits(realistic_rows, existing_results):
    existing_by_key = existing_result_map(existing_results)
    selected = []
    complete_count = 0
    for index, row in realistic_rows.iterrows():
        checkpoint_key = result_checkpoint_key(row)
        merged = merge_existing_result(row, existing_by_key.get(checkpoint_key))
        if row_needs_any_fit(merged):
            selected.append((index, merged))
        else:
            complete_count += 1
            logger.debug("Skipping fully fitted row checkpoint_key=%s", checkpoint_key)
    logger.info(
        "Fit-stage completion check selected %d row(s); %d row(s) already have "
        "photometry-only, photometry+RV, and RV-only fits",
        len(selected),
        complete_count,
    )
    if not selected:
        return realistic_rows.iloc[0:0].copy()
    return pandas.DataFrame([row for _, row in selected], index=[index for index, _ in selected])


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


class AsyncCheckpointWriter:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.condition = threading.Condition()
        self.latest_results = None
        self.stop_requested = False
        self.error = None
        self.thread = None
        if self.enabled:
            self.thread = threading.Thread(
                target=self._run,
                name="fit-checkpoint-writer",
                daemon=True,
            )
            self.thread.start()

    def submit(self, results):
        if not self.enabled:
            return
        with self.condition:
            self.latest_results = list(results)
            self.condition.notify()

    def check_error(self):
        if self.error is not None:
            raise RuntimeError("Background checkpoint writer failed") from self.error

    def close(self, final_results=None):
        if not self.enabled:
            if final_results is not None:
                save_results(final_results)
            return
        with self.condition:
            if final_results is not None:
                self.latest_results = list(final_results)
            self.stop_requested = True
            self.condition.notify()
        self.thread.join()
        self.check_error()

    def _run(self):
        while True:
            with self.condition:
                while self.latest_results is None and not self.stop_requested:
                    self.condition.wait()
                if self.latest_results is None and self.stop_requested:
                    return
                results = self.latest_results
                self.latest_results = None
            try:
                save_results(results)
            except Exception as exc:
                self.error = exc
                logger.exception("Background checkpoint writer failed")
                return
            with self.condition:
                if self.stop_requested and self.latest_results is None:
                    return


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
    new_ts = pandas.Series(ts).copy()
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


def add_fit_stage_columns(ts, fit_ts, stage_prefix):
    if fit_ts is None:
        return ts
    new_ts = pandas.Series(ts).copy()
    for column in FIT_RESULT_COLUMNS:
        if column in fit_ts:
            new_ts[f"{stage_prefix}_{column}"] = fit_ts[column]
    return new_ts


def add_prefit_columns(ts, prefit_ts):
    return add_fit_stage_columns(ts, prefit_ts, "prefit")


def add_rv_only_columns(ts, rvfit_ts):
    return add_fit_stage_columns(ts, rvfit_ts, "rvfit")


def fit_photometry_prefit(i, ts, curve):
    fit_lc = lightcurve_fit_window(curve, ts)
    mcmc_desc = f"MCMC prefit {short_id(ts.get('realistic_id', None))}"
    logger.info(
        "Starting photometry-only prefit for realistic_id=%s with %d binned LC samples",
        ts.get("realistic_id", None),
        len(fit_lc["lc"]),
    )
    sls, prefit_sampler, walker_state = unpack_fit_result(
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
            **blackjax_sampler_kwargs(
                return_sampler=FIT_SAVE_SAMPLERS,
                proposal_scale_fraction=FIT_PREFIT_PROPOSAL_SCALE_FRACTION,
            ),
        )
    )
    logger.info(
        "Finished photometry-only prefit for realistic_id=%s walker_state=%s",
        ts.get("realistic_id", None),
        walker_state is not None,
    )
    prefit_ts = fitted_timeseries_row(i, ts, truth_self_lensing_system(ts), sls)
    return prefit_ts, walker_state_coords(walker_state), sampler_arrays(prefit_sampler)


def fit(a):
    fit_result = run_fit(a)
    return materialize_fit_result(fit_result)


def run_fit(a):
    if len(a) == 6:
        i, ts, curve, initial_walker_state, mcmc_desc, fit_use_lc = a
    elif len(a) == 5:
        i, ts, curve, initial_walker_state, mcmc_desc = a
        fit_use_lc = FIT_USE_LC
    elif len(a) == 4:
        i, ts, curve, initial_walker_state = a
        mcmc_desc = f"MCMC joint {short_id(ts.get('realistic_id', None))}"
        fit_use_lc = FIT_USE_LC
    elif len(a) == 3:
        i, ts, curve = a
        initial_walker_state = None
        mcmc_desc = f"MCMC joint {short_id(ts.get('realistic_id', None))}"
        fit_use_lc = FIT_USE_LC
    else:
        i, ts = a
        curve = load_realistic_timeseries_curve(ts)
        initial_walker_state = None
        mcmc_desc = f"MCMC joint {short_id(ts.get('realistic_id', None))}"
        fit_use_lc = FIT_USE_LC
    logger.info(
        "Starting fit for row index=%s realistic_id=%s path=%s",
        i,
        ts.get("realistic_id", None),
        ts.get("realistic_path", None),
    )
    truth_sls = truth_self_lensing_system(ts)
    fit_lc = lightcurve_fit_window(curve, ts) if fit_use_lc else None
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
    sls, sampler, walker_state = unpack_fit_result(
        self_lensing_system_from_fit_blackjax(
            rv_times=Time(curve["rv_times"], format="jd"),
            observed_rv=curve["rv"] * units.m / units.s,
            observed_rv_err=curve["rv_err"] * units.m / units.s,
            lc_times=Time(fit_lc["lc_times"], format="jd") if fit_use_lc else None,
            observed_lc=(
                fit_lc["lc"] * units.dimensionless_unscaled if fit_use_lc else None
            ),
            observed_lc_err=(
                fit_lc["lc_err"] * units.dimensionless_unscaled if fit_use_lc else None
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
                return_sampler=FIT_SAVE_SAMPLERS,
                return_walker_state=FIT_SAVE_SAMPLERS,
            ),
        )
    )
    logger.info(
        "MCMC fit returned for row index=%s realistic_id=%s in %.2f s",
        i,
        ts.get("realistic_id", None),
        time.perf_counter() - fit_start,
    )
    return i, ts, truth_sls, sls, sampler, walker_state, curve


def run_rv_only_fit(i, ts, curve):
    return run_fit(
        (
            i,
            ts,
            curve,
            None,
            f"MCMC RV-only {short_id(ts.get('realistic_id', None))}",
            False,
        )
    )


def materialize_fit_result(
    fit_result,
    sampler_writer=None,
    prefit_sampler_state_path=None,
    prefit_ts=None,
):
    i, ts, truth_sls, sls, sampler, walker_state, curve = fit_result
    materialize_start = time.perf_counter()
    new_ts = fitted_timeseries_row(i, ts, truth_sls, sls)
    new_ts = add_prefit_columns(new_ts, prefit_ts)
    if sampler_writer is None:
        sampler_writer = SamplerStateWriter()
        close_sampler_writer = True
    else:
        close_sampler_writer = False
    sampler_state_path = sampler_writer.submit(
        new_ts,
        sampler,
        walker_state,
        prefit_sampler_state_path=prefit_sampler_state_path,
    )
    if close_sampler_writer:
        sampler_writer.close()
    if sampler_state_path is not None:
        new_ts["fit_sampler_state_path"] = str(sampler_state_path)
    if prefit_sampler_state_path is not None:
        new_ts["fit_prefit_sampler_state_path"] = str(prefit_sampler_state_path)
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


def materialize_rv_only_fit_result(fit_result, row):
    i, _, truth_sls, sls, sampler, walker_state, curve = fit_result
    rvfit_ts = fitted_timeseries_row(i, row, truth_sls, sls)
    new_ts = add_rv_only_columns(row, rvfit_ts)
    logger.info(
        "Materialized RV-only fit result for row index=%s: "
        "rvfit_fitted_a=%.6g rvfit_fitted_Macc=%.6g rvfit_fitted_e=%.6g",
        i,
        new_ts["rvfit_fitted_a"],
        new_ts["rvfit_fitted_Macc"],
        new_ts["rvfit_fitted_e"],
    )
    return new_ts, sampler, walker_state, curve


def finish_pending_fit(pending_fit):
    if pending_fit is None:
        return None
    return pending_fit.result()


def finish_pending_fit_results(pending_fit):
    if pending_fit is None:
        return []
    return pending_fit.result()


def can_use_batched_joint_fit():
    return FIT_BATCHED_JOINT


def can_use_batched_rv_only_fit():
    return FIT_BATCHED_RV_ONLY


def walker_state_for_batch_row(walker_state, batch_index):
    coords = walker_state_coords(walker_state)
    if coords is None:
        return None
    return coords[batch_index]


def padded_series(series, fill_value=0.0, fill_from_first=False, length=None):
    max_len = max(len(values) for values in series)
    if length is None:
        length = max_len
    elif max_len > length:
        raise ValueError(f"Cannot pad series of length {max_len} to fixed length {length}")
    data = numpy.full((len(series), length), fill_value, dtype=float)
    mask = numpy.zeros((len(series), length), dtype=bool)
    for index, values in enumerate(series):
        values = numpy.asarray(values, dtype=float)
        if fill_from_first and len(values) > 0:
            data[index, :] = values[0]
        data[index, : len(values)] = values
        mask[index, : len(values)] = True
    return data, mask


def batched_fit_inputs(rows, curves, include_lc):
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
        if include_lc:
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
    if include_lc:
        lc_times, lc_mask = padded_series(
            lc_times,
            fill_from_first=True,
            length=FIT_LC_BINS,
        )
        lc, _ = padded_series(lc, length=FIT_LC_BINS)
        lc_err, _ = padded_series(lc_err, fill_value=1.0, length=FIT_LC_BINS)
        inputs.update(
            {
                "lc_times": lc_times,
                "observed_lc": lc,
                "observed_lc_err": lc_err,
                "lc_mask": lc_mask,
            }
        )
    return inputs


def batched_joint_fit_inputs(rows, curves):
    return batched_fit_inputs(rows, curves, include_lc=FIT_USE_LC)


def run_joint_batch(
    rows,
    curves,
    initial_walker_state,
):
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
    systems, sampler, walker_state = self_lensing_systems_from_batched_fit_blackjax(
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
        proposal_scale_fraction=FIT_PROPOSAL_SCALE_FRACTION,
        return_sampler=FIT_SAVE_SAMPLERS,
        return_walker_state=FIT_SAVE_SAMPLERS,
    )
    logger.info(
        "Batched joint MCMC returned %d systems in %.2f s",
        len(systems),
        time.perf_counter() - start,
    )
    if FIT_SAVE_SAMPLERS and sampler is None:
        logger.warning("Sampler saving is enabled, but batched fit returned no sampler")
    return rows, systems, sampler, walker_state


def materialize_joint_batch_result(
    batch_fit_result,
    sampler_writer=None,
    prefit_sampler_state_path=None,
    prefit_ts=None,
):
    rows, systems, sampler, walker_state = batch_fit_result
    results = []
    materialize_start = time.perf_counter()
    for batch_index, ((i, ts), sls) in enumerate(zip(rows, systems)):
        new_ts = fitted_timeseries_row(i, ts, truth_self_lensing_system(ts), sls)
        new_ts = add_prefit_columns(new_ts, prefit_ts)
        row_sampler = sampler[batch_index] if sampler is not None else None
        sampler_state_path = None
        if sampler_writer is not None:
            sampler_state_path = sampler_writer.submit(
                new_ts,
                row_sampler,
                walker_state_for_batch_row(walker_state, batch_index),
                prefit_sampler_state_path=prefit_sampler_state_path,
            )
        if sampler_state_path is not None:
            new_ts["fit_sampler_state_path"] = str(sampler_state_path)
        if prefit_sampler_state_path is not None:
            new_ts["fit_prefit_sampler_state_path"] = str(prefit_sampler_state_path)
        results.append((new_ts, None, None))
    logger.info(
        "Materialized %d batched fit results in %.2f s",
        len(results),
        time.perf_counter() - materialize_start,
    )
    return results


def fit_joint_batch(
    rows,
    curves,
    initial_walker_state,
    sampler_writer,
    prefit_sampler_state_path=None,
    prefit_ts=None,
):
    return materialize_joint_batch_result(
        run_joint_batch(rows, curves, initial_walker_state),
        sampler_writer,
        prefit_sampler_state_path=prefit_sampler_state_path,
        prefit_ts=prefit_ts,
    )


def run_rv_only_batch(rows, curves):
    start = time.perf_counter()
    inputs = batched_fit_inputs(rows, curves, include_lc=False)
    row_table = pandas.DataFrame([ts for _, ts in rows])
    logger.info(
        "Running batched RV-only MCMC for %d rows: max_rv=%d",
        len(rows),
        inputs["rv_times"].shape[1],
    )
    systems, sampler, walker_state = self_lensing_systems_from_batched_fit_blackjax(
        rv_times=Time(inputs["rv_times"], format="jd"),
        observed_rv=inputs["observed_rv"] * units.m / units.s,
        observed_rv_err=inputs["observed_rv_err"] * units.m / units.s,
        rv_mask=inputs["rv_mask"],
        peak_time=Time(row_table["peak_time"].to_numpy(dtype=float), format="jd"),
        Mcomp=row_table["Mcomp"].to_numpy(dtype=float) * units.M_sun,
        Rcomp=row_table["Rcomp"].to_numpy(dtype=float) * units.R_sun,
        Teff=row_table["Teff"].to_numpy(dtype=float) * units.K,
        false_alarm_probability=0.5,
        nsteps=FIT_NSTEPS,
        nwalkers=FIT_NWALKERS,
        burn=FIT_BURN,
        progress=FIT_MCMC_PROGRESS,
        progress_desc=f"MCMC batched RV-only {short_id(rows[0][1].get('pristine_id', None))}",
        progress_chunk_size=FIT_PROGRESS_CHUNK_SIZE,
        lensing_method=FIT_LENSING_METHOD,
        center_lc_model=True,
        device=FIT_DEVICE,
        prefer_gpu=FIT_DEVICE == "auto",
        proposal_scale_fraction=FIT_PROPOSAL_SCALE_FRACTION,
        return_sampler=False,
        return_walker_state=False,
    )
    logger.info(
        "Batched RV-only MCMC returned %d systems in %.2f s",
        len(systems),
        time.perf_counter() - start,
    )
    return rows, systems, sampler, walker_state


def materialize_rv_only_batch_result(batch_fit_result):
    rows, systems, sampler, walker_state = batch_fit_result
    results = []
    materialize_start = time.perf_counter()
    for (i, row), sls in zip(rows, systems):
        rvfit_ts = fitted_timeseries_row(i, row, truth_self_lensing_system(row), sls)
        new_ts = add_rv_only_columns(row, rvfit_ts)
        logger.info(
            "Materialized batched RV-only fit result for row index=%s: "
            "rvfit_fitted_a=%.6g rvfit_fitted_Macc=%.6g rvfit_fitted_e=%.6g",
            i,
            new_ts["rvfit_fitted_a"],
            new_ts["rvfit_fitted_Macc"],
            new_ts["rvfit_fitted_e"],
        )
        results.append((new_ts, sampler, walker_state, None))
    logger.info(
        "Materialized %d batched RV-only fit results in %.2f s",
        len(results),
        time.perf_counter() - materialize_start,
    )
    return results


def fit_rv_only_batch(rows, curves):
    return materialize_rv_only_batch_result(
        run_rv_only_batch(rows, curves),
    )


def batched_row_chunks(rows):
    if FIT_BATCH_SIZE is None or FIT_BATCH_SIZE >= len(rows):
        yield rows
        return

    for start in range(0, len(rows), FIT_BATCH_SIZE):
        yield rows[start : start + FIT_BATCH_SIZE]


def store_materialized_results(results, rows_by_key, materialized_results):
    if not materialized_results:
        return
    results.extend(materialized_results)
    for materialized_result in materialized_results:
        rows_by_key[result_checkpoint_key(materialized_result[0])] = (
            materialized_result[0]
        )


def make_materialize_executor():
    if use_process_materializer():
        materialize_jax_platforms = os.environ.get(
            "SLRV_MATERIALIZE_JAX_PLATFORMS",
            "cpu",
        )
        os.environ["JAX_PLATFORMS"] = materialize_jax_platforms
        logger.info(
            "Using one background process for batched result materialization "
            "with JAX_PLATFORMS=%s",
            materialize_jax_platforms,
        )
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context(MP_START_METHOD),
        )
    logger.info("Using one background thread for batched result materialization")
    return ThreadPoolExecutor(max_workers=1)


def use_process_materializer():
    return FIT_MATERIALIZE_IN_PROCESS and not FIT_SAVE_SAMPLERS and FIT_MAX_WORKERS == 1


@dataclass
class PendingRealisticFileResult:
    rows: list
    rows_by_key: dict
    results: list
    rv_only_results: list
    pending_joint_materialization: object
    pending_rv_only_materialization: object
    path: object
    sampler_writer: object
    close_sampler_writer: bool
    materialize_executor: object


def finish_realistic_file_result(pending_result):
    if pending_result.pending_joint_materialization is not None:
        store_materialized_results(
            pending_result.results,
            pending_result.rows_by_key,
            finish_pending_fit_results(pending_result.pending_joint_materialization),
        )
        pending_result.pending_joint_materialization = None
    if pending_result.pending_rv_only_materialization is not None:
        store_materialized_results(
            pending_result.rv_only_results,
            pending_result.rows_by_key,
            finish_pending_fit_results(pending_result.pending_rv_only_materialization),
        )
        pending_result.pending_rv_only_materialization = None
    if pending_result.materialize_executor is not None:
        pending_result.materialize_executor.shutdown(wait=True)
        pending_result.materialize_executor = None
    logger.info("Finished all RV-only fits for %s", pending_result.path)
    if pending_result.close_sampler_writer:
        pending_result.sampler_writer.close()
    logger.info("Finished grouped fit for %s", pending_result.path)
    updated_keys = {
        result_checkpoint_key(ts)
        for _, ts in pending_result.rows
        if row_needs_prefit(ts) or row_needs_joint_fit(ts) or row_needs_rv_only_fit(ts)
    }
    return [(pending_result.rows_by_key[key], None, None) for key in updated_keys]


def fit_realistic_file(rows, defer_final_materialization=False):
    rows = list(rows)
    if not rows:
        return []

    path = rows[0][1]["realistic_path"]
    logger.info("Fitting %d realistic rows from %s", len(rows), path)
    logger.info("Loading grouped realistic curves from %s", path)
    curves = load_realistic_timeseries_curves(rows)
    logger.info("Loaded %d grouped realistic curves from %s", len(curves), path)
    rows_by_key = {result_checkpoint_key(ts): pandas.Series(ts).copy() for _, ts in rows}
    rows_needing_prefit = [
        (i, ts) for i, ts in rows if row_needs_prefit(ts) or row_needs_joint_fit(ts)
    ]
    rows_needing_joint = [(i, ts) for i, ts in rows if row_needs_joint_fit(ts)]
    rows_needing_rv_only = [(i, ts) for i, ts in rows if row_needs_rv_only_fit(ts)]
    logger.info(
        "Grouped fit stage needs for %s: prefit=%d joint=%d rv_only=%d",
        path,
        len(rows_needing_prefit),
        len(rows_needing_joint),
        len(rows_needing_rv_only),
    )
    initial_walker_state = None
    prefit_sampler_state = None
    prefit_sampler_state_path = None
    prefit_ts = None
    sampler_writer = shared_sampler_writer()
    close_sampler_writer = False
    if sampler_writer is None:
        sampler_writer = SamplerStateWriter()
        close_sampler_writer = True
    if FIT_USE_LC and rows_needing_prefit:
        prefit_i, prefit_ts = rows_needing_prefit[0]
        prefit_curve = curves[str(prefit_ts["realistic_id"])]
        logger.info(
            "Prefitting photometry once for %d-row realistic group from %s using row index=%s",
            len(rows),
            path,
            prefit_i,
        )
        prefit_ts, initial_walker_state, prefit_sampler_state = fit_photometry_prefit(
            prefit_i, prefit_ts, prefit_curve
        )
        prefit_sampler_state_path = sampler_writer.submit_prefit(
            prefit_ts,
            prefit_sampler_state,
        )
        for key, row in list(rows_by_key.items()):
            if row_needs_prefit(row):
                rows_by_key[key] = add_prefit_columns(row, prefit_ts)
    results = []
    materialize_executor = None
    if (
        (rows_needing_joint and can_use_batched_joint_fit())
        or (rows_needing_rv_only and can_use_batched_rv_only_fit())
    ):
        materialize_executor = make_materialize_executor()
    pending_joint_materialization = None
    if rows_needing_joint and can_use_batched_joint_fit():
        batches = list(batched_row_chunks(rows_needing_joint))
        if len(batches) > 1:
            logger.info(
                "Splitting %d realistic rows into %d batched joint chunks of up to %d",
                len(rows_needing_joint),
                len(batches),
                FIT_BATCH_SIZE,
            )
        for batch_number, batch_rows in enumerate(batches, start=1):
            if len(batches) > 1:
                logger.info(
                    "Starting batched joint chunk %d/%d with %d rows from %s",
                    batch_number,
                    len(batches),
                    len(batch_rows),
                    path,
                )
            batch_fit_result = run_joint_batch(
                batch_rows,
                curves,
                initial_walker_state,
            )
            store_materialized_results(
                results,
                rows_by_key,
                finish_pending_fit_results(pending_joint_materialization),
            )
            if use_process_materializer():
                pending_joint_materialization = materialize_executor.submit(
                    materialize_joint_batch_result_in_process,
                    batch_fit_result,
                    prefit_sampler_state_path,
                    prefit_ts,
                )
            else:
                pending_joint_materialization = materialize_executor.submit(
                    materialize_joint_batch_result,
                    batch_fit_result,
                    sampler_writer,
                    prefit_sampler_state_path,
                    prefit_ts,
                )
        if not (rows_needing_rv_only and can_use_batched_rv_only_fit()):
            store_materialized_results(
                results,
                rows_by_key,
                finish_pending_fit_results(pending_joint_materialization),
            )
            pending_joint_materialization = None
    elif rows_needing_joint:
        file_progress_desc = (
            f"RV joint fits {short_id(rows[0][1].get('pristine_id', Path(path).stem))}"
        )
        row_iterator = tqdm(
            rows_needing_joint,
            total=len(rows_needing_joint),
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
                    len(rows_needing_joint),
                    ts.get("realistic_id", None),
                )
                curve = curves[str(ts["realistic_id"])]
                mcmc_desc = (
                    f"MCMC joint {row_number}/{len(rows_needing_joint)} "
                    f"{short_id(ts.get('realistic_id', None))}"
                )
                fit_result = run_fit((i, ts, curve, initial_walker_state, mcmc_desc))
                finished_fit = finish_pending_fit(pending_fit)
                if finished_fit is not None:
                    results.append(finished_fit)
                    rows_by_key[result_checkpoint_key(finished_fit[0])] = finished_fit[0]
                pending_fit = row_executor.submit(
                    materialize_fit_result,
                    fit_result,
                    sampler_writer,
                    prefit_sampler_state_path,
                    prefit_ts,
                )
            finished_fit = finish_pending_fit(pending_fit)
            if finished_fit is not None:
                results.append(finished_fit)
                rows_by_key[result_checkpoint_key(finished_fit[0])] = finished_fit[0]
    logger.info("Finished all joint fits for %s", path)
    rv_only_results = []
    pending_rv_only_materialization = None
    if rows_needing_rv_only:
        if can_use_batched_rv_only_fit():
            current_rv_only_rows = [
                (i, rows_by_key.get(result_checkpoint_key(ts), ts))
                for i, ts in rows_needing_rv_only
            ]
            batches = list(batched_row_chunks(current_rv_only_rows))
            if len(batches) > 1:
                logger.info(
                    "Splitting %d realistic rows into %d batched RV-only chunks of up to %d",
                    len(rows_needing_rv_only),
                    len(batches),
                    FIT_BATCH_SIZE,
                )
            for batch_number, batch_rows in enumerate(batches, start=1):
                if len(batches) > 1:
                    logger.info(
                        "Starting batched RV-only chunk %d/%d with %d rows from %s",
                        batch_number,
                        len(batches),
                        len(batch_rows),
                        path,
                    )
                batch_fit_result = run_rv_only_batch(batch_rows, curves)
                store_materialized_results(
                    rv_only_results,
                    rows_by_key,
                    finish_pending_fit_results(pending_rv_only_materialization),
                )
                if pending_joint_materialization is not None:
                    store_materialized_results(
                        results,
                        rows_by_key,
                        finish_pending_fit_results(pending_joint_materialization),
                    )
                    pending_joint_materialization = None
                materialize_rows = [
                    (i, rows_by_key.get(result_checkpoint_key(ts), ts))
                    for i, ts in batch_rows
                ]
                rv_only_materializer = (
                    materialize_rv_only_batch_result_in_process
                    if use_process_materializer()
                    else materialize_rv_only_batch_result
                )
                pending_rv_only_materialization = materialize_executor.submit(
                    rv_only_materializer,
                    (
                        materialize_rows,
                        batch_fit_result[1],
                        batch_fit_result[2],
                        batch_fit_result[3],
                    ),
                )
            if not defer_final_materialization:
                store_materialized_results(
                    rv_only_results,
                    rows_by_key,
                    finish_pending_fit_results(pending_rv_only_materialization),
                )
                pending_rv_only_materialization = None
        else:
            if pending_joint_materialization is not None:
                store_materialized_results(
                    results,
                    rows_by_key,
                    finish_pending_fit_results(pending_joint_materialization),
                )
                pending_joint_materialization = None
            file_progress_desc = (
                f"RV-only fits {short_id(rows[0][1].get('pristine_id', Path(path).stem))}"
            )
            for row_number, (i, ts) in enumerate(
                tqdm(
                    rows_needing_rv_only,
                    total=len(rows_needing_rv_only),
                    desc=file_progress_desc,
                    leave=False,
                    dynamic_ncols=True,
                ),
                start=1,
            ):
                key = result_checkpoint_key(ts)
                row = rows_by_key.get(key, ts)
                logger.info(
                    "Starting RV-only fit %d/%d for realistic_id=%s",
                    row_number,
                    len(rows_needing_rv_only),
                    row.get("realistic_id", None),
                )
                curve = curves[str(row["realistic_id"])]
                rv_fit_result = run_rv_only_fit(i, row, curve)
                rv_only_result = materialize_rv_only_fit_result(rv_fit_result, row)
                rv_only_results.append(rv_only_result)
                rows_by_key[result_checkpoint_key(rv_only_result[0])] = rv_only_result[0]
    pending_result = PendingRealisticFileResult(
        rows=rows,
        rows_by_key=rows_by_key,
        results=results,
        rv_only_results=rv_only_results,
        pending_joint_materialization=pending_joint_materialization,
        pending_rv_only_materialization=pending_rv_only_materialization,
        path=path,
        sampler_writer=sampler_writer,
        close_sampler_writer=close_sampler_writer,
        materialize_executor=materialize_executor,
    )
    if defer_final_materialization:
        logger.info("Deferring final materialization for %s", path)
        return pending_result
    return finish_realistic_file_result(pending_result)


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


def clear_existing_results():
    paths = [
        TIMESERIES_TABLE_PATH,
        Path(f"{TIMESERIES_TABLE_PATH}.new.gz"),
    ]
    for path in paths:
        if path.exists():
            logger.info("Removing saved fit results at %s", path)
            path.unlink()
    if SAMPLER_STATE_PATH.exists():
        logger.info("Removing saved sampler states at %s", SAMPLER_STATE_PATH)
        shutil.rmtree(SAMPLER_STATE_PATH)


def record_group_results(
    group_results,
    results,
    checkpoint_writer,
    new_result_count,
    last_checkpoint_result_count,
):
    checkpoint_writer.check_error()
    new_results = [result[0] for result in group_results]
    results += new_results
    new_result_count += len(new_results)
    logger.info(
        "Fit job produced %d fit results; cumulative new results=%d "
        "cumulative results=%d",
        len(new_results),
        new_result_count,
        len(results),
    )
    if (
        FIT_CHECKPOINT_INTERVAL > 0
        and new_result_count - last_checkpoint_result_count >= FIT_CHECKPOINT_INTERVAL
    ):
        logger.info(
            "Checkpointing fit results after %d new rows",
            new_result_count,
        )
        checkpoint_writer.submit(results)
        last_checkpoint_result_count = new_result_count
    return new_result_count, last_checkpoint_result_count


if __name__ == "__main__":
    args = parse_args()
    configure_logging()
    logger.info("Starting realistic timeseries fitting script")
    if args.clear_results:
        clear_existing_results()
    fit_sample_size_label = FIT_SAMPLE_SIZE if FIT_SAMPLE_SIZE is not None else "all"
    logger.info(
        "Starting fitting script with FIT_MAX_WORKERS=%d FIT_SAMPLE_SIZE=%s "
        "CHECKPOINT_RESULTS=%s FIT_CHECKPOINT_INTERVAL=%d",
        FIT_MAX_WORKERS,
        fit_sample_size_label,
        CHECKPOINT_RESULTS,
        FIT_CHECKPOINT_INTERVAL,
    )
    logger.info("Fit batch file: %s", FIT_BATCH_FILE)
    logger.info(
        "MCMC settings: FIT_NSTEPS=%d FIT_NWALKERS=%d FIT_BURN=%d "
        "FIT_PROGRESS_CHUNK_SIZE=%d FIT_BATCH_SIZE=%s "
        "FIT_PROPOSAL_SCALE_FRACTION=%g "
        "FIT_PREFIT_PROPOSAL_SCALE_FRACTION=%g MP_START_METHOD=%s",
        FIT_NSTEPS,
        FIT_NWALKERS,
        FIT_BURN,
        FIT_PROGRESS_CHUNK_SIZE,
        FIT_BATCH_SIZE if FIT_BATCH_SIZE is not None else "all",
        FIT_PROPOSAL_SCALE_FRACTION,
        FIT_PREFIT_PROPOSAL_SCALE_FRACTION,
        MP_START_METHOD,
    )
    logger.info("Fit eccentricity limits: %s", DEFAULT_LIMITS["e"])
    logger.info(
        "Fit likelihood settings: FIT_USE_LC=%s "
        "FIT_MCMC_PROGRESS=%s sampler=blackjax",
        FIT_USE_LC,
        FIT_MCMC_PROGRESS,
    )
    logger.info("Fit device setting: FIT_DEVICE=%s", FIT_DEVICE)
    logger.info("Fit lensing method: FIT_LENSING_METHOD=%s", FIT_LENSING_METHOD)
    logger.info(
        "Batched fit settings: FIT_BATCHED_JOINT=%s FIT_BATCHED_RV_ONLY=%s",
        FIT_BATCHED_JOINT,
        FIT_BATCHED_RV_ONLY,
    )
    logger.info(
        "Async materialization: FIT_MATERIALIZE_IN_PROCESS=%s",
        FIT_MATERIALIZE_IN_PROCESS,
    )
    logger.info(
        "Sampler state saving: FIT_SAVE_SAMPLERS=%s SAMPLER_STATE_PATH=%s",
        FIT_SAVE_SAMPLERS,
        SAMPLER_STATE_PATH,
    )
    logger.info(
        "JAX backend: default=%s devices=[%s]",
        jax.default_backend(),
        jax_device_summary(),
    )
    logger.info(
        "JAX compilation cache: %s max_size=%s bytes",
        os.environ["JAX_COMPILATION_CACHE_DIR"],
        os.environ["JAX_COMPILATION_CACHE_MAX_SIZE"],
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
    logger.info(
        "Loaded %d realistic rows and %d existing fitted rows",
        len(rts),
        len(existing_results),
    )

    rts = rows_requiring_fits(rts, existing_results)
    logger.info("Filtered realistic rows down to %d rows with missing fit stages", len(rts))

    if len(rts) == 0:
        logger.info("No realistic rows left with missing fit stages; exiting")
        raise SystemExit

    if FIT_BATCH_FILE:
        logger.info("Using all %d incomplete manifest rows for fitting", len(rts))
    elif FIT_SAMPLE_SIZE is None:
        logger.info("Using all %d incomplete rows for standalone fitting", len(rts))
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

    if FIT_MAX_WORKERS == 1:
        ACTIVE_SAMPLER_WRITER = SamplerStateWriter()
        logger.info("Using one shared sampler state writer across serial fit jobs")

    checkpoint_writer = AsyncCheckpointWriter(enabled=CHECKPOINT_RESULTS)
    new_result_count = 0
    last_checkpoint_result_count = 0
    pending_file_result = None
    try:
        if FIT_MAX_WORKERS == 1:
            logger.info(
                "Starting serial realistic fit loop with deferred file materialization"
            )
            for job in tqdm(jobs, total=len(jobs), desc="Realistic simulation fits"):
                current_file_result = fit_realistic_file(
                    job,
                    defer_final_materialization=True,
                )
                if pending_file_result is not None:
                    finishing_file_result = pending_file_result
                    pending_file_result = None
                    group_results = finish_realistic_file_result(finishing_file_result)
                    (
                        new_result_count,
                        last_checkpoint_result_count,
                    ) = record_group_results(
                        group_results,
                        results,
                        checkpoint_writer,
                        new_result_count,
                        last_checkpoint_result_count,
                    )
                pending_file_result = current_file_result
            if pending_file_result is not None:
                finishing_file_result = pending_file_result
                pending_file_result = None
                group_results = finish_realistic_file_result(finishing_file_result)
                (
                    new_result_count,
                    last_checkpoint_result_count,
                ) = record_group_results(
                    group_results,
                    results,
                    checkpoint_writer,
                    new_result_count,
                    last_checkpoint_result_count,
                )
        else:
            for group_results in process_map_iter(
                fit_realistic_file,
                jobs,
                total=len(jobs),
                desc="Realistic simulation fits",
                max_workers=FIT_MAX_WORKERS,
            ):
                (
                    new_result_count,
                    last_checkpoint_result_count,
                ) = record_group_results(
                    group_results,
                    results,
                    checkpoint_writer,
                    new_result_count,
                    last_checkpoint_result_count,
                )
        if ACTIVE_SAMPLER_WRITER is not None:
            ACTIVE_SAMPLER_WRITER.close()
            ACTIVE_SAMPLER_WRITER = None
        checkpoint_writer.close(final_results=results)
    except BaseException:
        if pending_file_result is not None:
            finishing_file_result = pending_file_result
            pending_file_result = None
            group_results = finish_realistic_file_result(finishing_file_result)
            (
                new_result_count,
                last_checkpoint_result_count,
            ) = record_group_results(
                group_results,
                results,
                checkpoint_writer,
                new_result_count,
                last_checkpoint_result_count,
            )
        if ACTIVE_SAMPLER_WRITER is not None:
            ACTIVE_SAMPLER_WRITER.close()
            ACTIVE_SAMPLER_WRITER = None
        checkpoint_writer.close(final_results=results)
        raise
    logger.info(
        "Fit jobs produced %d fit results; cumulative results=%d",
        new_result_count,
        len(results),
    )
    logger.info("Finished fitting script")
