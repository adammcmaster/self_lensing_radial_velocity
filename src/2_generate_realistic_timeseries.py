"""
This takes the pristine RV timeseries, samples them at various observing cadences and then adds noise.
"""

import logging
import os
import multiprocessing
import numpy
import pandas
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("SLRV_JAX_PLATFORMS", "cpu"))

from astropy import units
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from pristine import (
    PRISTINE_METADATA_PATH,
    generate_pristine_timeseries,
    load_pristine_lightcurve,
    load_pristine_rv_curve,
    load_pristine_timeseries,
)
from spectrograph import Spectrograph

from realistic import (
    PARAMS,
    existing_realistic_keys,
    metadata_row_from_params,
    realistic_id_from_params,
    realistic_key_from_values,
    realistic_path_from_params,
    update_realistic_metadata,
    write_realistic_timeseries_file,
)
from utils import configure_logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="erfa.core")

PARAM_MAX_WORKERS = int(
    os.environ.get("SLRV_GENERATE_PARAM_MAX_WORKERS", os.cpu_count() or 1)
)
PARAM_CHUNKSIZE = int(os.environ.get("SLRV_GENERATE_PARAM_CHUNKSIZE", 32))
REALISTIC_MAX_WORKERS = int(
    os.environ.get("SLRV_GENERATE_REALISTIC_MAX_WORKERS", os.cpu_count() or 1)
)
VECTOR_BATCH_SIZE = int(os.environ.get("SLRV_GENERATE_VECTOR_BATCH_SIZE", 50))
MP_START_METHOD = os.environ.get("SLRV_MP_START_METHOD", "fork")
GENERATE_BATCH_FILE = os.environ.get("SLRV_GENERATE_BATCH_FILE")
GENERATE_METADATA_OUTPUT_PATH = os.environ.get("SLRV_GENERATE_METADATA_OUTPUT_PATH")
SKIP_PRISTINE_GENERATION = os.environ.get("SLRV_SKIP_PRISTINE_GENERATION", "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


_WORKER_EXISTING_KEYS = None


def _init_param_worker(existing_keys):
    global _WORKER_EXISTING_KEYS
    _WORKER_EXISTING_KEYS = existing_keys


def missing_params_for_pristine(row):
    i, pristine_id = row
    missing = []
    for baseline in PARAMS["campaign_baselines"]:
        for cadence in PARAMS["cadences"]:
            for rv_accuracy in PARAMS["rv_accuracies"]:
                key = realistic_key_from_values(
                    pristine_id,
                    cadence,
                    rv_accuracy,
                    baseline,
                )
                if key not in _WORKER_EXISTING_KEYS:
                    missing.append(
                        (
                            i,
                            cadence,
                            rv_accuracy,
                            baseline,
                        )
                    )
    return missing


def generate_missing_params(pristine, existing_keys):
    rows = [
        (i, row["simulation_id"])
        for i, (_, row) in enumerate(pristine.iterrows())
    ]
    if not rows:
        return []

    logger.debug(
        "Generating realistic sample params with %d workers, chunksize=%d",
        PARAM_MAX_WORKERS,
        PARAM_CHUNKSIZE,
    )
    if PARAM_MAX_WORKERS == 1:
        _init_param_worker(existing_keys)
        results = [
            missing_params_for_pristine(row)
            for row in tqdm(rows, desc="[2/4] Generating sample params")
        ]
    else:
        context = multiprocessing.get_context(MP_START_METHOD)
        with ProcessPoolExecutor(
            max_workers=PARAM_MAX_WORKERS,
            mp_context=context,
            initializer=_init_param_worker,
            initargs=(existing_keys,),
        ) as executor:
            results = list(
                tqdm(
                    executor.map(
                        missing_params_for_pristine,
                        rows,
                        chunksize=PARAM_CHUNKSIZE,
                    ),
                    total=len(rows),
                    desc="[2/4] Generating sample params",
                )
            )

    return [param for row_params in results for param in row_params]


def sample_indices(ts, step, max_len):
    step = int(step)
    max_len = min(int(max_len), len(ts))
    return numpy.arange(0, max_len, step, dtype=int)


def grouped_params_by_sampling(pristine_params):
    grouped = {}
    for cadence, rv_accuracy, baseline in pristine_params:
        grouped.setdefault((int(cadence), int(baseline)), []).append(float(rv_accuracy))
    return grouped


def curve_from_arrays(
    i,
    ts,
    cadence,
    rv_accuracy,
    baseline,
    rv_times,
    noisy_rv,
    rv_err,
    lc_times,
    lc,
    lc_err,
):
    realistic_id = realistic_id_from_params(ts, cadence, rv_accuracy, baseline)
    realistic_path = realistic_path_from_params(ts, cadence, rv_accuracy, baseline)
    row = metadata_row_from_params(
        realistic_id=realistic_id,
        realistic_path=realistic_path,
        pristine=ts,
        rv_accuracy=rv_accuracy,
        cadence=cadence,
        baseline=baseline,
    )
    return {
        "row": row,
        "rv_times": rv_times,
        "rv": noisy_rv,
        "rv_err": rv_err,
        "lc_times": lc_times,
        "lc": lc,
        "lc_err": lc_err,
    }


def realistic_curves_for_sampling(
    i,
    ts,
    cadence,
    baseline,
    rv_accuracies,
    pristine_rv,
    pristine_lc,
):
    indices = sample_indices(pristine_rv["rv"], cadence, baseline)
    rv_times = numpy.asarray(pristine_rv["rv_times"], dtype=float)[indices]
    pristine_sample = numpy.asarray(pristine_rv["rv"], dtype=float)[indices]
    lc_times = numpy.asarray(pristine_lc["lc_times"], dtype=float)
    lc = numpy.asarray(pristine_lc["lc"], dtype=float)
    lc_err = numpy.asarray(pristine_lc["lc_err"], dtype=float)
    stellar_noise = Spectrograph.STELLAR_NOISE.to_value(units.m / units.s)
    curves = []

    for start in range(0, len(rv_accuracies), VECTOR_BATCH_SIZE):
        accuracy_batch = numpy.asarray(
            rv_accuracies[start : start + VECTOR_BATCH_SIZE],
            dtype=float,
        )
        rv_noise = numpy.random.normal(
            0,
            stellar_noise,
            (len(accuracy_batch), len(pristine_sample)),
        )
        rv_noise += numpy.random.normal(
            0,
            accuracy_batch[:, None],
            (len(accuracy_batch), len(pristine_sample)),
        )
        noisy_rvs = pristine_sample[None, :] + rv_noise
        rv_errs = numpy.sqrt(stellar_noise**2 + accuracy_batch**2)

        for rv_accuracy, noisy_rv, rv_err in zip(
            accuracy_batch,
            noisy_rvs,
            rv_errs,
        ):
            curves.append(
                curve_from_arrays(
                    i,
                    ts,
                    cadence,
                    rv_accuracy,
                    baseline,
                    rv_times,
                    noisy_rv,
                    float(rv_err),
                    lc_times,
                    lc,
                    lc_err,
                )
            )

    return curves


def gen_pristine_ts(params):
    pristine_row_index, ts, pristine_params = params
    logger.debug(
        "Generating %d realistic timeseries for pristine row index=%s simulation_id=%s",
        len(pristine_params),
        pristine_row_index,
        ts["simulation_id"],
    )
    pristine_rv = load_pristine_rv_curve(ts)
    pristine_lc = load_pristine_lightcurve(ts)
    logger.debug(
        "Loaded pristine inputs for simulation_id=%s: %d RV samples, %d LC samples",
        ts["simulation_id"],
        len(pristine_rv["rv"]),
        len(pristine_lc["lc"]),
    )

    curves = []
    for (cadence, baseline), rv_accuracies in grouped_params_by_sampling(
        pristine_params
    ).items():
        curves.extend(
            realistic_curves_for_sampling(
                pristine_row_index,
                ts,
                cadence,
                baseline,
                rv_accuracies,
                pristine_rv,
                pristine_lc,
            )
        )

    if curves:
        write_realistic_timeseries_file(
            curves, realistic_path_from_params(ts, *pristine_params[0])
        )
    return [curve["row"] for curve in curves]


def group_params_by_pristine(params, pristine):
    grouped = {}
    for i, cadence, rv_accuracy, baseline in params:
        grouped.setdefault(i, []).append((cadence, rv_accuracy, baseline))
    groups = []
    for i, pristine_params in grouped.items():
        pristine_row = pristine.iloc[i]
        pristine_row_index = int(pristine_row.get("_pristine_row_index", i))
        groups.append((pristine_row_index, pristine_row, pristine_params))
    return groups


def filter_pristine_to_batch(pristine, path):
    batch = pandas.read_csv(path)
    if "simulation_id" not in batch.columns:
        raise ValueError(f"Batch file {path} must contain a simulation_id column")
    simulation_ids = set(batch["simulation_id"].dropna().astype(str))
    if not simulation_ids:
        return pristine.iloc[0:0].copy()
    return pristine[pristine["simulation_id"].astype(str).isin(simulation_ids)].copy()


def load_pristine_for_generation():
    if GENERATE_BATCH_FILE and PRISTINE_METADATA_PATH.exists():
        pristine = pandas.read_csv(PRISTINE_METADATA_PATH)
        pristine["_pristine_row_index"] = numpy.arange(len(pristine))
        return filter_pristine_to_batch(pristine, GENERATE_BATCH_FILE)

    pristine = load_pristine_timeseries()
    pristine["_pristine_row_index"] = numpy.arange(len(pristine))
    if GENERATE_BATCH_FILE:
        pristine = filter_pristine_to_batch(pristine, GENERATE_BATCH_FILE)
    return pristine


def write_metadata_shard(rows, path):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    table = pandas.DataFrame(rows)
    tmp_path = Path(f"{path}.new")
    table.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    configure_logging()
    logger.debug("Starting realistic timeseries generation script")
    logger.debug("Generate batch file: %s", GENERATE_BATCH_FILE)
    logger.debug("Generate metadata output path: %s", GENERATE_METADATA_OUTPUT_PATH)
    logger.debug("Skip pristine generation: %s", SKIP_PRISTINE_GENERATION)
    # NUM_SAMPLES = 100000
    # logger.debug("Configured NUM_SAMPLES=%d", NUM_SAMPLES)
    if not SKIP_PRISTINE_GENERATION:
        generate_pristine_timeseries(desc="[1/4] Generating timeseries")
    pristine = load_pristine_for_generation()
    if GENERATE_BATCH_FILE:
        print(
            f"Loaded {len(pristine)} pristine rows from batch file {GENERATE_BATCH_FILE}",
            flush=True,
        )
        logger.debug("Filtered pristine rows down to %d batch rows", len(pristine))

    existing_keys = existing_realistic_keys()
    logger.debug(
        "Loaded %d pristine rows and %d existing realistic keys",
        len(pristine),
        len(existing_keys),
    )

    params = generate_missing_params(pristine, existing_keys)
    print(
        f"Generated {len(params)} missing realistic parameter combinations "
        f"for {len(pristine)} pristine rows",
        flush=True,
    )
    logger.debug("Generated %d candidate realistic parameter combinations", len(params))
    # if len(params) > NUM_SAMPLES:
    #     params = random.sample(params, NUM_SAMPLES)
    #     logger.debug("Sampled candidate parameter combinations down to %d", len(params))
    params = group_params_by_pristine(params, pristine)
    logger.debug(
        "Grouped candidate combinations into %d pristine FITS files", len(params)
    )
    del pristine
    if not params:
        logger.debug("No realistic parameter combinations to generate")
        if GENERATE_METADATA_OUTPUT_PATH:
            write_metadata_shard([], GENERATE_METADATA_OUTPUT_PATH)
        else:
            update_realistic_metadata([])
    else:
        grouped_results = process_map(
            gen_pristine_ts,
            params,
            desc="[4/4] Sampling timeseries",
            max_workers=REALISTIC_MAX_WORKERS,
        )
        results = [row for rows in grouped_results for row in rows]
        logger.debug("Generated %d realistic metadata rows from workers", len(results))
        if GENERATE_METADATA_OUTPUT_PATH:
            write_metadata_shard(results, GENERATE_METADATA_OUTPUT_PATH)
        else:
            update_realistic_metadata(results)
    logger.debug("Finished realistic timeseries generation script")
