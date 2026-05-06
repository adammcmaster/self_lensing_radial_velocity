"""
This takes the pristine RV timeseries, samples them at various observing cadences and then adds noise.
"""

import logging
import numpy
import random
import warnings

from astropy import units
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from pristine import (
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
    metadata_row_from_realistic_path,
    realistic_id_from_params,
    realistic_key_from_values,
    realistic_path_from_params,
    update_realistic_metadata,
    write_realistic_timeseries,
)
from utils import configure_logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="erfa.core")


def sample_timeseries(ts, step, max_len):
    step = int(step)
    max_len = int(max_len)
    logger.debug("Sampling timeseries with len=%d step=%d max_len=%d", len(ts), step, max_len)
    if max_len > len(ts):
        max_len = len(ts)
    idx = numpy.arange(0, max_len, step)
    logger.debug("Sampled %d indices from timeseries", len(idx))
    return idx.tolist(), numpy.array(ts).take(idx).tolist()


def gen_ts(params):
    i, cadence, rv_accuracy, baseline, ts = params
    logger.debug(
        "Generating realistic timeseries for pristine_idx=%s cadence=%s rv_accuracy=%s baseline=%s",
        i,
        cadence,
        rv_accuracy,
        baseline,
    )
    realistic_id = realistic_id_from_params(ts, cadence, rv_accuracy, baseline)
    realistic_path = realistic_path_from_params(ts, cadence, rv_accuracy, baseline)
    if realistic_path.exists():
        logger.debug("Realistic timeseries already exists at %s", realistic_path)
        return metadata_row_from_realistic_path(realistic_path)

    pristine_rv = load_pristine_rv_curve(ts)
    pristine_lc = load_pristine_lightcurve(ts)
    logger.debug(
        "Loaded pristine inputs for realistic_id=%s: %d RV samples, %d LC samples",
        realistic_id,
        len(pristine_rv["rv"]),
        len(pristine_lc["lc"]),
    )

    sample_idx, pristine_sample = sample_timeseries(
        pristine_rv["rv"], cadence, baseline
    )
    pristine_sample *= units.m / units.s
    noisy_sample = Spectrograph.add_rv_noise(
        pristine_sample, rv_accuracy * (units.m / units.s)
    )
    noisy_err = Spectrograph.total_error(
        unit=pristine_sample.unit, accuracy=rv_accuracy * (units.m / units.s)
    )
    lc = pristine_lc["lc"]

    lc_noise = numpy.abs(numpy.median(lc[:5] - numpy.max(lc))) / 20
    noisy_lc = lc + numpy.random.normal(0, lc_noise, len(lc))
    rv_times = numpy.array(pristine_rv["rv_times"]).take(sample_idx)
    logger.debug(
        "Noise model for realistic_id=%s: rv_err=%s lc_noise=%.6g sampled_rv=%d",
        realistic_id,
        noisy_err,
        lc_noise,
        len(rv_times),
    )

    row = metadata_row_from_params(
        realistic_id=realistic_id,
        realistic_path=realistic_path,
        pristine_idx=i,
        pristine=ts,
        rv_accuracy=rv_accuracy,
        cadence=cadence,
        baseline=baseline,
        rv_err=noisy_err,
        lc_err=lc_noise,
    )
    write_realistic_timeseries(
        row=row,
        rv_times=rv_times,
        rv=noisy_sample.to_value(units.m / units.s),
        rv_err=noisy_err,
        lc_times=pristine_lc["lc_times"],
        lc=noisy_lc,
        lc_err=lc_noise,
    )
    logger.debug("Finished realistic timeseries generation for realistic_id=%s", realistic_id)
    return row


if __name__ == "__main__":
    configure_logging()
    logger.debug("Starting realistic timeseries generation script")
    NUM_SAMPLES = 100000
    logger.debug("Configured NUM_SAMPLES=%d", NUM_SAMPLES)
    generate_pristine_timeseries(desc="[1/4] Generating timeseries")
    pristine = load_pristine_timeseries()
    existing_keys = existing_realistic_keys()
    logger.debug(
        "Loaded %d pristine rows and %d existing realistic keys",
        len(pristine),
        len(existing_keys),
    )

    params = []

    for i in tqdm(range(len(pristine)), desc="[2/4] Generating sample params"):
        for baseline in PARAMS["campaign_baselines"]:
            for cadence in PARAMS["cadences"]:
                for rv_accuracy in PARAMS["rv_accuracies"]:
                    key = realistic_key_from_values(
                        pristine.iloc[i]["simulation_id"],
                        cadence,
                        rv_accuracy,
                        baseline,
                    )
                    if key not in existing_keys:
                        params.append(
                            (
                                i,
                                cadence,
                                rv_accuracy,
                                baseline,
                            )
                        )
    logger.debug("Generated %d candidate realistic parameter combinations", len(params))
    if len(params) > NUM_SAMPLES:
        params = random.sample(params, NUM_SAMPLES)
        logger.debug("Sampled candidate parameter combinations down to %d", len(params))
    params = [
        p + (pristine.iloc[p[0]],)
        for p in tqdm(params, desc="[3/4] Adding timeseries to params")
    ]
    del pristine
    if not params:
        logger.debug("No realistic parameter combinations to generate")
        update_realistic_metadata([])
    else:
        results = process_map(gen_ts, params, desc="[4/4] Sampling timeseries")
        logger.debug("Generated %d realistic metadata rows from workers", len(results))
        update_realistic_metadata(results)
    logger.debug("Finished realistic timeseries generation script")
