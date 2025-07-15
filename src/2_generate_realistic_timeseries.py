"""
This takes the pristine RV timeseries, samples them at various observing cadences and then adds noise.
"""

import numpy
import pandas
import random

from astropy import units
from tqdm.contrib.concurrent import process_map

from pristine import load_pristine_timeseries
from spectrograph import Spectrograph

from realistic import PARAMS, TIMESERIES_TABLE_PATH


def sample_timeseries(ts, step, max_len):
    if max_len > len(ts):
        max_len = len(ts)
    idx = numpy.arange(0, max_len, step)
    return idx.tolist(), numpy.array(ts).take(idx).tolist()


def gen_ts(params):
    i, ts, cadence, rv_accuracy, baseline = params
    pristine_idx, pristine_sample = sample_timeseries(ts["rv"], cadence, baseline)
    pristine_sample *= units.m / units.s
    noisy_sample = Spectrograph.add_rv_noise(
        pristine_sample, rv_accuracy * (units.m / units.s)
    )
    noisy_err = Spectrograph.total_error(
        unit=pristine_sample.unit, accuracy=rv_accuracy * (units.m / units.s)
    )
    new_ts = pandas.Series(ts)
    # new_ts["lc"] += numpy.random.normal(0, PARAMS["lc_noise"], len(new_ts["lc"]))
    new_ts["rv"] = list(noisy_sample.value)
    new_ts["rv_err"] = noisy_err
    new_ts["rv_times"] = list(numpy.array(ts["rv_times"]).take(pristine_idx))
    new_ts["RV accuracy"] = rv_accuracy
    new_ts["cadence"] = cadence
    new_ts["baseline"] = baseline
    new_ts["pristine idx"] = i
    return new_ts


if __name__ == "__main__":
    NUM_SAMPLES = 100000
    pristine = load_pristine_timeseries()

    params = []

    for baseline in PARAMS["campaign_baselines"]:
        for cadence in PARAMS["cadences"]:
            for i, ts in pristine.iterrows():
                for rv_accuracy in PARAMS["rv_accuracies"]:
                    params.append(
                        (
                            i,
                            ts,
                            cadence,
                            rv_accuracy,
                            baseline,
                        )
                    )
    if len(params) > NUM_SAMPLES:
        params = random.sample(params, NUM_SAMPLES)
    results = process_map(gen_ts, params)
    results = pandas.DataFrame(results)
    results.reset_index(inplace=True, drop=True)
    results.to_pickle(TIMESERIES_TABLE_PATH)
