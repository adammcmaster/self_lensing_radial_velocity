"""
This takes the pristine RV timeseries, samples them at various observing cadences and then adds noise.
"""

import numpy
import pandas
import random

from astropy import units
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from pristine import load_pristine_timeseries, generate_pristine_timeseries
from spectrograph import Spectrograph

from realistic import PARAMS, TIMESERIES_TABLE_PATH


def sample_timeseries(ts, step, max_len):
    step = int(step)
    max_len = int(max_len)
    if max_len > len(ts):
        max_len = len(ts)
    idx = numpy.arange(0, max_len, step)
    # print(ts)
    return idx.tolist(), numpy.array(ts).take(idx).tolist()


def gen_ts(params):
    i, cadence, rv_accuracy, baseline, ts = params
    pristine_idx, pristine_sample = sample_timeseries(ts["rv"], cadence, baseline)
    pristine_sample *= units.m / units.s
    noisy_sample = Spectrograph.add_rv_noise(
        pristine_sample, rv_accuracy * (units.m / units.s)
    )
    noisy_err = Spectrograph.total_error(
        unit=pristine_sample.unit, accuracy=rv_accuracy * (units.m / units.s)
    )
    new_ts = pandas.Series(ts)

    # lc_cadence = new_ts["tau_eff"] // 2
    # if lc_cadence < 1:
    #     lc_cadence = 1
    # lc_len = new_ts["Porb"] * 2
    # if lc_len < 10 * lc_cadence:
    #     lc_len = 10 * lc_cadence
    # if lc_len > 365:
    #     lc_len = 365
    # lc_idx, lc_sample = sample_timeseries(new_ts["lc"], lc_cadence, lc_len)
    # new_ts["lc"] = lc_sample
    # Add Gaussian noise with sigma=5% of magnification
    lc_noise = numpy.abs(numpy.median(new_ts["lc"][:5] - numpy.max(new_ts["lc"]))) / 20
    new_ts["lc"] += numpy.random.normal(0, lc_noise, len(new_ts["lc"]))
    new_ts["lc_err"] = lc_noise
    # new_ts["lc_times"] = list(numpy.array(ts["lc_times"]).take(lc_idx))

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
    pristine = generate_pristine_timeseries(desc="[1/4] Generating timeseries")

    params = []

    for i in tqdm(range(len(pristine)), desc="[2/4] Generating sample params"):
        for baseline in PARAMS["campaign_baselines"]:
            for cadence in PARAMS["cadences"]:
                for rv_accuracy in PARAMS["rv_accuracies"]:
                    params.append(
                        (
                            i,
                            # ts,
                            cadence,
                            rv_accuracy,
                            baseline,
                        )
                    )
    if len(params) > NUM_SAMPLES:
        params = random.sample(params, NUM_SAMPLES)
    params = [
        p + (pristine.iloc[p[0]],)
        for p in tqdm(params, desc="[3/4] Adding timeseries to params")
    ]
    del pristine
    results = process_map(gen_ts, params, desc="[4/4] Sampling timeseries")
    results = pandas.DataFrame(results)
    results.reset_index(inplace=True, drop=True)
    results.to_pickle(TIMESERIES_TABLE_PATH)
