"""
This takes the pristine RV timeseries, samples them at various observing cadences and then adds noise.
"""

import numpy
import pandas

import concurrent.futures

from astropy import units
from tqdm import tqdm

from pristine import load_pristine_timeseries
from spectrograph import Spectrograph

from realistic import PARAMS, TIMESERIES_TABLE_PATH

from utils import PROC_LIMIT


def sample_timeseries(ts, step, max_len):
    if max_len > len(ts):
        max_len = len(ts)
    idx = numpy.arange(0, max_len, step)
    return idx.tolist(), numpy.array(ts).take(idx).tolist()


def gen_ts(i, ts, cadence, pristine_sample, pristine_idx, rv_accuracy, baseline):
    noisy_sample = Spectrograph.add_rv_noise(
        pristine_sample, rv_accuracy * (units.m / units.s)
    )
    noisy_err = Spectrograph.total_error(
        unit=pristine_sample.unit, accuracy=rv_accuracy * (units.m / units.s)
    )
    new_ts = pandas.Series(ts)
    new_ts["rv"] = list(noisy_sample.value)
    new_ts["rv_err"] = noisy_err
    new_ts["rv_times"] = list(numpy.array(ts["rv_times"]).take(pristine_idx))
    new_ts["RV accuracy"] = rv_accuracy
    new_ts["cadence"] = cadence
    new_ts["baseline"] = baseline
    new_ts["pristine idx"] = i
    return new_ts


if __name__ == "__main__":
    pristine = load_pristine_timeseries()

    results = []
    total = (
        len(PARAMS["cadences"])
        * len(PARAMS["campaign_baselines"])
        * len(pristine)
        * len(PARAMS["rv_accuracies"])
    )

    with tqdm(total=total) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            futures = set()
            for baseline in PARAMS["campaign_baselines"]:
                for cadence in PARAMS["cadences"]:
                    for i, ts in pristine.iterrows():
                        pristine_idx, pristine_sample = sample_timeseries(
                            ts["rv"], cadence, baseline
                        )
                        pristine_sample *= units.m / units.s

                        for rv_accuracy in PARAMS["rv_accuracies"]:
                            if len(futures) >= PROC_LIMIT:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    res = f.result()
                                    if res is not None:
                                        results.append(res)
                            futures.add(
                                executor.submit(
                                    gen_ts,
                                    i,
                                    ts,
                                    cadence,
                                    pristine_sample,
                                    pristine_idx,
                                    rv_accuracy,
                                    baseline,
                                )
                            )
                            pbar.update(1)
            for f in futures:
                res = f.result()
                if res is not None:
                    results.append(res)

    results = pandas.DataFrame(results)
    results.reset_index(inplace=True, drop=True)
    results.to_parquet(TIMESERIES_TABLE_PATH)
