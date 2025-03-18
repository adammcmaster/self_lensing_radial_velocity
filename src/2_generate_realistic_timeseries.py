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


def sample_timeseries(ts, step):
    idx = numpy.arange(0, len(ts), step)
    return idx.tolist(), numpy.array(ts).take(idx).tolist()


def gen_ts(i, ts, cadence, pristine_sample, pristine_idx, spectrograph):
    noisy_sample = spectrograph.add_rv_noise(pristine_sample)
    noisy_err = spectrograph.total_error(unit=pristine_sample.unit)
    new_ts = pandas.Series(ts)
    new_ts["rv"] = list(noisy_sample.value)
    new_ts["rv_err"] = noisy_err
    new_ts["rv_times"] = numpy.array(ts["rv_times"]).take(pristine_idx)
    new_ts["RV accuracy"] = spectrograph.RV_ACCURACY.to(units.m / units.s).value
    new_ts["cadence"] = cadence
    new_ts["pristine idx"] = i
    return new_ts


if __name__ == "__main__":
    pristine = load_pristine_timeseries()

    results = []
    total = len(PARAMS["cadences"]) * len(pristine) * len(Spectrograph.all)

    with tqdm(total=total) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = set()
            for cadence in PARAMS["cadences"]:
                for i, ts in pristine.iterrows():
                    pristine_idx, pristine_sample = sample_timeseries(ts["rv"], cadence)
                    pristine_sample *= units.m / units.s

                    for spectrograph in Spectrograph.all:
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
                                spectrograph,
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
