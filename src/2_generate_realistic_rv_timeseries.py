"""
This takes the pristine RV timeseries, samples them at various observing cadences and then adds noise.
"""

import numpy
import pandas

from astropy import units
from tqdm import tqdm

from pathlib import Path

from pristine import load_pristine_timeseries
from spectrograph import Spectrograph

from realistic import PARAMS, TIMESERIES_TABLE_PATH


def sample_timeseries(ts, step):
    idx = numpy.arange(0, len(ts), step)
    return idx.tolist(), numpy.array(ts).take(idx).tolist()


if __name__ == "__main__":
    pristine = load_pristine_timeseries()
    results = []

    total = len(PARAMS["cadences"]) * len(pristine) * len(Spectrograph.all)

    with tqdm(total=total) as pbar:
        for cadence in PARAMS["cadences"]:
            for i, ts in pristine.iterrows():
                pristine_idx, pristine_sample = sample_timeseries(ts["rv"], cadence)
                pristine_sample *= units.m / units.s

                for spectrograph in Spectrograph.all:
                    noisy_sample = spectrograph.add_rv_noise(pristine_sample)
                    noisy_err = spectrograph.total_error(unit=pristine_sample.unit)
                    new_ts = pandas.Series(ts)
                    new_ts["rv"] = list(noisy_sample.value)
                    new_ts["rv_err"] = noisy_err
                    new_ts["times"] = pristine_idx
                    new_ts["RV accuracy"] = spectrograph.RV_ACCURACY.to(
                        units.m / units.s
                    ).value
                    new_ts["cadence"] = cadence
                    new_ts["pristine idx"] = i
                    results.append(new_ts)
                    pbar.update(1)

    results = pandas.DataFrame(results)
    results.to_parquet(TIMESERIES_TABLE_PATH)
