"""
This generates a set of RV timeseries with no errors and a uniform observing cadence.

These timeseries can then form the basis of more realistic ones generated later.
"""

import pandas

from tqdm.contrib.concurrent import process_map

from pristine import TIMESERIES_TABLE_PATH, PARAMS, calc_rv


if __name__ == "__main__":
    params = []
    for a in PARAMS["a_sample"]:
        for b in PARAMS["b_sample"]:
            for e in PARAMS["e_sample"]:
                for periapsis_phase in PARAMS["periapsis_phase_sample"]:
                    for macc in PARAMS["macc_sample"]:
                        params.append((a, b, e, periapsis_phase, macc))

    results = process_map(calc_rv, params)

    results = pandas.DataFrame(results)
    results.to_parquet(TIMESERIES_TABLE_PATH)
