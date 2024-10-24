"""
This generates a set of RV timeseries with no errors and a uniform observing cadence.

These timeseries can then form the basis of more realistic ones generated later.
"""

import concurrent.futures
import pandas

from tqdm import tqdm

from pristine import TIMESERIES_TABLE_PATH, PARAMS, calc_rv


if __name__ == "__main__":
    results = []
    with tqdm(total=PARAMS["total"]) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = set()
            for a in PARAMS["a_sample"]:
                for e in PARAMS["e_sample"]:
                    for periapsis_phase in PARAMS["periapsis_phase_sample"]:
                        for macc in PARAMS["macc_sample"]:
                            if len(futures) >= 10:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    results.append(f.result())
                            futures.add(
                                executor.submit(calc_rv, a, e, periapsis_phase, macc)
                            )
                            pbar.update(1)
                for f in futures:
                    results.append(f.result())
            results = pandas.DataFrame(results)
            results.to_parquet(TIMESERIES_TABLE_PATH, compression="gzip")
