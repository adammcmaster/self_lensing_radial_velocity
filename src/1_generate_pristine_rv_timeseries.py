import numpy

from astropy import units
from astropy.time import Time
from tqdm import tqdm

from pathlib import Path

import concurrent.futures

from pristine import DATA_PATH, PARAMS, gen_sls
from self_lensing import LensingEvent


def calc_rv(a, b, macc):
    sls = gen_sls(a, b, macc)
    le = LensingEvent(sls, PARAMS["start_time"])

    rv = le.rv_range(PARAMS["start_time"], PARAMS["end_time"])
    return numpy.concatenate([[a, b, macc], rv.value])


if __name__ == "__main__":
    with tqdm(total=PARAMS["total"]) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = set()
            for a in PARAMS["a_sample"]:
                a_results = []
                for b in PARAMS["b_sample"]:
                    for macc in PARAMS["macc_sample"]:
                        try:
                            if len(futures) >= 10:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    a_results.append(f.result())
                            futures.add(executor.submit(calc_rv, a, b, macc))
                        finally:
                            pbar.update(1)
                for f in futures:
                    a_results.append(f.result())
                a_results = numpy.array(a_results)
                numpy.savez_compressed(DATA_PATH / f"{a}.npz", a_results)
