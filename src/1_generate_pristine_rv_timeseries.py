import numpy

from tqdm import tqdm

import concurrent.futures

from pristine import DATA_PATH, PARAMS, gen_sls, calc_rv


if __name__ == "__main__":
    with tqdm(total=PARAMS["total"]) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = set()
            for a in PARAMS["a_sample"]:
                a_results = []
                for e in PARAMS["e_sample"]:
                    for periapsis_phase in PARAMS["periapsis_phase_sample"]:
                        for macc in PARAMS["macc_sample"]:
                            if len(futures) >= 10:
                                completed, futures = concurrent.futures.wait(
                                    futures,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                for f in completed:
                                    a_results.append(f.result())
                            futures.add(
                                executor.submit(calc_rv, a, e, periapsis_phase, macc)
                            )
                            pbar.update(1)
                for f in futures:
                    a_results.append(f.result())
                a_results = numpy.array(a_results)
                numpy.savez_compressed(DATA_PATH / f"{a}.npz", a_results)
