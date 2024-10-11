import numpy

from astropy import units
from astropy.time import Time
from tqdm import tqdm

from pathlib import Path

import concurrent.futures

from self_lensing import SelfLensingSystem, LensingEvent


DATA_PATH = Path("../data/pristine_timeseries")
DATA_PATH.mkdir(exist_ok=True, parents=True)

a_sample = numpy.linspace(0.1, 200, 1000)
b_sample = range(0, 10)
macc_sample = numpy.linspace(0.1, 20, 2000)

# We assume a Sun-like star. This shouldn't really matter for RV calculations
Mcomp = 1 * units.M_sun
Rcomp = 1 * units.R_sun
Teff = 5500 * units.K

# Pick an arbitrary start and end time
start_time = Time("2025-01-01")
end_time = Time("2026-01-01")

total = len(a_sample) * len(b_sample) * len(macc_sample)


def calc_rv(a, b, macc):
    sls = SelfLensingSystem(
        a=a * units.au,
        b=b * units.dimensionless_unscaled,
        Macc=macc * units.M_sun,
        Mcomp=Mcomp,
        Rcomp=Rcomp,
        Teff=Teff,
    )
    le = LensingEvent(sls, start_time)

    rv = le.rv_range(start_time, end_time)
    return numpy.concatenate([[a, b, macc], rv.value])


if __name__ == "__main__":
    with tqdm(total=total) as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = set()
            for a in a_sample:
                a_results = []
                for b in b_sample:
                    for macc in macc_sample:
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
