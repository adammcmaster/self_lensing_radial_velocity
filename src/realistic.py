import numpy
import pandas

from pathlib import Path


PARAMS = {
    "cadences": [1, 30, 60, 120, 150, 300],
    "campaign_baselines": 365 * numpy.array([1, 3, 5]),
    "rv_accuracies": numpy.concatenate(
        [
            numpy.linspace(0.1, 5, 5),
            numpy.linspace(5, 6000, 5),
        ]
    ),
}

DATA_PATH = Path(__file__).parent / Path("../data/")
DATA_PATH.mkdir(exist_ok=True, parents=True)

TIMESERIES_TABLE_PATH = DATA_PATH / "realistic_timeseries.parquet"
LOMBSCARGLE_TABLE_PATH = DATA_PATH / "realistic_rv_lombscargle.parquet"


def load_realistic_timeseries():
    return pandas.read_parquet(TIMESERIES_TABLE_PATH)


def load_realistic_lombscargle():
    return pandas.read_parquet(LOMBSCARGLE_TABLE_PATH)
