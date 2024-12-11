import numpy
import pandas

from pathlib import Path

PARAMS = {"cadences": numpy.arange(1, 100, 10)}

DATA_PATH = Path(__file__).parent / Path("../data/realistic_timeseries")
DATA_PATH.mkdir(exist_ok=True, parents=True)

TIMESERIES_TABLE_PATH = DATA_PATH / "realistic_rv_timeseries.parquet"
LOMBSCARGLE_TABLE_PATH = DATA_PATH / "realistic_rv_lombscargle.parquet"


def load_realistic_timeseries():
    return pandas.read_parquet(TIMESERIES_TABLE_PATH)


def load_realistic_lombscargle():
    return pandas.read_parquet(LOMBSCARGLE_TABLE_PATH)
