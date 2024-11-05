import numpy

from pathlib import Path

PARAMS = {"cadences": numpy.arange(1, 100, 10)}

DATA_PATH = Path(__file__).parent / Path("../data/realistic_timeseries")
DATA_PATH.mkdir(exist_ok=True, parents=True)

TIMESERIES_TABLE_PATH = DATA_PATH / "realistic_timeseries.parquet"
