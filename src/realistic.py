import numpy
import pandas

from utils import DATA_PATH


PARAMS = {
    "cadences": [30, 150, 300],
    "campaign_baselines": 365 * numpy.array([1, 3, 5]),
    "rv_accuracies": numpy.logspace(-1, 3, 50),
}

TIMESERIES_TABLE_PATH = DATA_PATH / "realistic_timeseries.gz"


def load_realistic_timeseries():
    return pandas.read_pickle(TIMESERIES_TABLE_PATH)
