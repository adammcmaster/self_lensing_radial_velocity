import logging
import os
from pathlib import Path

from utils import DATA_PATH

logger = logging.getLogger(__name__)


TIMESERIES_TABLE_PATH = Path(
    os.environ.get("SLRV_FIT_OUTPUT_PATH", DATA_PATH / "fitted_timeseries.gz")
)
SAMPLER_STATE_PATH = Path(
    os.environ.get("SLRV_FIT_SAMPLER_STATE_PATH", DATA_PATH / "fit_sampler_states")
)
