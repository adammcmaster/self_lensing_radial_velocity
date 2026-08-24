import logging
import os
import atexit
from pathlib import Path

from filelock import FileLock, Timeout

from utils import DATA_PATH

logger = logging.getLogger(__name__)


TIMESERIES_TABLE_PATH = Path(
    os.environ.get("SLRV_FIT_OUTPUT_PATH", DATA_PATH / "fitted_timeseries.gz")
)
PREFIT_RESULTS_PATH = Path(
    os.environ.get("SLRV_PREFIT_OUTPUT_PATH", DATA_PATH / "photometry_prefits.gz")
)
PREFIT_POSTERIOR_PATH = Path(
    os.environ.get("SLRV_PREFIT_POSTERIOR_PATH", DATA_PATH / "photometry_posteriors")
)
SCHEDULE_TABLE_PATH = Path(
    os.environ.get("SLRV_SCHEDULE_OUTPUT_PATH", DATA_PATH / "rv_schedules.gz")
)
SAMPLER_STATE_PATH = Path(
    os.environ.get("SLRV_FIT_SAMPLER_STATE_PATH", DATA_PATH / "fit_sampler_states")
)
FIT_LOCK_PATH = Path(
    os.environ.get("SLRV_FIT_LOCK_PATH", f"{TIMESERIES_TABLE_PATH}.lock")
)
FIT_LOCK_TIMEOUT = float(os.environ.get("SLRV_FIT_LOCK_TIMEOUT", "0"))


def acquire_fit_results_lock(owner):
    FIT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(FIT_LOCK_PATH)
    try:
        lock.acquire(timeout=FIT_LOCK_TIMEOUT)
    except Timeout as exc:
        raise RuntimeError(
            f"{owner} could not acquire fit-results lock {FIT_LOCK_PATH}. "
            "Another fitting script is probably running against the same results "
            "table. Stop that process, or set SLRV_FIT_LOCK_TIMEOUT to wait."
        ) from exc
    logger.info("%s acquired fit-results lock %s", owner, FIT_LOCK_PATH)
    atexit.register(lock.release)
    return lock
