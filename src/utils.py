import logging
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", module="erfa.core")

from astropy import units
from astropy.time import Time
import numpy


def _env_flag(name):
    value = os.environ.get(name, "")
    return value.lower() not in ("", "0", "false", "no", "off")


def configure_logging():
    """Configure application logging from environment variables."""
    level = logging.DEBUG if _env_flag("DEBUG") else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger().setLevel(logging.WARNING)
    for logger_name in (
        "__main__",
        "__mp_main__",
        "utils",
        "pristine",
        "realistic",
        "spectrograph",
        "fitting",
    ):
        logging.getLogger(logger_name).setLevel(level)
    logging.getLogger(__name__).debug(
        "Logging configured at %s", logging.getLevelName(level)
    )


logger = logging.getLogger(__name__)
configure_logging()


DEFAULTS = {
    # We just fix impact param as it doesn't make enough difference to change it
    # "b": 0 * units.dimensionless_unscaled,
    # We assume a Sun-like star. This shouldn't really matter for RV calculations
    "Mcomp": 1 * units.M_sun,
    "Rcomp": 1 * units.R_sun,
    "Teff": 5500 * units.K,
    # Pick an arbitrary start and end time
    "rv_start_time": Time("2025-01-01"),
    "rv_end_time": Time("2030-01-01"),
    "lc_start_time": Time("2024-01-01"),
    "lc_end_time": Time("2025-01-01"),
    "peak_time": Time("2024-06-01"),
}


default_rv_times = numpy.linspace(
    DEFAULTS["rv_start_time"],
    DEFAULTS["rv_end_time"],
    int((DEFAULTS["rv_end_time"] - DEFAULTS["rv_start_time"]).value),
)

DATA_PATH = Path(__file__).parent / Path("../data/")
DATA_PATH.mkdir(exist_ok=True, parents=True)
logger.debug("Data path initialised at %s", DATA_PATH.resolve())
