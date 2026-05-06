"""
This generates a set of RV timeseries with no errors and a uniform observing cadence.

These timeseries can then form the basis of more realistic ones generated later.
"""

import logging

from pristine import generate_pristine_timeseries
from utils import configure_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    configure_logging()
    logger.debug("Starting pristine timeseries generation script")
    generate_pristine_timeseries()
    logger.debug("Finished pristine timeseries generation script")
