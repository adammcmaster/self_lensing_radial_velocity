import logging
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", module="erfa.core")


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that writes without corrupting active tqdm bars."""

    def emit(self, record):
        try:
            from tqdm import tqdm

            message = self.format(record)
            tqdm.write(message, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def _env_flag(name):
    value = os.environ.get(name, "")
    return value.lower() not in ("", "0", "false", "no", "off")


def configure_logging():
    """Configure application logging from environment variables."""
    level = logging.DEBUG if _env_flag("DEBUG") else logging.INFO
    handler = TqdmLoggingHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
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


DATA_PATH = Path(__file__).parent / Path("../data/")
DATA_PATH.mkdir(exist_ok=True, parents=True)
logger.debug("Data path initialised at %s", DATA_PATH.resolve())
