import tempfile
import unittest
from pathlib import Path
from unittest import mock

import fitting


class FitResultsLockTest(unittest.TestCase):
    def test_fit_results_lock_fails_fast_when_already_held(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "fitted_timeseries.gz.lock"
            with mock.patch.object(fitting, "FIT_LOCK_PATH", lock_path), mock.patch.object(
                fitting,
                "FIT_LOCK_TIMEOUT",
                0,
            ):
                lock = fitting.acquire_fit_results_lock("first script")
                try:
                    with self.assertRaisesRegex(RuntimeError, "could not acquire"):
                        fitting.acquire_fit_results_lock("second script")
                finally:
                    lock.release()


if __name__ == "__main__":
    unittest.main()
