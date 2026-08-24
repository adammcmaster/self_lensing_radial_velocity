import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import prefit
import rv_timeseries as rv


class ScheduledPipelineConfigTest(unittest.TestCase):
    def test_schedule_grid_uses_nobs_and_a_single_five_year_baseline(self):
        self.assertEqual(rv.PARAMS["nobs"], tuple(range(3, 11)))
        self.assertEqual(tuple(rv.PARAMS["campaign_baselines"]), (1826.25,))
        self.assertEqual(len(rv.PARAM_GRID), 8 * 5)
        self.assertEqual(prefit.RV_CAMPAIGN_BASELINE_DAYS, 1826.25)

    def test_rv_metadata_uses_nobs_not_cadence(self):
        self.assertIn("nobs", rv.RV_TIMESERIES_METADATA_COLUMNS)
        self.assertNotIn("cadence", rv.RV_TIMESERIES_METADATA_COLUMNS)


if __name__ == "__main__":
    unittest.main()
