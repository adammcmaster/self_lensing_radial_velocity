import os
import sys
import unittest
from pathlib import Path

import pandas

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

    def test_distance_survives_rv_metadata_storage_and_expansion(self):
        photometry = {name: 1.0 for name in rv.PHOTOMETRY_SHARED_COLUMNS}
        photometry.update(
            simulation_id="simulation_000001",
            photometry_path="simulation_000001.fits.gz",
            d=5.845761303332092,
        )
        row = rv.metadata_row_from_params(
            "test-rv", "test.fits.gz", photometry, 10.0, 3, 1826.25
        )
        self.assertEqual(row["d"], photometry["d"])
        self.assertEqual(rv._metadata_header(row, 10.0)["D"], photometry["d"])

        # Existing caches store only RV fields; stage 3 restores shared truth
        # values from the prefit table each time it loads them.
        cached = pandas.DataFrame([row], columns=rv.RV_TIMESERIES_METADATA_COLUMNS)
        expanded = rv._expand_metadata_table_with_photometry(
            cached, pandas.DataFrame([photometry])
        )
        self.assertEqual(expanded.iloc[0]["d"], photometry["d"])


if __name__ == "__main__":
    unittest.main()
