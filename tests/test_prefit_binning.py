import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import prefit


class PrefitBinningTest(unittest.TestCase):
    def test_crossing_time_selects_requested_bin_widths(self):
        self.assertEqual(prefit.prefit_bin_width_days(0.5 / 24.0), 10.0 / 1440.0)
        self.assertEqual(prefit.prefit_bin_width_days(0.5), 1.0 / 24.0)
        self.assertEqual(prefit.prefit_bin_width_days(2.0), 6.0 / 24.0)

    def test_batches_never_mix_bin_widths(self):
        rows = [
            {"simulation_id": "short", "tau_eff": 0.5 / 24.0},
            {"simulation_id": "medium", "tau_eff": 0.5},
            {"simulation_id": "long", "tau_eff": 2.0},
        ]

        batches = prefit.photometry_prefit_batches(rows)

        self.assertEqual(len(batches), 3)
        for batch in batches:
            widths = {prefit.prefit_bin_width_days(row["tau_eff"]) for row in batch}
            self.assertEqual(len(widths), 1)

    def test_default_binning_preserves_ten_minute_cadence(self):
        cadence = 600.0 / 86400.0
        start = 2460000.0
        curve = {
            "times": start + numpy.arange(12) * cadence,
            "flux": numpy.arange(12, dtype=float),
            "err": numpy.ones(12),
        }

        binned = prefit.bin_full_lightcurve_hourly(
            curve, bin_width_days=10.0 / 1440.0
        )

        self.assertEqual(numpy.count_nonzero(binned["mask"]), len(curve["times"]))
        numpy.testing.assert_allclose(
            binned["flux"][binned["mask"]], curve["flux"]
        )
