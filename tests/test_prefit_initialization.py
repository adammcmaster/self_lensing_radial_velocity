import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy
import pandas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import prefit


class ObservableSeedInitialisationTest(unittest.TestCase):
    def test_seeded_walkers_cover_distinct_neighbour_modes_and_respect_bounds(self):
        candidates = pandas.DataFrame(
            {
                "semi_major_axis_au": [1.0, 1.1, 10.0, 10.2],
                "inclination_rad": [1.57070, 1.57071, 1.57075, 1.57076],
                "lens_mass_msun": [3.0, 3.1, 20.0, 20.2],
                "eccentricity": [0.1, 0.11, 0.5, 0.51],
                "periapsis_phase_rad": [0.1, 0.12, 3.1, 3.12],
            }
        )
        row = {
            "simulation_id": "seed-test",
            "source_amp_sl": 0.02,
            "tau_eff": 2.0,
            "Mcomp": 1.0,
            "Rcomp": 1.0,
            "Teff": 5772.0,
        }
        with (
            patch.object(prefit, "FIT_NWALKERS", 8),
            patch.object(prefit, "PREFIT_SEED_MODE_COUNT", 2),
            patch.object(prefit, "PREFIT_SEED_NEIGHBORS", 4),
            patch("estimation.nearest_self_lensing_observable_space", return_value=candidates),
        ):
            walkers = prefit.observable_seed_walker_positions([row], random_seed=5)

        self.assertEqual(walkers.shape, (1, 8, 5))
        self.assertTrue(numpy.all(walkers[..., 0] >= prefit.DEFAULT_LIMITS["a"][0]))
        self.assertTrue(numpy.all(walkers[..., 0] <= prefit.DEFAULT_LIMITS["a"][1]))
        self.assertTrue(numpy.all(walkers[..., 2] >= prefit.DEFAULT_LIMITS["Macc"][0]))
        self.assertTrue(numpy.all(walkers[..., 2] <= prefit.DEFAULT_LIMITS["Macc"][1]))
        self.assertGreater(numpy.ptp(walkers[0, :, 0]), 5.0)

