import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy
from numpy.testing import assert_allclose

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import estimation


class BatchEstimatorConsistencyTest(unittest.TestCase):
    def test_companion_properties_default_to_selflensing_main_sequence_relations(self):
        mass, radius, teff = estimation.companion_stellar_properties(1.0)

        self.assertEqual(mass, 1.0)
        assert_allclose(radius, 1.0)
        assert_allclose(teff, 5772.0, rtol=1e-5)

    def setUp(self):
        self.observable_space = {
            "peak_amplification_ppm": numpy.array([100.0, 1_000.0, 10_000.0, 100_000.0]),
            "crossing_time_hr": numpy.array([1.0, 10.0, 100.0, 1_000.0]),
            "lens_mass_msun": numpy.array([3.0, 10.0, 20.0, 30.0]),
            "orbital_period_yr": numpy.array([1.0, 2.0, 3.0, 4.0]),
            "semi_major_axis_au": numpy.array([0.3, 1.0, 10.0, 100.0]),
            "eccentricity": numpy.array([0.0, 0.2, 0.4, 0.6]),
            "periapsis_phase_rad": numpy.array([0.0, 1.0, 2.0, 3.0]),
            "impact_fraction": numpy.array([0.1, 0.3, 0.5, 0.7]),
            "inclination_rad": numpy.array([1.2, 1.3, 1.4, 1.5]),
            "valid": numpy.ones(4, dtype=bool),
        }
        self.estimator_kwargs = {
            "n_neighbors": 3,
            "sample_size": 4,
            "random_seed": 123,
            "companion_mass_msun": 1.1,
            "companion_radius_rsun": 1.2,
            "companion_teff_k": 6000.0,
        }

    def assert_scalar_and_batch_match(self, peak_amplification_ppm, crossing_time_hr):
        with patch.object(
            estimation,
            "calc_self_lensing_observable_space",
            return_value=self.observable_space,
        ):
            scalar = estimation.estimate_lens_mass_and_orbital_period(
                peak_amplification_ppm,
                crossing_time_hr,
                **self.estimator_kwargs,
            )
            batch = estimation.estimate_lens_mass_and_orbital_period_batch(
                numpy.array([peak_amplification_ppm]),
                numpy.array([crossing_time_hr]),
                batch_size=1,
                **self.estimator_kwargs,
            )

        for quantity in (
            "lens_mass_msun",
            "lens_mass_uncertainty_msun",
            "orbital_period_yr",
            "orbital_period_uncertainty_yr",
            "effective_n_neighbors",
        ):
            assert_allclose(batch[quantity], [scalar[quantity]])

    def test_batch_matches_scalar_for_interpolated_observables(self):
        self.assert_scalar_and_batch_match(1_500.0, 15.0)

    def test_batch_matches_scalar_when_an_exact_neighbour_exists(self):
        self.assert_scalar_and_batch_match(1_000.0, 10.0)


if __name__ == "__main__":
    unittest.main()
