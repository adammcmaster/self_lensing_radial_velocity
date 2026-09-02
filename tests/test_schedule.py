import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy
from numpy.testing import assert_allclose
import jax.numpy as jnp

from selflensing.lightcurves import radial_velocity_m_per_s

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import schedule


# a, i, Macc, Mcomp, Rcomp, Teff, e, periapsis_phase
PARAMETERS = numpy.array([1.0, numpy.pi / 2, 10.0, 1.0, 1.0, 5772.0, 0.0, 0.0])


class RVScheduleTest(unittest.TestCase):
    def test_evenly_spaced_helper_covers_the_full_campaign(self):
        assert_allclose(
            schedule.evenly_spaced_rv_observations(50.0, 150.0, 6),
            numpy.linspace(50.0, 200.0, 6),
        )

    def test_schedules_peak_triplet_and_even_remaining_observations(self):
        times = schedule.schedule_rv_observations(
            t0=100.0,
            porb=100.0,
            baseline=150.0,
            campaign_start=50.0,
            nobs=6,
            sigma_rv=100.0,
            estimated_parameters=PARAMETERS,
        )

        self.assertEqual(times.shape, (6,))
        self.assertEqual(numpy.unique(times).size, 6)
        self.assertTrue(numpy.all(numpy.diff(times) >= 0))
        self.assertTrue(numpy.any(numpy.isclose(times, 100.0)))
        self.assertTrue(numpy.all((times >= 50.0) & (times <= 200.0)))

        anchor_index = numpy.flatnonzero(numpy.isclose(times, 100.0))[0]
        rv_parameters = jnp.asarray(PARAMETERS).at[-1].set(jnp.pi)
        rv0 = radial_velocity_m_per_s(jnp.array([100.0]), rv_parameters, 100.0)[0]
        rv_triplet = radial_velocity_m_per_s(
            jnp.asarray(times[anchor_index - 1 : anchor_index + 2]),
            rv_parameters,
            100.0,
        )
        self.assertGreaterEqual(abs(float(rv_triplet[0] - rv0)), 500.0)
        self.assertGreaterEqual(abs(float(rv_triplet[2] - rv0)), 500.0)

    def test_uses_even_schedule_when_no_lensing_epoch_is_in_campaign(self):
        times = schedule.schedule_rv_observations(
            t0=0.0,
            porb=100.0,
            baseline=20.0,
            campaign_start=21.0,
            nobs=4,
            sigma_rv=100.0,
            estimated_parameters=PARAMETERS,
        )

        assert_allclose(times, [21.0, 27.6666666667, 34.3333333333, 41.0])

    def test_posterior_schedule_honours_joint_samples(self):
        samples = numpy.repeat(PARAMETERS[numpy.newaxis, :], 8, axis=0)
        samples[:, 2] = numpy.linspace(8.0, 12.0, len(samples))
        periods = numpy.linspace(95.0, 105.0, len(samples))
        times = schedule.schedule_rv_observations_from_posterior(
            t0=100.0,
            porb_samples=periods,
            baseline=160.0,
            campaign_start=50.0,
            nobs=5,
            sigma_rv=100.0,
            posterior_parameter_samples=samples,
            credibility=0.75,
        )

        self.assertEqual(times.shape, (5,))
        self.assertTrue(numpy.any(numpy.isclose(times, 100.0)))

    def test_unreachable_significance_uses_largest_in_campaign_offset(self):
        times = schedule.schedule_rv_observations(
            t0=100.0,
            porb=100.0,
            baseline=150.0,
            campaign_start=50.0,
            nobs=5,
            sigma_rv=1e9,
            estimated_parameters=PARAMETERS,
        )

        self.assertEqual(times.shape, (5,))
        self.assertTrue(numpy.all((times >= 50.0) & (times <= 200.0)))
        self.assertTrue(numpy.any(numpy.isclose(times, 100.0)))

    def test_posterior_context_matches_individual_schedules(self):
        samples = numpy.repeat(PARAMETERS[numpy.newaxis, :], 8, axis=0)
        periods = numpy.linspace(95.0, 105.0, len(samples))
        context = schedule.schedule_rv_observation_context_from_posterior(
            100.0, periods, 160.0, 50.0, 100.0, samples, credibility=0.75
        )
        for nobs in range(3, 11):
            direct = schedule.schedule_rv_observations_from_posterior(
                100.0, periods, 160.0, 50.0, nobs, 100.0, samples,
                credibility=0.75,
            )
            assert_allclose(
                schedule.schedule_rv_observations_from_context(context, nobs),
                direct,
            )

    def test_mode_aware_schedule_distinguishes_separated_posterior_clouds(self):
        first = numpy.repeat(PARAMETERS[numpy.newaxis, :], 16, axis=0)
        second = first.copy()
        first[:, 0] = 1.0
        first[:, 2] = 10.0
        second[:, 0] = 12.0
        second[:, 2] = 30.0
        samples = numpy.concatenate((first, second))
        periods = numpy.concatenate((numpy.full(16, 100.0), numpy.full(16, 1000.0)))
        labels = schedule.posterior_mode_labels(samples)
        self.assertGreaterEqual(numpy.unique(labels).size, 2)
        context = schedule.schedule_rv_observation_context_from_posterior(
            100.0, periods, 1600.0, 50.0, 10.0, samples
        )
        self.assertIn("mode_rv_design", context)
        times = schedule.schedule_rv_observations_from_context(context, 6)
        self.assertEqual(times.shape, (6,))
        self.assertEqual(numpy.unique(times).size, 6)
        self.assertTrue(numpy.all((times >= 50.0) & (times <= 1650.0)))
        self.assertGreater(numpy.max(numpy.diff(times)), 100.0)


if __name__ == "__main__":
    unittest.main()
