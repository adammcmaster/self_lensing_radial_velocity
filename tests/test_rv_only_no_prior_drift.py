"""Regression checks against the upstream BlackJAX fitter.

These tests do not run MCMC.  They pin the local RV-no-prior adapter to the
upstream sampler conventions that it intentionally mirrors.
"""

import importlib.util
import inspect
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy
from numpy.testing import assert_allclose
from astropy import units

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import rv_only_no_prior
from selflensing.fitting import blackjax as upstream_blackjax
from selflensing.fitting.likelihood import combined_chi2 as upstream_logprob
from selflensing.lightcurves import radial_velocity_m_per_s
from selflensing.models import DEFAULT_LIMITS
from selflensing.models import SelfLensingSystem


def load_rv_no_prior_script():
    path = SRC_PATH / "4_rv_even_spaced_fit.py"
    spec = importlib.util.spec_from_file_location("rv_no_prior_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RVOnlyNoPriorDriftTest(unittest.TestCase):
    def test_initial_walkers_match_upstream_physical_coordinates(self):
        nwalkers = 16
        numpy.random.seed(123)
        upstream = upstream_blackjax._initial_walker_positions(nwalkers)
        numpy.random.seed(123)
        local = rv_only_no_prior._initial_walker_positions(nwalkers)

        self.assertEqual(local.shape, (nwalkers, 6))
        assert_allclose(local[:, :5], upstream)
        self.assertTrue(numpy.all(local[:, 5] >= 0.0))
        self.assertTrue(numpy.all(local[:, 5] <= 2.0 * numpy.pi))

    def test_initial_walker_state_validation_tracks_upstream_dimension_plus_phase(self):
        with self.assertRaisesRegex(ValueError, r"nwalkers, 6"):
            rv_only_no_prior._initial_walker_positions(
                4,
                initial_walker_state=numpy.zeros((4, 5)),
            )

        coords = numpy.zeros((4, 6))
        assert_allclose(
            rv_only_no_prior._initial_walker_positions(
                4,
                initial_walker_state=coords,
            ),
            coords,
        )

    def test_proposal_scale_matches_upstream_for_physical_coordinates(self):
        for scale_fraction in (0.005, 0.02):
            upstream = upstream_blackjax._default_blackjax_proposal_scale(
                scale_fraction
            )
            local = rv_only_no_prior._default_proposal_scale(scale_fraction)

            self.assertEqual(local.shape, (6,))
            assert_allclose(local[:5], upstream)
            assert_allclose(local[5], scale_fraction * 2.0 * numpy.pi)

    def test_observation_count_scaling_matches_upstream_for_unbatched_rv(self):
        proposal_scale = rv_only_no_prior._default_proposal_scale(0.005)
        observed_rv = numpy.arange(13.0)

        upstream_args = [proposal_scale[:5], observed_rv, None, None, None, None]
        upstream = upstream_blackjax._scaled_proposal_scale(*upstream_args)
        local = rv_only_no_prior._scaled_rv_proposal_scale(
            proposal_scale,
            observed_rv,
            None,
        )

        assert_allclose(local[:5], upstream)
        assert_allclose(local[5], proposal_scale[5])

    def test_physical_prior_bounds_are_the_upstream_default_limits(self):
        source = inspect.getsource(rv_only_no_prior.rv_no_prior_logprob)
        for parameter in ("a", "i", "Macc", "e", "periapsis_phase"):
            self.assertIn(f'DEFAULT_LIMITS["{parameter}"]', source)
        self.assertEqual(rv_only_no_prior.RV_PHASE_LIMITS, (0.0, 2.0 * numpy.pi))

    def test_local_rv_logprob_matches_upstream_at_equivalent_peak_epoch(self):
        """The local phase parameter must reproduce upstream's fixed-epoch RV model."""
        physical = numpy.asarray([1.0, numpy.pi / 2, 8.0, 0.2, 0.4])
        mcomp = 1.0 * units.M_sun
        rcomp = 1.0 * units.R_sun
        system = SelfLensingSystem(
            a=physical[0] * units.au,
            i=physical[1] * units.rad,
            Macc=physical[2] * units.M_sun,
            Mcomp=mcomp,
            Rcomp=rcomp,
            e=physical[3],
            periapsis_phase=physical[4] * units.rad,
        )
        reference_time = 2450000.0
        peak_time = reference_time + 0.37 * system.porb.to_value(units.day)
        phase = 2.0 * numpy.pi * (peak_time - reference_time) / system.porb.to_value(units.day)
        times = reference_time + numpy.linspace(1.0, 60.0, 8)
        model_params = numpy.asarray([*physical[:3], 1.0, 1.0, 5772.0, *physical[3:]])
        observed_rv = numpy.asarray(
            radial_velocity_m_per_s(times, model_params, peak_time), dtype=float
        )
        observed_rv_err = numpy.full_like(observed_rv, 10.0)

        local = rv_only_no_prior.rv_no_prior_logprob(
            numpy.append(physical, phase),
            mcomp.to_value(units.kg),
            rcomp.to_value(units.m),
            1000.0,
            times,
            observed_rv,
            observed_rv_err,
            reference_time,
            None,
        )
        upstream = upstream_logprob(
            physical,
            mcomp.to_value(units.kg),
            rcomp.to_value(units.m),
            None,
            1000.0,
            peak_time,
            times,
            observed_rv,
            observed_rv_err,
            None,
            None,
            None,
            None,
            numpy.asarray([]),
            None,
        )
        assert_allclose(numpy.asarray(local), numpy.asarray(upstream), rtol=1e-6, atol=1e-6)

    def test_single_fit_keeps_upstream_sampler_control_surface(self):
        upstream_signature = inspect.signature(
            upstream_blackjax.self_lensing_system_from_fit_blackjax
        )
        local_signature = inspect.signature(
            rv_only_no_prior.self_lensing_system_from_rv_fit_no_peak_prior_blackjax
        )
        shared_controls = {
            "nwalkers",
            "nsteps",
            "burn",
            "lensing_method",
            "progress",
            "progress_desc",
            "prefer_gpu",
            "device",
            "initial_walker_state",
            "proposal_scale",
            "proposal_scale_fraction",
            "proposal_reference_observations",
            "random_seed",
            "progress_chunk_size",
            "posterior_summary_top_k",
            "return_sampler",
            "return_walker_state",
        }

        for name in shared_controls:
            self.assertIn(name, local_signature.parameters)
            self.assertEqual(
                local_signature.parameters[name].default,
                upstream_signature.parameters[name].default,
                name,
            )
        self.assertNotIn("peak_time", local_signature.parameters)

    def test_merge_adds_only_rvnoprior_columns(self):
        script = load_rv_no_prior_script()
        base = script.pandas.DataFrame(
            [
                {"rv_id": "a", "fitted_Macc": 7.0, "existing": "keep"},
                {"rv_id": "b", "fitted_Macc": 8.0, "existing": "also_keep"},
            ]
        )
        new_rows = [
            script.pandas.Series(
                {
                    "rv_id": "a",
                    "fitted_Macc": 99.0,
                    "rvnoprior_fitted_Macc": 6.5,
                }
            )
        ]

        merged = script.merge_rv_even_spaced_results(base, new_rows)

        self.assertEqual(list(merged["rv_id"]), ["a", "b"])
        self.assertEqual(merged.loc[0, "fitted_Macc"], 7.0)
        self.assertEqual(merged.loc[1, "fitted_Macc"], 8.0)
        self.assertEqual(merged.loc[1, "existing"], "also_keep")
        self.assertEqual(merged.loc[0, "rvnoprior_fitted_Macc"], 6.5)
        self.assertTrue(script.pandas.isna(merged.loc[1, "rvnoprior_fitted_Macc"]))


if __name__ == "__main__":
    unittest.main()
