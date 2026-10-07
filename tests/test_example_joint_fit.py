"""Check production fitting delegation and memory-bounded plotting, without MCMC."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import plot_example_joint_fit as example
from selflensing.fitting.numeric import deterministic_models, parameter_row


class ExampleJointFitTest(unittest.TestCase):
    def setUp(self):
        self.row = pd.Series(dict(
            simulation_id="example", a=20., i=np.pi / 2 - 1e-5,
            Macc=10., Mcomp=5., Rcomp=3., Teff=10000., d=1.,
            e=.2, periapsis_phase=2., tau_eff=3., peak_time=2460000.,
        ))

    def test_fit_uses_actual_stage3_arguments_and_saved_walkers(self):
        pipeline = example.production_pipeline("cpu")
        initial = np.zeros((pipeline.FIT_NWALKERS, 5))
        times = self.row.peak_time + np.arange(-1., 1., .125)
        rv = dict(lc_times=times, lc=np.zeros(len(times)),
                  lc_err=np.full(len(times), .001),
                  rv_times=times[:3], rv=np.zeros(3), rv_err=np.ones(3))
        variant = dict(rv_id="variant", system_id="example")
        sampler = object()
        final = SimpleNamespace(coords=initial[None, :, :])
        with patch.object(pipeline, "saved_prefit_walker_state", return_value=initial), \
             patch.object(pipeline, "self_lensing_systems_from_batched_fit_blackjax",
                          return_value=([None], [sampler], final)) as backend:
            actual, _, _ = example.fit_example(pipeline, self.row, variant, rv)
        self.assertIs(actual, sampler)
        kwargs = backend.call_args.kwargs
        self.assertEqual(kwargs["nwalkers"], pipeline.FIT_NWALKERS)
        self.assertEqual(kwargs["nsteps"], pipeline.FIT_NSTEPS)
        self.assertEqual(kwargs["burn"], pipeline.FIT_BURN)
        self.assertEqual(kwargs["proposal_scale_fraction"], pipeline.FIT_PROPOSAL_SCALE_FRACTION)
        self.assertEqual(kwargs["progress_chunk_size"], pipeline.FIT_PROGRESS_CHUNK_SIZE)
        self.assertTrue(kwargs["return_sampler"])
        self.assertTrue(kwargs["center_lc_model"])
        np.testing.assert_array_equal(kwargs["initial_walker_state"], initial[None, :, :])
        prepared = pipeline.binned_full_lightcurve(rv, self.row)
        np.testing.assert_allclose(kwargs["observed_lc"].value[0], prepared["lc"])
        np.testing.assert_allclose(kwargs["lc_times"].jd[0], prepared["lc_times"])

    def test_chunked_plots_match_production_model_across_time_boundary(self):
        sample = np.array([self.row[k] for k in example.PREFIT_PARAMETER_NAMES])
        samples = np.stack([sample, sample])
        times = self.row.peak_time + np.linspace(-7, 7, 133)
        params = parameter_row(jnp.asarray(sample), self.row.Mcomp, self.row.Rcomp)
        with jax.default_device(jax.devices("cpu")[0]):
            phot, radial = deterministic_models(jnp.asarray(times), jnp.asarray(times),
                                                params, self.row.peak_time)
            expected_phot, expected_rv = np.asarray(phot) - 1, np.asarray(radial)
            actual_phot = example.model_curves(samples, times, self.row)
            actual_rv = example.rv_models(samples, times, self.row)
        for i in range(2):
            np.testing.assert_allclose(actual_phot[i], expected_phot, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(actual_rv[i], expected_rv, rtol=1e-10, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
