"""Check bounded likelihood evaluation against the unmodified fitter."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax.numpy as jnp
import numpy as np
from astropy import units
from astropy.time import Time
from blackjax_memory import bounded_walker_logprob, install_blackjax_memory_bounds
from jax_compat import install_quax_jax_stage_patch

install_quax_jax_stage_patch()
from selflensing.fitting import blackjax


class BlackjaxMemoryTest(unittest.TestCase):
    def test_order_and_remainder(self):
        for walkers in (3, 16, 17, 32, 128):
            params = jnp.arange(2 * walkers * 5, dtype=float).reshape(2, walkers, 5)
            traced_widths = []
            def logprob(x):
                traced_widths.append(x.shape[1])
                return -jnp.sum(x ** 2, axis=-1)
            actual = bounded_walker_logprob(logprob, 16)(params)
            self.assertLessEqual(max(traced_widths), 16)
            np.testing.assert_array_equal(
                actual, logprob(params)
            )
        with self.assertRaises(ValueError):
            bounded_walker_logprob(logprob, 0)

    def test_joint_chain_and_summary_match(self):
        nfits, nwalkers = 2, 17
        rng = np.random.default_rng(21)
        initial = np.broadcast_to(
            [1., np.pi / 2, 5., .1, .3], (nfits, nwalkers, 5)
        ).copy()
        initial += rng.normal(size=initial.shape) * [0.01, 1e-5, .01, .001, .001]
        times = np.broadcast_to(2450000. + np.linspace(-2, 2, 11), (nfits, 11))
        kwargs = dict(
            Mcomp=np.ones(nfits) * units.M_sun,
            Rcomp=np.ones(nfits) * units.R_sun,
            Teff=np.full(nfits, 5772.) * units.K,
            peak_time=Time(np.full(nfits, 2450000.), format="jd"),
            rv_times=Time(times, format="jd"),
            observed_rv=np.zeros_like(times) * units.m / units.s,
            observed_rv_err=np.full_like(times, 100.) * units.m / units.s,
            lc_times=Time(times, format="jd"),
            observed_lc=np.ones_like(times),
            observed_lc_err=np.full_like(times, .01),
            center_lc_model=True,
            initial_walker_state=initial,
            nwalkers=nwalkers, nsteps=6, burn=2,
            progress=True, progress_chunk_size=2,
            device="cpu", prefer_gpu=False, random_seed=12,
            return_walker_state=True,
        )
        factory = blackjax.make_batched_numeric_combined_chi2_vec_jax
        update = blackjax._update_batched_top_posterior_samples
        for save_sampler in (False, True):
            reference = blackjax.self_lensing_systems_from_batched_fit_blackjax(
                **kwargs, return_sampler=save_sampler
            )
            with (
                patch.object(blackjax, "make_batched_numeric_combined_chi2_vec_jax", factory),
                patch.object(blackjax, "_update_batched_top_posterior_samples", update),
                patch.object(blackjax, "_slrv_memory_bounds_installed", False, create=True),
            ):
                install_blackjax_memory_bounds(16)
                actual = blackjax.self_lensing_systems_from_batched_fit_blackjax(
                    **kwargs, return_sampler=save_sampler
                )
            np.testing.assert_array_equal(actual[2].coords, reference[2].coords)
            for expected_system, actual_system in zip(reference[0], actual[0]):
                for name in ("a", "i", "Macc", "e", "periapsis_phase"):
                    np.testing.assert_array_equal(getattr(actual_system, name), getattr(expected_system, name))
            if save_sampler:
                for expected_sampler, actual_sampler in zip(reference[1], actual[1]):
                    np.testing.assert_array_equal(actual_sampler.chain, expected_sampler.chain)
                    # XLA can round differently for the smaller remainder block.
                    # Draws, accepted positions and summaries must match exactly.
                    np.testing.assert_allclose(
                        actual_sampler.log_prob, expected_sampler.log_prob,
                        rtol=1e-12, atol=1e-10,
                    )
                    np.testing.assert_array_equal(actual_sampler.acceptance, expected_sampler.acceptance)


if __name__ == "__main__":
    unittest.main()
