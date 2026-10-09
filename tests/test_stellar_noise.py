import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ['SLRV_JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORMS'] = 'cpu'
SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import numpy as np


class StellarNoiseTest(unittest.TestCase):
    def test_sampled_parameters_reach_count_space_gp(self):
        spec = importlib.util.spec_from_file_location('generate_photometry', SRC / '0_generate_photometry.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import jax.numpy as jnp

        class FluxModel:
            def mass_to_mag(self, mass, distance):
                return jnp.asarray(10.0), jnp.asarray(0.001)

        draws = np.array([[50., 100., 100.], [300., 10000., 1000.]])
        with patch.object(module.flux_finder, 'FluxModel', FluxModel):
            generate = module.make_tess_gp_input_generator(
                np.random.default_rng(42), noise_sampler=lambda rng, rows: draws)
            params, counts, noise = generate({'Mcomp': np.array([1., 1.]), 'd': np.array([1., 1.])})
        np.testing.assert_allclose(params[:, 3], draws[:, 1])
        np.testing.assert_allclose(10.**params[:, 1] / counts * 1e6, draws[:, 0], rtol=2e-6)
        np.testing.assert_allclose(10.**params[:, 4] / counts * 1e6, draws[:, 2], rtol=2e-6)
        np.testing.assert_allclose(noise / counts, 0.001)

        # Verify the production path delegates all draws to upstream helpers,
        # using the same RNG and the batch size.
        rng = np.random.default_rng(42)
        with (
            patch.object(module.flux_finder, 'FluxModel', FluxModel),
            patch.object(module.gp_priors, 'sample_log_sigma',
                         side_effect=[np.log10(draws[:, 0] / 1e6),
                                      np.log10(draws[:, 2] / 1e6)]) as amplitude_sampler,
            patch.object(module.gp_priors, 'sample_quality_factor',
                         return_value=draws[:, 1]) as quality_sampler,
        ):
            generate = module.make_tess_gp_input_generator(rng)
            default_params, default_counts, _ = generate(
                {'Mcomp': np.array([1., 1.]), 'd': np.array([1., 1.])})
        np.testing.assert_allclose(default_params, params)
        np.testing.assert_allclose(default_counts, counts)
        amplitude_sampler.assert_has_calls([
            unittest.mock.call(0.0, 'G', rng, size=2),
            unittest.mock.call(0.0, 'S', rng, size=2),
        ])
        quality_sampler.assert_called_once_with(rng, size=2)

        with patch.object(module.flux_finder, 'FluxModel', FluxModel):
            rows = {'Mcomp': np.ones(4), 'd': np.ones(4)}
            first = module.make_tess_gp_input_generator(np.random.default_rng(42))(rows)
            second = module.make_tess_gp_input_generator(np.random.default_rng(42))(rows)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertTrue(np.all(np.isfinite(first[0])))
        self.assertTrue(np.all((first[0][:, 3] >= 100) & (first[0][:, 3] <= 10000)))

        from astropy.io import fits
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'simulation_000000.fits.gz'
            hdu = fits.BinTableHDU()
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
            with self.assertRaisesRegex(RuntimeError, 'previous stellar-noise'):
                module.existing_simulation_indices(Path(directory))
            hdu.header['GPPRIOR'] = 'sampled-v1'
            fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
            self.assertEqual(module.existing_simulation_indices(Path(directory)), {0})


if __name__ == '__main__':
    unittest.main()
