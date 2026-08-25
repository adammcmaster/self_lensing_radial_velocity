import unittest

import jax.numpy as jnp
import numpy

from adaptive_blackjax import adaptive_batched_mala, adaptive_batched_random_walk


class TestAdaptiveBatchedRandomWalk(unittest.TestCase):
    def test_returns_production_chain_and_per_fit_adaptation(self):
        initial_positions = numpy.array(
            [
                [[-4.0], [3.0], [1.0], [-2.0]],
                [[-1.0], [2.0], [4.0], [-3.0]],
            ]
        )

        def logprob(positions):
            centres = jnp.asarray([0.0, 2.0])[:, None, None]
            return -0.5 * jnp.sum((positions - centres) ** 2, axis=-1)

        result = adaptive_batched_random_walk(
            logprob,
            initial_positions,
            proposal_scale=numpy.array([0.5]),
            warmup_steps=20,
            adaptation_rounds=2,
            production_steps=40,
            random_seed=4,
        )

        self.assertEqual(result.chain.shape, (40, 2, 4, 1))
        self.assertEqual(result.log_prob.shape, (40, 2, 4))
        self.assertEqual(result.acceptance.shape, (40, 2, 4))
        self.assertEqual(result.warmup_acceptance.shape, (2, 2))
        self.assertEqual(result.proposal_scale.shape, (2, 1))
        self.assertTrue(numpy.all(numpy.isfinite(result.chain)))

    def test_mala_returns_a_batched_production_chain(self):
        initial_positions = numpy.array([[[1.0], [-1.0], [2.0], [-2.0]]])

        result = adaptive_batched_mala(
            lambda positions: -0.5 * jnp.sum(positions**2, axis=-1),
            initial_positions,
            metric_scale=numpy.array([1.0]),
            warmup_steps=20,
            adaptation_rounds=2,
            production_steps=40,
            initial_step_size=0.1,
            random_seed=9,
        )

        self.assertEqual(result.chain.shape, (40, 1, 4, 1))
        self.assertTrue(numpy.all(numpy.isfinite(result.chain)))
        self.assertTrue(numpy.all(numpy.isfinite(result.acceptance)))


if __name__ == "__main__":
    unittest.main()
