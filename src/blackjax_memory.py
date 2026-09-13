"""Memory bounds for script 3's external selflensing BlackJAX backend."""

from functools import wraps

import jax
import jax.numpy as jnp


def bounded_walker_logprob(logprob, walkers_per_chunk):
    """Evaluate independent walkers in blocks, preserving their order."""
    if walkers_per_chunk < 1:
        raise ValueError("walkers_per_chunk must be positive")

    @jax.jit
    def evaluate(params):
        nfits, nwalkers, ndim = params.shape
        if nwalkers <= walkers_per_chunk:
            return logprob(params)
        full_walkers = nwalkers // walkers_per_chunk * walkers_per_chunk
        blocks = params[:, :full_walkers].reshape(
            nfits, -1, walkers_per_chunk, ndim
        ).transpose(1, 0, 2, 3)
        values = jax.lax.map(logprob, blocks).transpose(1, 0, 2)
        values = values.reshape(nfits, full_walkers)
        if full_walkers < nwalkers:
            values = jnp.concatenate(
                (values, logprob(params[:, full_walkers:])), axis=1
            )
        return values

    return evaluate


def install_blackjax_memory_bounds(walkers_per_chunk=16):
    """Bound likelihood intermediates and outstanding summary chunks.

    Installed only by script 3. Keep the upstream sampler and RNG schedule
    intact; changing MCMC chunk sizes would change its random draws.
    """
    from selflensing.fitting import blackjax

    if walkers_per_chunk < 1:
        raise ValueError("walkers_per_chunk must be positive")
    if getattr(blackjax, "_slrv_memory_bounds_installed", False):
        return
    factory = blackjax.make_batched_numeric_combined_chi2_vec_jax
    update = blackjax._update_batched_top_posterior_samples

    @wraps(factory)
    def bounded_factory(*args, **kwargs):
        return bounded_walker_logprob(factory(*args, **kwargs), walkers_per_chunk)

    @wraps(update)
    def synchronized_update(*args, **kwargs):
        # A summary depends on the completed scan. Waiting here prevents the
        # Python loop from queuing multiple scans and retaining their buffers.
        return jax.block_until_ready(update(*args, **kwargs))

    blackjax.make_batched_numeric_combined_chi2_vec_jax = bounded_factory
    blackjax._update_batched_top_posterior_samples = synchronized_update
    blackjax._slrv_memory_bounds_installed = True
