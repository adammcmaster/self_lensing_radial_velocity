"""GPU-batched, two-stage adaptive random-walk fitting.

This is intentionally structured as a small companion to
``selflensing.fitting.blackjax``.  The generic sampler below has no
self-lensing dependencies; the thin self-lensing wrapper only constructs the
batched likelihood and converts the final positions to systems.  It can
therefore be moved upstream with the private-helper imports made relative to
the upstream ``blackjax`` module.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jax_numpy
import numpy
from astropy import units
from quaxed import numpy as jnp

from selflensing.fitting import blackjax as upstream
from selflensing.fitting import likelihood as upstream_likelihood
from selflensing.fitting._utils import _device_array, _extract_value
from selflensing.experimental.photometry import (
    PhotometryRow,
    batched_photometric_logprob_and_grad,
)
from selflensing.generators import SelfLensingLightcurveGenerator
from selflensing.models import SelfLensingSystem


@dataclass
class AdaptiveRandomWalkResult:
    """Host-side output of a batched adaptive random-walk run."""

    chain: numpy.ndarray
    log_prob: numpy.ndarray
    acceptance: numpy.ndarray
    final_positions: numpy.ndarray
    proposal_scale: numpy.ndarray
    warmup_acceptance: numpy.ndarray


def _per_fit_scale(scale, nfits, ndim):
    scale = jax_numpy.asarray(scale)
    if scale.ndim == 1:
        scale = jnp.broadcast_to(scale, (nfits, 1, ndim))
    elif scale.ndim == 2:
        scale = scale[:, None, :]
    if scale.shape != (nfits, 1, ndim):
        raise ValueError(
            "proposal_scale must have shape (ndim,), (nfits, ndim), or "
            f"(nfits, 1, ndim); got {scale.shape}"
        )
    return scale


def _elite_positions(positions, log_prob, count):
    """Return the best positions of each fit from a scan output.

    Keeping elites gives every production walker a viable starting point
    without introducing CPU work or breaking the leading fit-axis batch.
    """
    nfits = positions.shape[1]
    ndim = positions.shape[-1]
    flat_positions = jnp.transpose(positions, (1, 0, 2, 3)).reshape(
        (nfits, -1, ndim)
    )
    flat_log_prob = jnp.transpose(log_prob, (1, 0, 2)).reshape((nfits, -1))
    safe_log_prob = jnp.where(jnp.isfinite(flat_log_prob), flat_log_prob, -jnp.inf)
    _, indices = jax.lax.top_k(safe_log_prob, count)
    return jnp.take_along_axis(flat_positions, indices[..., None], axis=1)


def _covariance_proposal(elites, current_transform, regularization):
    """Estimate a regularized, per-fit Cholesky proposal transform."""
    ndim = elites.shape[-1]
    centred = elites - jnp.mean(elites, axis=1, keepdims=True)
    empirical = jnp.einsum("fki,fkj->fij", centred, centred) / max(
        elites.shape[1] - 1, 1
    )
    current_covariance = jnp.einsum(
        "fij,fkj->fik", current_transform, current_transform
    )
    diagonal = jnp.diagonal(empirical, axis1=-2, axis2=-1)
    fallback_diagonal = jnp.diagonal(current_covariance, axis1=-2, axis2=-1)
    diagonal = jnp.where(diagonal > 0, diagonal, fallback_diagonal)
    diagonal_covariance = jax.vmap(jnp.diag)(diagonal)
    covariance = (1.0 - regularization) * empirical + regularization * diagonal_covariance
    # A small scale-relative diagonal floor makes the Cholesky decomposition
    # stable when many elite states are duplicate rejected proposals.
    covariance += regularization * 1.0e-3 * jax.vmap(jnp.diag)(fallback_diagonal)
    return jax.vmap(jnp.linalg.cholesky)(covariance * (2.38**2 / ndim))


def adaptive_batched_random_walk(
    logprob_vec,
    initial_positions,
    proposal_scale,
    *,
    warmup_steps=400,
    adaptation_rounds=2,
    target_acceptance=0.25,
    adaptation_gain=2.0,
    scale_change_limit=4.0,
    adapt_covariance=False,
    covariance_elite_multiplier=4,
    covariance_regularization=0.1,
    production_steps=1000,
    random_seed=0,
):
    """Run batched exploration, per-fit scale adaptation, and production.

    ``logprob_vec`` takes ``(nfits, nwalkers, ndim)`` positions and returns
    ``(nfits, nwalkers)`` log probabilities.  Each warm-up round starts from
    that fit's best explored states; its mean acceptance adjusts only that
    fit's proposal scale.  The production chain is the only chain returned.
    """
    position = jax_numpy.asarray(initial_positions)
    if position.ndim != 3:
        raise ValueError("initial_positions must have shape (nfits, nwalkers, ndim)")
    nfits, nwalkers, ndim = position.shape
    if warmup_steps < 1 or adaptation_rounds < 1 or production_steps < 1:
        raise ValueError("warmup_steps, adaptation_rounds, and production_steps must be positive")
    if not 0 < target_acceptance < 1:
        raise ValueError("target_acceptance must be between zero and one")
    scale = _per_fit_scale(proposal_scale, nfits, ndim)
    transform = jax.vmap(jnp.diag)(scale[:, 0, :])
    log_prob = logprob_vec(position)

    def one_step(carry, rng_key, current_transform):
        current_position, current_log_prob = carry
        proposal_key, accept_key = jax.random.split(rng_key)
        noise = jax.random.normal(proposal_key, shape=current_position.shape)
        proposal = current_position + jnp.einsum(
            "fwi,fji->fwj", noise, current_transform
        )
        proposal_log_prob = logprob_vec(proposal)
        accepted = jnp.log(jax.random.uniform(accept_key, shape=current_log_prob.shape)) < (
            proposal_log_prob - current_log_prob
        )
        new_position = jnp.where(accepted[..., None], proposal, current_position)
        new_log_prob = jnp.where(accepted, proposal_log_prob, current_log_prob)
        return (new_position, new_log_prob), (new_position, new_log_prob, accepted)

    def run_steps(start_position, start_log_prob, current_transform, rng_key, steps):
        step_keys = jax.random.split(rng_key, steps)

        @jax.jit
        def scan(position, log_prob, keys, transform):
            return jax.lax.scan(
                lambda carry, key: one_step(carry, key, transform),
                (position, log_prob),
                keys,
            )

        return scan(start_position, start_log_prob, step_keys, current_transform)

    rng_key = jax.random.PRNGKey(random_seed)
    warmup_acceptance = []
    for _ in range(adaptation_rounds):
        rng_key, round_key = jax.random.split(rng_key)
        (position, log_prob), (positions, log_probs, accepted) = run_steps(
            position, log_prob, transform, round_key, warmup_steps
        )
        acceptance = jnp.mean(accepted, axis=(0, 2))
        warmup_acceptance.append(acceptance)
        elite_count = min(
            positions.shape[0] * nwalkers,
            max(nwalkers, int(covariance_elite_multiplier) * nwalkers),
        )
        elites = _elite_positions(positions, log_probs, elite_count)
        position = elites[:, :nwalkers]
        log_prob = logprob_vec(position)
        multiplier = jnp.exp(adaptation_gain * (acceptance - target_acceptance))
        multiplier = jnp.clip(multiplier, 1.0 / scale_change_limit, scale_change_limit)
        if adapt_covariance:
            transform = _covariance_proposal(
                elites, transform, covariance_regularization
            )
        transform = transform * multiplier[:, None, None]

    rng_key, production_key = jax.random.split(rng_key)
    (position, log_prob), (chain, log_prob_chain, accepted) = run_steps(
        position, log_prob, transform, production_key, production_steps
    )
    return AdaptiveRandomWalkResult(
        chain=numpy.asarray(jax.device_get(chain)),
        log_prob=numpy.asarray(jax.device_get(log_prob_chain)),
        acceptance=numpy.asarray(jax.device_get(accepted)),
        final_positions=numpy.asarray(jax.device_get(position)),
        proposal_scale=numpy.asarray(jax.device_get(jnp.diagonal(transform, axis1=-2, axis2=-1))),
        warmup_acceptance=numpy.asarray(jax.device_get(jnp.stack(warmup_acceptance))),
    )


def adaptive_batched_mala(
    logprob_vec,
    initial_positions,
    metric_scale,
    *,
    warmup_steps=400,
    adaptation_rounds=2,
    target_acceptance=0.574,
    adaptation_gain=2.0,
    scale_change_limit=4.0,
    initial_step_size=0.1,
    production_steps=1000,
    random_seed=0,
    value_and_grad_vec=None,
):
    """Batched preconditioned MALA with per-fit step-size adaptation.

    ``metric_scale`` supplies a diagonal position scale for each fit.  The
    gradient is evaluated jointly over the complete batch, while each walker
    retains its own Metropolis acceptance decision.
    """
    position = jax_numpy.asarray(initial_positions)
    if position.ndim != 3:
        raise ValueError("initial_positions must have shape (nfits, nwalkers, ndim)")
    nfits, nwalkers, ndim = position.shape
    if warmup_steps < 1 or adaptation_rounds < 1 or production_steps < 1:
        raise ValueError("warmup_steps, adaptation_rounds, and production_steps must be positive")
    metric_scale = _per_fit_scale(metric_scale, nfits, ndim)[:, 0, :]
    metric_variance = metric_scale**2

    def evaluate(positions):
        if value_and_grad_vec is not None:
            return value_and_grad_vec(positions)
        # Generic fallback. The self-lensing wrapper supplies a more memory-
        # efficient per-row forward-mode implementation below.
        _, gradient = jax.value_and_grad(
            lambda value: jnp.sum(logprob_vec(value))
        )(positions)
        return logprob_vec(positions), gradient

    log_prob, gradient = evaluate(position)

    def one_step(carry, rng_key, step_size):
        current_position, current_log_prob, current_gradient = carry
        proposal_key, accept_key = jax.random.split(rng_key)
        drift = step_size[:, None, None] * metric_variance[:, None, :] * current_gradient
        noise = jax.random.normal(proposal_key, shape=current_position.shape)
        proposal = current_position + drift + (
            jnp.sqrt(2.0 * step_size)[:, None, None]
            * metric_scale[:, None, :]
            * noise
        )
        proposal_log_prob, proposal_gradient = evaluate(proposal)
        forward = proposal - current_position - drift
        reverse = current_position - proposal - (
            step_size[:, None, None] * metric_variance[:, None, :] * proposal_gradient
        )
        denominator = 4.0 * step_size[:, None, None] * metric_variance[:, None, :]
        forward_energy = jnp.sum(forward**2 / denominator, axis=-1)
        reverse_energy = jnp.sum(reverse**2 / denominator, axis=-1)
        log_accept_ratio = proposal_log_prob - current_log_prob + forward_energy - reverse_energy
        accepted = jnp.log(jax.random.uniform(accept_key, shape=current_log_prob.shape)) < log_accept_ratio
        new_position = jnp.where(accepted[..., None], proposal, current_position)
        new_log_prob = jnp.where(accepted, proposal_log_prob, current_log_prob)
        new_gradient = jnp.where(accepted[..., None], proposal_gradient, current_gradient)
        return (new_position, new_log_prob, new_gradient), (new_position, new_log_prob, accepted)

    def run_steps(start_position, start_log_prob, start_gradient, step_size, rng_key, steps):
        step_keys = jax.random.split(rng_key, steps)

        @jax.jit
        def scan(position, log_prob, gradient, keys, sizes):
            return jax.lax.scan(
                lambda carry, key: one_step(carry, key, sizes),
                (position, log_prob, gradient),
                keys,
            )

        return scan(start_position, start_log_prob, start_gradient, step_keys, step_size)

    step_size = jnp.full((nfits,), float(initial_step_size))
    rng_key = jax.random.PRNGKey(random_seed)
    warmup_acceptance = []
    for _ in range(adaptation_rounds):
        rng_key, round_key = jax.random.split(rng_key)
        (position, log_prob, gradient), (positions, log_probs, accepted) = run_steps(
            position, log_prob, gradient, step_size, round_key, warmup_steps
        )
        acceptance = jnp.mean(accepted, axis=(0, 2))
        warmup_acceptance.append(acceptance)
        position = _elite_positions(positions, log_probs, nwalkers)
        log_prob, gradient = evaluate(position)
        multiplier = jnp.exp(adaptation_gain * (acceptance - target_acceptance))
        step_size = step_size * jnp.clip(
            multiplier, 1.0 / scale_change_limit, scale_change_limit
        )

    rng_key, production_key = jax.random.split(rng_key)
    (position, log_prob, gradient), (chain, log_prob_chain, accepted) = run_steps(
        position, log_prob, gradient, step_size, production_key, production_steps
    )
    return AdaptiveRandomWalkResult(
        chain=numpy.asarray(jax.device_get(chain)),
        log_prob=numpy.asarray(jax.device_get(log_prob_chain)),
        acceptance=numpy.asarray(jax.device_get(accepted)),
        final_positions=numpy.asarray(jax.device_get(position)),
        proposal_scale=numpy.asarray(jax.device_get(metric_scale * jnp.sqrt(step_size)[:, None])),
        warmup_acceptance=numpy.asarray(jax.device_get(jnp.stack(warmup_acceptance))),
    )


def make_batched_scalar_logprob_and_forward_gradient(
    *,
    Mcomp, Rcomp, tau_eff, d, peak_time,
    rv_times, observed_rv, observed_rv_err,
    lc_times, observed_lc, observed_lc_err,
    detected_periods, lensing_method, center_lc_model,
    period_min, period_max, model_sls, rv_mask, lc_mask, device,
):
    """Build a batched scalar likelihood with parameter-wise forward gradients.

    This deliberately avoids ``grad(sum(batched_logprob))``.  Each five
    dimensional parameter vector is differentiated in forward mode before
    batching over walkers and systems, preventing reverse mode from retaining
    the light-curve's large time-domain intermediates for every walker.
    """
    values = [
        _extract_value(value)
        for value in (
            Mcomp, Rcomp, tau_eff, d, peak_time, rv_times, observed_rv,
            observed_rv_err, lc_times, observed_lc, observed_lc_err,
            detected_periods, rv_mask, lc_mask,
        )
    ]
    (
        Mcomp, Rcomp, tau_eff, d, peak_time, rv_times, observed_rv,
        observed_rv_err, lc_times, observed_lc, observed_lc_err,
        detected_periods, rv_mask, lc_mask,
    ) = [_device_array(value, device) for value in values]

    def scalar(params, row_Mcomp, row_Rcomp, row_tau_eff, row_d, row_peak_time,
               row_rv_times, row_observed_rv, row_observed_rv_err,
               row_lc_times, row_observed_lc, row_observed_lc_err,
               row_periods, row_rv_mask, row_lc_mask):
        return upstream_likelihood.combined_chi2(
            params, row_Mcomp, row_Rcomp, row_tau_eff, row_d, row_peak_time,
            row_rv_times, row_observed_rv, row_observed_rv_err, row_lc_times,
            row_observed_lc, row_observed_lc_err, None, row_periods,
            lensing_method, center_lc_model, period_min, period_max, model_sls,
            row_rv_mask, row_lc_mask,
        )

    def rows_with_tau(params):
        def row(row_params, *row_inputs):
            def one(position):
                return scalar(position, *row_inputs)
            return jax.vmap(one)(row_params), jax.vmap(jax.jacfwd(one))(row_params)

        return jax.vmap(row)(
            params, Mcomp, Rcomp, tau_eff, d, peak_time, rv_times, observed_rv,
            observed_rv_err, lc_times, observed_lc, observed_lc_err,
            detected_periods, rv_mask, lc_mask,
        )

    if tau_eff is not None:
        return jax.jit(rows_with_tau)

    def rows_without_tau(params):
        def row(row_params, row_Mcomp, row_Rcomp, row_d, row_peak_time,
                row_rv_times, row_observed_rv, row_observed_rv_err,
                row_lc_times, row_observed_lc, row_observed_lc_err,
                row_periods, row_rv_mask, row_lc_mask):
            def one(position):
                return scalar(
                    position, row_Mcomp, row_Rcomp, None, row_d, row_peak_time,
                    row_rv_times, row_observed_rv, row_observed_rv_err,
                    row_lc_times, row_observed_lc, row_observed_lc_err,
                    row_periods, row_rv_mask, row_lc_mask,
                )
            return jax.vmap(one)(row_params), jax.vmap(jax.jacfwd(one))(row_params)

        return jax.vmap(row)(
            params, Mcomp, Rcomp, d, peak_time, rv_times, observed_rv,
            observed_rv_err, lc_times, observed_lc, observed_lc_err,
            detected_periods, rv_mask, lc_mask,
        )

    return jax.jit(rows_without_tau)


def self_lensing_systems_from_adaptive_batched_fit_blackjax(
    *,
    Mcomp,
    peak_time,
    Rcomp,
    Teff,
    lc_times=None,
    observed_lc=None,
    observed_lc_err=None,
    lc_mask=None,
    rv_times=None,
    observed_rv=None,
    observed_rv_err=None,
    rv_mask=None,
    tau_eff=None,
    d=SelfLensingSystem.DEFAULT_D,
    false_alarm_probability=None,
    periods=(),
    period_min=None,
    period_max=None,
    lensing_method=None,
    center_lc_model=False,
    weight_lc_peak=False,
    nwalkers=64,
    nsteps=1000,
    burn=200,
    warmup_steps=400,
    adaptation_rounds=2,
    target_acceptance=0.25,
    adaptation_gain=2.0,
    scale_change_limit=4.0,
    adapt_covariance=False,
    covariance_elite_multiplier=4,
    covariance_regularization=0.1,
    kernel="random_walk",
    mala_initial_step_size=0.1,
    mala_gradient_mode="reverse",
    proposal_scale=None,
    proposal_scale_fraction=0.02,
    proposal_reference_observations=None,
    device=None,
    prefer_gpu=True,
    initial_walker_state=None,
    random_seed=0,
    progress=False,
    progress_desc=None,
    return_sampler=True,
    return_walker_state=True,
):
    """Self-lensing wrapper for :func:`adaptive_batched_random_walk`.

    This follows the upstream batched fitter's input/output conventions.  It
    deliberately supports the common RV+photometry arguments so it can be
    ported as a sibling of the upstream BlackJAX implementation.
    """
    if observed_rv is None and observed_lc is None:
        raise RuntimeError("You must provide at least one of observed_rv, observed_lc")
    if burn >= nsteps:
        raise ValueError("burn must be smaller than nsteps")
    fit_device = upstream._select_fit_device(device=device, prefer_gpu=prefer_gpu)
    if lensing_method is None:
        lensing_method = SelfLensingLightcurveGenerator.default_lensing_method_for_device(
            fit_device
        )
    Mcomp = Mcomp.to(SelfLensingSystem.param_units["Mcomp"])
    nfits = int(numpy.asarray(Mcomp.value).shape[0])
    Mcomp = upstream._batched_quantity(Mcomp, SelfLensingSystem.param_units["Mcomp"], nfits, "Mcomp")
    Rcomp = upstream._batched_quantity(Rcomp, SelfLensingSystem.param_units["Rcomp"], nfits, "Rcomp")
    Teff = upstream._batched_quantity(Teff, SelfLensingSystem.param_units["Teff"], nfits, "Teff")
    d = upstream._batched_quantity(d, SelfLensingSystem.param_units["d"], nfits, "d")
    if tau_eff is not None:
        tau_eff = upstream._batched_quantity(tau_eff, SelfLensingSystem.param_units["tau_eff"], nfits, "tau_eff")
    if observed_rv is not None:
        observed_rv = observed_rv.to(units.m / units.s)
        observed_rv_err = observed_rv_err.to(units.m / units.s)
    if observed_lc is not None and hasattr(observed_lc, "to"):
        observed_lc = observed_lc.to(units.dimensionless_unscaled)
        observed_lc_err = observed_lc_err.to(units.dimensionless_unscaled)
    rv_times, observed_rv, observed_rv_err, rv_mask = upstream._batched_observation_block(
        rv_times,
        observed_rv.value if observed_rv is not None else None,
        observed_rv_err.value if observed_rv_err is not None else None,
        rv_mask,
        nfits,
    )
    lc_times, observed_lc, observed_lc_err, lc_mask = upstream._batched_observation_block(
        lc_times,
        observed_lc.value if hasattr(observed_lc, "value") else observed_lc,
        observed_lc_err.value if hasattr(observed_lc, "value") else observed_lc_err,
        lc_mask,
        nfits,
    )
    default_sls = SelfLensingSystem(
        0 * SelfLensingSystem.param_units["Macc"],
        0 * SelfLensingSystem.param_units["Mcomp"],
        0 * SelfLensingSystem.param_units["Rcomp"],
        0 * SelfLensingSystem.param_units["a"],
    )
    likelihood_args = {
        "Mcomp": Mcomp.value,
        "Rcomp": Rcomp.value,
        "tau_eff": tau_eff.value if tau_eff is not None else None,
        "d": d.value,
        "peak_time": peak_time,
        "rv_times": rv_times,
        "observed_rv": observed_rv,
        "observed_rv_err": observed_rv_err,
        "lc_times": lc_times,
        "observed_lc": observed_lc,
        "observed_lc_err": observed_lc_err,
        "detected_periods": jnp.array(upstream._batched_periods(periods, nfits)),
        "lensing_method": lensing_method,
        "center_lc_model": center_lc_model,
        "period_min": period_min,
        "period_max": period_max,
        "model_sls": default_sls,
        "rv_mask": rv_mask,
        "lc_mask": lc_mask,
        "device": fit_device,
    }
    forward_value_and_grad = None
    if kernel == "mala":
        experimental_rows = PhotometryRow(
            times_days=jax.device_put(
                jax_numpy.asarray(_extract_value(lc_times)), fit_device
            ),
            flux=jax.device_put(jax_numpy.asarray(observed_lc), fit_device),
            flux_err=jax.device_put(jax_numpy.asarray(observed_lc_err), fit_device),
            mask=jax.device_put(jax_numpy.asarray(lc_mask, dtype=bool), fit_device),
            known=jax.device_put(
                jax_numpy.column_stack((Mcomp.value, Rcomp.value, Teff.value)),
                fit_device,
            ),
            peak_time_days=jax.device_put(
                jax_numpy.asarray(_extract_value(peak_time)), fit_device
            ),
        )

        def forward_value_and_grad(positions):
            return batched_photometric_logprob_and_grad(positions, experimental_rows)

        logprob_vec = lambda positions: forward_value_and_grad(positions)[0]
    elif mala_gradient_mode == "forward":
        forward_value_and_grad = make_batched_scalar_logprob_and_forward_gradient(
            **likelihood_args
        )
        logprob_vec = lambda positions: forward_value_and_grad(positions)[0]
    else:
        logprob_vec = upstream.make_batched_combined_chi2_vec_jax(
            likelihood_args["Mcomp"], likelihood_args["Rcomp"], likelihood_args["tau_eff"],
            likelihood_args["d"], likelihood_args["peak_time"], likelihood_args["rv_times"],
            likelihood_args["observed_rv"], likelihood_args["observed_rv_err"],
            likelihood_args["lc_times"], likelihood_args["observed_lc"], likelihood_args["observed_lc_err"], None,
            likelihood_args["detected_periods"], likelihood_args["lensing_method"],
            center_lc_model, weight_lc_peak, period_min, period_max, default_sls,
            rv_mask, lc_mask, fit_device,
        )
    p0 = upstream._initial_batched_walker_positions(nfits, int(nwalkers), initial_walker_state)
    if proposal_scale is None:
        proposal_scale = upstream._default_blackjax_proposal_scale(proposal_scale_fraction)
    proposal_scale = upstream._scaled_proposal_scale(
        proposal_scale, observed_rv, observed_rv_err, rv_mask, observed_lc, lc_mask,
        proposal_reference_observations,
    )
    sampler_args = (
        logprob_vec,
        jax.device_put(jax_numpy.asarray(p0), fit_device),
        jax.device_put(jax_numpy.asarray(proposal_scale), fit_device),
    )
    common_args = {
        "warmup_steps": warmup_steps,
        "adaptation_rounds": adaptation_rounds,
        "target_acceptance": target_acceptance,
        "adaptation_gain": adaptation_gain,
        "scale_change_limit": scale_change_limit,
        "production_steps": nsteps,
        "random_seed": random_seed,
    }
    if kernel == "random_walk":
        result = adaptive_batched_random_walk(
            *sampler_args,
            adapt_covariance=adapt_covariance,
            covariance_elite_multiplier=covariance_elite_multiplier,
            covariance_regularization=covariance_regularization,
            **common_args,
        )
    elif kernel == "mala":
        result = adaptive_batched_mala(
            *sampler_args,
            initial_step_size=mala_initial_step_size,
            value_and_grad_vec=forward_value_and_grad,
            **common_args,
        )
    else:
        raise ValueError("kernel must be 'random_walk' or 'mala'")
    flat_positions = result.chain[burn:].transpose(1, 0, 2, 3).reshape(nfits, -1, result.chain.shape[-1])
    flat_log_prob = result.log_prob[burn:].transpose(1, 0, 2).reshape(nfits, -1)
    top_k = min(upstream.DEFAULT_POSTERIOR_SUMMARY_TOP_K, flat_log_prob.shape[1])
    top_indices = numpy.argpartition(flat_log_prob, -top_k, axis=1)[:, -top_k:]
    top_positions = numpy.take_along_axis(flat_positions, top_indices[..., None], axis=1)
    top_log_prob = numpy.take_along_axis(flat_log_prob, top_indices, axis=1)
    summary_positions = upstream._posterior_summaries_from_flat_samples(
        jnp.asarray(top_positions), jnp.asarray(top_log_prob), jnp.asarray(top_positions[:, 0, :])
    )
    systems = upstream._best_fit_systems_from_positions(
        jax.device_get(summary_positions), Rcomp, Mcomp, Teff, d
    )
    samplers = None
    if return_sampler:
        samplers = []
        for fit_index in range(nfits):
            sampler = upstream.BlackJAXSamplerResult(
                result.chain[:, fit_index], result.log_prob[:, fit_index],
                acceptance=result.acceptance[:, fit_index],
            )
            sampler.warmup_acceptance = result.warmup_acceptance[:, fit_index]
            sampler.proposal_scale = result.proposal_scale[fit_index]
            samplers.append(sampler)
    walker_state = (
        upstream.BlackJAXFitState(coords=result.final_positions)
        if return_walker_state else None
    )
    return systems, samplers, walker_state
