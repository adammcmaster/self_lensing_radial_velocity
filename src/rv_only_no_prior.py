"""BlackJAX RV-only fits with a free RV phase.

This module intentionally lives in this project instead of the upstream
``selflensing`` package.  It mirrors the BlackJAX random-walk Metropolis
Hastings fitter used upstream, but samples an additional RV phase so the fit is
not conditioned on the photometric peak-time prior.
"""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jax_numpy
import numpy
import unxt
from astropy import units
from quax import quaxify
from quaxed import numpy as jnp
from tqdm import tqdm

from selflensing.fitting._utils import _device_array, _extract_value, _select_fit_device
from selflensing.fitting.blackjax import (
    DEFAULT_POSTERIOR_SUMMARY_TOP_K,
    DEFAULT_PROPOSAL_REFERENCE_OBSERVATIONS,
    BlackJAXSamplerResult,
    _batched_quantity,
    _count_observations,
    _update_batched_top_posterior_samples,
    _update_top_posterior_samples,
)
from selflensing.generators import RVGenerator, SelfLensingLightcurveGenerator
from selflensing.gp.PriorsSamplers import log_prior_bounds
from selflensing.models import DEFAULT_LIMITS, SelfLensingSystem

logger = logging.getLogger(__name__)

RV_PHASE_INDEX = 5
N_DIM = 6
RV_PHASE_LIMITS = (0.0, 2.0 * numpy.pi)
CIRCULAR_INDICES = (4, 5)


@dataclass
class RVNoPriorFitState:
    coords: numpy.ndarray


@dataclass
class RVNoPriorFitResult:
    system: SelfLensingSystem
    rv_phase: float
    rv_reference_time: float
    rv_peak_time: float


def _reference_time(rv_times):
    values = numpy.asarray(_extract_value(rv_times), dtype=float)
    values = values[numpy.isfinite(values)]
    if values.size == 0:
        raise ValueError("RV-only no-prior fit requires at least one finite RV time")
    return float(numpy.min(values))


def _initial_walker_positions(nwalkers, initial_walker_state=None):
    if initial_walker_state is not None:
        initial_walker_coords = getattr(
            initial_walker_state,
            "coords",
            initial_walker_state,
        )
        initial_walker_coords = numpy.asarray(initial_walker_coords)
        if initial_walker_coords.ndim != 2 or initial_walker_coords.shape[1] != N_DIM:
            raise ValueError(
                "initial_walker_state must have coordinates with shape "
                f"(nwalkers, {N_DIM})"
            )
        return initial_walker_coords

    p0 = [
        numpy.random.uniform(bounds[0], bounds[1], nwalkers)
        for bounds in (
            DEFAULT_LIMITS["i"],
            DEFAULT_LIMITS["Macc"],
            DEFAULT_LIMITS["e"],
            DEFAULT_LIMITS["periapsis_phase"],
            RV_PHASE_LIMITS,
        )
    ]
    p0.insert(
        0,
        jnp.logspace(
            jnp.log10(DEFAULT_LIMITS["a"][0]),
            jnp.log10(DEFAULT_LIMITS["a"][1]),
            nwalkers,
        ),
    )
    return numpy.asarray(list(zip(*p0)), dtype=float)


def _initial_batched_walker_positions(
    nfits,
    nwalkers,
    initial_walker_state=None,
):
    if initial_walker_state is not None:
        initial_walker_coords = getattr(
            initial_walker_state,
            "coords",
            initial_walker_state,
        )
        initial_walker_coords = numpy.asarray(initial_walker_coords)
        if initial_walker_coords.ndim == 2:
            if initial_walker_coords.shape[1] != N_DIM:
                raise ValueError(
                    "initial_walker_state must have coordinates with shape "
                    f"(nwalkers, {N_DIM}) or (nfits, nwalkers, {N_DIM})"
                )
            if initial_walker_coords.shape[0] != nwalkers:
                nwalkers = initial_walker_coords.shape[0]
            return numpy.broadcast_to(
                initial_walker_coords[None, :, :],
                (nfits, nwalkers, N_DIM),
            ).copy()
        if (
            initial_walker_coords.ndim == 3
            and initial_walker_coords.shape[0] == nfits
            and initial_walker_coords.shape[2] == N_DIM
        ):
            return initial_walker_coords
        raise ValueError(
            "initial_walker_state must have coordinates with shape "
            f"(nwalkers, {N_DIM}) or (nfits, nwalkers, {N_DIM})"
        )

    return numpy.stack(
        [_initial_walker_positions(nwalkers) for _ in range(nfits)],
        axis=0,
    )


def _default_proposal_scale(scale_fraction):
    return jnp.asarray(
        [
            scale_fraction * (DEFAULT_LIMITS["a"][1] - DEFAULT_LIMITS["a"][0]),
            scale_fraction * (DEFAULT_LIMITS["i"][1] - DEFAULT_LIMITS["i"][0]),
            scale_fraction * (DEFAULT_LIMITS["Macc"][1] - DEFAULT_LIMITS["Macc"][0]),
            scale_fraction * (DEFAULT_LIMITS["e"][1] - DEFAULT_LIMITS["e"][0]),
            scale_fraction
            * (
                DEFAULT_LIMITS["periapsis_phase"][1]
                - DEFAULT_LIMITS["periapsis_phase"][0]
            ),
            scale_fraction * (RV_PHASE_LIMITS[1] - RV_PHASE_LIMITS[0]),
        ],
        dtype=float,
    )


def _proposal_observation_counts(observed_rv, rv_mask):
    rv_count = _count_observations(observed_rv, rv_mask)
    return 1 if rv_count is None else rv_count


def _posterior_summary_from_flat_samples(flat_samples, flat_log_prob, fallback_position):
    finite = jnp.isfinite(flat_log_prob)
    count = jnp.sum(finite)
    weights = finite.astype(float)
    safe_count = jnp.maximum(count, 1)
    regular_sum = jnp.sum(flat_samples * weights[:, None], axis=0)
    summary = regular_sum / safe_count
    for index in CIRCULAR_INDICES:
        phase = jnp.mod(
            jnp.arctan2(
                jnp.sum(jnp.sin(flat_samples[:, index]) * weights),
                jnp.sum(jnp.cos(flat_samples[:, index]) * weights),
            ),
            2 * jnp.pi,
        )
        summary = summary.at[index].set(phase)
    return jnp.where(count > 0, summary, fallback_position)


def _posterior_summaries_from_flat_samples(
    flat_positions,
    flat_log_prob,
    fallback_position,
):
    finite = jnp.isfinite(flat_log_prob)
    counts = jnp.sum(finite, axis=1)
    weights = finite.astype(float)
    safe_counts = jnp.maximum(counts, 1)
    regular_sum = jnp.sum(flat_positions * weights[:, :, None], axis=1)
    summaries = regular_sum / safe_counts[:, None]
    for index in CIRCULAR_INDICES:
        phases = jnp.mod(
            jnp.arctan2(
                jnp.sum(jnp.sin(flat_positions[:, :, index]) * weights, axis=1),
                jnp.sum(jnp.cos(flat_positions[:, :, index]) * weights, axis=1),
            ),
            2 * jnp.pi,
        )
        summaries = summaries.at[:, index].set(phases)
    return jnp.where(counts[:, None] > 0, summaries, fallback_position)


def _scaled_rv_proposal_scale(
    proposal_scale,
    observed_rv,
    rv_mask,
    reference_observations=DEFAULT_PROPOSAL_REFERENCE_OBSERVATIONS,
):
    counts = numpy.asarray(_proposal_observation_counts(observed_rv, rv_mask), dtype=float)
    counts = numpy.maximum(counts, 1.0)
    if reference_observations is None:
        finite_counts = counts[numpy.isfinite(counts)]
        reference_observations = numpy.min(finite_counts) if finite_counts.size else 1.0
    reference_observations = max(float(reference_observations), 1.0)
    factor = numpy.minimum(1.0, numpy.sqrt(reference_observations / counts))
    proposal_scale = numpy.asarray(proposal_scale, dtype=float)
    if factor.ndim == 0:
        return proposal_scale * float(factor)
    if proposal_scale.ndim == 1:
        proposal_scale = proposal_scale[None, :]
    return factor[:, None, None] * proposal_scale[:, None, :]


@quaxify
def rv_no_prior_logprob(
    params,
    Mcomp,
    Rcomp,
    d,
    rv_times,
    observed_rv,
    observed_rv_err,
    rv_reference_time,
    lensing_method,
    rv_mask=None,
):
    Mcomp = unxt.Quantity(Mcomp, SelfLensingSystem.param_units["Mcomp"])
    Rcomp = unxt.Quantity(Rcomp, SelfLensingSystem.param_units["Rcomp"])
    d = unxt.Quantity(d, SelfLensingSystem.param_units["d"])
    observed_rv = unxt.Quantity(observed_rv, units.m / units.s)
    observed_rv_err = unxt.Quantity(observed_rv_err, units.m / units.s)

    physical_params = params[:RV_PHASE_INDEX]
    rv_phase = params[RV_PHASE_INDEX]
    lp_bounds = log_prior_bounds(
        physical_params,
        (
            DEFAULT_LIMITS["a"],
            DEFAULT_LIMITS["i"],
            DEFAULT_LIMITS["Macc"],
            DEFAULT_LIMITS["e"],
            DEFAULT_LIMITS["periapsis_phase"],
        ),
    )
    phase_ok = jnp.logical_and(
        rv_phase >= RV_PHASE_LIMITS[0],
        rv_phase < RV_PHASE_LIMITS[1],
    )

    def return_neg_inf(*args, **kwargs):
        return -jnp.inf

    @quaxify
    def compute_full_logprob(_):
        a, i, Macc, e, periapsis_phase = physical_params
        model_sls = SelfLensingSystem(
            a=unxt.Quantity(a, SelfLensingSystem.param_units["a"]),
            i=unxt.Quantity(i, SelfLensingSystem.param_units["i"]),
            Macc=unxt.Quantity(Macc, SelfLensingSystem.param_units["Macc"]),
            Mcomp=Mcomp,
            Rcomp=Rcomp,
            e=unxt.Quantity(e, SelfLensingSystem.param_units["e"]),
            periapsis_phase=unxt.Quantity(
                periapsis_phase,
                SelfLensingSystem.param_units["periapsis_phase"],
            ),
            d=d,
        )

        def condition_value(condition):
            return jnp.asarray(
                condition.value if hasattr(condition, "value") else condition
            )

        def calc_rv_chi2():
            rv_epoch = rv_reference_time + (
                rv_phase / (2.0 * jnp.pi)
            ) * model_sls.porb.to(units.day).value
            rv_gen = RVGenerator(model_sls, times=rv_times, peak_time=rv_epoch)
            model_rv = rv_gen.rv.to(units.m / units.s).value
            if rv_mask is None:
                weight = 1.0
            else:
                weight = jnp.asarray(rv_mask, dtype=bool).astype(float)
            return -jnp.sum(weight * (observed_rv.value - model_rv) ** 2 / observed_rv_err.value**2)

        return jax.lax.cond(
            condition_value(model_sls.sl_possible),
            calc_rv_chi2,
            return_neg_inf,
        )

    return jax.lax.cond(
        jnp.logical_and(jnp.isfinite(lp_bounds), phase_ok),
        compute_full_logprob,
        return_neg_inf,
        operand=None,
    )


def make_rv_no_prior_logprob_vec(
    Mcomp,
    Rcomp,
    d,
    rv_times,
    observed_rv,
    observed_rv_err,
    rv_reference_time,
    lensing_method,
    rv_mask=None,
    device=None,
):
    rv_times = _device_array(_extract_value(rv_times), device)
    observed_rv = _device_array(_extract_value(observed_rv), device)
    observed_rv_err = _device_array(_extract_value(observed_rv_err), device)
    Mcomp = _device_array(_extract_value(Mcomp), device)
    Rcomp = _device_array(_extract_value(Rcomp), device)
    d = _device_array(_extract_value(d), device)
    rv_reference_time = _device_array(rv_reference_time, device)
    rv_mask = _device_array(_extract_value(rv_mask), device)

    @quaxify
    def _logprob_vec(params):
        params = jax.device_put(jax_numpy.asarray(params), device)
        return jnp.asarray(
            jax.vmap(
                lambda p: rv_no_prior_logprob(
                    p,
                    Mcomp,
                    Rcomp,
                    d,
                    rv_times,
                    observed_rv,
                    observed_rv_err,
                    rv_reference_time,
                    lensing_method,
                    rv_mask,
                )
            )(params)
        )

    logprob_vec = jax.jit(_logprob_vec)

    def logprob(params):
        return logprob_vec(jax.device_put(params, device))

    return logprob


def make_batched_rv_no_prior_logprob_vec(
    lensing_method,
    device=None,
):
    @quaxify
    def _batched_logprob_vec(
        params,
        Mcomp,
        Rcomp,
        d,
        rv_times,
        observed_rv,
        observed_rv_err,
        rv_reference_time,
        rv_mask,
    ):
        params = jax.device_put(jax_numpy.asarray(params), device)
        return jnp.asarray(
            jax.vmap(
                lambda row_params, row_Mcomp, row_Rcomp, row_d, row_rv_times,
                row_observed_rv, row_observed_rv_err, row_rv_reference_time,
                row_rv_mask: jax.vmap(
                    lambda p: rv_no_prior_logprob(
                        p,
                        row_Mcomp,
                        row_Rcomp,
                        row_d,
                        row_rv_times,
                        row_observed_rv,
                        row_observed_rv_err,
                        row_rv_reference_time,
                        lensing_method,
                        row_rv_mask,
                    )
                )(row_params)
            )(
                params,
                Mcomp,
                Rcomp,
                d,
                rv_times,
                observed_rv,
                observed_rv_err,
                rv_reference_time,
                rv_mask,
            )
        )

    return jax.jit(_batched_logprob_vec)


def _fit_result_from_position(position, Rcomp, Mcomp, Teff, d, rv_reference_time):
    out_a, out_i, out_Macc, out_e, out_periapsis_phase, out_rv_phase = position
    system = SelfLensingSystem(
        a=out_a * SelfLensingSystem.param_units["a"],
        i=out_i * SelfLensingSystem.param_units["i"],
        Macc=out_Macc * SelfLensingSystem.param_units["Macc"],
        Rcomp=Rcomp,
        Mcomp=Mcomp,
        Teff=Teff,
        e=out_e * SelfLensingSystem.param_units["e"],
        periapsis_phase=out_periapsis_phase
        * SelfLensingSystem.param_units["periapsis_phase"],
        d=d,
    )
    rv_peak_time = float(
        rv_reference_time + (out_rv_phase / (2.0 * numpy.pi)) * system.porb.value
    )
    return RVNoPriorFitResult(
        system=system,
        rv_phase=float(out_rv_phase),
        rv_reference_time=float(rv_reference_time),
        rv_peak_time=rv_peak_time,
    )


def _fit_results_from_positions(positions, Rcomp, Mcomp, Teff, d, rv_reference_time):
    return [
        _fit_result_from_position(
            position,
            Rcomp[index],
            Mcomp[index],
            Teff[index],
            d[index],
            rv_reference_time[index],
        )
        for index, position in enumerate(numpy.asarray(positions))
    ]


@units.quantity_input
def self_lensing_system_from_rv_fit_no_peak_prior_blackjax(
    Mcomp: units.kg,
    rv_times,
    observed_rv,
    observed_rv_err,
    Rcomp: units.m,
    Teff: units.K = None,
    d: units.pc = SelfLensingSystem.DEFAULT_D,
    nwalkers=100,
    nsteps=5000,
    burn=500,
    lensing_method=None,
    progress=True,
    progress_desc=None,
    prefer_gpu=True,
    device=None,
    initial_walker_state=None,
    proposal_scale=None,
    proposal_scale_fraction=0.02,
    proposal_reference_observations=DEFAULT_PROPOSAL_REFERENCE_OBSERVATIONS,
    random_seed=0,
    progress_chunk_size=100,
    posterior_summary_top_k=DEFAULT_POSTERIOR_SUMMARY_TOP_K,
    return_sampler=True,
    return_walker_state=True,
):
    import blackjax

    fit_device = _select_fit_device(device=device, prefer_gpu=prefer_gpu)
    if lensing_method is None:
        lensing_method = (
            SelfLensingLightcurveGenerator.default_lensing_method_for_device(fit_device)
        )
    logger.info(
        "blackjax RV-only no-prior fit: using JAX %s device %s with lensing_method=%s",
        fit_device.platform,
        fit_device,
        lensing_method,
    )

    Mcomp = Mcomp.to(SelfLensingSystem.param_units["Mcomp"])
    Rcomp = Rcomp.to(SelfLensingSystem.param_units["Rcomp"])
    Teff = None if Teff is None else Teff.to(SelfLensingSystem.param_units["Teff"])
    d = d.to(SelfLensingSystem.param_units["d"])
    observed_rv = observed_rv.to(units.m / units.s)
    observed_rv_err = observed_rv_err.to(units.m / units.s)
    rv_reference_time = _reference_time(rv_times)

    nwalkers = int(nwalkers)
    nsteps = int(nsteps)
    burn = int(burn)
    p0 = _initial_walker_positions(nwalkers, initial_walker_state)
    nwalkers = p0.shape[0]

    logprob_vec = make_rv_no_prior_logprob_vec(
        Mcomp.value,
        Rcomp.value,
        d.value,
        rv_times,
        observed_rv.value,
        observed_rv_err.value,
        rv_reference_time,
        lensing_method,
        device=fit_device,
    )

    def logprob(position):
        return logprob_vec(position[jnp.newaxis, :])[0]

    if proposal_scale is None:
        proposal_scale = _default_proposal_scale(proposal_scale_fraction)
    proposal_scale = _scaled_rv_proposal_scale(
        proposal_scale,
        observed_rv.value,
        None,
        proposal_reference_observations,
    )
    proposal_scale = jax.device_put(jax_numpy.asarray(proposal_scale), fit_device)
    random_walk = blackjax.additive_step_random_walk(
        logprob,
        blackjax.mcmc.random_walk.normal(proposal_scale),
    )
    state = jax.vmap(random_walk.init)(jax.device_put(jax_numpy.asarray(p0), fit_device))

    def one_step(state, rng_key):
        walker_keys = jax.random.split(rng_key, nwalkers)
        new_state, info = jax.vmap(random_walk.step)(walker_keys, state)
        return new_state, (new_state.position, new_state.logdensity, info.is_accepted)

    @jax.jit
    def run_steps(state, rng_keys):
        return jax.lax.scan(one_step, state, rng_keys)

    rng_key = jax.random.PRNGKey(random_seed)
    chain_chunks = []
    log_prob_chunks = []
    acceptance_chunks = []
    posterior_summary_top_k = int(posterior_summary_top_k)
    top_positions = jnp.zeros((posterior_summary_top_k, N_DIM))
    top_log_prob = jnp.full((posterior_summary_top_k,), -jnp.inf)
    if not return_sampler and burn >= nsteps:
        raise ValueError("burn must be smaller than nsteps when return_sampler=False")
    remaining_steps = nsteps
    completed_steps = 0
    progress_iter = tqdm(total=nsteps, desc=progress_desc) if progress else None
    try:
        chunk_size = nsteps if not progress else max(1, int(progress_chunk_size))
        while remaining_steps > 0:
            current_chunk_size = min(chunk_size, remaining_steps)
            rng_key, sample_key = jax.random.split(rng_key)
            step_keys = jax.random.split(sample_key, current_chunk_size)
            state, (positions, log_prob, accepted) = run_steps(state, step_keys)
            if return_sampler:
                chain_chunks.append(positions)
                log_prob_chunks.append(log_prob)
                acceptance_chunks.append(accepted)
            else:
                keep_steps = (
                    jnp.arange(completed_steps, completed_steps + current_chunk_size)
                    >= burn
                )
                top_positions, top_log_prob = _update_top_posterior_samples(
                    top_positions,
                    top_log_prob,
                    positions,
                    log_prob,
                    keep_steps,
                )
            remaining_steps -= current_chunk_size
            completed_steps += current_chunk_size
            if progress_iter is not None:
                progress_iter.update(current_chunk_size)
    finally:
        if progress_iter is not None:
            progress_iter.close()

    final_walker_state = (
        RVNoPriorFitState(coords=jax.device_get(state.position))
        if return_walker_state
        else None
    )
    if return_sampler:
        chain = jax.device_get(jnp.concatenate(chain_chunks, axis=0))
        log_prob = jax.device_get(jnp.concatenate(log_prob_chunks, axis=0))
        acceptance = jax.device_get(jnp.concatenate(acceptance_chunks, axis=0))
        sampler = BlackJAXSamplerResult(chain, log_prob, acceptance=acceptance)
        flat_samples = jnp.asarray(sampler.get_chain(flat=True, discard=burn))
        flat_log_prob = jnp.asarray(sampler.get_log_prob(flat=True, discard=burn))
        summary_top_k = min(posterior_summary_top_k, flat_log_prob.shape[0])
        top_log_prob, top_indices = jax.lax.top_k(flat_log_prob, summary_top_k)
        top_positions = flat_samples[top_indices]
        summary_position = _posterior_summary_from_flat_samples(
            top_positions,
            top_log_prob,
            top_positions[0],
        )
    else:
        sampler = None
        summary_position = _posterior_summary_from_flat_samples(
            top_positions,
            top_log_prob,
            top_positions[0],
        )
    fit_result = _fit_result_from_position(
        jax.device_get(summary_position),
        Rcomp,
        Mcomp,
        Teff,
        d,
        rv_reference_time,
    )
    return fit_result, sampler, final_walker_state


@units.quantity_input
def self_lensing_systems_from_batched_rv_fit_no_peak_prior_blackjax(
    Mcomp: units.kg,
    rv_times,
    observed_rv,
    observed_rv_err,
    rv_mask,
    Rcomp: units.m,
    Teff: units.K = None,
    d: units.pc = SelfLensingSystem.DEFAULT_D,
    nwalkers=100,
    nsteps=5000,
    burn=500,
    lensing_method=None,
    progress=True,
    progress_desc=None,
    prefer_gpu=True,
    device=None,
    initial_walker_state=None,
    proposal_scale=None,
    proposal_scale_fraction=0.02,
    proposal_reference_observations=None,
    random_seed=0,
    progress_chunk_size=100,
    posterior_summary_top_k=DEFAULT_POSTERIOR_SUMMARY_TOP_K,
    return_sampler=False,
    return_walker_state=False,
):
    if return_sampler:
        raise NotImplementedError("Batched RV no-prior fit does not return full samplers")

    fit_device = _select_fit_device(device=device, prefer_gpu=prefer_gpu)
    if lensing_method is None:
        lensing_method = (
            SelfLensingLightcurveGenerator.default_lensing_method_for_device(fit_device)
        )
    logger.info(
        "batched blackjax RV-only no-prior fit: using JAX %s device %s with lensing_method=%s",
        fit_device.platform,
        fit_device,
        lensing_method,
    )

    Mcomp = Mcomp.to(SelfLensingSystem.param_units["Mcomp"])
    nfits = int(numpy.asarray(Mcomp.value).shape[0])
    Rcomp = _batched_quantity(
        Rcomp,
        SelfLensingSystem.param_units["Rcomp"],
        nfits,
        "Rcomp",
    )
    Mcomp = _batched_quantity(
        Mcomp,
        SelfLensingSystem.param_units["Mcomp"],
        nfits,
        "Mcomp",
    )
    Teff = _batched_quantity(
        Teff,
        SelfLensingSystem.param_units["Teff"],
        nfits,
        "Teff",
    )
    d = _batched_quantity(d, SelfLensingSystem.param_units["d"], nfits, "d")
    observed_rv = observed_rv.to(units.m / units.s)
    observed_rv_err = observed_rv_err.to(units.m / units.s)
    rv_times_value = numpy.asarray(_extract_value(rv_times), dtype=float)
    rv_mask_value = numpy.asarray(rv_mask, dtype=bool)
    rv_reference_time = numpy.asarray(
        [
            _reference_time(rv_times_value[index][rv_mask_value[index]])
            for index in range(nfits)
        ],
        dtype=float,
    )

    nwalkers = int(nwalkers)
    nsteps = int(nsteps)
    burn = int(burn)
    p0 = _initial_batched_walker_positions(
        nfits,
        nwalkers,
        initial_walker_state,
    )
    nwalkers = p0.shape[1]
    position = jax.device_put(jax_numpy.asarray(p0), fit_device)
    Mcomp_value = _device_array(Mcomp.value, fit_device)
    Rcomp_value = _device_array(Rcomp.value, fit_device)
    d_value = _device_array(d.value, fit_device)
    rv_times_value = _device_array(rv_times_value, fit_device)
    observed_rv_value = _device_array(observed_rv.value, fit_device)
    observed_rv_err_value = _device_array(observed_rv_err.value, fit_device)
    rv_reference_time = _device_array(rv_reference_time, fit_device)
    rv_mask_value = _device_array(rv_mask_value, fit_device)
    logprob_vec = make_batched_rv_no_prior_logprob_vec(
        lensing_method,
        fit_device,
    )
    logprob_inputs = (
        Mcomp_value,
        Rcomp_value,
        d_value,
        rv_times_value,
        observed_rv_value,
        observed_rv_err_value,
        rv_reference_time,
        rv_mask_value,
    )
    log_prob = logprob_vec(position, *logprob_inputs)

    if proposal_scale is None:
        proposal_scale = _default_proposal_scale(proposal_scale_fraction)
    proposal_scale = _scaled_rv_proposal_scale(
        proposal_scale,
        observed_rv.value,
        rv_mask_value,
        proposal_reference_observations,
    )
    proposal_scale = jax.device_put(jax_numpy.asarray(proposal_scale), fit_device)

    @jax.jit
    def run_steps(position, log_prob, rng_keys, proposal_scale, *logprob_inputs):
        def one_step(carry, rng_key):
            position, log_prob = carry
            proposal_key, accept_key = jax.random.split(rng_key)
            proposal = position + jax.random.normal(
                proposal_key,
                shape=position.shape,
            ) * proposal_scale
            proposal_log_prob = logprob_vec(proposal, *logprob_inputs)
            log_accept_ratio = proposal_log_prob - log_prob
            accepted = (
                jnp.log(jax.random.uniform(accept_key, shape=log_prob.shape))
                < log_accept_ratio
            )
            new_position = jnp.where(accepted[..., None], proposal, position)
            new_log_prob = jnp.where(accepted, proposal_log_prob, log_prob)
            return (new_position, new_log_prob), (
                new_position,
                new_log_prob,
                accepted,
            )

        return jax.lax.scan(one_step, (position, log_prob), rng_keys)

    rng_key = jax.random.PRNGKey(random_seed)
    posterior_summary_top_k = int(posterior_summary_top_k)
    top_positions = jnp.zeros((nfits, posterior_summary_top_k, N_DIM))
    top_log_prob = jnp.full((nfits, posterior_summary_top_k), -jnp.inf)
    if not return_sampler and burn >= nsteps:
        raise ValueError("burn must be smaller than nsteps when return_sampler=False")
    remaining_steps = nsteps
    completed_steps = 0
    progress_iter = tqdm(total=nsteps, desc=progress_desc) if progress else None
    try:
        chunk_size = nsteps if not progress else max(1, int(progress_chunk_size))
        while remaining_steps > 0:
            current_chunk_size = min(chunk_size, remaining_steps)
            rng_key, sample_key = jax.random.split(rng_key)
            step_keys = jax.random.split(sample_key, current_chunk_size)
            (position, log_prob), (positions, chunk_log_prob, _accepted) = run_steps(
                position, log_prob, step_keys, proposal_scale, *logprob_inputs
            )
            keep_steps = (
                jnp.arange(completed_steps, completed_steps + current_chunk_size) >= burn
            )
            top_positions, top_log_prob = _update_batched_top_posterior_samples(
                top_positions,
                top_log_prob,
                positions,
                chunk_log_prob,
                keep_steps,
            )
            remaining_steps -= current_chunk_size
            completed_steps += current_chunk_size
            if progress_iter is not None:
                progress_iter.update(current_chunk_size)
    finally:
        if progress_iter is not None:
            progress_iter.close()

    final_walker_state = (
        RVNoPriorFitState(coords=jax.device_get(position)) if return_walker_state else None
    )
    summary_position = _posterior_summaries_from_flat_samples(
        top_positions,
        top_log_prob,
        top_positions[:, 0, :],
    )
    fit_results = _fit_results_from_positions(
        jax.device_get(summary_position),
        Rcomp,
        Mcomp,
        Teff,
        d,
        rv_reference_time,
    )
    return fit_results, None, final_walker_state
