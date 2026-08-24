"""Estimating self-lensing model parameters directly from observables."""

import secrets

import jax
import jax.numpy as jnp
import numpy
import pandas

from selflensing.batch import make_batched_light_curve_generator
from selflensing.gp.PhysicalRelations import effective_temperature_K, stellar_radius_Rsun
from selflensing.lightcurves import PARAMETER_ORDER
from selflensing.orbits import (
    AU_M,
    DAY_S,
    M_SUN_KG,
    R_SUN_M,
    einstein_radius_m,
    orbital_period_days,
)


OBSERVABLE_SPACE_SAMPLE_SIZE = 10_000
# A fresh default seed is selected whenever this module is loaded. Callers that
# require reproducibility should pass an explicit seed.
OBSERVABLE_SPACE_RANDOM_SEED = secrets.randbits(32)


def weighted_mean_and_std(values, weights):
    """Return the weighted mean and local weighted dispersion of values."""
    mean = numpy.average(values, weights=weights)
    std = numpy.sqrt(numpy.average((values - mean) ** 2, weights=weights))
    return mean, std


def companion_stellar_properties(
    companion_mass_msun,
    companion_radius_rsun=None,
    companion_teff_k=None,
):
    """Return companion properties, deriving main-sequence values when omitted."""
    if companion_radius_rsun is None:
        companion_radius_rsun = float(stellar_radius_Rsun(companion_mass_msun))
    if companion_teff_k is None:
        companion_teff_k = float(effective_temperature_K(companion_mass_msun))
    return companion_mass_msun, companion_radius_rsun, companion_teff_k


def calc_self_lensing_observable_space(
    sample_size=OBSERVABLE_SPACE_SAMPLE_SIZE,
    random_seed=OBSERVABLE_SPACE_RANDOM_SEED,
    *,
    companion_mass_msun=1.0,
    companion_radius_rsun=None,
    companion_teff_k=None,
):
    """Sample self-lensing systems for the supplied companion-star properties."""
    jax.config.update("jax_enable_x64", True)
    (
        companion_mass_msun,
        companion_radius_rsun,
        companion_teff_k,
    ) = companion_stellar_properties(
        companion_mass_msun,
        companion_radius_rsun,
        companion_teff_k,
    )

    key = jax.random.key(random_seed)
    keys = jax.random.split(key, 5)

    def uniform(draw_key, low, high):
        return jax.random.uniform(draw_key, (sample_size,), minval=low, maxval=high)

    def log_uniform(draw_key, low, high):
        return 10 ** uniform(draw_key, numpy.log10(low), numpy.log10(high))

    semi_major_axis_au = log_uniform(keys[0], 0.3, 100.0)
    lens_mass_msun = log_uniform(keys[1], 3.0, 50.0)
    companion_mass = jnp.full(sample_size, companion_mass_msun)
    companion_radius = jnp.full(sample_size, companion_radius_rsun)
    companion_teff = jnp.full(sample_size, companion_teff_k)
    eccentricity = uniform(keys[2], 0.0, 0.6)
    periapsis_phase = uniform(keys[3], 0.0, 2.0 * numpy.pi)
    impact_fraction = uniform(keys[4], 0.0, 0.95)

    semi_major_axis_m = semi_major_axis_au * AU_M
    lens_mass_kg = lens_mass_msun * M_SUN_KG
    companion_mass_kg = companion_mass * M_SUN_KG
    companion_radius_m = companion_radius * R_SUN_M

    edge_on_einstein_radius_m = einstein_radius_m(
        semi_major_axis_m,
        jnp.ones(sample_size),
        lens_mass_kg,
        eccentricity,
        periapsis_phase,
    )
    maximum_impact = 1.0 + companion_radius_m / edge_on_einstein_radius_m
    impact_einstein_radii = impact_fraction * maximum_impact
    cos_inclination = jnp.clip(
        impact_einstein_radii * edge_on_einstein_radius_m / semi_major_axis_m,
        0.0,
        0.2,
    )
    inclination = jnp.arccos(cos_inclination)

    param_matrix = jnp.stack(
        [
            {
                "a": semi_major_axis_au,
                "i": inclination,
                "Macc": lens_mass_msun,
                "Mcomp": companion_mass,
                "Rcomp": companion_radius,
                "Teff": companion_teff,
                "e": eccentricity,
                "periapsis_phase": periapsis_phase,
            }[name]
            for name in PARAMETER_ORDER
        ],
        axis=-1,
    )

    sin_inclination = jnp.sin(inclination)
    orbital_period_days_value = orbital_period_days(
        semi_major_axis_m,
        lens_mass_kg + companion_mass_kg,
    )
    orbital_period_seconds = orbital_period_days_value * DAY_S
    einstein_radius_m_value = einstein_radius_m(
        semi_major_axis_m,
        sin_inclination,
        lens_mass_kg,
        eccentricity,
        periapsis_phase,
    )
    impact_parameter = jnp.abs(
        semi_major_axis_m * jnp.cos(inclination) / einstein_radius_m_value
    )
    maximum_impact = 1.0 + companion_radius_m / einstein_radius_m_value
    crossing_time_days = (
        (orbital_period_seconds * (einstein_radius_m_value + companion_radius_m))
        / (jnp.pi * semi_major_axis_m * sin_inclination)
        * jnp.sqrt(
            jnp.clip(1.0 - (impact_parameter / maximum_impact) ** 2, 0.0, 1.0)
        )
    ) / DAY_S

    generate = make_batched_light_curve_generator(
        jnp.array([0.0]),
        peak_times=0.0,
        sl=True,
        ev=False,
        db=False,
        device=jax.devices()[0],
    )
    peak_flux = numpy.asarray(generate(param_matrix).block_until_ready())[:, 0]
    peak_amplification_ppm = 1e6 * (peak_flux - 1.0)
    crossing_time_hr = numpy.asarray(crossing_time_days) * 24.0
    orbital_period_yr = numpy.asarray(orbital_period_days_value) / 365.25
    semi_major_axis = numpy.asarray(semi_major_axis_au)
    lens_mass = numpy.asarray(lens_mass_msun)
    eccentricity = numpy.asarray(eccentricity)
    periapsis_phase = numpy.asarray(periapsis_phase)
    impact_fraction = numpy.asarray(impact_fraction)
    inclination = numpy.asarray(inclination)

    valid = (
        numpy.isfinite(peak_amplification_ppm)
        & numpy.isfinite(crossing_time_hr)
        & numpy.isfinite(orbital_period_yr)
        & (peak_amplification_ppm > 1.0)
        & (crossing_time_hr > 0.0)
        & (orbital_period_yr > 0.0)
    )
    if not valid.any():
        raise ValueError("No valid self-lensing observable-space samples were generated")

    return {
        "sample_size": sample_size,
        "random_seed": random_seed,
        "companion_mass_msun": companion_mass_msun,
        "companion_radius_rsun": companion_radius_rsun,
        "companion_teff_k": companion_teff_k,
        "peak_amplification_ppm": peak_amplification_ppm,
        "crossing_time_hr": crossing_time_hr,
        "orbital_period_yr": orbital_period_yr,
        "semi_major_axis_au": semi_major_axis,
        "lens_mass_msun": lens_mass,
        "eccentricity": eccentricity,
        "periapsis_phase_rad": periapsis_phase,
        "impact_fraction": impact_fraction,
        "inclination_rad": inclination,
        "valid": valid,
    }


def nearest_self_lensing_observable_space(
    peak_amplification_ppm,
    crossing_time_hr,
    *,
    n_neighbors=50,
    sample_size=OBSERVABLE_SPACE_SAMPLE_SIZE,
    random_seed=OBSERVABLE_SPACE_RANDOM_SEED,
    companion_mass_msun=1.0,
    companion_radius_rsun=None,
    companion_teff_k=None,
):
    """Return the sampled systems nearest an observed self-lensing signal.

    Distances are Euclidean in log10 peak-amplification and log10 crossing-time
    space, giving both observable axes equal weight across their dynamic ranges.
    """
    if peak_amplification_ppm <= 0.0 or crossing_time_hr <= 0.0:
        raise ValueError("Peak amplification and crossing time must be positive")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least one")

    estimates = calc_self_lensing_observable_space(
        sample_size=sample_size,
        random_seed=random_seed,
        companion_mass_msun=companion_mass_msun,
        companion_radius_rsun=companion_radius_rsun,
        companion_teff_k=companion_teff_k,
    )
    valid_indices = numpy.flatnonzero(estimates["valid"])
    neighbor_count = min(n_neighbors, len(valid_indices))
    log_distance = numpy.hypot(
        numpy.log10(estimates["peak_amplification_ppm"][valid_indices])
        - numpy.log10(peak_amplification_ppm),
        numpy.log10(estimates["crossing_time_hr"][valid_indices])
        - numpy.log10(crossing_time_hr),
    )
    nearest_indices = valid_indices[numpy.argsort(log_distance)[:neighbor_count]]

    neighbor_columns = (
        "peak_amplification_ppm",
        "crossing_time_hr",
        "semi_major_axis_au",
        "lens_mass_msun",
        "orbital_period_yr",
        "eccentricity",
        "periapsis_phase_rad",
        "impact_fraction",
        "inclination_rad",
    )
    neighbors = pandas.DataFrame(
        {column: estimates[column][nearest_indices] for column in neighbor_columns}
    )
    neighbors.insert(0, "observable_space_index", nearest_indices)
    neighbors["log_observable_distance"] = numpy.sort(log_distance)[:neighbor_count]
    neighbors.attrs.update(
        {
            "target_peak_amplification_ppm": peak_amplification_ppm,
            "target_crossing_time_hr": crossing_time_hr,
            "random_seed": random_seed,
            "companion_mass_msun": companion_mass_msun,
            "companion_radius_rsun": companion_radius_rsun,
            "companion_teff_k": companion_teff_k,
        }
    )
    return neighbors


def estimate_lens_mass_and_orbital_period(
    peak_amplification_ppm,
    crossing_time_hr,
    *,
    n_neighbors=500,
    sample_size=OBSERVABLE_SPACE_SAMPLE_SIZE,
    random_seed=OBSERVABLE_SPACE_RANDOM_SEED,
    companion_mass_msun=1.0,
    companion_radius_rsun=None,
    companion_teff_k=None,
):
    """Estimate mass, period, and local uncertainty from weighted neighbours.

    Neighbours are weighted by the inverse of their distance in log-observable
    space. If an exact match is present, only exact matches contribute. The
    uncertainty is the weighted neighbour dispersion, not a formal standard
    error because the simulated neighbours are not independent draws.
    """
    neighbors = nearest_self_lensing_observable_space(
        peak_amplification_ppm,
        crossing_time_hr,
        n_neighbors=n_neighbors,
        sample_size=sample_size,
        random_seed=random_seed,
        companion_mass_msun=companion_mass_msun,
        companion_radius_rsun=companion_radius_rsun,
        companion_teff_k=companion_teff_k,
    )
    distances = neighbors["log_observable_distance"].to_numpy()
    exact_matches = numpy.isclose(distances, 0.0)
    if exact_matches.any():
        weights = exact_matches.astype(float)
    else:
        weights = 1.0 / distances
    weights /= weights.sum()

    lens_mass, lens_mass_uncertainty = weighted_mean_and_std(
        neighbors["lens_mass_msun"], weights
    )
    orbital_period, orbital_period_uncertainty = weighted_mean_and_std(
        neighbors["orbital_period_yr"], weights
    )
    return {
        "lens_mass_msun": float(lens_mass),
        "lens_mass_uncertainty_msun": float(lens_mass_uncertainty),
        "orbital_period_yr": float(orbital_period),
        "orbital_period_uncertainty_yr": float(orbital_period_uncertainty),
        "effective_n_neighbors": float(1.0 / numpy.square(weights).sum()),
    }


def estimate_lens_mass_and_orbital_period_batch(
    peak_amplification_ppm,
    crossing_time_hr,
    *,
    n_neighbors=500,
    sample_size=OBSERVABLE_SPACE_SAMPLE_SIZE,
    random_seed=OBSERVABLE_SPACE_RANDOM_SEED,
    companion_mass_msun=1.0,
    companion_radius_rsun=None,
    companion_teff_k=None,
    batch_size=256,
):
    """Estimate masses, periods, and local uncertainty for many signals.

    This is the vectorized counterpart of
    :func:`estimate_lens_mass_and_orbital_period`. It uses the identical
    inverse-distance weighting but calculates the sampled reference space once.
    The returned uncertainty is the weighted neighbour dispersion, not a formal
    standard error: the nearby simulated systems are not independent draws.
    """
    peak_amplification_ppm, crossing_time_hr = numpy.broadcast_arrays(
        numpy.asarray(peak_amplification_ppm, dtype=float),
        numpy.asarray(crossing_time_hr, dtype=float),
    )
    if (
        peak_amplification_ppm.ndim != 1
        or not numpy.isfinite(peak_amplification_ppm).all()
        or not numpy.isfinite(crossing_time_hr).all()
        or (peak_amplification_ppm <= 0.0).any()
        or (crossing_time_hr <= 0.0).any()
    ):
        raise ValueError("Observables must be finite, positive one-dimensional arrays")
    if n_neighbors < 1 or batch_size < 1:
        raise ValueError("n_neighbors and batch_size must be at least one")

    estimates = calc_self_lensing_observable_space(
        sample_size=sample_size,
        random_seed=random_seed,
        companion_mass_msun=companion_mass_msun,
        companion_radius_rsun=companion_radius_rsun,
        companion_teff_k=companion_teff_k,
    )
    valid = estimates["valid"]
    reference_peak = numpy.log10(estimates["peak_amplification_ppm"][valid])
    reference_crossing_time = numpy.log10(estimates["crossing_time_hr"][valid])
    reference_mass = estimates["lens_mass_msun"][valid]
    reference_period = estimates["orbital_period_yr"][valid]
    neighbor_count = min(n_neighbors, len(reference_peak))
    estimated_mass = numpy.empty(len(peak_amplification_ppm))
    mass_uncertainty = numpy.empty(len(peak_amplification_ppm))
    estimated_period = numpy.empty(len(peak_amplification_ppm))
    period_uncertainty = numpy.empty(len(peak_amplification_ppm))
    effective_n_neighbors = numpy.empty(len(peak_amplification_ppm))

    for start in range(0, len(peak_amplification_ppm), batch_size):
        stop = min(start + batch_size, len(peak_amplification_ppm))
        distances = numpy.hypot(
            numpy.log10(peak_amplification_ppm[start:stop, None]) - reference_peak,
            numpy.log10(crossing_time_hr[start:stop, None]) - reference_crossing_time,
        )
        nearest = numpy.argpartition(distances, neighbor_count - 1, axis=1)[
            :, :neighbor_count
        ]
        nearest_distances = numpy.take_along_axis(distances, nearest, axis=1)
        exact_matches = numpy.isclose(nearest_distances, 0.0)
        has_exact_match = exact_matches.any(axis=1, keepdims=True)
        inverse_distances = 1.0 / numpy.maximum(nearest_distances, numpy.finfo(float).eps)
        weights = numpy.where(
            has_exact_match,
            exact_matches.astype(float),
            inverse_distances,
        )
        weights /= weights.sum(axis=1, keepdims=True)
        estimated_mass[start:stop] = numpy.sum(weights * reference_mass[nearest], axis=1)
        estimated_period[start:stop] = numpy.sum(
            weights * reference_period[nearest], axis=1
        )
        mass_uncertainty[start:stop] = numpy.sqrt(
            numpy.sum(
                weights * (reference_mass[nearest] - estimated_mass[start:stop, None]) ** 2,
                axis=1,
            )
        )
        period_uncertainty[start:stop] = numpy.sqrt(
            numpy.sum(
                weights
                * (reference_period[nearest] - estimated_period[start:stop, None]) ** 2,
                axis=1,
            )
        )
        effective_n_neighbors[start:stop] = 1.0 / numpy.square(weights).sum(axis=1)

    return {
        "lens_mass_msun": estimated_mass,
        "lens_mass_uncertainty_msun": mass_uncertainty,
        "orbital_period_yr": estimated_period,
        "orbital_period_uncertainty_yr": period_uncertainty,
        "effective_n_neighbors": effective_n_neighbors,
    }
