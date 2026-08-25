"""RV follow-up schedules for self-lensing events.

Times are floating-point Julian dates (or any other common day-based epoch).
The parameter vectors use :data:`selflensing.lightcurves.PARAMETER_ORDER` and
its public units: AU, solar masses/radii, K, and radians.  Keeping the public
calculation numeric makes the posterior version directly usable with JAX/GPU
batches.
"""

import math

import jax
import jax.numpy as jnp
import numpy

from selflensing.lightcurves import (
    PARAMETER_ORDER,
    as_light_curve_parameter_array,
    radial_velocity_m_per_s,
)
from selflensing.orbits import AU_M, M_SUN_KG, orbital_period_days


# A dense, fixed grid is deliberate: it is readily vmapped/jitted and avoids
# scalar root finders in the eventual GPU batch workflow.
DELTA_GRID_SIZE = 4097


def _value_in_days(value):
    """Extract a numeric day value from a Time, Quantity, or float."""
    if hasattr(value, "jd"):
        return float(value.jd)
    if hasattr(value, "to_value"):
        from astropy import units

        return float(value.to_value(units.day))
    return float(value)


def _value_in_m_per_s(value):
    if hasattr(value, "to_value"):
        from astropy import units

        return float(value.to_value(units.m / units.s))
    return float(value)


def _parameter_rows(parameters):
    """Return selflensing parameter rows, accepting a system-like object too."""
    if all(hasattr(parameters, name) for name in PARAMETER_ORDER):
        parameters = {name: getattr(parameters, name) for name in PARAMETER_ORDER}
    return as_light_curve_parameter_array(parameters)


def _with_periapsis_phase_pi(parameter_rows):
    """Use selflensing's *phase angle* convention, not argument of periastron."""
    return parameter_rows.at[:, PARAMETER_ORDER.index("periapsis_phase")].set(jnp.pi)


def _period_days_from_parameters(parameter_rows):
    a_m = parameter_rows[:, PARAMETER_ORDER.index("a")] * AU_M
    total_mass_kg = (
        parameter_rows[:, PARAMETER_ORDER.index("Macc")]
        + parameter_rows[:, PARAMETER_ORDER.index("Mcomp")]
    ) * M_SUN_KG
    return orbital_period_days(a_m, total_mass_kg)


def _set_period(parameter_rows, periods_days):
    """Rescale ``a`` so a posterior period draw is honoured exactly."""
    model_periods = _period_days_from_parameters(parameter_rows)
    scale = (jnp.asarray(periods_days) / model_periods) ** (2.0 / 3.0)
    return parameter_rows.at[:, PARAMETER_ORDER.index("a")].multiply(scale)


def _rv_changes_about_epoch(parameter_rows, epoch, offsets):
    """Absolute RV changes at symmetric offsets for every parameter row."""
    parameter_rows = _with_periapsis_phase_pi(parameter_rows)
    offsets = jnp.asarray(offsets)

    def one_row(row):
        rv0 = radial_velocity_m_per_s(jnp.asarray([epoch]), row, epoch)[0]
        rv_plus = radial_velocity_m_per_s(epoch + offsets, row, epoch)
        rv_minus = radial_velocity_m_per_s(epoch - offsets, row, epoch)
        # One delta must make both flanking observations distinguishable.
        return jnp.minimum(jnp.abs(rv_plus - rv0), jnp.abs(rv_minus - rv0))

    return jax.vmap(one_row)(parameter_rows)


def _delta_for_significance(parameter_rows, epoch, target_change, max_delta, credibility):
    offsets = jnp.linspace(0.0, max_delta, DELTA_GRID_SIZE)
    changes = _rv_changes_about_epoch(parameter_rows, epoch, offsets)
    # The lower tail gives a delta that succeeds for ``credibility`` of draws.
    conservative_change = jnp.quantile(changes, 1.0 - credibility, axis=0)
    reached = conservative_change >= target_change
    index = int(numpy.asarray(jnp.argmax(reached)))
    if not bool(numpy.asarray(jnp.any(reached))):
        # Some posterior draws predict an RV amplitude too small to achieve
        # the requested significance within the available observing window.
        # Keep the informative targeted triplet, at the largest permissible
        # separation, rather than aborting all schedules for that system.
        return float(max_delta)
    if index == 0:
        return 0.0

    lower_offset, upper_offset = offsets[index - 1], offsets[index]
    lower_change, upper_change = (
        conservative_change[index - 1],
        conservative_change[index],
    )
    fraction = (target_change - lower_change) / (upper_change - lower_change)
    return float(numpy.asarray(lower_offset + fraction * (upper_offset - lower_offset)))


def _peak_epoch_in_campaign(t0, porb, campaign_start, campaign_end):
    """Return the in-campaign lensing epoch closest to the campaign midpoint."""
    first = math.ceil((campaign_start - t0) / porb)
    last = math.floor((campaign_end - t0) / porb)
    if first > last:
        return None
    midpoint = 0.5 * (campaign_start + campaign_end)
    cycles = numpy.arange(first, last + 1)
    epochs = t0 + cycles * porb
    return float(epochs[numpy.argmin(numpy.abs(epochs - midpoint))])


def _even_times(start, duration, count):
    if count == 0:
        return numpy.empty(0)
    if count == 1:
        return numpy.array([start + 0.5 * duration])
    return numpy.linspace(start, start + duration, count)


def _even_times_outside_triplet(start, duration, count, anchor, delta):
    """Place observations evenly over the usable window excluding the triplet."""
    if count == 0:
        return numpy.empty(0)
    end = start + duration
    left_duration = anchor - delta - start
    right_duration = end - (anchor + delta)
    available_duration = left_duration + right_duration
    if available_duration <= 0:
        return numpy.empty(0)

    # Midpoints never land on either edge of the excluded targeted interval.
    positions = (numpy.arange(count) + 0.5) * available_duration / count
    return numpy.where(
        positions < left_duration,
        start + positions,
        anchor + delta + positions - left_duration,
    )


def _schedule(parameter_rows, t0, porb, baseline, campaign_start, nobs, sigma_rv, credibility):
    if nobs < 1:
        raise ValueError("nobs must be at least one")
    if porb <= 0 or baseline <= 0:
        raise ValueError("porb and baseline must be positive")
    if sigma_rv <= 0:
        raise ValueError("sigma_rv must be positive")
    if not 0.0 < credibility <= 1.0:
        raise ValueError("credibility must be in (0, 1]")

    campaign_end = campaign_start + baseline
    useful_duration = min(porb, baseline)
    anchor = _peak_epoch_in_campaign(t0, porb, campaign_start, campaign_end)
    if anchor is None or nobs < 3:
        return _even_times(campaign_start, useful_duration, nobs)

    max_delta = min(anchor - campaign_start, campaign_end - anchor, 0.5 * porb)
    if max_delta <= 0:
        return _even_times(campaign_start, useful_duration, nobs)
    delta = _delta_for_significance(
        parameter_rows, anchor, 5.0 * sigma_rv, max_delta, credibility
    )
    remaining = _even_times_outside_triplet(
        campaign_start, useful_duration, nobs - 3, anchor, delta
    )
    if remaining.shape[0] != nobs - 3:
        # This can only occur when an in-campaign peak lies outside the first
        # ``min(porb, baseline)`` window.  The targeted triplet still takes
        # priority, and the remaining visits use the full campaign instead.
        remaining = _even_times_outside_triplet(
            campaign_start, baseline, nobs - 3, anchor, delta
        )
    if remaining.shape[0] != nobs - 3:
        # A maximal offset can put targeted visits at both campaign edges,
        # leaving no excluded interval in which to place the remaining visits.
        # In that degenerate case retain the triplet and distribute the rest
        # across the usable window.
        remaining = _even_times(campaign_start, useful_duration, nobs - 3)
    return numpy.sort(numpy.concatenate((remaining, [anchor - delta, anchor, anchor + delta])))


def schedule_rv_observations(
    t0,
    porb,
    baseline,
    campaign_start,
    nobs,
    sigma_rv,
    estimated_parameters,
):
    """Schedule deterministic RV observations.

    ``t0``, ``porb``, ``baseline``, and ``campaign_start`` are in days.  A
    lensing epoch ``t0 + x * porb`` inside the campaign is selected, and the
    three targeted observations are placed at that epoch and at symmetric
    offsets where the selflensing RV model changes by ``5 * sigma_rv``.  The
    remaining observations are evenly distributed over ``min(porb, baseline)``.

    ``estimated_parameters`` can be a selflensing parameter vector/mapping or
    a ``SelfLensingSystem``.  Its supplied periapsis phase is intentionally
    ignored: the scheduling calculation fixes ``periapsis_phase = pi`` in the
    project's orbital-longitude convention.
    """
    rows = _parameter_rows(estimated_parameters)
    if rows.shape[0] != 1:
        raise ValueError("estimated_parameters must describe exactly one system")
    return _schedule(
        rows,
        _value_in_days(t0),
        _value_in_days(porb),
        _value_in_days(baseline),
        _value_in_days(campaign_start),
        int(nobs),
        _value_in_m_per_s(sigma_rv),
        1.0,
    )


def schedule_rv_observations_from_posterior(
    t0,
    porb_samples,
    baseline,
    campaign_start,
    nobs,
    sigma_rv,
    posterior_parameter_samples,
    *,
    credibility=0.9,
):
    """Schedule RV observations robustly from MCMC posterior samples.

    Each row of ``posterior_parameter_samples`` is a selflensing parameter
    sample.  Stellar properties may simply be repeated in every row when known
    exactly.  ``porb_samples`` supplies the period posterior explicitly; each
    row's semi-major axis is rescaled to honour that sampled period before the
    JAX RV calculation.  This propagates period, lens-mass, inclination,
    eccentricity, and any other sampled parameters jointly.

    The returned schedule uses the median predicted lensing epoch.  Its offset
    is the first one for which at least ``credibility`` of posterior draws have
    a >= 5-sigma RV change on *both* sides of that epoch.
    """
    rows = _parameter_rows(posterior_parameter_samples)
    period_samples = jnp.asarray(porb_samples)
    if period_samples.ndim != 1 or period_samples.shape[0] != rows.shape[0]:
        raise ValueError("porb_samples must contain one period for each posterior row")
    if not bool(numpy.asarray(jnp.all(period_samples > 0))):
        raise ValueError("porb_samples must be positive")
    rows = _set_period(rows, period_samples)
    return _schedule(
        rows,
        _value_in_days(t0),
        float(numpy.median(numpy.asarray(period_samples))),
        _value_in_days(baseline),
        _value_in_days(campaign_start),
        int(nobs),
        _value_in_m_per_s(sigma_rv),
        float(credibility),
    )


# A concise alias for interactive use; keep the descriptive name for imports.
schedule = schedule_rv_observations
