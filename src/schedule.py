"""RV follow-up schedules for self-lensing events.

Times are floating-point Julian dates (or any other common day-based epoch).
The parameter vectors use :data:`selflensing.lightcurves.PARAMETER_ORDER` and
its public units: AU, solar masses/radii, K, and radians.  Keeping the public
calculation numeric makes the posterior version directly usable with JAX/GPU
batches.
"""

import math
import os

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
MODE_DESIGN_GRID_SIZE = int(os.environ.get("SLRV_MODE_DESIGN_GRID_SIZE", 513))
MAX_POSTERIOR_MODES = int(os.environ.get("SLRV_MAX_POSTERIOR_MODES", 8))
MODE_MIN_FRACTION = float(os.environ.get("SLRV_MODE_MIN_FRACTION", 0.03))
MODE_SEPARATION = float(os.environ.get("SLRV_MODE_SEPARATION", 0.15))


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


def _mode_features(parameter_rows):
    """Scale parameter coordinates for identifying separated posterior modes."""
    names = {name: PARAMETER_ORDER.index(name) for name in PARAMETER_ORDER}
    rows = numpy.asarray(parameter_rows, dtype=float)
    phase = rows[:, names["periapsis_phase"]]
    return numpy.column_stack(
        (
            numpy.log(rows[:, names["a"]]) / numpy.log(200.0 / 0.1),
            (rows[:, names["i"]] - 0.5 * numpy.pi) / 1.0e-4,
            numpy.log(rows[:, names["Macc"]]) / numpy.log(50.0 / 0.5),
            rows[:, names["e"]],
            numpy.sin(phase),
            numpy.cos(phase),
        )
    )


def posterior_mode_labels(parameter_rows, max_modes=MAX_POSTERIOR_MODES):
    """Cluster posterior samples into separated, non-negligible mode clouds.

    This deliberately uses a physical coordinate scale rather than each
    sample's own variance: a broad single mode must not be split merely
    because it has a narrow dimension such as inclination.
    """
    features = _mode_features(parameter_rows)
    if len(features) == 0:
        raise ValueError("Cannot identify modes from an empty posterior")
    median = numpy.median(features, axis=0)
    centres = [features[numpy.argmin(numpy.sum((features - median) ** 2, axis=1))]]
    while len(centres) < min(max_modes, len(features)):
        distances = numpy.min(
            numpy.sum((features[:, None, :] - numpy.asarray(centres)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        candidate = int(numpy.argmax(distances))
        if numpy.sqrt(distances[candidate]) < MODE_SEPARATION:
            break
        centres.append(features[candidate])
    centres = numpy.asarray(centres)
    labels = numpy.zeros(len(features), dtype=int)
    for _ in range(20):
        labels = numpy.argmin(
            numpy.sum((features[:, None, :] - centres[None, :, :]) ** 2, axis=2), axis=1
        )
        updated = numpy.array(
            [
                features[labels == index].mean(axis=0)
                if numpy.any(labels == index) else centres[index]
                for index in range(len(centres))
            ]
        )
        if numpy.allclose(updated, centres):
            break
        centres = updated
    counts = numpy.bincount(labels, minlength=len(centres))
    keep = counts >= max(2, int(numpy.ceil(MODE_MIN_FRACTION * len(features))))
    # Reassign negligible clusters to their closest retained mode.
    if numpy.any(~keep) and numpy.any(keep):
        retained = centres[keep]
        labels = numpy.argmin(
            numpy.sum((features[:, None, :] - retained[None, :, :]) ** 2, axis=2), axis=1
        )
    return labels


def _mode_rv_design(parameter_rows, peak_time, campaign_start, baseline, sigma_rv):
    """Build a noise-aware, mode-separated RV design context."""
    rows = numpy.asarray(parameter_rows, dtype=float)
    labels = posterior_mode_labels(rows)
    mode_count = int(labels.max()) + 1
    if mode_count < 2:
        return None
    times = numpy.linspace(campaign_start, campaign_start + baseline, MODE_DESIGN_GRID_SIZE)
    curves = numpy.asarray(
        jax.vmap(
            lambda row: radial_velocity_m_per_s(jnp.asarray(times), row, peak_time)
        )(jnp.asarray(rows)),
        dtype=float,
    )
    means = numpy.stack([curves[labels == index].mean(axis=0) for index in range(mode_count)])
    variances = numpy.stack([
        curves[labels == index].var(axis=0) if numpy.sum(labels == index) > 1
        else numpy.zeros(len(times))
        for index in range(mode_count)
    ])
    pairwise = []
    for first in range(mode_count):
        for second in range(first + 1, mode_count):
            variance = sigma_rv**2 + variances[first] + variances[second]
            pairwise.append((means[first] - means[second]) ** 2 / variance)
    return {
        "times": times,
        "pairwise_separation": numpy.asarray(pairwise),
        "mode_count": mode_count,
        "mode_weights": numpy.bincount(labels, minlength=mode_count) / len(labels),
    }


def _mode_design_schedule(context, nobs):
    """Greedily distinguish every mode, then use remaining visits for coverage."""
    nobs = int(nobs)
    design = context["mode_rv_design"]
    times = design["times"]
    separation = design["pairwise_separation"]
    if nobs < 1:
        raise ValueError("nobs must be at least one")
    diagnostic_count = min(nobs, max(3, (nobs + 1) // 2))
    selected = []
    cumulative = numpy.zeros(separation.shape[0])
    available = numpy.ones(len(times), dtype=bool)
    for _ in range(diagnostic_count):
        objective = numpy.min(cumulative[:, None] + separation, axis=0)
        objective[~available] = -numpy.inf
        index = int(numpy.argmax(objective))
        selected.append(index)
        cumulative += separation[:, index]
        available[index] = False
    # Once the diagnostic core is selected, make the remaining allocation a
    # maximin temporal design: each visit reduces the largest phase/time gap.
    while len(selected) < nobs:
        distance = numpy.min(numpy.abs(times[:, None] - times[numpy.asarray(selected)][None, :]), axis=1)
        distance[~available] = -numpy.inf
        index = int(numpy.argmax(distance))
        selected.append(index)
        available[index] = False
    return numpy.sort(times[numpy.asarray(selected)])


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


def _schedule_context(parameter_rows, t0, porb, baseline, campaign_start, sigma_rv, credibility):
    if porb <= 0 or baseline <= 0:
        raise ValueError("porb and baseline must be positive")
    if sigma_rv <= 0:
        raise ValueError("sigma_rv must be positive")
    if not 0.0 < credibility <= 1.0:
        raise ValueError("credibility must be in (0, 1]")

    campaign_end = campaign_start + baseline
    useful_duration = min(porb, baseline)
    anchor = _peak_epoch_in_campaign(t0, porb, campaign_start, campaign_end)
    if anchor is None:
        return {
            "campaign_start": campaign_start,
            "baseline": baseline,
            "useful_duration": useful_duration,
            "anchor": None,
            "delta": None,
        }

    max_delta = min(anchor - campaign_start, campaign_end - anchor, 0.5 * porb)
    if max_delta <= 0:
        return {
            "campaign_start": campaign_start,
            "baseline": baseline,
            "useful_duration": useful_duration,
            "anchor": None,
            "delta": None,
        }
    delta = _delta_for_significance(
        parameter_rows, anchor, 5.0 * sigma_rv, max_delta, credibility
    )
    return {
        "campaign_start": campaign_start,
        "baseline": baseline,
        "useful_duration": useful_duration,
        "anchor": anchor,
        "delta": delta,
    }


def schedule_rv_observations_from_context(context, nobs):
    """Expand a precomputed scheduling context for one observation count."""
    nobs = int(nobs)
    if nobs < 1:
        raise ValueError("nobs must be at least one")
    if "mode_rv_design" in context:
        return _mode_design_schedule(context, nobs)
    campaign_start = context["campaign_start"]
    baseline = context["baseline"]
    useful_duration = context["useful_duration"]
    anchor = context["anchor"]
    delta = context["delta"]
    if anchor is None or nobs < 3:
        return _even_times(campaign_start, useful_duration, nobs)
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


def _schedule(parameter_rows, t0, porb, baseline, campaign_start, nobs, sigma_rv, credibility):
    context = _schedule_context(
        parameter_rows, t0, porb, baseline, campaign_start, sigma_rv, credibility
    )
    return schedule_rv_observations_from_context(context, nobs)


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
    """Schedule RV observations to distinguish viable posterior modes.

    Each row of ``posterior_parameter_samples`` is a selflensing parameter
    sample.  Stellar properties may simply be repeated in every row when known
    exactly.  ``porb_samples`` supplies the period posterior explicitly; each
    row's semi-major axis is rescaled to honour that sampled period before the
    JAX RV calculation.  This propagates period, lens-mass, inclination,
    eccentricity, and any other sampled parameters jointly.

    Distinct posterior modes are represented by their posterior-predictive RV
    curves over the whole campaign.  The first visits greedily maximize the
    weakest pairwise noise-normalized separation, and remaining visits reduce
    the largest time/phase gaps.  A single-mode posterior falls back to the
    historical targeted-triplet design.
    """
    rows = _parameter_rows(posterior_parameter_samples)
    period_samples = jnp.asarray(porb_samples)
    if period_samples.ndim != 1 or period_samples.shape[0] != rows.shape[0]:
        raise ValueError("porb_samples must contain one period for each posterior row")
    if not bool(numpy.asarray(jnp.all(period_samples > 0))):
        raise ValueError("porb_samples must be positive")
    rows = _set_period(rows, period_samples)
    context = schedule_rv_observation_context_from_posterior(
        t0, period_samples, baseline, campaign_start, sigma_rv, rows,
        credibility=credibility,
    )
    return schedule_rv_observations_from_context(context, nobs)


def schedule_rv_observation_context_from_posterior(
    t0,
    porb_samples,
    baseline,
    campaign_start,
    sigma_rv,
    posterior_parameter_samples,
    *,
    credibility=0.9,
):
    """Compute a reusable posterior-mode RV-design context once.

    The result can be expanded efficiently for multiple ``nobs`` values with
    :func:`schedule_rv_observations_from_context`.
    """
    rows = _parameter_rows(posterior_parameter_samples)
    period_samples = jnp.asarray(porb_samples)
    if period_samples.ndim != 1 or period_samples.shape[0] != rows.shape[0]:
        raise ValueError("porb_samples must contain one period for each posterior row")
    if not bool(numpy.asarray(jnp.all(period_samples > 0))):
        raise ValueError("porb_samples must be positive")
    rows = _set_period(rows, period_samples)
    mode_design = _mode_rv_design(
        rows,
        _value_in_days(t0),
        _value_in_days(campaign_start),
        _value_in_days(baseline),
        _value_in_m_per_s(sigma_rv),
    )
    if mode_design is not None:
        return {"mode_rv_design": mode_design}
    return _schedule_context(
        rows,
        _value_in_days(t0),
        float(numpy.median(numpy.asarray(period_samples))),
        _value_in_days(baseline),
        _value_in_days(campaign_start),
        _value_in_m_per_s(sigma_rv),
        float(credibility),
    )


# A concise alias for interactive use; keep the descriptive name for imports.
schedule = schedule_rv_observations
