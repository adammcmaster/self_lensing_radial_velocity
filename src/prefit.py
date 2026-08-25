"""Photometry-only fitting and posterior-backed RV schedule construction."""

import hashlib
import os
from pathlib import Path

import numpy
import pandas
from astropy import units
from astropy.io import fits
from astropy.time import Time

from fitting import PREFIT_POSTERIOR_PATH, PREFIT_RESULTS_PATH, SCHEDULE_TABLE_PATH
from schedule import schedule_rv_observations_from_posterior
from selflensing.fitting import (
    self_lensing_system_from_fit_blackjax,
    self_lensing_systems_from_batched_fit_blackjax,
)
from selflensing.lightcurves import PARAMETER_ORDER


DAYS_PER_YEAR = 365.25
RV_CAMPAIGN_BASELINE_DAYS = 5.0 * DAYS_PER_YEAR
NOBS_VALUES = tuple(range(3, 11))
RV_ACCURACIES = tuple(10.0**power for power in (-1, 0, 1, 2, 3))
POSTERIOR_SCHEDULE_SAMPLES = int(os.environ.get("SLRV_SCHEDULE_POSTERIOR_SAMPLES", 512))
SCHEDULE_CREDIBILITY = float(os.environ.get("SLRV_SCHEDULE_CREDIBILITY", 0.9))

FIT_NSTEPS = int(os.environ.get("SLRV_PREFIT_NSTEPS", os.environ.get("SLRV_FIT_NSTEPS", 1000)))
FIT_NWALKERS = int(os.environ.get("SLRV_PREFIT_NWALKERS", os.environ.get("SLRV_FIT_NWALKERS", 128)))
FIT_BURN = int(os.environ.get("SLRV_PREFIT_BURN", os.environ.get("SLRV_FIT_BURN", 200)))
PREFIT_LC_BIN_WIDTH_DAYS = 1.0 / 24.0
FIT_DEVICE = os.environ.get("SLRV_PREFIT_DEVICE", os.environ.get("SLRV_FIT_DEVICE", "gpu:0"))
FIT_LENSING_METHOD = os.environ.get("SLRV_PREFIT_LENSING_METHOD")
FIT_MCMC_PROGRESS = os.environ.get("SLRV_PREFIT_MCMC_PROGRESS", "0").lower() in (
    "1", "true", "yes", "on"
)
# Full-campaign hourly photometry substantially increases the likelihood tensor
# size.  Keep the default conservative for GPU memory; callers can raise it
# after profiling their accelerator.
PREFIT_BATCH_SIZE = int(os.environ.get("SLRV_PREFIT_BATCH_SIZE", 4))

PREFIT_PARAMETER_NAMES = ("a", "i", "Macc", "e", "periapsis_phase")
PREFIT_RESULT_COLUMNS = tuple(f"prefit_fitted_{name}" for name in (
    "a", "i", "Macc", "e", "periapsis_phase", "porb",
))


def _metadata_from_photometry_path(path):
    """Read only the truth values required by prefit and RV simulation."""
    path = Path(path)
    with fits.open(path) as hdul:
        header = hdul[1].header
    return {
        "simulation_id": path.name.removesuffix(".fits.gz"),
        "photometry_path": str(path),
        "a": float(header["A"]),
        "e": float(header["E"]),
        "periapsis_phase": float(header["PERI"]),
        "Macc": float(header["MBH"]),
        "Porb": float(header["P"]),
        "tau_eff": float(header["TAU_EFF"]),
        "source_amp_sl": float(header["AMP_SL"]) if "AMP_SL" in header else numpy.nan,
        "Mcomp": float(header["MSTAR"]),
        "Rcomp": float(header["RSTAR"]),
        "Teff": float(header["TEFF"]),
        "i": float(header["I"]),
        "peak_time": float(header["PEAK_T"]),
    }


def photometry_rows(paths):
    return pandas.DataFrame([_metadata_from_photometry_path(path) for path in paths])


def load_lightcurve(row):
    with fits.open(row["photometry_path"]) as hdul:
        data = hdul[1].data
        flux = numpy.asarray(data["flux"], dtype=float)
        flux_err = numpy.asarray(data["flux_err"], dtype=float)
        times = numpy.asarray(data["time"], dtype=float)
    if times.ndim == 2:
        times = times.sum(axis=1)
    normalisation = numpy.median(flux)
    normalised_flux = flux / normalisation - 1.0
    normalised_err = flux_err / normalisation
    if not numpy.any(numpy.isfinite(normalised_err) & (normalised_err > 0)):
        estimated_err = numpy.nanstd(normalised_flux[numpy.isfinite(normalised_flux)])
        if not numpy.isfinite(estimated_err) or estimated_err <= 0:
            raise ValueError("Photometry curve has no usable flux uncertainties")
        normalised_err = numpy.full_like(normalised_flux, estimated_err)
    return {"times": times, "flux": normalised_flux, "err": normalised_err}


def bin_full_lightcurve_hourly(curve):
    """Inverse-variance bin the complete photometry curve into hourly bins.

    The bin grid is anchored to absolute Julian-date hour boundaries, so
    simulations with the common survey cadence produce identically shaped
    arrays for the batched prefit likelihood.  No phase-dependent clipping is
    applied: the prefit sees the entire observed photometry campaign.
    """
    mask = (
        numpy.isfinite(curve["times"])
        & numpy.isfinite(curve["flux"])
        & numpy.isfinite(curve["err"])
        & (curve["err"] > 0)
    )
    if not numpy.any(mask):
        raise ValueError("Photometry prefit contains no valid samples")
    times, flux, err = (curve[key][mask] for key in ("times", "flux", "err"))
    first_bin = int(numpy.floor(numpy.min(times) / PREFIT_LC_BIN_WIDTH_DAYS))
    last_bin = int(numpy.floor(numpy.max(times) / PREFIT_LC_BIN_WIDTH_DAYS))
    bins = numpy.floor(times / PREFIT_LC_BIN_WIDTH_DAYS).astype(int) - first_bin
    output = []
    for index in range(last_bin - first_bin + 1):
        current = bins == index
        if not numpy.any(current):
            continue
        weights = 1.0 / err[current] ** 2
        output.append(
            (
                numpy.average(times[current], weights=weights),
                numpy.average(flux[current], weights=weights),
                numpy.sqrt(1.0 / weights.sum()),
            )
        )
    return {
        "times": numpy.asarray([value[0] for value in output]),
        "flux": numpy.asarray([value[1] for value in output]),
        "err": numpy.asarray([value[2] for value in output]),
    }


def _posterior_path(simulation_id):
    return PREFIT_POSTERIOR_PATH / f"{simulation_id}.npz"


def _campaign_start(row):
    """Reproduce the historical 1--10 year post-photometry start distribution."""
    curve = load_lightcurve(row)
    digest = hashlib.sha256(str(row["simulation_id"]).encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    offset_years = numpy.random.default_rng(seed).uniform(1.0, 10.0)
    return float(numpy.nanmax(curve["times"]) + offset_years * DAYS_PER_YEAR), offset_years


def _posterior_parameter_rows(flat_chain, row):
    """Expand the five fitted coordinates with the known companion properties."""
    values = {name: flat_chain[:, index] for index, name in enumerate(PREFIT_PARAMETER_NAMES)}
    values.update({"Mcomp": numpy.full(len(flat_chain), row["Mcomp"]), "Rcomp": numpy.full(len(flat_chain), row["Rcomp"]), "Teff": numpy.full(len(flat_chain), row["Teff"])})
    return numpy.column_stack([values[name] for name in PARAMETER_ORDER])


def _prefit_result_row(row, sls, campaign_start, offset_years, posterior_path):
    output = dict(row)
    output.update({
        "campaign_start": campaign_start,
        "campaign_start_offset_years": offset_years,
        "prefit_posterior_path": str(posterior_path),
        "prefit_fitted_a": float(sls.a.value),
        "prefit_fitted_i": float(sls.i.value),
        "prefit_fitted_Macc": float(sls.Macc.value),
        "prefit_fitted_e": float(sls.e.value),
        "prefit_fitted_periapsis_phase": float(sls.periapsis_phase.value),
        "prefit_fitted_porb": float(sls.porb.value),
    })
    return output


def fit_photometry_row(row):
    curve = bin_full_lightcurve_hourly(load_lightcurve(row))
    result = self_lensing_system_from_fit_blackjax(
        lc_times=Time(curve["times"], format="jd"),
        observed_lc=curve["flux"] * units.dimensionless_unscaled,
        observed_lc_err=curve["err"] * units.dimensionless_unscaled,
        peak_time=Time(row["peak_time"], format="jd"),
        Mcomp=row["Mcomp"] * units.M_sun,
        Rcomp=row["Rcomp"] * units.R_sun,
        Teff=row["Teff"] * units.K,
        false_alarm_probability=0.5,
        nsteps=FIT_NSTEPS,
        nwalkers=FIT_NWALKERS,
        burn=FIT_BURN,
        progress=FIT_MCMC_PROGRESS,
        lensing_method=FIT_LENSING_METHOD,
        center_lc_model=True,
        device=FIT_DEVICE,
        prefer_gpu=True,
        proposal_scale_fraction=0.02,
        return_sampler=True,
        return_walker_state=True,
    )
    sls, sampler, walker_state = result
    posterior_path = _posterior_path(row["simulation_id"])
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    chain = numpy.asarray(sampler.get_chain(), dtype=float)
    flat_chain = numpy.asarray(sampler.get_chain(flat=True, discard=FIT_BURN), dtype=float)
    final_walker_coords = numpy.asarray(
        getattr(walker_state, "coords", walker_state), dtype=float
    )
    numpy.savez_compressed(
        posterior_path,
        chain=chain,
        flat_chain=flat_chain,
        final_walker_coords=final_walker_coords,
        parameter_names=numpy.asarray(PREFIT_PARAMETER_NAMES),
    )
    campaign_start, offset_years = _campaign_start(row)
    return _prefit_result_row(row, sls, campaign_start, offset_years, posterior_path)


def _padded_lightcurve_inputs(rows):
    """Pad independently binned photometry curves for a batched likelihood."""
    curves = [
        bin_full_lightcurve_hourly(load_lightcurve(row))
        for row in rows
    ]
    width = max(len(curve["times"]) for curve in curves)
    times = numpy.zeros((len(rows), width), dtype=float)
    flux = numpy.zeros((len(rows), width), dtype=float)
    err = numpy.ones((len(rows), width), dtype=float)
    mask = numpy.zeros((len(rows), width), dtype=bool)
    for index, curve in enumerate(curves):
        size = len(curve["times"])
        times[index, :size] = curve["times"]
        if size < width:
            times[index, size:] = curve["times"][-1]
        flux[index, :size] = curve["flux"]
        err[index, :size] = curve["err"]
        mask[index, :size] = True
    return times, flux, err, mask


def _save_prefit_result(row, sls, sampler, walker_coords):
    posterior_path = _posterior_path(row["simulation_id"])
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    chain = numpy.asarray(sampler.get_chain(), dtype=float)
    flat_chain = numpy.asarray(sampler.get_chain(flat=True, discard=FIT_BURN), dtype=float)
    numpy.savez_compressed(
        posterior_path,
        chain=chain,
        flat_chain=flat_chain,
        final_walker_coords=numpy.asarray(walker_coords, dtype=float),
        parameter_names=numpy.asarray(PREFIT_PARAMETER_NAMES),
    )
    campaign_start, offset_years = _campaign_start(row)
    return _prefit_result_row(row, sls, campaign_start, offset_years, posterior_path)


def _fit_values(sls):
    return {
        "a": float(sls.a.value),
        "i": float(sls.i.value),
        "Macc": float(sls.Macc.value),
        "e": float(sls.e.value),
        "periapsis_phase": float(sls.periapsis_phase.value),
        "porb": float(sls.porb.value),
    }


def _prefit_result_from_values(row, values, posterior_path):
    output = dict(row)
    campaign_start, offset_years = _campaign_start(row)
    output.update(
        {
            "campaign_start": campaign_start,
            "campaign_start_offset_years": offset_years,
            "prefit_posterior_path": str(posterior_path),
            **{f"prefit_fitted_{name}": value for name, value in values.items()},
        }
    )
    return output


def _prefit_batch_payload(rows, systems, samplers, walker_coords):
    """Copy a completed GPU batch into a picklable CPU-sidecar payload."""
    payload = []
    for index, (row, system, sampler) in enumerate(zip(rows, systems, samplers)):
        chain = numpy.asarray(sampler.get_chain(), dtype=float)
        payload.append(
            {
                "row": dict(row),
                "fit_values": _fit_values(system),
                "chain": chain,
                "flat_chain": numpy.asarray(
                    sampler.get_chain(flat=True, discard=FIT_BURN), dtype=float
                ),
                "final_walker_coords": numpy.asarray(walker_coords[index], dtype=float),
            }
        )
    return payload


def materialize_prefit_batch(payload):
    """CPU-sidecar task: persist posterior samples and create RV schedules."""
    prefits = []
    for item in payload:
        row = item["row"]
        posterior_path = _posterior_path(row["simulation_id"])
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        numpy.savez_compressed(
            posterior_path,
            chain=item["chain"],
            flat_chain=item["flat_chain"],
            final_walker_coords=item["final_walker_coords"],
            parameter_names=numpy.asarray(PREFIT_PARAMETER_NAMES),
        )
        prefits.append(_prefit_result_from_values(row, item["fit_values"], posterior_path))
    schedules = [schedule for row in prefits for schedule in schedule_rows(row)]
    return prefits, schedules


def initialise_prefit_sidecar():
    """Ensure schedule calculations in a spawned sidecar use CPU, not the GPU."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def fit_photometry_batch(rows):
    """Run one GPU MCMC batch and return a CPU-materialisation payload."""
    if not rows:
        return []
    lc_times, observed_lc, observed_lc_err, lc_mask = _padded_lightcurve_inputs(rows)
    systems, samplers, walker_state = self_lensing_systems_from_batched_fit_blackjax(
        lc_times=Time(lc_times, format="jd"),
        observed_lc=observed_lc * units.dimensionless_unscaled,
        observed_lc_err=observed_lc_err * units.dimensionless_unscaled,
        lc_mask=lc_mask,
        peak_time=Time([row["peak_time"] for row in rows], format="jd"),
        Mcomp=numpy.asarray([row["Mcomp"] for row in rows]) * units.M_sun,
        Rcomp=numpy.asarray([row["Rcomp"] for row in rows]) * units.R_sun,
        Teff=numpy.asarray([row["Teff"] for row in rows]) * units.K,
        false_alarm_probability=0.5,
        nsteps=FIT_NSTEPS,
        nwalkers=FIT_NWALKERS,
        burn=FIT_BURN,
        progress=FIT_MCMC_PROGRESS,
        lensing_method=FIT_LENSING_METHOD,
        center_lc_model=True,
        device=FIT_DEVICE,
        prefer_gpu=True,
        proposal_scale_fraction=0.02,
        return_sampler=True,
        return_walker_state=True,
    )
    return _prefit_batch_payload(
        rows, systems, samplers, numpy.asarray(walker_state.coords, dtype=float)
    )


def fit_photometry_rows(rows):
    """Fit photometry prefits in GPU batches and save one posterior per system."""
    if not rows:
        return []
    results = []
    for start in range(0, len(rows), PREFIT_BATCH_SIZE):
        prefits, _ = materialize_prefit_batch(
            fit_photometry_batch(rows[start : start + PREFIT_BATCH_SIZE])
        )
        results.extend(prefits)
    return results


def load_posterior_parameters(row):
    with numpy.load(row["prefit_posterior_path"]) as samples:
        chain = numpy.asarray(
            samples["flat_chain"] if "flat_chain" in samples.files else samples["chain"],
            dtype=float,
        )
    if chain.ndim == 3:
        chain = chain[FIT_BURN:].reshape((-1, chain.shape[-1]))
    if len(chain) == 0:
        raise ValueError(f"Photometry posterior for {row['simulation_id']} is empty")
    count = min(POSTERIOR_SCHEDULE_SAMPLES, len(chain))
    indices = numpy.linspace(0, len(chain) - 1, count, dtype=int)
    parameters = _posterior_parameter_rows(chain[indices], row)
    # Porb is derived from a and the sampled lens mass; it remains joint with
    # every other posterior coordinate.
    from selflensing.orbits import AU_M, M_SUN_KG, orbital_period_days
    periods = numpy.asarray(orbital_period_days(parameters[:, 0] * AU_M, (parameters[:, 2] + parameters[:, 3]) * M_SUN_KG))
    return parameters, periods


def schedule_rows(prefit_row):
    parameters, periods = load_posterior_parameters(prefit_row)
    rows = []
    for nobs in NOBS_VALUES:
        for rv_accuracy in RV_ACCURACIES:
            times = schedule_rv_observations_from_posterior(
                prefit_row["peak_time"], periods, RV_CAMPAIGN_BASELINE_DAYS,
                prefit_row["campaign_start"], nobs, rv_accuracy, parameters,
                credibility=SCHEDULE_CREDIBILITY,
            )
            rows.append({
                "simulation_id": prefit_row["simulation_id"], "nobs": nobs,
                "RV accuracy": rv_accuracy, "baseline": RV_CAMPAIGN_BASELINE_DAYS,
                "campaign_start": prefit_row["campaign_start"],
                "rv_times": numpy.asarray(times, dtype=float),
            })
    return rows


def save_table(table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(f"{path}.new.gz")
    os.replace(f"{path}.new.gz", path)


def load_table(path):
    return pandas.read_pickle(path) if Path(path).exists() else pandas.DataFrame()


def save_prefits_and_schedules(prefits, schedules=None):
    """Atomically checkpoint new prefit summaries and their RV schedules."""
    existing = load_table(PREFIT_RESULTS_PATH)
    combined = pandas.concat([existing, pandas.DataFrame(prefits)], ignore_index=True)
    combined.drop_duplicates("simulation_id", keep="last", inplace=True)
    save_table(combined, PREFIT_RESULTS_PATH)
    if schedules is None:
        schedules = [
            schedule for _, row in combined.iterrows() for schedule in schedule_rows(row)
        ]
    existing_schedules = load_table(SCHEDULE_TABLE_PATH)
    combined_schedules = pandas.concat(
        [existing_schedules, pandas.DataFrame(schedules)], ignore_index=True
    )
    if not combined_schedules.empty:
        combined_schedules.drop_duplicates(
            ["simulation_id", "nobs", "RV accuracy", "baseline"],
            keep="last",
            inplace=True,
        )
    save_table(combined_schedules, SCHEDULE_TABLE_PATH)
    return combined
