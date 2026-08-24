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
FIT_LC_BINS = int(os.environ.get("SLRV_PREFIT_LC_BINS", 100))
FIT_DEVICE = os.environ.get("SLRV_PREFIT_DEVICE", os.environ.get("SLRV_FIT_DEVICE", "gpu:0"))
FIT_LENSING_METHOD = os.environ.get("SLRV_PREFIT_LENSING_METHOD")
FIT_MCMC_PROGRESS = os.environ.get("SLRV_PREFIT_MCMC_PROGRESS", "0").lower() in (
    "1", "true", "yes", "on"
)
PREFIT_BATCH_SIZE = int(os.environ.get("SLRV_PREFIT_BATCH_SIZE", 50))

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
    return {"times": times, "flux": flux / normalisation - 1.0, "err": flux_err / normalisation}


def lightcurve_fit_window(curve, peak_time, tau_eff):
    mask = (
        numpy.isfinite(curve["times"])
        & numpy.isfinite(curve["flux"])
        & numpy.isfinite(curve["err"])
        & (curve["err"] > 0)
        & (curve["times"] >= peak_time - tau_eff)
        & (curve["times"] <= peak_time + tau_eff)
    )
    if not numpy.any(mask):
        raise ValueError("Photometry prefit window contains no valid samples")
    times, flux, err = (curve[key][mask] for key in ("times", "flux", "err"))
    edges = numpy.linspace(peak_time - tau_eff, peak_time + tau_eff, FIT_LC_BINS + 1)
    bins = numpy.clip(numpy.searchsorted(edges, times, side="right") - 1, 0, FIT_LC_BINS - 1)
    output = []
    for index in range(FIT_LC_BINS):
        current = bins == index
        if not numpy.any(current):
            continue
        weights = 1.0 / err[current] ** 2
        output.append((numpy.average(times[current], weights=weights), numpy.average(flux[current], weights=weights), numpy.sqrt(1.0 / weights.sum())))
    return {"times": numpy.asarray([value[0] for value in output]), "flux": numpy.asarray([value[1] for value in output]), "err": numpy.asarray([value[2] for value in output])}


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
    curve = lightcurve_fit_window(load_lightcurve(row), row["peak_time"], row["tau_eff"])
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
        lightcurve_fit_window(load_lightcurve(row), row["peak_time"], row["tau_eff"])
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


def fit_photometry_rows(rows):
    """Fit photometry prefits in GPU batches and save one posterior per system."""
    if not rows:
        return []
    results = []
    for start in range(0, len(rows), PREFIT_BATCH_SIZE):
        batch = rows[start : start + PREFIT_BATCH_SIZE]
        lc_times, observed_lc, observed_lc_err, lc_mask = _padded_lightcurve_inputs(batch)
        systems, samplers, walker_state = self_lensing_systems_from_batched_fit_blackjax(
            lc_times=Time(lc_times, format="jd"),
            observed_lc=observed_lc * units.dimensionless_unscaled,
            observed_lc_err=observed_lc_err * units.dimensionless_unscaled,
            lc_mask=lc_mask,
            peak_time=Time([row["peak_time"] for row in batch], format="jd"),
            Mcomp=numpy.asarray([row["Mcomp"] for row in batch]) * units.M_sun,
            Rcomp=numpy.asarray([row["Rcomp"] for row in batch]) * units.R_sun,
            Teff=numpy.asarray([row["Teff"] for row in batch]) * units.K,
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
        walker_coords = numpy.asarray(walker_state.coords, dtype=float)
        results.extend(
            _save_prefit_result(row, system, sampler, walker_coords[index])
            for index, (row, system, sampler) in enumerate(zip(batch, systems, samplers))
        )
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


def save_prefits_and_schedules(prefits):
    existing = load_table(PREFIT_RESULTS_PATH)
    combined = pandas.concat([existing, pandas.DataFrame(prefits)], ignore_index=True)
    combined.drop_duplicates("simulation_id", keep="last", inplace=True)
    save_table(combined, PREFIT_RESULTS_PATH)
    schedules = [schedule for _, row in combined.iterrows() for schedule in schedule_rows(row)]
    save_table(pandas.DataFrame(schedules), SCHEDULE_TABLE_PATH)
    return combined
