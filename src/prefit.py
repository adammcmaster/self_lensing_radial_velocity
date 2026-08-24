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
from selflensing.fitting import self_lensing_system_from_fit_blackjax
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
FIT_DEVICE = os.environ.get("SLRV_PREFIT_DEVICE", os.environ.get("SLRV_FIT_DEVICE", "auto"))
FIT_LENSING_METHOD = os.environ.get("SLRV_PREFIT_LENSING_METHOD")
FIT_MCMC_PROGRESS = os.environ.get("SLRV_PREFIT_MCMC_PROGRESS", "0").lower() in (
    "1", "true", "yes", "on"
)

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
        prefer_gpu=FIT_DEVICE == "auto",
        proposal_scale_fraction=0.02,
        return_sampler=True,
        return_walker_state=False,
    )
    sls, sampler = result[:2]
    posterior_path = _posterior_path(row["simulation_id"])
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    chain = numpy.asarray(sampler.get_chain(flat=True, discard=FIT_BURN), dtype=float)
    numpy.savez_compressed(posterior_path, chain=chain, parameter_names=numpy.asarray(PREFIT_PARAMETER_NAMES))
    campaign_start, offset_years = _campaign_start(row)
    return _prefit_result_row(row, sls, campaign_start, offset_years, posterior_path)


def load_posterior_parameters(row):
    with numpy.load(row["prefit_posterior_path"]) as samples:
        chain = numpy.asarray(samples["chain"], dtype=float)
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
