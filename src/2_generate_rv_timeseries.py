"""Generate noisy RV curves at the posterior-informed observing timestamps."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# JAX backend names use "cuda" rather than "gpu".  Do not inherit a cluster
# default such as ROCm when this virtual environment contains CUDA JAX.
os.environ["JAX_PLATFORMS"] = os.environ.get("SLRV_JAX_PLATFORMS", "cuda")

import jax
from jax import numpy as jax_numpy
import numpy
import pandas
from astropy import units

from fitting import PREFIT_RESULTS_PATH, SCHEDULE_TABLE_PATH
from prefit import load_lightcurve, load_table
from rv_timeseries import (
    existing_rv_keys,
    metadata_row_from_params,
    rv_id_from_params,
    rv_key_from_values,
    rv_path_from_params,
    update_rv_metadata,
    write_rv_timeseries_file,
)
from selflensing.lightcurves import radial_velocity_m_per_s
from spectrograph import Spectrograph
from utils import configure_logging

RV_GENERATION_BATCH_SIZE = int(os.environ.get("SLRV_RV_GENERATION_BATCH_SIZE", 50))
REQUIRE_GPU = os.environ.get("SLRV_REQUIRE_GPU", "1").lower() in (
    "1", "true", "yes", "on"
)


@jax.jit
def batched_radial_velocities(parameter_rows, times, peak_times):
    return jax.vmap(radial_velocity_m_per_s)(times, parameter_rows, peak_times)


def truth_parameter_row(prefit_row):
    return numpy.asarray(
        [prefit_row[name] for name in ("a", "i", "Macc", "Mcomp", "Rcomp", "Teff", "e", "periapsis_phase")],
        dtype=float,
    )


def make_curves(prefit_row, schedule_rows):
    """Generate all requested schedules for one system in one GPU kernel."""
    schedule_rows = list(schedule_rows)
    max_nobs = max(len(row["rv_times"]) for row in schedule_rows)
    rv_times = numpy.empty((len(schedule_rows), max_nobs), dtype=float)
    for index, schedule_row in enumerate(schedule_rows):
        times = numpy.asarray(schedule_row["rv_times"], dtype=float)
        rv_times[index, :len(times)] = times
        if len(times) < max_nobs:
            rv_times[index, len(times):] = times[-1]
    parameter_rows = numpy.repeat(truth_parameter_row(prefit_row)[None, :], len(schedule_rows), axis=0)
    photometry_rvs = numpy.asarray(
        batched_radial_velocities(
            jax_numpy.asarray(parameter_rows),
            jax_numpy.asarray(rv_times),
            jax_numpy.full(len(schedule_rows), float(prefit_row["peak_time"])),
        )
    )
    photometry = load_lightcurve(prefit_row)
    curves = []
    for index, schedule_row in enumerate(schedule_rows):
        nobs = int(schedule_row["nobs"])
        accuracy = float(schedule_row["RV accuracy"])
        baseline = float(schedule_row["baseline"])
        times = rv_times[index, :nobs]
        photometry_rv = photometry_rvs[index, :nobs]
        stellar_noise = Spectrograph.STELLAR_NOISE.to_value(units.m / units.s)
        rv_err = numpy.sqrt(stellar_noise**2 + accuracy**2)
        noisy_rv = photometry_rv + numpy.random.normal(0.0, rv_err, len(times))
        metadata_photometry = pandas.Series(prefit_row).copy()
        metadata_photometry["rv_path"] = ""
        row = metadata_row_from_params(
            rv_id=rv_id_from_params(metadata_photometry, nobs, accuracy, baseline),
            rv_path=rv_path_from_params(metadata_photometry, nobs, accuracy, baseline),
            photometry=metadata_photometry, rv_accuracy=accuracy, nobs=nobs, baseline=baseline,
        )
        curves.append({"row": row, "rv_times": times, "rv": noisy_rv, "rv_err": rv_err,
                       "lc_times": photometry["times"], "lc": photometry["flux"], "lc_err": photometry["err"]})
    return curves


def main():
    configure_logging()
    if REQUIRE_GPU and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Stage 2 requires a JAX GPU; set SLRV_REQUIRE_GPU=0 for debugging.")
    prefits = load_table(PREFIT_RESULTS_PATH)
    schedules = load_table(SCHEDULE_TABLE_PATH)
    if prefits.empty or schedules.empty:
        raise RuntimeError("Run src/1_prefit_and_schedule.py before RV generation")
    existing = existing_rv_keys()
    rows = []
    for _, prefit_row in prefits.iterrows():
        system_schedules = schedules[schedules["simulation_id"].astype(str) == str(prefit_row["simulation_id"])]
        schedule_rows = []
        for _, schedule_row in system_schedules.iterrows():
            key = rv_key_from_values(prefit_row["simulation_id"], schedule_row["nobs"], schedule_row["RV accuracy"], schedule_row["baseline"])
            if key not in existing:
                schedule_rows.append(schedule_row)
        curves = []
        for start in range(0, len(schedule_rows), RV_GENERATION_BATCH_SIZE):
            curves.extend(make_curves(prefit_row, schedule_rows[start : start + RV_GENERATION_BATCH_SIZE]))
        if curves:
            write_rv_timeseries_file(curves, curves[0]["row"]["rv_path"])
            rows.extend(curve["row"] for curve in curves)
    update_rv_metadata(rows)


if __name__ == "__main__":
    main()
