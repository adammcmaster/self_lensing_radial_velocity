"""Generate noisy RV curves at the posterior-informed observing timestamps."""

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# JAX backend names use "cuda" rather than "gpu".  Do not inherit a cluster
# default such as ROCm when this virtual environment contains CUDA JAX.
if os.environ.get("SLRV_RV_CPU_SIDECAR") == "1":
    # Spawned sidecars must never initialise or contend for the generation GPU.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["JAX_PLATFORMS"] = os.environ.get("SLRV_JAX_PLATFORMS", "cuda")

import jax
from jax import numpy as jax_numpy
import numpy
import pandas
from astropy import units
from tqdm import tqdm

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

RV_GENERATION_GPU_BATCH_SIZE = max(1, int(
    os.environ.get("SLRV_RV_GENERATION_GPU_BATCH_SIZE", 32)
))
RV_GENERATION_CPU_MAX_WORKERS = max(1, int(
    os.environ.get("SLRV_RV_GENERATION_CPU_MAX_WORKERS", 16)
))
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


def make_curves(prefit_row, schedule_rows, photometry_rvs):
    """Add noise and package one system's GPU-generated RV curves for FITS."""
    schedule_rows = list(schedule_rows)
    photometry = load_lightcurve(prefit_row)
    curves = []
    for index, schedule_row in enumerate(schedule_rows):
        nobs = int(schedule_row["nobs"])
        accuracy = float(schedule_row["RV accuracy"])
        baseline = float(schedule_row["baseline"])
        times = numpy.asarray(schedule_row["rv_times"], dtype=float)
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


def gpu_radial_velocity_batch(system_batches):
    """Evaluate all schedules for up to 32 systems in one GPU dispatch."""
    counts = []
    parameter_rows = []
    times_by_schedule = []
    peak_times = []
    for prefit_row, schedule_rows in system_batches:
        parameter_row = truth_parameter_row(prefit_row)
        counts.append(len(schedule_rows))
        for schedule_row in schedule_rows:
            parameter_rows.append(parameter_row)
            times_by_schedule.append(
                numpy.asarray(schedule_row["rv_times"], dtype=float)
            )
            peak_times.append(float(prefit_row["peak_time"]))

    max_nobs = max(len(times) for times in times_by_schedule)
    times = numpy.empty((len(times_by_schedule), max_nobs), dtype=float)
    for index, schedule_times in enumerate(times_by_schedule):
        times[index, : len(schedule_times)] = schedule_times
        if len(schedule_times) < max_nobs:
            times[index, len(schedule_times) :] = schedule_times[-1]

    radial_velocities = numpy.asarray(
        batched_radial_velocities(
            jax_numpy.asarray(parameter_rows),
            jax_numpy.asarray(times),
            jax_numpy.asarray(peak_times),
        )
    )
    offsets = numpy.cumsum([0, *counts])
    return [
        radial_velocities[offsets[index] : offsets[index + 1]].copy()
        for index in range(len(system_batches))
    ]


def write_system_curves(prefit_row, schedule_rows, photometry_rvs):
    """CPU sidecar entry point: build and write one system's FITS file."""
    curves = make_curves(prefit_row, schedule_rows, photometry_rvs)
    write_rv_timeseries_file(curves, curves[0]["row"]["rv_path"])
    return [curve["row"] for curve in curves]


@contextmanager
def cpu_sidecar_pool():
    """Create spawn-based workers that inherit a CPU-only JAX configuration."""
    previous = os.environ.get("SLRV_RV_CPU_SIDECAR")
    os.environ["SLRV_RV_CPU_SIDECAR"] = "1"
    try:
        with ProcessPoolExecutor(
            max_workers=RV_GENERATION_CPU_MAX_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            yield executor
    finally:
        if previous is None:
            del os.environ["SLRV_RV_CPU_SIDECAR"]
        else:
            os.environ["SLRV_RV_CPU_SIDECAR"] = previous


def main():
    configure_logging()
    if REQUIRE_GPU and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Stage 2 requires a JAX GPU; set SLRV_REQUIRE_GPU=0 for debugging.")
    prefits = load_table(PREFIT_RESULTS_PATH)
    schedules = load_table(SCHEDULE_TABLE_PATH)
    if prefits.empty or schedules.empty:
        raise RuntimeError("Run src/1_prefit_and_schedule.py before RV generation")
    existing = existing_rv_keys()
    schedules_by_system = {
        str(simulation_id): group
        for simulation_id, group in schedules.groupby("simulation_id", sort=False)
    }

    def submit_system_batch(executor, system_batches):
        radial_velocity_batches = gpu_radial_velocity_batch(system_batches)
        return [
            executor.submit(write_system_curves, prefit_row, schedule_rows, radial_velocities)
            for (prefit_row, schedule_rows), radial_velocities in zip(
                system_batches, radial_velocity_batches
            )
        ]

    def checkpoint_futures(futures):
        metadata_rows = []
        for future in as_completed(futures):
            metadata_rows.extend(future.result())
        # Keep the resume index consistent with the completed FITS sidecars,
        # but checkpoint several GPU batches at once to avoid repeated CSV rewrites.
        update_rv_metadata(metadata_rows)

    with tqdm(total=len(prefits), desc="RV timeseries systems", unit="system") as progress:
        with cpu_sidecar_pool() as executor:
            system_batches = []
            futures = []
            max_in_flight = max(
                2 * RV_GENERATION_GPU_BATCH_SIZE,
                2 * RV_GENERATION_CPU_MAX_WORKERS,
            )
            for _, prefit_row in prefits.iterrows():
                system_schedules = schedules_by_system.get(
                    str(prefit_row["simulation_id"])
                )
                schedule_rows = []
                if system_schedules is not None:
                    for schedule_row in system_schedules.to_dict("records"):
                        key = rv_key_from_values(
                            prefit_row["simulation_id"],
                            schedule_row["nobs"],
                            schedule_row["RV accuracy"],
                            schedule_row["baseline"],
                        )
                        if key not in existing:
                            schedule_rows.append(schedule_row)
                            existing.add(key)
                if not schedule_rows:
                    progress.update(1)
                    continue

                system_batches.append((prefit_row.to_dict(), schedule_rows))
                if len(system_batches) == RV_GENERATION_GPU_BATCH_SIZE:
                    futures.extend(submit_system_batch(executor, system_batches))
                    system_batches = []
                    if len(futures) >= max_in_flight:
                        checkpoint_futures(futures)
                        progress.update(len(futures))
                        futures = []

            if system_batches:
                futures.extend(submit_system_batch(executor, system_batches))
            if futures:
                checkpoint_futures(futures)
                progress.update(len(futures))


if __name__ == "__main__":
    main()
