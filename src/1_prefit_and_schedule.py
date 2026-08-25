"""Fit photometry and build posterior-informed RV observing schedules."""

import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# JAX uses backend names ("cuda"), not device-platform names ("gpu").
# Override cluster-provided settings such as ``JAX_PLATFORMS=rocm``: this
# project environment ships a CUDA JAX backend.
os.environ["JAX_PLATFORMS"] = os.environ.get("SLRV_JAX_PLATFORMS", "cuda")

import jax
from tqdm import tqdm

from utils import configure_logging


REQUIRE_GPU = os.environ.get("SLRV_REQUIRE_GPU", "1").lower() in (
    "1", "true", "yes", "on"
)
PREFIT_SIDECAR_WORKERS = int(os.environ.get("SLRV_PREFIT_SIDECAR_WORKERS", 1))
PREFIT_MAX_PENDING_BATCHES = int(os.environ.get("SLRV_PREFIT_MAX_PENDING_BATCHES", 2))


def main():
    # Spawned sidecars import this module but do not execute main.  Delaying
    # these JAX-dependent imports lets their initializer select a CPU backend.
    from photometry import PHOTOMETRY_SIMULATION_PATH, photometry_simulation_paths
    from prefit import (
        PREFIT_RESULTS_PATH,
        PREFIT_BATCH_SIZE,
        SCHEDULE_TABLE_PATH,
        NOBS_VALUES,
        RV_ACCURACIES,
        fit_photometry_batch,
        load_table,
        photometry_rows,
        save_prefits_and_schedules,
    )
    from prefit_sidecar import initialise, materialize

    configure_logging()
    if REQUIRE_GPU and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Stage 1 requires a JAX GPU; set SLRV_REQUIRE_GPU=0 for debugging.")
    paths = photometry_simulation_paths(PHOTOMETRY_SIMULATION_PATH)
    if not paths:
        raise FileNotFoundError(
            "No photometry FITS files found in "
            f"{PHOTOMETRY_SIMULATION_PATH}. Ensure Searchlight's DATA_PATH "
            "contains simulations/1.0/lensing with the input *.fits.gz files."
        )
    rows = photometry_rows(paths)
    existing = load_table(PREFIT_RESULTS_PATH)
    existing_schedules = load_table(SCHEDULE_TABLE_PATH)
    expected_schedule_count = len(NOBS_VALUES) * len(RV_ACCURACIES)
    scheduled_counts = (
        existing_schedules.assign(
            simulation_id=existing_schedules["simulation_id"].astype(str)
        ).groupby("simulation_id").size()
        if "simulation_id" in existing_schedules
        else {}
    )
    completed = (
        {
            str(simulation_id)
            for simulation_id in existing["simulation_id"]
            if scheduled_counts.get(str(simulation_id), 0) >= expected_schedule_count
        }
        if "simulation_id" in existing
        else set()
    )
    rows = rows[~rows["simulation_id"].astype(str).isin(completed)]
    pending_rows = [row for _, row in rows.iterrows()]
    batch_starts = range(0, len(pending_rows), PREFIT_BATCH_SIZE)
    if PREFIT_SIDECAR_WORKERS < 1 or PREFIT_MAX_PENDING_BATCHES < 1:
        raise ValueError("Prefit sidecar worker and pending-batch counts must be positive")
    completed_count = 0

    def checkpoint(future):
        nonlocal completed_count
        batch_prefits, batch_schedules = future.result()
        # Only this main process writes the shared checkpoint tables.
        save_prefits_and_schedules(batch_prefits, batch_schedules)
        completed_count += len(batch_prefits)
        print(
            f"Checkpointed {completed_count} of {len(pending_rows)} photometry prefits",
            flush=True,
        )

    with ProcessPoolExecutor(
        max_workers=PREFIT_SIDECAR_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialise,
    ) as sidecar:
        pending = set()
        for start in tqdm(
            batch_starts,
            total=(len(pending_rows) + PREFIT_BATCH_SIZE - 1) // PREFIT_BATCH_SIZE,
            desc="Photometry prefit batches",
            unit="batch",
        ):
            payload = fit_photometry_batch(
                pending_rows[start : start + PREFIT_BATCH_SIZE]
            )
            pending.add(sidecar.submit(materialize, payload))
            if len(pending) >= PREFIT_MAX_PENDING_BATCHES:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    checkpoint(future)
        while pending:
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                checkpoint(future)
    print(f"Saved photometry prefits to {PREFIT_RESULTS_PATH} and RV schedules to {SCHEDULE_TABLE_PATH}", flush=True)


if __name__ == "__main__":
    main()
