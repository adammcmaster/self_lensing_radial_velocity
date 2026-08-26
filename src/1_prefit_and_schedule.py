"""Fit photometry and build posterior-informed RV observing schedules."""

import os
from collections import Counter
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
# A deterministic subset is useful for repeatable performance and convergence
# benchmarks.  It is deliberately opt-in so production runs retain all rows.
PREFIT_SAMPLE_SIZE = int(os.environ.get("SLRV_PREFIT_SAMPLE_SIZE", 0))


def main():
    # Spawned sidecars import this module but do not execute main.  Delaying
    # these JAX-dependent imports lets their initializer select a CPU backend.
    from photometry import PHOTOMETRY_SIMULATION_PATH, photometry_simulation_paths
    from prefit import (
        PREFIT_RESULTS_PATH,
        PREFIT_BATCH_SIZE,
        PREFIT_POSTERIOR_SAMPLE_COUNT,
        FIT_BURN,
        FIT_NSTEPS,
        FIT_NWALKERS,
        FIT_SAMPLER,
        SCHEDULE_TABLE_PATH,
        NOBS_VALUES,
        RV_ACCURACIES,
        fit_photometry_batch,
        photometry_prefit_batches,
        prefit_bin_width_days,
        load_table,
        photometry_rows,
        save_prefits_and_schedules,
    )
    from prefit_sidecar import initialise, materialize

    configure_logging()
    if REQUIRE_GPU and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Stage 1 requires a JAX GPU; set SLRV_REQUIRE_GPU=0 for debugging.")
    print(
        "Stage-1 prefit settings: "
        f"sampler={FIT_SAMPLER}, max_batch_size={PREFIT_BATCH_SIZE}, "
        f"nwalkers={FIT_NWALKERS}, nsteps={FIT_NSTEPS}, burn={FIT_BURN}, "
        f"retained_posterior_samples={PREFIT_POSTERIOR_SAMPLE_COUNT}, "
        f"sidecar_workers={PREFIT_SIDECAR_WORKERS}",
        flush=True,
    )
    paths = photometry_simulation_paths(PHOTOMETRY_SIMULATION_PATH)
    if not paths:
        raise FileNotFoundError(
            "No photometry FITS files found in "
            f"{PHOTOMETRY_SIMULATION_PATH}. Run src/0_generate_photometry.py "
            "to create the local simulations/2.0/lensing input library."
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
    if PREFIT_SAMPLE_SIZE < 0:
        raise ValueError("SLRV_PREFIT_SAMPLE_SIZE must be zero or positive")
    if PREFIT_SAMPLE_SIZE:
        rows = rows.iloc[:PREFIT_SAMPLE_SIZE].copy()
    pending_rows = [row for _, row in rows.iterrows()]
    batches = photometry_prefit_batches(pending_rows)
    batch_summary = Counter(
        24.0 * prefit_bin_width_days(batch[0]["tau_eff"])
        for batch in batches
    )
    print(
        "Photometry prefit batches by bin width: "
        + ", ".join(
            f"{width:g} h={count}" for width, count in sorted(batch_summary.items())
        ),
        flush=True,
    )
    if PREFIT_SIDECAR_WORKERS < 1 or PREFIT_MAX_PENDING_BATCHES < 1:
        raise ValueError("Prefit sidecar worker and pending-batch counts must be positive")
    def checkpoint(future):
        batch_prefits, batch_schedules = future.result()
        # Only this main process writes the shared checkpoint tables.
        save_prefits_and_schedules(batch_prefits, batch_schedules)

    with ProcessPoolExecutor(
        max_workers=PREFIT_SIDECAR_WORKERS,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialise,
    ) as sidecar:
        pending = set()
        for batch in tqdm(
            batches,
            total=len(batches),
            desc="Photometry prefit batches",
            unit="batch",
        ):
            payload = fit_photometry_batch(batch)
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
