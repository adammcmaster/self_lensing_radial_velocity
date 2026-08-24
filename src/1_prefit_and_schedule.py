"""Fit photometry and build posterior-informed RV observing schedules."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# JAX uses backend names ("cuda"), not device-platform names ("gpu").
# Override cluster-provided settings such as ``JAX_PLATFORMS=rocm``: this
# project environment ships a CUDA JAX backend.
os.environ["JAX_PLATFORMS"] = os.environ.get("SLRV_JAX_PLATFORMS", "cuda")

import jax

from prefit import (
    PREFIT_RESULTS_PATH,
    SCHEDULE_TABLE_PATH,
    fit_photometry_rows,
    load_table,
    photometry_rows,
    save_prefits_and_schedules,
)
from photometry import PHOTOMETRY_SIMULATION_PATH, photometry_simulation_paths
from utils import configure_logging


REQUIRE_GPU = os.environ.get("SLRV_REQUIRE_GPU", "1").lower() in (
    "1", "true", "yes", "on"
)


def main():
    configure_logging()
    if REQUIRE_GPU and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Stage 1 requires a JAX GPU; set SLRV_REQUIRE_GPU=0 for debugging.")
    rows = photometry_rows(photometry_simulation_paths(PHOTOMETRY_SIMULATION_PATH))
    existing = load_table(PREFIT_RESULTS_PATH)
    completed = (
        set(existing["simulation_id"].astype(str))
        if "simulation_id" in existing
        else set()
    )
    rows = rows[~rows["simulation_id"].astype(str).isin(completed)]
    prefits = fit_photometry_rows([row for _, row in rows.iterrows()])
    save_prefits_and_schedules(prefits)
    print(
        f"Saved photometry prefits to {PREFIT_RESULTS_PATH} and RV schedules to {SCHEDULE_TABLE_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
