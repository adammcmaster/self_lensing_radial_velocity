"""Fit photometry and build posterior-informed RV observing schedules."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from prefit import (
    PREFIT_RESULTS_PATH,
    SCHEDULE_TABLE_PATH,
    fit_photometry_row,
    load_table,
    photometry_rows,
    save_prefits_and_schedules,
)
from pristine import PHOTOMETRY_SIMULATION_PATH, photometry_simulation_paths
from utils import configure_logging


def main():
    configure_logging()
    rows = photometry_rows(photometry_simulation_paths(PHOTOMETRY_SIMULATION_PATH))
    existing = load_table(PREFIT_RESULTS_PATH)
    completed = (
        set(existing["simulation_id"].astype(str))
        if "simulation_id" in existing
        else set()
    )
    rows = rows[~rows["simulation_id"].astype(str).isin(completed)]
    prefits = [fit_photometry_row(row) for _, row in rows.iterrows()]
    save_prefits_and_schedules(prefits)
    print(
        f"Saved photometry prefits to {PREFIT_RESULTS_PATH} and RV schedules to {SCHEDULE_TABLE_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
