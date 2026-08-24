"""Generate noisy RV curves at the posterior-informed observing timestamps."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("SLRV_JAX_PLATFORMS", "cpu"))

import numpy
import pandas
from astropy import units
from astropy.time import Time

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
from selflensing.generators import RVGenerator
from selflensing.models import SelfLensingSystem
from spectrograph import Spectrograph
from utils import configure_logging


def truth_system(row):
    return SelfLensingSystem(
        a=row["a"] * units.au, i=row["i"] * units.rad,
        Macc=row["Macc"] * units.M_sun, Mcomp=row["Mcomp"] * units.M_sun,
        Rcomp=row["Rcomp"] * units.R_sun, Teff=row["Teff"] * units.K,
        e=row["e"] * units.dimensionless_unscaled,
        periapsis_phase=row["periapsis_phase"] * units.rad,
    )


def make_curve(prefit_row, schedule_row):
    nobs = int(schedule_row["nobs"])
    accuracy = float(schedule_row["RV accuracy"])
    baseline = float(schedule_row["baseline"])
    rv_times = numpy.asarray(schedule_row["rv_times"], dtype=float)
    system = truth_system(prefit_row)
    photometry_rv = RVGenerator(system, times=Time(rv_times, format="jd"), peak_time=Time(prefit_row["peak_time"], format="jd")).rv.to_value(units.m / units.s)
    stellar_noise = Spectrograph.STELLAR_NOISE.to_value(units.m / units.s)
    rv_err = numpy.sqrt(stellar_noise**2 + accuracy**2)
    noisy_rv = photometry_rv + numpy.random.normal(0.0, rv_err, len(rv_times))
    photometry = load_lightcurve(prefit_row)
    metadata_photometry = pandas.Series(prefit_row).copy()
    metadata_photometry["rv_path"] = ""
    row = metadata_row_from_params(
        rv_id=rv_id_from_params(metadata_photometry, nobs, accuracy, baseline),
        rv_path=rv_path_from_params(metadata_photometry, nobs, accuracy, baseline),
        photometry=metadata_photometry, rv_accuracy=accuracy, nobs=nobs, baseline=baseline,
    )
    return {"row": row, "rv_times": rv_times, "rv": noisy_rv, "rv_err": rv_err,
            "lc_times": photometry["times"], "lc": photometry["flux"], "lc_err": photometry["err"]}


def main():
    configure_logging()
    prefits = load_table(PREFIT_RESULTS_PATH)
    schedules = load_table(SCHEDULE_TABLE_PATH)
    if prefits.empty or schedules.empty:
        raise RuntimeError("Run src/1_prefit_and_schedule.py before RV generation")
    existing = existing_rv_keys()
    rows = []
    for _, prefit_row in prefits.iterrows():
        system_schedules = schedules[schedules["simulation_id"].astype(str) == str(prefit_row["simulation_id"])]
        curves = []
        for _, schedule_row in system_schedules.iterrows():
            key = rv_key_from_values(prefit_row["simulation_id"], schedule_row["nobs"], schedule_row["RV accuracy"], schedule_row["baseline"])
            if key not in existing:
                curves.append(make_curve(prefit_row, schedule_row))
        if curves:
            write_rv_timeseries_file(curves, curves[0]["row"]["rv_path"])
            rows.extend(curve["row"] for curve in curves)
    update_rv_metadata(rows)


if __name__ == "__main__":
    main()
