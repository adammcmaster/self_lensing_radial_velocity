"""Materialization helpers for rv fit results.

This module is intentionally independent of JAX so it can run in a background
process without creating a competing GPU context.
"""

import logging
import time

import pandas
from astropy import units

from selflensing.models import SelfLensingSystem

logger = logging.getLogger(__name__)


def truth_self_lensing_system(ts):
    return SelfLensingSystem(
        a=ts["a"] * units.au,
        i=ts["i"] * units.rad,
        Macc=ts["Macc"] * units.M_sun,
        e=ts["e"] * units.dimensionless_unscaled,
        periapsis_phase=ts["periapsis_phase"] * units.rad,
        Mcomp=ts["Mcomp"] * units.M_sun,
        Rcomp=ts["Rcomp"] * units.R_sun,
        Teff=ts["Teff"] * units.K,
    )


def fitted_timeseries_row(i, ts, truth_sls, sls):
    new_ts = pandas.Series(ts).copy()
    new_ts["rv_index"] = i
    new_ts["b_max"] = truth_sls.b_max.value
    new_ts["fitted_a"] = sls.a.value
    new_ts["fitted_b"] = sls.b.value
    new_ts["fitted_b_max"] = sls.b_max.value
    new_ts["fitted_i"] = sls.i.value
    new_ts["fitted_Macc"] = sls.Macc.value
    new_ts["fitted_e"] = sls.e.value
    new_ts["fitted_periapsis_phase"] = sls.periapsis_phase.value
    new_ts["fitted_porb"] = sls.porb.value
    new_ts["fitted_tau_eff"] = sls.tau_eff.value
    return new_ts


def add_fit_stage_columns(ts, fit_ts, stage_prefix):
    if fit_ts is None:
        return ts
    new_ts = pandas.Series(ts).copy()
    fit_result_columns = (
        "b_max",
        "fitted_a",
        "fitted_b",
        "fitted_b_max",
        "fitted_i",
        "fitted_Macc",
        "fitted_e",
        "fitted_periapsis_phase",
        "fitted_porb",
        "fitted_tau_eff",
    )
    for column in fit_result_columns:
        if column in fit_ts:
            new_ts[f"{stage_prefix}_{column}"] = fit_ts[column]
    return new_ts


def add_prefit_columns(ts, prefit_ts):
    return add_fit_stage_columns(ts, prefit_ts, "prefit")


def add_rv_only_columns(ts, rvfit_ts):
    return add_fit_stage_columns(ts, rvfit_ts, "rvfit")


def materialize_joint_batch_result(
    batch_fit_result,
    prefit_sampler_state_path=None,
    prefit_ts=None,
):
    rows, systems, sampler, walker_state = batch_fit_result
    del sampler, walker_state
    results = []
    materialize_start = time.perf_counter()
    for (i, ts), sls in zip(rows, systems):
        new_ts = fitted_timeseries_row(i, ts, truth_self_lensing_system(ts), sls)
        new_ts = add_prefit_columns(new_ts, prefit_ts)
        if prefit_sampler_state_path is not None:
            new_ts["fit_prefit_sampler_state_path"] = str(prefit_sampler_state_path)
        results.append((new_ts, None, None))
    logger.info(
        "Materialized %d batched fit results in %.2f s",
        len(results),
        time.perf_counter() - materialize_start,
    )
    return results


def materialize_rv_only_batch_result(batch_fit_result):
    rows, systems, sampler, walker_state = batch_fit_result
    results = []
    materialize_start = time.perf_counter()
    for (i, row), sls in zip(rows, systems):
        rvfit_ts = fitted_timeseries_row(i, row, truth_self_lensing_system(row), sls)
        new_ts = add_rv_only_columns(row, rvfit_ts)
        logger.info(
            "Materialized batched RV-only fit result for row index=%s: "
            "rvfit_fitted_a=%.6g rvfit_fitted_Macc=%.6g rvfit_fitted_e=%.6g",
            i,
            new_ts["rvfit_fitted_a"],
            new_ts["rvfit_fitted_Macc"],
            new_ts["rvfit_fitted_e"],
        )
        results.append((new_ts, sampler, walker_state, None))
    logger.info(
        "Materialized %d batched RV-only fit results in %.2f s",
        len(results),
        time.perf_counter() - materialize_start,
    )
    return results
