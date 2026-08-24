"""
Generate fit diagnostic plots for already-fitted rv simulations.
"""

import argparse
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault(
    "JAX_PLATFORMS",
    os.environ.get(
        "SLRV_PLOT_JAX_PLATFORMS",
        os.environ.get("SLRV_JAX_PLATFORMS", "cpu"),
    ),
)
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("SLRV_MPLCONFIGDIR", "/tmp"))

logger = logging.getLogger(__name__)
DEFAULT_FIT_RESULTS = Path(__file__).parent / Path("../data/fitted_timeseries.gz")
DEFAULT_PREFIT_RESULTS = Path(__file__).parent / Path("../data/photometry_prefits.gz")
DEFAULT_FIG_PATH = Path(__file__).parent / Path("../data/figures")
FIG_PATH = Path(os.environ.get("SLRV_FIG_PATH", DEFAULT_FIG_PATH))
PLOT_MAX_WORKERS = int(os.environ.get("SLRV_PLOT_MAX_WORKERS", min(os.cpu_count(), 16) or 1))
MP_START_METHOD = os.environ.get("SLRV_MP_START_METHOD", "spawn")
FIT_PARAMETER_NAMES = ("a", "i", "Macc", "e", "periapsis_phase")
FIT_PARAMETER_LABELS = (
    "$a$ [AU]",
    "$i$ [rad]",
    "$M_\\mathrm{BH}$ [$M_\\odot$]",
    "$e$",
    "$\\varphi$ [rad]",
)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate diagnostic PNGs for saved fitted timeseries."
    )
    parser.add_argument(
        "simulation_ids",
        nargs="*",
        help="Explicit simulation IDs to plot. Defaults to rv_id values.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every row in the fitted results table.",
    )
    parser.add_argument(
        "--sample",
        type=positive_int,
        help="Plot a random sample of this many fitted rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed to use with --sample.",
    )
    parser.add_argument(
        "--id-column",
        default="rv_id",
        help="Column in the fitted results table to match for explicit IDs.",
    )
    parser.add_argument(
        "--fit-results",
        type=Path,
        default=DEFAULT_FIT_RESULTS,
        help=f"Path to fitted results table. Default: {DEFAULT_FIT_RESULTS}.",
    )
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        default=PLOT_MAX_WORKERS,
        help=f"Maximum plot rendering worker processes. Default: {PLOT_MAX_WORKERS}.",
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        choices=(
            "fit",
            "prefit",
            "prefit_walkers",
            "prefit_corner",
            "trace",
            "walkers",
            "corner",
            "all",
        ),
        default=("fit",),
        help="Plot types to render. 'trace' is accepted as an alias for walkers; 'prefit' renders all prefit diagnostics. Default: fit.",
    )
    return parser.parse_args()


def load_plotting_stack():
    global Time
    global pyplot
    global numpy
    global pandas
    global seaborn
    global tqdm
    global units
    global corner
    global LightcurveGenerator
    global RVGenerator
    global SelfLensingSystem
    global load_rv_timeseries_curve

    import corner
    import numpy
    import pandas
    import seaborn
    from astropy import units
    from astropy.time import Time
    from matplotlib import pyplot
    from tqdm import tqdm

    from jax_compat import install_quax_jax_stage_patch

    install_quax_jax_stage_patch()

    from selflensing.generators import LightcurveGenerator, RVGenerator
    from selflensing.models import SelfLensingSystem

    from rv_timeseries import load_rv_timeseries_curve

    logging.getLogger().addFilter(
        lambda record: record.getMessage() != "Too few points to create valid contours"
    )


def load_fitted_results(path):
    import pandas

    if not path.exists():
        raise FileNotFoundError(f"Fitted results table does not exist: {path}")
    table = pandas.read_pickle(path)
    logger.info("Loaded %d fitted rows from %s", len(table), path)
    return table


def deduplicate_fitted_results(table):
    if "rv_id" not in table.columns:
        return table.copy()
    deduped = table.drop_duplicates(subset="rv_id", keep="last").copy()
    removed = len(table) - len(deduped)
    if removed:
        logger.info("Dropped %d older duplicate fitted row(s)", removed)
    return deduped


def validate_selection(args):
    modes = int(args.all) + int(args.sample is not None) + int(bool(args.simulation_ids))
    if modes != 1:
        raise ValueError(
            "Choose exactly one plotting mode: --all, --sample N, or explicit IDs."
        )


def rows_for_ids(table, id_column, simulation_ids):
    if id_column not in table.columns:
        raise ValueError(
            f"Fitted results table has no {id_column!r} column. "
            f"Available columns: {', '.join(table.columns)}"
        )

    requested = [str(simulation_id) for simulation_id in simulation_ids]
    requested_set = set(requested)
    id_values = table[id_column].astype(str)
    rows = table[id_values.isin(requested_set)].copy()
    found = set(rows[id_column].astype(str))
    missing = [simulation_id for simulation_id in requested if simulation_id not in found]
    if missing:
        raise KeyError(
            f"Could not find {id_column} value(s) in fitted results: "
            f"{', '.join(missing)}"
        )
    return rows


def select_rows(table, args):
    table = deduplicate_fitted_results(table)
    if args.all:
        rows = table.copy()
        logger.info("Selected all %d fitted row(s)", len(rows))
        return rows
    if args.sample is not None:
        if args.sample > len(table):
            raise ValueError(
                f"Cannot sample {args.sample} rows from only {len(table)} fitted rows"
            )
        rows = table.sample(n=args.sample, random_state=args.seed).copy()
        logger.info(
            "Selected random sample of %d/%d fitted row(s)", len(rows), len(table)
        )
        return rows

    rows = rows_for_ids(table, args.id_column, args.simulation_ids)
    logger.info(
        "Selected %d fitted row(s) matching %d explicit ID(s)",
        len(rows),
        len(args.simulation_ids),
    )
    return rows


def truth_self_lensing_system(row):
    return SelfLensingSystem(
        a=scalar_value(row["a"]) * units.au,
        i=scalar_value(row["i"]) * units.rad,
        Macc=scalar_value(row["Macc"]) * units.M_sun,
        e=scalar_value(row["e"]) * units.dimensionless_unscaled,
        periapsis_phase=scalar_value(row["periapsis_phase"]) * units.rad,
        Mcomp=scalar_value(row["Mcomp"]) * units.M_sun,
        Rcomp=scalar_value(row["Rcomp"]) * units.R_sun,
        Teff=scalar_value(row["Teff"]) * units.K,
    )


def quantity_values(values, unit=None):
    if unit is not None and hasattr(values, "to"):
        values = values.to(unit)
    if hasattr(values, "value"):
        values = values.value
    return numpy.asarray(values, dtype=float)


def scalar_value(value):
    return float(numpy.asarray(value))


def format_value_token(value):
    token = f"{scalar_value(value):g}"
    return token.replace("-", "m").replace(".", "p")


def format_filename_token(name, value, suffix=""):
    return f"{name}{format_value_token(value)}{suffix}"


def safe_path_token(value):
    if value is None or pandas.isna(value):
        return "unknown"
    return str(value).replace("/", "_")


def normalized_plot_types(plot_types):
    plot_types = set(plot_types)
    if "all" in plot_types:
        return (
            "fit",
            "prefit",
            "prefit_walkers",
            "prefit_corner",
            "walkers",
            "corner",
        )
    if "trace" in plot_types:
        plot_types.add("walkers")
    if "prefit" in plot_types:
        plot_types.update(("prefit_walkers", "prefit_corner"))
    return tuple(
        plot_type
        for plot_type in (
            "fit",
            "prefit",
            "prefit_walkers",
            "prefit_corner",
            "walkers",
            "corner",
        )
        if plot_type in plot_types
    )


def sorted_xy(x, y):
    x = quantity_values(x)
    y = quantity_values(y)
    order = numpy.argsort(x)
    return x[order], y[order]


def align_lc_model_to_data(model_lc, lc_data):
    model_lc = quantity_values(model_lc)
    lc_data = quantity_values(lc_data)
    finite_model = numpy.isfinite(model_lc)
    finite_data = numpy.isfinite(lc_data)
    if not finite_model.any() or not finite_data.any():
        return model_lc
    return model_lc - numpy.nanmedian(model_lc) + numpy.nanmedian(lc_data)


def plot_fit(row):
    logger.debug(
        "Plotting fit for rv_index=%s rv_id=%s",
        row.get("rv_index"),
        row.get("rv_id"),
    )
    truth_sls = truth_self_lensing_system(row)
    sls = SelfLensingSystem(
        a=scalar_value(row["fitted_a"]) * units.au,
        i=scalar_value(row["fitted_i"]) * units.rad,
        Macc=scalar_value(row["fitted_Macc"]) * units.M_sun,
        e=scalar_value(row["fitted_e"]) * units.dimensionless_unscaled,
        periapsis_phase=scalar_value(row["fitted_periapsis_phase"]) * units.rad,
        Mcomp=scalar_value(row["Mcomp"]) * units.M_sun,
        Rcomp=scalar_value(row["Rcomp"]) * units.R_sun,
        Teff=scalar_value(row["Teff"]) * units.K,
    )
    peak_time = Time(row["peak_time"], format="jd")
    rvg = RVGenerator(
        sls, times=Time(row["rv_times"], format="jd"), peak_time=peak_time
    )
    lcg = LightcurveGenerator(
        sls,
        times=Time(row["lc_times"], format="jd"),
        peak_time=peak_time,
    )
    truth_lcg = LightcurveGenerator(
        truth_sls,
        times=Time(row["lc_times"], format="jd"),
        peak_time=peak_time,
    )
    rv_times = quantity_values(row["rv_times"])
    rv_data = quantity_values(row["rv"])
    rv_model = quantity_values(rvg.rv, units.m / units.s)
    lc_times = quantity_values(row["lc_times"])
    lc_data = quantity_values(row["lc"])
    time_zero = numpy.nanmin(lc_times)
    rv_plot_times = rv_times - time_zero
    lc_plot_times = lc_times - time_zero
    model_lc = align_lc_model_to_data(quantity_values(lcg.flux_array) - 1.0, lc_data)
    truth_model_lc = align_lc_model_to_data(
        quantity_values(truth_lcg.flux_array) - 1.0,
        lc_data,
    )
    rv_model_times, rv_model_sorted = sorted_xy(rv_plot_times, rv_model)
    lc_model_times, model_lc_sorted = sorted_xy(lc_plot_times, model_lc)
    _, truth_model_lc_sorted = sorted_xy(lc_plot_times, truth_model_lc)

    fig, (
        rv_axis,
        rv_resid_axis,
        lc_axis,
        lc_full_resid_axis,
        lc_window_axis,
        lc_window_resid_axis,
    ) = pyplot.subplots(
        6,
        1,
        figsize=(14, 15),
        gridspec_kw={"height_ratios": (4, 1, 4, 1, 2, 1)},
    )

    seaborn.scatterplot(x=rv_plot_times, y=rv_data, color="blue", ax=rv_axis)
    rv_axis.errorbar(
        rv_plot_times,
        rv_data,
        yerr=row["RV accuracy"],
        fmt="none",
        ecolor="gray",
        alpha=0.7,
    )
    rv_axis.plot(rv_model_times, rv_model_sorted, color="red")
    rv_axis.set_title("Radial Velocity")
    rv_axis.set_xlabel("Days Since Photometry Start")
    rv_axis.set_ylabel("RV")

    seaborn.scatterplot(
        x=lc_plot_times, y=lc_data, color="green", ax=lc_axis, label="Data"
    )
    lc_axis.plot(lc_model_times, model_lc_sorted, color="red", label="Fit")
    lc_axis.plot(lc_model_times, truth_model_lc_sorted, color="blue", label="Truth")
    lc_axis.set_title("Lightcurve")
    lc_axis.set_xlabel("Days Since Photometry Start")
    lc_axis.set_ylabel("LC")
    lc_axis.legend()

    fit_window_start = row["peak_time"] - row["tau_eff"]
    fit_window_end = row["peak_time"] + row["tau_eff"]
    fit_window_start_plot = fit_window_start - time_zero
    fit_window_end_plot = fit_window_end - time_zero
    inset_lc_mask = (lc_times >= fit_window_start) & (lc_times <= fit_window_end)
    seaborn.scatterplot(
        x=lc_plot_times[inset_lc_mask],
        y=lc_data[inset_lc_mask],
        color="green",
        ax=lc_window_axis,
        label="Data",
    )
    window_fit_times, window_fit_lc = sorted_xy(
        lc_plot_times[inset_lc_mask], model_lc[inset_lc_mask]
    )
    _, window_truth_lc = sorted_xy(
        lc_plot_times[inset_lc_mask], truth_model_lc[inset_lc_mask]
    )
    lc_window_axis.plot(window_fit_times, window_fit_lc, color="red", label="Fit")
    lc_window_axis.plot(
        window_fit_times, window_truth_lc, color="blue", label="Truth"
    )
    lc_window_axis.axvline(
        fit_window_start_plot, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_axis.axvline(
        fit_window_end_plot, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_axis.set_xlim(fit_window_start_plot, fit_window_end_plot)
    lc_window_axis.set_title("Lightcurve Fit Window")
    lc_window_axis.set_xlabel("Days Since Photometry Start")
    lc_window_axis.set_ylabel("LC")
    lc_window_axis.legend()

    rv_residuals = rv_data - rv_model
    lc_residuals = lc_data - model_lc

    seaborn.scatterplot(
        x=rv_plot_times, y=rv_residuals, color="blue", ax=rv_resid_axis
    )
    rv_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    rv_resid_axis.set_ylabel("RV Residuals")

    seaborn.scatterplot(
        x=lc_plot_times, y=lc_residuals, color="green", ax=lc_full_resid_axis
    )
    lc_full_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    lc_full_resid_axis.set_ylabel("LC Residuals")

    seaborn.scatterplot(
        x=lc_plot_times[inset_lc_mask],
        y=lc_residuals[inset_lc_mask],
        color="green",
        ax=lc_window_resid_axis,
    )
    lc_window_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    lc_window_resid_axis.axvline(
        fit_window_start_plot, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_resid_axis.axvline(
        fit_window_end_plot, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_resid_axis.set_xlim(fit_window_start_plot, fit_window_end_plot)
    lc_window_resid_axis.set_ylabel("Window Residuals")

    b_max = row.get("b_max", numpy.nan)
    truth_text = (
        "Truth\n"
        f"$a$={scalar_value(row['a']):.2f} AU\n"
        f"$M_\\text{{BH}}$={scalar_value(row['Macc']):.2f} M$_\\odot$\n"
        f"$P_\\text{{orb}}$={scalar_value(row['Porb']):.2f} d\n"
        f"$\\tau_\\text{{eff}}$={scalar_value(row['tau_eff']):.2f} d\n"
        f"$i$={scalar_value(row['i']):.2f}\n"
        f"$b$={scalar_value(row['b']):.2f}\n"
        f"$b_\\text{{max}}$={scalar_value(b_max):.2f}\n"
        f"$e$={scalar_value(row['e']):.2f}\n"
        f"$\\varphi$={scalar_value(row['periapsis_phase']):.2f} rad\n"
        f"Baseline={scalar_value(row['baseline']):.2f} d\n"
        f"Nobs={scalar_value(row['nobs']):.0f}"
    )
    fit_text = (
        "Fit\n"
        f"$a$={scalar_value(row['fitted_a']):.2f} AU\n"
        f"$M_\\text{{BH}}$={scalar_value(row['fitted_Macc']):.2f} M$_\\odot$\n"
        f"$P_\\text{{orb}}$={scalar_value(row['fitted_porb']):.2f} d\n"
        f"$\\tau_\\text{{eff}}$={scalar_value(row['fitted_tau_eff']):.2f} d\n"
        f"$i$={scalar_value(row['fitted_i']):.2f}\n"
        f"$b$={scalar_value(row['fitted_b']):.2f}\n"
        f"$b_\\text{{max}}$={scalar_value(row['fitted_b_max']):.2f}\n"
        f"$e$={scalar_value(row['fitted_e']):.2f}\n"
        f"$\\varphi$={scalar_value(row['fitted_periapsis_phase']):.2f} rad\n"
        "\n"
        f"RV acc.={scalar_value(row['RV accuracy']):.2g} m/s"
    )
    pyplot.subplots_adjust(right=0.74, hspace=0.32)
    fig.text(
        0.755,
        0.5,
        truth_text,
        ha="left",
        va="center",
        fontsize=12,
        wrap=True,
        color="blue",
    )
    fig.text(
        0.875,
        0.5,
        fit_text,
        ha="left",
        va="center",
        fontsize=12,
        wrap=True,
        color="red",
    )
    return fig


def plot_prefit(row):
    logger.debug(
        "Plotting photometry prefit for system_id=%s rv_id=%s",
        row.get("system_id"),
        row.get("rv_id"),
    )
    truth_sls = truth_self_lensing_system(row)
    prefit_sls = self_lensing_system_from_fit_values(row, prefit_parameter_values(row))
    peak_time = Time(row["peak_time"], format="jd")
    truth_lcg = LightcurveGenerator(
        truth_sls,
        times=Time(row["lc_times"], format="jd"),
        peak_time=peak_time,
    )
    prefit_lcg = LightcurveGenerator(
        prefit_sls,
        times=Time(row["lc_times"], format="jd"),
        peak_time=peak_time,
    )
    lc_times = quantity_values(row["lc_times"])
    lc_data = quantity_values(row["lc"])
    time_zero = numpy.nanmin(lc_times)
    lc_plot_times = lc_times - time_zero
    prefit_lc = align_lc_model_to_data(
        quantity_values(prefit_lcg.flux_array) - 1.0,
        lc_data,
    )
    truth_model_lc = align_lc_model_to_data(
        quantity_values(truth_lcg.flux_array) - 1.0,
        lc_data,
    )
    lc_model_times, prefit_lc_sorted = sorted_xy(lc_plot_times, prefit_lc)
    _, truth_model_lc_sorted = sorted_xy(lc_plot_times, truth_model_lc)

    fig, (
        lc_axis,
        lc_resid_axis,
        lc_window_axis,
        lc_window_resid_axis,
    ) = pyplot.subplots(
        4,
        1,
        figsize=(14, 10),
        gridspec_kw={"height_ratios": (4, 1, 3, 1)},
    )

    seaborn.scatterplot(
        x=lc_plot_times, y=lc_data, color="green", ax=lc_axis, label="Data"
    )
    lc_axis.plot(lc_model_times, prefit_lc_sorted, color="magenta", label="Prefit")
    lc_axis.plot(lc_model_times, truth_model_lc_sorted, color="blue", label="Truth")
    lc_axis.set_title("Photometry Prefit")
    lc_axis.set_xlabel("Days Since Photometry Start")
    lc_axis.set_ylabel("LC")
    lc_axis.legend()

    fit_window_start = row["peak_time"] - row["tau_eff"]
    fit_window_end = row["peak_time"] + row["tau_eff"]
    fit_window_start_plot = fit_window_start - time_zero
    fit_window_end_plot = fit_window_end - time_zero
    inset_lc_mask = (lc_times >= fit_window_start) & (lc_times <= fit_window_end)
    window_fit_times, window_prefit_lc = sorted_xy(
        lc_plot_times[inset_lc_mask], prefit_lc[inset_lc_mask]
    )
    _, window_truth_lc = sorted_xy(
        lc_plot_times[inset_lc_mask], truth_model_lc[inset_lc_mask]
    )

    prefit_residuals = lc_data - prefit_lc
    seaborn.scatterplot(
        x=lc_plot_times,
        y=prefit_residuals,
        color="green",
        ax=lc_resid_axis,
    )
    lc_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    lc_resid_axis.set_ylabel("LC Residuals")

    seaborn.scatterplot(
        x=lc_plot_times[inset_lc_mask],
        y=lc_data[inset_lc_mask],
        color="green",
        ax=lc_window_axis,
        label="Data",
    )
    lc_window_axis.plot(
        window_fit_times, window_prefit_lc, color="magenta", label="Prefit"
    )
    lc_window_axis.plot(window_fit_times, window_truth_lc, color="blue", label="Truth")
    lc_window_axis.axvline(
        fit_window_start_plot, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_axis.axvline(
        fit_window_end_plot, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_axis.set_xlim(fit_window_start_plot, fit_window_end_plot)
    lc_window_axis.set_title("Prefit Window")
    lc_window_axis.set_xlabel("Days Since Photometry Start")
    lc_window_axis.set_ylabel("LC")
    lc_window_axis.legend()

    seaborn.scatterplot(
        x=lc_plot_times[inset_lc_mask],
        y=prefit_residuals[inset_lc_mask],
        color="green",
        ax=lc_window_resid_axis,
    )
    lc_window_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    lc_window_resid_axis.axvline(
        fit_window_start_plot, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_resid_axis.axvline(
        fit_window_end_plot, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_resid_axis.set_xlim(fit_window_start_plot, fit_window_end_plot)
    lc_window_resid_axis.set_ylabel("Window Residuals")

    truth_text = (
        "Truth\n"
        f"$a$={scalar_value(row['a']):.2f} AU\n"
        f"$M_\\text{{BH}}$={scalar_value(row['Macc']):.2f} M$_\\odot$\n"
        f"$P_\\text{{orb}}$={scalar_value(row['Porb']):.2f} d\n"
        f"$\\tau_\\text{{eff}}$={scalar_value(row['tau_eff']):.2f} d\n"
        f"$i$={scalar_value(row['i']):.2f}\n"
        f"$e$={scalar_value(row['e']):.2f}\n"
        f"$\\varphi$={scalar_value(row['periapsis_phase']):.2f} rad"
    )
    prefit_text = (
        "Prefit\n"
        f"$a$={prefit_sls.a.value:.2f} AU\n"
        f"$M_\\text{{BH}}$={prefit_sls.Macc.value:.2f} M$_\\odot$\n"
        f"$P_\\text{{orb}}$={prefit_sls.porb.value:.2f} d\n"
        f"$\\tau_\\text{{eff}}$={prefit_sls.tau_eff.value:.2f} d\n"
        f"$i$={prefit_sls.i.value:.2f}\n"
        f"$e$={prefit_sls.e.value:.2f}\n"
        f"$\\varphi$={prefit_sls.periapsis_phase.value:.2f} rad"
    )
    pyplot.subplots_adjust(right=0.74, hspace=0.38)
    fig.text(
        0.755,
        0.5,
        truth_text,
        ha="left",
        va="center",
        fontsize=12,
        wrap=True,
        color="blue",
    )
    fig.text(
        0.875,
        0.5,
        prefit_text,
        ha="left",
        va="center",
        fontsize=12,
        wrap=True,
        color="magenta",
    )
    return fig


def figure_shard_keys(ts):
    return (
        f"a1e{int(numpy.floor(numpy.log10(ts['a'])))}",
        f"b{int(ts['b'])}",
        f"Macc{int(ts['Macc'])}",
    )


def target_id_for_row(ts):
    return safe_path_token(ts.get("rv_id", ts.get("rv_index", "unknown")))


def observational_token_for_row(ts):
    return "_".join(
        (
            format_filename_token("baseline", ts["baseline"], "d"),
            format_filename_token("nobs", ts["nobs"], ""),
            format_filename_token("rv", ts["RV accuracy"], "ms"),
        )
    )


def rv_plot_dir_name(ts):
    return "_".join((observational_token_for_row(ts), target_id_for_row(ts)))


def original_id_for_row(ts):
    return safe_path_token(ts.get("system_id", "unknown"))


def original_plot_dir_for_row(ts):
    a_key, b_key, macc_key = figure_shard_keys(ts)
    return FIG_PATH / a_key / b_key / macc_key / original_id_for_row(ts)


def target_plot_dir_for_row(ts):
    system_id = safe_path_token(ts.get("system_id", "unknown"))
    a_key, b_key, macc_key = figure_shard_keys(ts)
    return FIG_PATH / a_key / b_key / macc_key / system_id / rv_plot_dir_name(ts)


def prefit_plot_path_for_row(ts):
    original_id = original_id_for_row(ts)
    return original_plot_dir_for_row(ts) / f"prefit_{original_id}.png"


def prefit_sampler_plot_path_for_row(ts, plot_type):
    original_id = original_id_for_row(ts)
    return original_plot_dir_for_row(ts) / f"prefit_{plot_type}_{original_id}.png"


def plot_path_for_row(ts, plot_type):
    target_id = target_id_for_row(ts)
    name = "_".join(
        (
            plot_type,
            observational_token_for_row(ts),
            target_id,
        )
    )
    return target_plot_dir_for_row(ts) / f"{name}.png"


def default_sampler_state_path(ts):
    from fitting import SAMPLER_STATE_PATH

    if "rv_id" in ts and not pandas.isna(ts["rv_id"]):
        token = safe_path_token(ts["rv_id"])
    elif "rv_index" in ts and not pandas.isna(ts["rv_index"]):
        token = f"idx_{safe_path_token(ts['rv_index'])}"
    else:
        token = f"row_{safe_path_token(getattr(ts, 'name', None))}"
    system_id = safe_path_token(ts.get("system_id", "unknown"))
    a_key, b_key, macc_key = figure_shard_keys(ts)
    return SAMPLER_STATE_PATH / a_key / b_key / macc_key / system_id / f"{token}.npz"


def default_prefit_sampler_state_path(ts):
    from fitting import SAMPLER_STATE_PATH

    system_id = safe_path_token(ts.get("system_id", "unknown"))
    a_key, b_key, macc_key = figure_shard_keys(ts)
    return SAMPLER_STATE_PATH / a_key / b_key / macc_key / system_id / "_prefit.npz"


def legacy_sampler_state_path(ts):
    from fitting import SAMPLER_STATE_PATH

    if "rv_id" in ts and not pandas.isna(ts["rv_id"]):
        token = safe_path_token(ts["rv_id"])
    elif "rv_index" in ts and not pandas.isna(ts["rv_index"]):
        token = f"idx_{safe_path_token(ts['rv_index'])}"
    else:
        token = f"row_{safe_path_token(getattr(ts, 'name', None))}"
    return SAMPLER_STATE_PATH / f"{token}.npz"


def sampler_state_path_for_row(ts):
    if "fit_sampler_state_path" in ts and not pandas.isna(ts["fit_sampler_state_path"]):
        path = Path(ts["fit_sampler_state_path"])
        if path.exists():
            return path
    for path in (default_sampler_state_path(ts), legacy_sampler_state_path(ts)):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No sampler state found for rv_id={ts.get('rv_id', None)}"
    )


def path_string(value):
    return str(numpy.asarray(value).item())


def prefit_record_for_row(ts):
    """Load the separate stage-1 record for this RV system."""
    path = Path(os.environ.get("SLRV_PREFIT_OUTPUT_PATH", DEFAULT_PREFIT_RESULTS))
    if not path.exists():
        raise FileNotFoundError(f"Photometry-prefit table does not exist: {path}")
    prefits = pandas.read_pickle(path)
    matches = prefits[prefits["simulation_id"].astype(str) == str(ts["system_id"])]
    if len(matches) != 1:
        raise KeyError(f"Expected one prefit for system_id={ts['system_id']}, found {len(matches)}")
    return matches.iloc[0]


def prefit_sampler_state_path_for_row(ts, state):
    record = prefit_record_for_row(ts)
    path = Path(record["prefit_posterior_path"])
    if path.exists():
        return path
    if (
        "fit_prefit_sampler_state_path" in ts
        and not pandas.isna(ts["fit_prefit_sampler_state_path"])
    ):
        path = Path(ts["fit_prefit_sampler_state_path"])
        if path.exists():
            return path
    if "prefit_sampler_state_path" in state:
        path = Path(path_string(state["prefit_sampler_state_path"]))
        if path.exists():
            return path
    path = default_prefit_sampler_state_path(ts)
    if path.exists():
        return path
    return None


def attach_prefit_sampler_state(ts, state):
    if "prefit_chain" in state:
        return state
    path = prefit_sampler_state_path_for_row(ts, state)
    if path is None:
        return state
    with numpy.load(path) as data:
        state["prefit_chain"] = data["chain"].copy()
        state["prefit_parameter_names"] = data["parameter_names"].copy()
    state["prefit_path"] = path
    return state


def load_sampler_state(ts):
    path = sampler_state_path_for_row(ts)
    with numpy.load(path) as data:
        state = {key: data[key].copy() for key in data.files}
    state["path"] = path
    return attach_prefit_sampler_state(ts, state)


def sampler_param_indices(parameter_names):
    names = [str(name) for name in parameter_names]
    return [names.index(name) for name in FIT_PARAMETER_NAMES]


def truth_parameter_values(row):
    return numpy.asarray([scalar_value(row[name]) for name in FIT_PARAMETER_NAMES])


def fitted_parameter_values(row):
    return numpy.asarray(
        [
            scalar_value(row["fitted_a"]),
            scalar_value(row["fitted_i"]),
            scalar_value(row["fitted_Macc"]),
            scalar_value(row["fitted_e"]),
            scalar_value(row["fitted_periapsis_phase"]),
        ]
    )


def has_prefit_fit_columns(row):
    return all(
        column in row and not pandas.isna(row[column])
        for column in (
            "prefit_fitted_a",
            "prefit_fitted_i",
            "prefit_fitted_Macc",
            "prefit_fitted_e",
            "prefit_fitted_periapsis_phase",
        )
    )


def prefit_parameter_values(row):
    record = prefit_record_for_row(row)
    if has_prefit_fit_columns(record):
        return numpy.asarray([scalar_value(record[f"prefit_fitted_{name}"]) for name in FIT_PARAMETER_NAMES])

    state = load_sampler_state(row)
    if "prefit_chain" not in state:
        raise FileNotFoundError(
            f"No prefit fit columns or prefit sampler chain found for system_id={row.get('system_id', None)}"
        )
    parameter_indices = sampler_param_indices(
        state.get("prefit_parameter_names", state["parameter_names"])
    )
    prefit_chain = numpy.asarray(state["prefit_chain"], dtype=float)
    samples = prefit_chain[:, :, parameter_indices].reshape(
        (-1, len(parameter_indices))
    )
    return numpy.nanmedian(samples, axis=0)


def self_lensing_system_from_fit_values(row, values):
    return SelfLensingSystem(
        a=scalar_value(values[0]) * units.au,
        i=scalar_value(values[1]) * units.rad,
        Macc=scalar_value(values[2]) * units.M_sun,
        e=scalar_value(values[3]) * units.dimensionless_unscaled,
        periapsis_phase=scalar_value(values[4]) * units.rad,
        Mcomp=scalar_value(row["Mcomp"]) * units.M_sun,
        Rcomp=scalar_value(row["Rcomp"]) * units.R_sun,
        Teff=scalar_value(row["Teff"]) * units.K,
    )


def plot_label(ts):
    return (
        f"rv_id={ts.get('rv_id')} | "
        f"baseline={float(ts['baseline']):g} d | "
        f"nobs={float(ts['nobs']):g} | "
        f"RV accuracy={float(ts['RV accuracy']):g} m/s"
    )


def add_plot_title(fig, ts):
    fig.suptitle(f"Fit\n{plot_label(ts)}", fontsize=10)
    fig.subplots_adjust(top=0.9)


def add_prefit_plot_title(fig, ts):
    fig.suptitle(
        f"Photometry Prefit\nsystem_id={ts.get('system_id')}",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.9)


def add_sampler_plot_title(fig, ts, plot_type):
    fig.suptitle(f"{plot_type.title()}\n{plot_label(ts)}", fontsize=10)


def posterior_samples(state):
    chain = numpy.asarray(state["chain"], dtype=float)
    parameter_indices = sampler_param_indices(state["parameter_names"])
    burn = int(numpy.asarray(state.get("burn", 0)))
    if burn < 0 or burn >= chain.shape[0]:
        burn = 0
    return chain[burn:, :, :][:, :, parameter_indices].reshape((-1, len(parameter_indices)))


def walker_trace_chain_and_prefit_end(state):
    chain = numpy.asarray(state["chain"], dtype=float)
    parameter_indices = sampler_param_indices(state["parameter_names"])
    joint_chain = chain[:, :, parameter_indices]
    if "prefit_chain" not in state:
        return joint_chain, None

    prefit_chain = numpy.asarray(state["prefit_chain"], dtype=float)
    prefit_parameter_names = state.get("prefit_parameter_names", state["parameter_names"])
    prefit_parameter_indices = sampler_param_indices(prefit_parameter_names)
    prefit_chain = prefit_chain[:, :, prefit_parameter_indices]
    if prefit_chain.shape[1:] != joint_chain.shape[1:]:
        logger.warning(
            "Skipping prefit trace overlay for %s because prefit shape %s does not match joint shape %s",
            state.get("path", "unknown"),
            prefit_chain.shape,
            joint_chain.shape,
        )
        return joint_chain, None
    return numpy.concatenate((prefit_chain, joint_chain), axis=0), prefit_chain.shape[0]


def plot_walker_traces(row, state):
    chain, prefit_end = walker_trace_chain_and_prefit_end(state)
    truth_values = truth_parameter_values(row)
    fit_values = fitted_parameter_values(row)
    fig, axes = pyplot.subplots(
        len(FIT_PARAMETER_NAMES),
        1,
        figsize=(13, 10),
        sharex=True,
    )
    steps = numpy.arange(chain.shape[0])
    for index, axis in enumerate(numpy.ravel(axes)):
        axis.plot(steps, chain[:, :, index], color="black", alpha=0.18, linewidth=0.5)
        if prefit_end is not None:
            axis.axvline(
                prefit_end - 0.5,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="Prefit ends" if index == 0 else None,
            )
        axis.axhline(truth_values[index], color="blue", linewidth=1.4, label="Truth")
        axis.axhline(fit_values[index], color="red", linewidth=1.4, label="Fit")
        axis.set_ylabel(FIT_PARAMETER_LABELS[index])
        if index == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("MCMC step")
    add_sampler_plot_title(fig, row, "walker traces")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def prefit_chain_for_state(state):
    if "prefit_chain" not in state:
        raise FileNotFoundError(
            f"No prefit sampler chain found in sampler state {state.get('path', 'unknown')}"
        )
    parameter_names = state.get("prefit_parameter_names", state["parameter_names"])
    parameter_indices = sampler_param_indices(parameter_names)
    return numpy.asarray(state["prefit_chain"], dtype=float)[:, :, parameter_indices]


def prefit_posterior_samples(state):
    chain = prefit_chain_for_state(state)
    return chain.reshape((-1, chain.shape[-1]))


def plot_prefit_walker_traces(row, state):
    chain = prefit_chain_for_state(state)
    truth_values = truth_parameter_values(row)
    prefit_values = prefit_parameter_values(row)
    fig, axes = pyplot.subplots(
        len(FIT_PARAMETER_NAMES),
        1,
        figsize=(13, 10),
        sharex=True,
    )
    steps = numpy.arange(chain.shape[0])
    for index, axis in enumerate(numpy.ravel(axes)):
        axis.plot(steps, chain[:, :, index], color="black", alpha=0.18, linewidth=0.5)
        axis.axhline(truth_values[index], color="blue", linewidth=1.4, label="Truth")
        axis.axhline(
            prefit_values[index],
            color="magenta",
            linewidth=1.4,
            label="Prefit",
        )
        axis.set_ylabel(FIT_PARAMETER_LABELS[index])
        if index == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("MCMC prefit step")
    add_sampler_plot_title(fig, row, "prefit walker traces")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_prefit_corner(row, state):
    samples = prefit_posterior_samples(state)
    truth_values = truth_parameter_values(row)
    prefit_values = prefit_parameter_values(row)
    if samples.shape[0] <= samples.shape[1]:
        fig, axis = pyplot.subplots(figsize=(8, 4))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            (
                "Not enough prefit posterior samples for a corner plot\n"
                f"samples={samples.shape[0]}, dimensions={samples.shape[1]}"
            ),
            ha="center",
            va="center",
            fontsize=12,
        )
        add_sampler_plot_title(fig, row, "prefit corner")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig
    fig = corner.corner(
        samples,
        labels=FIT_PARAMETER_LABELS,
        truths=truth_values,
        truth_color="blue",
        color="black",
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 10},
    )
    axes = numpy.asarray(fig.axes).reshape((len(FIT_PARAMETER_NAMES), -1))
    for row_index in range(len(FIT_PARAMETER_NAMES)):
        for col_index in range(row_index + 1):
            axis = axes[row_index, col_index]
            if row_index == col_index:
                axis.axvline(prefit_values[col_index], color="magenta", linewidth=1.2)
            else:
                axis.plot(
                    prefit_values[col_index],
                    prefit_values[row_index],
                    marker="o",
                    color="magenta",
                    markersize=4,
                )
    add_sampler_plot_title(fig, row, "prefit corner")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_corner(row, state):
    samples = posterior_samples(state)
    truth_values = truth_parameter_values(row)
    fit_values = fitted_parameter_values(row)
    if samples.shape[0] <= samples.shape[1]:
        fig, axis = pyplot.subplots(figsize=(8, 4))
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            (
                "Not enough posterior samples for a corner plot\n"
                f"samples={samples.shape[0]}, dimensions={samples.shape[1]}"
            ),
            ha="center",
            va="center",
            fontsize=12,
        )
        add_sampler_plot_title(fig, row, "corner")
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        return fig
    fig = corner.corner(
        samples,
        labels=FIT_PARAMETER_LABELS,
        truths=truth_values,
        truth_color="blue",
        color="black",
        show_titles=True,
        title_fmt=".3g",
        title_kwargs={"fontsize": 10},
    )
    axes = numpy.asarray(fig.axes).reshape((len(FIT_PARAMETER_NAMES), -1))
    for row_index in range(len(FIT_PARAMETER_NAMES)):
        for col_index in range(row_index + 1):
            axis = axes[row_index, col_index]
            if row_index == col_index:
                axis.axvline(fit_values[col_index], color="red", linewidth=1.2)
            else:
                axis.plot(
                    fit_values[col_index],
                    fit_values[row_index],
                    marker="o",
                    color="red",
                    markersize=4,
                )
    add_sampler_plot_title(fig, row, "corner")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def save_trace_png(ts):
    state = load_sampler_state(ts)
    out_path = plot_path_for_row(ts, "walkers")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig = plot_walker_traces(ts, state)
    fig.canvas.draw()
    fig.savefig(out_path, dpi=150)
    pyplot.close(fig)
    return out_path


def save_corner_png(ts):
    state = load_sampler_state(ts)
    out_path = plot_path_for_row(ts, "corner")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig = plot_corner(ts, state)
    fig.canvas.draw()
    fig.savefig(out_path, dpi=150)
    pyplot.close(fig)
    return out_path


def save_prefit_png(ts):
    curve = load_rv_timeseries_curve(ts)
    out_path = prefit_plot_path_for_row(ts)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    logger.debug("Writing photometry prefit diagnostic plot to %s", out_path)
    fig = plot_prefit(plot_ts_with_curve(ts, curve))
    add_prefit_plot_title(fig, ts)
    fig.canvas.draw()
    fig.savefig(out_path, dpi=150)
    pyplot.close(fig)
    return out_path


def save_prefit_trace_png(ts):
    state = load_sampler_state(ts)
    out_path = prefit_sampler_plot_path_for_row(ts, "walkers")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig = plot_prefit_walker_traces(ts, state)
    fig.canvas.draw()
    fig.savefig(out_path, dpi=150)
    pyplot.close(fig)
    return out_path


def save_prefit_corner_png(ts):
    state = load_sampler_state(ts)
    out_path = prefit_sampler_plot_path_for_row(ts, "corner")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig = plot_prefit_corner(ts, state)
    fig.canvas.draw()
    fig.savefig(out_path, dpi=150)
    pyplot.close(fig)
    return out_path


def save_requested_plots(ts, plot_types):
    output_paths = []
    if "fit" in plot_types:
        output_paths.append(save_plot_png(ts, None))
    if "prefit" in plot_types:
        output_paths.append(save_prefit_png(ts))
    if "prefit_walkers" in plot_types:
        output_paths.append(save_prefit_trace_png(ts))
    if "prefit_corner" in plot_types:
        output_paths.append(save_prefit_corner_png(ts))
    if "walkers" in plot_types:
        output_paths.append(save_trace_png(ts))
    if "corner" in plot_types:
        output_paths.append(save_corner_png(ts))
    return output_paths


def plot_ts_with_curve(ts, curve):
    plot_ts = pandas.Series(ts)
    plot_ts["rv_times"] = curve["rv_times"]
    plot_ts["rv"] = curve["rv"]
    plot_ts["rv_err"] = curve["rv_err"]
    plot_ts["lc_times"] = curve["lc_times"]
    plot_ts["lc"] = curve["lc"]
    plot_ts["lc_err"] = curve["lc_err"]
    if "b_max" not in plot_ts or pandas.isna(plot_ts["b_max"]):
        plot_ts["b_max"] = truth_self_lensing_system(plot_ts).b_max.value
    return plot_ts


def save_plot_png(ts, curve):
    if curve is None:
        curve = load_rv_timeseries_curve(ts)
    out_path = plot_path_for_row(ts, "fit")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    logger.debug("Writing fit diagnostic plot to %s", out_path)
    fit_fig = plot_fit(plot_ts_with_curve(ts, curve))
    add_plot_title(fit_fig, ts)
    fit_fig.canvas.draw()
    fit_fig.savefig(out_path, dpi=150)
    pyplot.close(fit_fig)
    return out_path


def save_plot_pngs(results):
    if not results:
        return []

    output_paths = []
    for ts, curve in tqdm(
        results,
        desc="Plotting diagnostics",
        unit="fit",
        position=0,
        dynamic_ncols=True,
        file=sys.stderr,
    ):
        output_paths.append(save_plot_png(ts, curve))
    return output_paths


def init_plot_worker():
    from utils import configure_logging

    configure_logging()
    load_plotting_stack()


def plot_row_worker(ts):
    curve = load_rv_timeseries_curve(ts)
    return save_plot_png(ts, curve)


def requested_plot_row_worker(args):
    ts, plot_types = args
    return save_requested_plots(ts, plot_types)


def plot_group(rows):
    results = []
    for _, row in rows.iterrows():
        curve = load_rv_timeseries_curve(row)
        results.append((row, curve))
    return save_plot_pngs(results)


def row_groups(rows):
    if "rv_path" not in rows.columns:
        return [rows]
    return [group for _, group in rows.groupby("rv_path", sort=True)]


def rows_for_unique_prefits(rows):
    if "system_id" in rows.columns:
        return rows.drop_duplicates(subset="system_id", keep="last").copy()
    if "rv_path" in rows.columns:
        return rows.drop_duplicates(subset="rv_path", keep="last").copy()
    return rows.head(1).copy()


def plot_rows_serial(rows, plot_types):
    load_plotting_stack()
    output_paths = []
    row_items = [row for _, row in rows.iterrows()]
    logger.info("Plotting %d fitted row(s) serially", len(row_items))
    for row in tqdm(
        row_items,
        total=len(row_items),
        desc="Plotting diagnostics",
        unit="fit",
        position=0,
        dynamic_ncols=True,
        file=sys.stderr,
    ):
        row_paths = save_requested_plots(row, plot_types)
        output_paths.extend(row_paths)
        for output_path in row_paths:
            tqdm.write(str(output_path), file=sys.stderr)
    return output_paths


def plot_rows_parallel(rows, max_workers, plot_types):
    from tqdm import tqdm

    row_items = [row for _, row in rows.iterrows()]
    if not row_items:
        return []
    max_workers = min(max_workers, len(row_items))
    logger.info(
        "Plotting %d fitted row(s) with %d worker process(es)",
        len(row_items),
        max_workers,
    )
    context = multiprocessing.get_context(MP_START_METHOD)
    output_paths = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=init_plot_worker,
    ) as executor:
        worker_args = [(row, plot_types) for row in row_items]
        for row_paths in tqdm(
            executor.map(requested_plot_row_worker, worker_args),
            total=len(row_items),
            desc="Plotting diagnostics",
            unit="fit",
            position=0,
            dynamic_ncols=True,
            file=sys.stderr,
        ):
            output_paths.extend(row_paths)
            for output_path in row_paths:
                tqdm.write(str(output_path), file=sys.stderr)
    return output_paths


def plot_rows(rows, max_workers, plot_types):
    output_paths = []
    regular_plot_types = tuple(
        plot_type for plot_type in plot_types if not plot_type.startswith("prefit")
    )
    prefit_plot_types = tuple(
        plot_type for plot_type in plot_types if plot_type.startswith("prefit")
    )
    if regular_plot_types:
        if max_workers == 1:
            output_paths.extend(plot_rows_serial(rows, regular_plot_types))
        else:
            output_paths.extend(plot_rows_parallel(rows, max_workers, regular_plot_types))
    if not prefit_plot_types:
        return output_paths

    prefit_rows = rows_for_unique_prefits(rows)
    logger.info(
        "Selected %d unique photometry prefit row(s) from %d fitted row(s)",
        len(prefit_rows),
        len(rows),
    )
    if max_workers == 1:
        output_paths.extend(plot_rows_serial(prefit_rows, prefit_plot_types))
    else:
        output_paths.extend(
            plot_rows_parallel(prefit_rows, max_workers, prefit_plot_types)
        )
    return output_paths


def main():
    args = parse_args()

    from utils import configure_logging

    configure_logging()
    validate_selection(args)
    table = load_fitted_results(args.fit_results)
    rows = select_rows(table, args)
    if rows.empty:
        logger.info("No fitted rows selected; exiting")
        return
    plot_types = normalized_plot_types(args.plot_types)
    output_paths = plot_rows(rows, args.max_workers, plot_types)
    logger.info("Generated %d diagnostic PNG(s)", len(output_paths))


if __name__ == "__main__":
    main()
