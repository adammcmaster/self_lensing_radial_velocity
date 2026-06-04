"""
Generate fit diagnostic plots for already-fitted realistic simulations.
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
DEFAULT_FIG_PATH = Path(__file__).parent / Path("../data/figures")
FIG_PATH = Path(os.environ.get("SLRV_FIG_PATH", DEFAULT_FIG_PATH))
PLOT_MAX_WORKERS = int(os.environ.get("SLRV_PLOT_MAX_WORKERS", min(os.cpu_count(), 16) or 1))
MP_START_METHOD = os.environ.get("SLRV_MP_START_METHOD", "spawn")


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
        help="Explicit simulation IDs to plot. Defaults to realistic_id values.",
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
        default="realistic_id",
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
    return parser.parse_args()


def load_plotting_stack():
    global Time
    global pyplot
    global numpy
    global pandas
    global seaborn
    global tqdm
    global units
    global LightcurveGenerator
    global RVGenerator
    global SelfLensingSystem
    global load_realistic_timeseries_curve

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

    from realistic import load_realistic_timeseries_curve

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
    if "realistic_id" not in table.columns:
        return table.copy()
    deduped = table.drop_duplicates(subset="realistic_id", keep="last").copy()
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


def sorted_xy(x, y):
    x = quantity_values(x)
    y = quantity_values(y)
    order = numpy.argsort(x)
    return x[order], y[order]


def plot_fit(row):
    logger.debug(
        "Plotting fit for realistic_idx=%s realistic_id=%s",
        row.get("realistic_idx"),
        row.get("realistic_id"),
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
    model_lc = quantity_values(lcg.flux_array) - 1.0
    truth_model_lc = quantity_values(truth_lcg.flux_array) - 1.0
    rv_model_times, rv_model_sorted = sorted_xy(rv_times, rv_model)
    lc_model_times, model_lc_sorted = sorted_xy(lc_times, model_lc)
    _, truth_model_lc_sorted = sorted_xy(lc_times, truth_model_lc)

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
        figsize=(14, 13),
        gridspec_kw={"height_ratios": (4, 1, 4, 1, 2, 1)},
    )

    seaborn.scatterplot(x=rv_times, y=rv_data, color="blue", ax=rv_axis)
    rv_axis.errorbar(
        rv_times,
        rv_data,
        yerr=row["RV accuracy"],
        fmt="none",
        ecolor="gray",
        alpha=0.7,
    )
    rv_axis.plot(rv_model_times, rv_model_sorted, color="red")
    rv_axis.set_title("Radial Velocity")
    rv_axis.set_xlabel("RV Times")
    rv_axis.set_ylabel("RV")

    seaborn.scatterplot(
        x=lc_times, y=lc_data, color="green", ax=lc_axis, label="Data"
    )
    lc_axis.plot(lc_model_times, model_lc_sorted, color="red", label="Fit")
    lc_axis.plot(lc_model_times, truth_model_lc_sorted, color="blue", label="Truth")
    lc_axis.set_title("Lightcurve")
    lc_axis.set_xlabel("LC Times")
    lc_axis.set_ylabel("LC")
    lc_axis.legend()

    fit_window_start = row["peak_time"] - row["tau_eff"]
    fit_window_end = row["peak_time"] + row["tau_eff"]
    inset_lc_mask = (lc_times >= fit_window_start) & (lc_times <= fit_window_end)
    seaborn.scatterplot(
        x=lc_times[inset_lc_mask],
        y=lc_data[inset_lc_mask],
        color="green",
        ax=lc_window_axis,
        label="Data",
    )
    window_fit_times, window_fit_lc = sorted_xy(
        lc_times[inset_lc_mask], model_lc[inset_lc_mask]
    )
    _, window_truth_lc = sorted_xy(
        lc_times[inset_lc_mask], truth_model_lc[inset_lc_mask]
    )
    lc_window_axis.plot(window_fit_times, window_fit_lc, color="red", label="Fit")
    lc_window_axis.plot(
        window_fit_times, window_truth_lc, color="blue", label="Truth"
    )
    lc_window_axis.axvline(
        fit_window_start, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_axis.axvline(
        fit_window_end, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_axis.set_xlim(fit_window_start, fit_window_end)
    lc_window_axis.set_title("Lightcurve Fit Window")
    lc_window_axis.set_xlabel("LC Times")
    lc_window_axis.set_ylabel("LC")
    lc_window_axis.legend()

    rv_residuals = rv_data - rv_model
    lc_residuals = lc_data - model_lc

    seaborn.scatterplot(
        x=rv_times, y=rv_residuals, color="blue", ax=rv_resid_axis
    )
    rv_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    rv_resid_axis.set_ylabel("RV Residuals")

    seaborn.scatterplot(
        x=lc_times, y=lc_residuals, color="green", ax=lc_full_resid_axis
    )
    lc_full_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    lc_full_resid_axis.set_ylabel("LC Residuals")

    seaborn.scatterplot(
        x=lc_times[inset_lc_mask],
        y=lc_residuals[inset_lc_mask],
        color="green",
        ax=lc_window_resid_axis,
    )
    lc_window_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    lc_window_resid_axis.axvline(
        fit_window_start, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_resid_axis.axvline(
        fit_window_end, color="black", linestyle="--", linewidth=0.8
    )
    lc_window_resid_axis.set_xlim(fit_window_start, fit_window_end)
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
        f"Cadence={scalar_value(row['cadence']):.2f} d"
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
    pyplot.subplots_adjust(right=0.74, hspace=0.08)
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


def figure_shard_keys(ts):
    return (
        f"a1e{int(numpy.floor(numpy.log10(ts['a'])))}",
        f"b{int(ts['b'])}",
        f"Macc{int(ts['Macc'])}",
    )


def plot_path_for_row(ts):
    pristine_id = str(ts.get("pristine_id", "unknown"))
    realistic_id = str(ts.get("realistic_id", ts.get("realistic_idx", "unknown")))
    a_key, b_key, macc_key = figure_shard_keys(ts)
    name = "_".join(
        (
            format_filename_token("baseline", ts["baseline"], "d"),
            format_filename_token("cadence", ts["cadence"], "d"),
            format_filename_token("rv", ts["RV accuracy"], "ms"),
            realistic_id,
        )
    )
    return FIG_PATH / a_key / b_key / macc_key / pristine_id / f"{name}.png"


def plot_label(ts):
    return (
        f"realistic_id={ts.get('realistic_id')} | "
        f"baseline={float(ts['baseline']):g} d | "
        f"cadence={float(ts['cadence']):g} d | "
        f"RV accuracy={float(ts['RV accuracy']):g} m/s"
    )


def add_plot_title(fig, ts):
    fig.suptitle(f"Fit\n{plot_label(ts)}", fontsize=10)
    fig.subplots_adjust(top=0.9)


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
        curve = load_realistic_timeseries_curve(ts)
    out_path = plot_path_for_row(ts)
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
        desc="Plotting fits",
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
    curve = load_realistic_timeseries_curve(ts)
    return save_plot_png(ts, curve)


def plot_group(rows):
    results = []
    for _, row in rows.iterrows():
        curve = load_realistic_timeseries_curve(row)
        results.append((row, curve))
    return save_plot_pngs(results)


def row_groups(rows):
    if "realistic_path" not in rows.columns:
        return [rows]
    return [group for _, group in rows.groupby("realistic_path", sort=True)]


def plot_rows_serial(rows):
    load_plotting_stack()
    output_paths = []
    row_items = [row for _, row in rows.iterrows()]
    logger.info("Plotting %d fitted row(s) serially", len(row_items))
    for row in tqdm(
        row_items,
        total=len(row_items),
        desc="Plotting fits",
        unit="fit",
        position=0,
        dynamic_ncols=True,
        file=sys.stderr,
    ):
        output_path = save_plot_png(row, None)
        output_paths.append(output_path)
        tqdm.write(str(output_path), file=sys.stderr)
    return output_paths


def plot_rows_parallel(rows, max_workers):
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
        for output_path in tqdm(
            executor.map(plot_row_worker, row_items),
            total=len(row_items),
            desc="Plotting fits",
            unit="fit",
            position=0,
            dynamic_ncols=True,
            file=sys.stderr,
        ):
            output_paths.append(output_path)
            tqdm.write(str(output_path), file=sys.stderr)
    return output_paths


def plot_rows(rows, max_workers):
    if max_workers == 1:
        return plot_rows_serial(rows)
    return plot_rows_parallel(rows, max_workers)


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
    output_paths = plot_rows(rows, args.max_workers)
    logger.info("Generated %d diagnostic PNG(s)", len(output_paths))


if __name__ == "__main__":
    main()
