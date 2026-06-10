"""Generate paper-ready recovery figures and tables from fitted simulations.

The fit campaign may still be running. Outputs from this script are a snapshot
of the fitted subset present in fitted_timeseries.gz at execution time.
Re-run after the fit completes to refresh the paper outputs.
"""

from pathlib import Path
import logging
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from matplotlib.lines import Line2D

from utils import configure_logging


logger = logging.getLogger(__name__)

seaborn.set_theme(context="paper", style="whitegrid", font_scale=1.0)
pyplot.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
)

PROJECT_ROOT = Path.cwd()
FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"

RECOVERY_THRESHOLD = 0.10
SUCCESS_THRESHOLD = 0.80
COMPACT_OBJECT_CLASS_BINS = [0.0, 3.0, 5.0, numpy.inf]
COMPACT_OBJECT_CLASS_LABELS = ["WD/NS", "mass gap", "BH"]

PARAMETERS = {
    "Macc": {
        "fitted": "fitted_Macc",
        "label": r"$M_\mathrm{acc}$",
        "table_label": r"$M_\mathrm{acc}$",
        "unit": r"$M_\odot$",
    },
    "a": {
        "fitted": "fitted_a",
        "label": r"$a$",
        "table_label": r"$a$",
        "unit": "AU",
    },
    "Porb": {
        "fitted": "fitted_porb",
        "label": r"$P_\mathrm{orb}$",
        "table_label": r"$P_\mathrm{orb}$",
        "unit": "d",
    },
    "b": {
        "fitted": "fitted_b",
        "label": r"$b$",
        "table_label": r"$b$",
        "unit": "",
    },
}


def candidate_result_paths():
    env_path = os.environ.get("SLRV_FIT_OUTPUT_PATH")
    paths = []
    if env_path:
        paths.append(Path(env_path))
    paths.extend(
        [
            PROJECT_ROOT / "data" / "fitted_timeseries.gz",
            Path("/scratch/anm1u24/self_lensing_radial_velocity/data/fitted_timeseries.gz"),
        ]
    )
    return paths


def result_path():
    path = next((path for path in candidate_result_paths() if path.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "Could not find fitted_timeseries.gz. Set SLRV_FIT_OUTPUT_PATH to "
            "the fitted table."
        )
    return path


def relative_error(truth, fitted):
    truth = numpy.asarray(truth, dtype=float)
    fitted = numpy.asarray(fitted, dtype=float)
    scale = numpy.abs(truth).copy()
    scale[~numpy.isfinite(scale) | (scale == 0)] = 1.0
    return numpy.abs(fitted - truth) / scale


def interquartile_range(values):
    return numpy.nanpercentile(values, 75) - numpy.nanpercentile(values, 25)


def scalar_number(value):
    if value is None:
        return numpy.nan
    try:
        array = numpy.asarray(value)
        if array.shape == ():
            return float(array)
        if array.size == 1:
            return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return numpy.nan


def numeric_scalar_series(values):
    return pandas.to_numeric(values.map(scalar_number), errors="coerce")


def load_results(path):
    logger.info("Loading fitted results from %s", path)
    results = pandas.read_pickle(path).copy()

    required_columns = {"pristine_id", "baseline", "cadence", "RV accuracy"}
    for param, spec in PARAMETERS.items():
        required_columns.update({param, spec["fitted"]})
    missing = sorted(required_columns.difference(results.columns))
    if missing:
        raise KeyError(f"Fitted results table is missing required columns: {missing}")

    numeric_columns = [
        "baseline",
        "cadence",
        "RV accuracy",
        *PARAMETERS.keys(),
        *(spec["fitted"] for spec in PARAMETERS.values()),
    ]
    for column in numeric_columns:
        results[column] = numeric_scalar_series(results[column])

    results = results.dropna(subset=["baseline", "cadence", "RV accuracy"]).copy()
    results["baseline_years"] = results["baseline"] / 365.0
    results["n_rv"] = numpy.ceil(results["baseline"] / results["cadence"]).astype(int)
    results["strategy"] = (
        results["baseline_years"].map(lambda value: f"{value:g} yr")
        + ", "
        + results["cadence"].map(lambda value: f"{value:g} d")
    )
    logger.info(
        "Loaded %d fitted rows across %d pristine systems",
        len(results),
        results["pristine_id"].nunique(),
    )
    return results


def run_step(label, fn, *args, **kwargs):
    logger.info("Starting %s", label)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    logger.info("Finished %s in %.1f s", label, time.perf_counter() - start)
    return result


def summarise_strategy_recovery(results):
    for param, spec in PARAMETERS.items():
        error_column = f"{param}_relative_error"
        recovered_column = f"{param}_recovered"
        results[error_column] = relative_error(results[param], results[spec["fitted"]])
        results[recovered_column] = results[error_column] <= RECOVERY_THRESHOLD

    strategy_columns = ["baseline", "baseline_years", "cadence", "n_rv", "RV accuracy"]
    summary_aggs = {
        "pristine_id": pandas.NamedAgg(column="pristine_id", aggfunc="nunique")
    }
    for param in PARAMETERS:
        summary_aggs[f"{param}_recovery_fraction"] = pandas.NamedAgg(
            column=f"{param}_recovered", aggfunc="mean"
        )
        summary_aggs[f"{param}_median_relative_error"] = pandas.NamedAgg(
            column=f"{param}_relative_error", aggfunc="median"
        )

    return (
        results.groupby(strategy_columns, dropna=False)
        .agg(**summary_aggs)
        .rename(columns={"pristine_id": "n_systems"})
        .reset_index()
        .sort_values(strategy_columns)
    )


def prefit_comparison_parameters(results):
    parameters = {}
    for param, spec in PARAMETERS.items():
        prefit_column = f"prefit_{spec['fitted']}"
        if prefit_column in results.columns:
            parameters[param] = {**spec, "prefit": prefit_column}
    if not parameters:
        raise KeyError(
            "Fitted results table has no photometry-only prefit columns. "
            "Expected columns such as prefit_fitted_Macc."
        )
    return parameters


def prefit_joint_error_rows(results):
    parameters = prefit_comparison_parameters(results)

    rows = []
    for param, spec in parameters.items():
        needed = [param, spec["fitted"], spec["prefit"], "pristine_id", "n_rv"]
        finite = results.dropna(subset=needed).copy()
        if finite.empty:
            continue

        finite[param] = numeric_scalar_series(finite[param])
        finite[spec["fitted"]] = numeric_scalar_series(finite[spec["fitted"]])
        finite[spec["prefit"]] = numeric_scalar_series(finite[spec["prefit"]])
        finite = finite[
            numpy.isfinite(finite[param])
            & numpy.isfinite(finite[spec["fitted"]])
            & numpy.isfinite(finite[spec["prefit"]])
        ]
        if finite.empty:
            continue

        for fit_stage, value_column in (
            ("Photometry only", spec["prefit"]),
            ("Photometry + RV", spec["fitted"]),
        ):
            errors = relative_error(finite[param], finite[value_column])
            stage_rows = pandas.DataFrame(
                {
                    "parameter": param,
                    "parameter_label": spec["label"],
                    "fit_stage": fit_stage,
                    "n_rv": finite["n_rv"].astype(int).to_numpy(),
                    "pristine_id": finite["pristine_id"].astype(str).to_numpy(),
                    "relative_error": errors,
                    "recovered": errors <= RECOVERY_THRESHOLD,
                }
            )
            rows.append(stage_rows)

    if not rows:
        raise ValueError(
            "No finite paired photometry-only prefit and joint-fit values were "
            "available in the fitted results table."
        )
    return pandas.concat(rows, ignore_index=True)


def summarise_prefit_joint_errors(error_rows):
    return (
        error_rows.groupby(
            ["parameter", "parameter_label", "fit_stage", "n_rv"],
            dropna=False,
        )
        .agg(
            median_relative_error=pandas.NamedAgg(
                column="relative_error", aggfunc="median"
            ),
            q25_relative_error=pandas.NamedAgg(
                column="relative_error", aggfunc=lambda values: numpy.nanpercentile(values, 25)
            ),
            q75_relative_error=pandas.NamedAgg(
                column="relative_error", aggfunc=lambda values: numpy.nanpercentile(values, 75)
            ),
            recovery_fraction=pandas.NamedAgg(column="recovered", aggfunc="mean"),
            n_rows=pandas.NamedAgg(column="relative_error", aggfunc="size"),
            n_systems=pandas.NamedAgg(column="pristine_id", aggfunc="nunique"),
        )
        .reset_index()
        .sort_values(["parameter", "fit_stage", "n_rv"])
    )


def plot_prefit_improvement_by_n_rv(results):
    error_rows = prefit_joint_error_rows(results)
    summary = summarise_prefit_joint_errors(error_rows)
    summary.to_csv(TABLE_DIR / "prefit_improvement_by_n_rv.csv", index=False)

    plot_params = [
        param for param in ("Macc", "a", "Porb", "b") if param in set(summary["parameter"])
    ]
    if not plot_params:
        raise ValueError("No configured parameters are available for prefit comparison")

    ncols = min(2, len(plot_params))
    nrows = int(numpy.ceil(len(plot_params) / ncols))
    fig, axes = pyplot.subplots(
        nrows,
        ncols,
        figsize=(7.2, 2.9 * nrows),
        sharex=True,
        constrained_layout=True,
    )
    axes = numpy.atleast_1d(axes).ravel()

    stage_styles = {
        "Photometry only": {"color": "#CC79A7", "marker": None, "linestyle": "--"},
        "Photometry + RV": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    }
    for axis, param in zip(axes, plot_params):
        param_data = summary[summary["parameter"] == param]
        for stage, style in stage_styles.items():
            stage_data = param_data[param_data["fit_stage"] == stage].sort_values("n_rv")
            if stage_data.empty:
                continue
            x = stage_data["n_rv"].to_numpy(dtype=float)
            median = stage_data["median_relative_error"].to_numpy(dtype=float)
            q25 = stage_data["q25_relative_error"].to_numpy(dtype=float)
            q75 = stage_data["q75_relative_error"].to_numpy(dtype=float)
            axis.plot(
                x,
                median,
                label=stage,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.2,
                markersize=4,
            )
            axis.fill_between(x, q25, q75, color=style["color"], alpha=0.16, linewidth=0)

        label = param_data["parameter_label"].iloc[0]
        axis.set_title(label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.axhline(RECOVERY_THRESHOLD, color="0.35", linestyle=":", linewidth=0.9)
        axis.grid(True, which="both", axis="both", alpha=0.25)
        axis.set_ylabel("Relative error")

    for axis in axes[len(plot_params):]:
        axis.set_visible(False)
    for axis in axes[-ncols:]:
        axis.set_xlabel(r"$N_\mathrm{RV}$")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            markersize=4,
            label=stage,
        )
        for stage, style in stage_styles.items()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle=":",
            linewidth=0.9,
            label=rf"{RECOVERY_THRESHOLD:g} recovery threshold",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"prefit_improvement_by_n_rv.{suffix}",
            bbox_inches="tight",
        )
    pyplot.close(fig)


def percent_matrix(table, value_column, baseline):
    subset = table[table["baseline"] == baseline]
    matrix = subset.pivot(index="cadence", columns="RV accuracy", values=value_column)
    matrix = matrix.sort_index().sort_index(axis=1)
    return 100.0 * matrix


def plot_mass_recovery_by_strategy(strategy_summary):
    baselines = numpy.sort(strategy_summary["baseline"].dropna().unique())
    fig, axes = pyplot.subplots(
        1,
        len(baselines),
        figsize=(7.2, 2.8),
        sharey=True,
        constrained_layout=True,
    )
    if len(baselines) == 1:
        axes = [axes]

    for axis, baseline in zip(axes, baselines):
        matrix = percent_matrix(strategy_summary, "Macc_recovery_fraction", baseline)
        seaborn.heatmap(
            matrix,
            ax=axis,
            vmin=0,
            vmax=100,
            cmap="viridis",
            annot=True,
            fmt=".0f",
            cbar=axis is axes[-1],
            cbar_kws={"label": "Recovered systems [per cent]"},
        )
        axis.set_title(f"Baseline {baseline / 365:g} yr")
        axis.set_xlabel(r"RV accuracy [m s$^{-1}$]")
        axis.set_ylabel("Cadence [d]" if axis is axes[0] else "")

    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"mass_recovery_by_strategy.{suffix}", bbox_inches="tight")
    pyplot.close(fig)


def range_label(left, right, unit):
    if unit:
        return f"{left:.2g}-{right:.2g} {unit}"
    return f"{left:.2g}-{right:.2g}"


def period_value_label(days):
    if days < 365.25:
        return f"{days:.0f} d"
    years = days / 365.25
    if years < 10:
        value = f"{years:.1f}".rstrip("0").rstrip(".")
        return f"{value} yr"
    return f"{years:.0f} yr"


def period_range_label(left, right, unit):
    return f"{period_value_label(left)}-{period_value_label(right)}"


def binned_recovery_table(
    results,
    value_column,
    recovered_column,
    unit,
    n_bins=4,
    bins=None,
    labels=None,
    label_formatter=range_label,
):
    finite = results[numpy.isfinite(results[value_column])].copy()
    if finite.empty:
        raise ValueError(f"No finite values available for {value_column}")

    if bins is None:
        bin_count = min(n_bins, finite[value_column].nunique())
        if bin_count < 1:
            raise ValueError(f"Could not define bins for {value_column}")

        finite["_value_bin"] = pandas.qcut(
            finite[value_column],
            q=bin_count,
            duplicates="drop",
        )
        finite["_value_bin_label"] = finite["_value_bin"].map(
            lambda interval: label_formatter(interval.left, interval.right, unit)
        )
    else:
        finite["_value_bin"] = pandas.cut(
            finite[value_column],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        finite = finite.dropna(subset=["_value_bin"]).copy()
        finite["_value_bin_label"] = finite["_value_bin"].astype(str)

    table = (
        finite.groupby(
            [
                "baseline",
                "baseline_years",
                "cadence",
                "RV accuracy",
                "_value_bin",
                "_value_bin_label",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            recovery_fraction=pandas.NamedAgg(column=recovered_column, aggfunc="mean"),
            n_systems=pandas.NamedAgg(column="pristine_id", aggfunc="nunique"),
        )
        .reset_index()
        .sort_values(["baseline", "cadence", "RV accuracy", "_value_bin"])
    )
    return table


def binned_metric_table(
    results,
    value_column,
    metric_column,
    metric_name,
    unit,
    n_bins=4,
    bins=None,
    labels=None,
    label_formatter=range_label,
    aggfunc="median",
):
    finite = results[
        numpy.isfinite(results[value_column]) & numpy.isfinite(results[metric_column])
    ].copy()
    if finite.empty:
        raise ValueError(f"No finite values available for {value_column} and {metric_column}")

    if bins is None:
        bin_count = min(n_bins, finite[value_column].nunique())
        if bin_count < 1:
            raise ValueError(f"Could not define bins for {value_column}")

        finite["_value_bin"] = pandas.qcut(
            finite[value_column],
            q=bin_count,
            duplicates="drop",
        )
        finite["_value_bin_label"] = finite["_value_bin"].map(
            lambda interval: label_formatter(interval.left, interval.right, unit)
        )
    else:
        finite["_value_bin"] = pandas.cut(
            finite[value_column],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        finite = finite.dropna(subset=["_value_bin"]).copy()
        finite["_value_bin_label"] = finite["_value_bin"].astype(str)

    table = (
        finite.groupby(
            [
                "baseline",
                "baseline_years",
                "cadence",
                "RV accuracy",
                "_value_bin",
                "_value_bin_label",
            ],
            observed=True,
            dropna=False,
        )
        .agg(
            **{
                metric_name: pandas.NamedAgg(column=metric_column, aggfunc=aggfunc),
                "n_systems": pandas.NamedAgg(column="pristine_id", aggfunc="nunique"),
            }
        )
        .reset_index()
        .sort_values(["baseline", "cadence", "RV accuracy", "_value_bin"])
    )
    return table


def plot_recovery_by_truth_range(
    results,
    param,
    value_column,
    unit,
    filename,
    x_label,
    bins=None,
    labels=None,
    label_formatter=range_label,
    recovered_column=None,
    colorbar_label="Recovered systems [per cent]",
):
    if recovered_column is None:
        recovered_column = f"{param}_recovered"
    table = binned_recovery_table(
        results,
        value_column=value_column,
        recovered_column=recovered_column,
        unit=unit,
        bins=bins,
        labels=labels,
        label_formatter=label_formatter,
    )
    table.to_csv(TABLE_DIR / f"{filename}.csv", index=False)

    baselines = numpy.sort(table["baseline"].dropna().unique())
    cadences = numpy.sort(table["cadence"].dropna().unique())
    bin_labels = list(labels) if labels is not None else list(table["_value_bin_label"].drop_duplicates())

    fig, axes = pyplot.subplots(
        len(baselines),
        len(cadences),
        figsize=(8.2, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(baselines) == 1 and len(cadences) == 1:
        axes = numpy.asarray([[axes]])
    elif len(baselines) == 1:
        axes = numpy.asarray([axes])
    elif len(cadences) == 1:
        axes = numpy.asarray([[axis] for axis in axes])

    last_mesh = None
    for row_index, baseline in enumerate(baselines):
        for col_index, cadence in enumerate(cadences):
            axis = axes[row_index, col_index]
            subset = table[
                (table["baseline"] == baseline) & (table["cadence"] == cadence)
            ]
            matrix = (
                subset.pivot(
                    index="RV accuracy",
                    columns="_value_bin_label",
                    values="recovery_fraction",
                )
                .reindex(columns=bin_labels)
                .sort_index()
            )
            matrix = 100.0 * matrix
            heatmap = seaborn.heatmap(
                matrix,
                ax=axis,
                vmin=0,
                vmax=100,
                cmap="viridis",
                annot=True,
                fmt=".0f",
                cbar=False,
            )
            last_mesh = heatmap.collections[0]
            if row_index == 0:
                axis.set_title(f"Cadence {cadence:g} d")
            if col_index == 0:
                axis.set_ylabel(
                    f"Baseline {baseline / 365:g} yr\n"
                    + r"RV accuracy [m s$^{-1}$]"
                )
            else:
                axis.set_ylabel("")
            if row_index == len(baselines) - 1:
                axis.set_xlabel(x_label)
            else:
                axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=35)

    if last_mesh is not None:
        fig.colorbar(
            last_mesh,
            ax=axes,
            location="right",
            shrink=0.78,
            label=colorbar_label,
        )
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{filename}.{suffix}", bbox_inches="tight")
    pyplot.close(fig)


def plot_metric_by_truth_range(
    results,
    value_column,
    metric_column,
    metric_name,
    unit,
    filename,
    x_label,
    colorbar_label,
    bins=None,
    labels=None,
    label_formatter=range_label,
    aggfunc="median",
    percent_scale=False,
    cmap="viridis_r",
):
    table = binned_metric_table(
        results,
        value_column=value_column,
        metric_column=metric_column,
        metric_name=metric_name,
        unit=unit,
        bins=bins,
        labels=labels,
        label_formatter=label_formatter,
        aggfunc=aggfunc,
    )
    table.to_csv(TABLE_DIR / f"{filename}.csv", index=False)

    baselines = numpy.sort(table["baseline"].dropna().unique())
    cadences = numpy.sort(table["cadence"].dropna().unique())
    bin_labels = list(labels) if labels is not None else list(table["_value_bin_label"].drop_duplicates())

    fig, axes = pyplot.subplots(
        len(baselines),
        len(cadences),
        figsize=(8.2, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(baselines) == 1 and len(cadences) == 1:
        axes = numpy.asarray([[axes]])
    elif len(baselines) == 1:
        axes = numpy.asarray([axes])
    elif len(cadences) == 1:
        axes = numpy.asarray([[axis] for axis in axes])

    last_mesh = None
    for row_index, baseline in enumerate(baselines):
        for col_index, cadence in enumerate(cadences):
            axis = axes[row_index, col_index]
            subset = table[
                (table["baseline"] == baseline) & (table["cadence"] == cadence)
            ].copy()
            if percent_scale:
                subset[metric_name] = 100.0 * subset[metric_name]
            matrix = (
                subset.pivot(
                    index="RV accuracy",
                    columns="_value_bin_label",
                    values=metric_name,
                )
                .reindex(columns=bin_labels)
                .sort_index()
            )
            heatmap = seaborn.heatmap(
                matrix,
                ax=axis,
                vmin=0,
                cmap=cmap,
                annot=True,
                fmt=".1f",
                cbar=False,
            )
            last_mesh = heatmap.collections[0]
            if row_index == 0:
                axis.set_title(f"Cadence {cadence:g} d")
            if col_index == 0:
                axis.set_ylabel(
                    f"Baseline {baseline / 365:g} yr\n"
                    + r"RV accuracy [m s$^{-1}$]"
                )
            else:
                axis.set_ylabel("")
            if row_index == len(baselines) - 1:
                axis.set_xlabel(x_label)
            else:
                axis.set_xlabel("")
            axis.tick_params(axis="x", rotation=35)

    if last_mesh is not None:
        fig.colorbar(
            last_mesh,
            ax=axes,
            location="right",
            shrink=0.78,
            label=colorbar_label,
        )
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{filename}.{suffix}", bbox_inches="tight")
    pyplot.close(fig)


def compact_object_class(masses):
    return pandas.cut(
        masses,
        bins=COMPACT_OBJECT_CLASS_BINS,
        labels=COMPACT_OBJECT_CLASS_LABELS,
        include_lowest=True,
        right=False,
    )


def plot_lens_mass_recovery_by_mass(results):
    plot_recovery_by_truth_range(
        results,
        param="Macc",
        value_column="Macc",
        unit=r"$M_\odot$",
        filename="lens_mass_recovery_by_lens_mass",
        x_label=r"True lens mass range",
        bins=[0.0, 3.0, 5.0, 20.0, 50.0],
        labels=[
            r"$<3\,M_\odot$",
            r"$3$--$5\,M_\odot$",
            r"$5$--$20\,M_\odot$",
            r"$20$--$50\,M_\odot$",
        ],
    )


def plot_lens_mass_class_recovery_by_mass(results):
    classified = results.copy()
    true_class = compact_object_class(classified["Macc"])
    fitted_class = compact_object_class(classified["fitted_Macc"])
    classified["Macc_compact_object_class_recovered"] = (
        true_class.notna() & fitted_class.notna() & (true_class == fitted_class)
    )

    plot_recovery_by_truth_range(
        classified,
        param="Macc",
        value_column="Macc",
        unit=r"$M_\odot$",
        filename="lens_mass_class_recovery_by_lens_mass",
        x_label=r"True lens mass range",
        bins=[0.0, 3.0, 5.0, 20.0, 50.0],
        labels=[
            r"$<3\,M_\odot$",
            r"$3$--$5\,M_\odot$",
            r"$5$--$20\,M_\odot$",
            r"$20$--$50\,M_\odot$",
        ],
        recovered_column="Macc_compact_object_class_recovered",
        colorbar_label="Correct compact-object class [per cent]",
    )


def plot_lens_mass_relative_error_by_mass(results):
    plot_metric_by_truth_range(
        results,
        value_column="Macc",
        metric_column="Macc_relative_error",
        metric_name="median_relative_error",
        unit=r"$M_\odot$",
        filename="lens_mass_relative_error_by_lens_mass",
        x_label=r"True lens mass range",
        colorbar_label=r"Median $|\Delta M_\mathrm{acc}|/M_\mathrm{acc}$ [per cent]",
        bins=[0.0, 3.0, 5.0, 20.0, 50.0],
        labels=[
            r"$<3\,M_\odot$",
            r"$3$--$5\,M_\odot$",
            r"$5$--$20\,M_\odot$",
            r"$20$--$50\,M_\odot$",
        ],
        percent_scale=True,
    )


def plot_lens_mass_relative_error_iqr_by_mass(results):
    plot_metric_by_truth_range(
        results,
        value_column="Macc",
        metric_column="Macc_relative_error",
        metric_name="iqr_relative_error",
        unit=r"$M_\odot$",
        filename="lens_mass_relative_error_iqr_by_lens_mass",
        x_label=r"True lens mass range",
        colorbar_label=(
            r"IQR of $|\Delta M_\mathrm{acc}|/M_\mathrm{acc}$ [per cent]"
        ),
        bins=[0.0, 3.0, 5.0, 20.0, 50.0],
        labels=[
            r"$<3\,M_\odot$",
            r"$3$--$5\,M_\odot$",
            r"$5$--$20\,M_\odot$",
            r"$20$--$50\,M_\odot$",
        ],
        aggfunc=interquartile_range,
        percent_scale=True,
    )


def plot_orbital_period_recovery_by_period(results):
    plot_recovery_by_truth_range(
        results,
        param="Porb",
        value_column="Porb",
        unit="d",
        filename="orbital_period_recovery_by_period",
        x_label=r"True orbital period range",
        label_formatter=period_range_label,
    )


def plot_parameter_recovery_curves(strategy_summary):
    plot_rows = []
    for _, row in strategy_summary.iterrows():
        for param, spec in PARAMETERS.items():
            plot_rows.append(
                {
                    "parameter": spec["label"],
                    "baseline_years": row["baseline_years"],
                    "cadence": row["cadence"],
                    "RV accuracy": row["RV accuracy"],
                    "recovery_percent": 100.0 * row[f"{param}_recovery_fraction"],
                }
            )
    plot_data = pandas.DataFrame(plot_rows)

    figure = seaborn.relplot(
        data=plot_data,
        x="RV accuracy",
        y="recovery_percent",
        hue="cadence",
        style="cadence",
        row="parameter",
        col="baseline_years",
        kind="line",
        marker="o",
        facet_kws={"sharey": True, "margin_titles": True},
        height=1.8,
        aspect=1.15,
        palette="colorblind",
    )
    figure.set(
        xscale="log",
        ylim=(-3, 103),
        xlabel=r"RV accuracy [m s$^{-1}$]",
        ylabel="Recovered [per cent]",
    )
    figure.set_titles(row_template="{row_name}", col_template="{col_name:g} yr")
    for axis in figure.axes.flat:
        axis.axhline(
            100 * SUCCESS_THRESHOLD,
            color="0.35",
            linestyle="--",
            linewidth=0.7,
        )
        axis.grid(True, which="both", axis="x", alpha=0.25)
    if figure._legend is not None:
        figure._legend.set_title("Cadence [d]")
    figure.fig.subplots_adjust(top=0.95)
    for suffix in ("pdf", "png"):
        figure.fig.savefig(
            FIGURE_DIR / f"parameter_recovery_curves.{suffix}",
            bbox_inches="tight",
        )
    pyplot.close(figure.fig)


def latex_cell(value):
    if pandas.isna(value):
        return "--"
    if isinstance(value, (float, numpy.floating)):
        return f"{value:.3g}"
    if isinstance(value, (int, numpy.integer)):
        return str(int(value))
    return str(value)


def write_latex_table(
    frame,
    name,
    caption,
    label,
    column_format=None,
    index=False,
    environment="table",
):
    tex_path = TABLE_DIR / f"{name}.tex"
    csv_path = TABLE_DIR / f"{name}.csv"
    frame.to_csv(csv_path, index=index)

    table = frame.reset_index() if index else frame.copy()
    if column_format is None:
        column_format = "l" * len(table.columns)

    lines = [
        rf"\begin{{{environment}}}",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\hline",
        " & ".join(map(str, table.columns)) + r" \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        lines.append(" & ".join(latex_cell(value) for value in row) + r" \\")
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\end{{{environment}}}",
            "",
        ]
    )
    tex_path.write_text("\n".join(lines))
    return tex_path


def write_sample_summary(results):
    sample_summary = pandas.DataFrame(
        [
            ("Fitted observing configurations", len(results)),
            ("Unique pristine systems", results["pristine_id"].nunique()),
            (
                "Campaign baselines [yr]",
                ", ".join(f"{value:g}" for value in sorted(results["baseline_years"].unique())),
            ),
            (
                "RV cadences [d]",
                ", ".join(f"{value:g}" for value in sorted(results["cadence"].unique())),
            ),
            (
                r"RV accuracies [m s$^{-1}$]",
                ", ".join(f"{value:g}" for value in sorted(results["RV accuracy"].unique())),
            ),
            ("Recovery criterion", rf"$|\Delta x|/x \leq {RECOVERY_THRESHOLD:g}$"),
        ],
        columns=["Quantity", "Value"],
    )

    write_latex_table(
        sample_summary,
        "sample_summary",
        "Summary of the currently fitted observing-strategy subset used in this "
        "snapshot of the recovery analysis.",
        "tab:recovery-sample-summary",
        column_format="ll",
    )


def write_mass_recovery_table(strategy_summary):
    mass_table = strategy_summary.copy()
    mass_table["Baseline [yr]"] = mass_table["baseline_years"].map(
        lambda value: f"{value:g}"
    )
    mass_table["Cadence [d]"] = mass_table["cadence"].map(lambda value: f"{value:g}")
    mass_table[r"$N_\mathrm{RV}$"] = mass_table["n_rv"].astype(int)
    mass_table["RV accuracy"] = mass_table["RV accuracy"].map(lambda value: f"{value:g}")
    mass_table["Recovered [per cent]"] = (
        100.0 * mass_table["Macc_recovery_fraction"]
    ).round(1)

    mass_pivot = mass_table.pivot_table(
        index=["Baseline [yr]", "Cadence [d]", r"$N_\mathrm{RV}$"],
        columns="RV accuracy",
        values="Recovered [per cent]",
        aggfunc="first",
    ).reset_index()
    mass_pivot.columns.name = None
    rv_columns = [
        column
        for column in mass_pivot.columns
        if column not in ["Baseline [yr]", "Cadence [d]", r"$N_\mathrm{RV}$"]
    ]
    mass_pivot = mass_pivot[
        ["Baseline [yr]", "Cadence [d]", r"$N_\mathrm{RV}$", *sorted(rv_columns, key=float)]
    ]

    write_latex_table(
        mass_pivot,
        "mass_recovery_by_strategy",
        "Lens mass recovery fraction by observing strategy for the "
        "currently fitted subset. Values are percentages of fitted systems with "
        rf"$|\Delta M_\mathrm{{acc}}|/M_\mathrm{{acc}} \leq {RECOVERY_THRESHOLD:g}$.",
        "tab:mass-recovery-by-strategy",
        column_format="rrr" + "r" * len(rv_columns),
        environment="table*",
    )


def requirement_for_group(group, param):
    value_column = f"{param}_recovery_fraction"
    successful = group[group[value_column] >= SUCCESS_THRESHOLD].sort_values(
        "RV accuracy"
    )
    if successful.empty:
        return numpy.nan
    return successful["RV accuracy"].max()


def write_rv_accuracy_requirement_table(strategy_summary):
    requirement_rows = []
    group_columns = ["baseline", "baseline_years", "cadence", "n_rv"]
    for param, spec in PARAMETERS.items():
        for (_, baseline_years, cadence, n_rv), group in strategy_summary.groupby(
            group_columns
        ):
            requirement_rows.append(
                {
                    "Parameter": spec["table_label"],
                    "Baseline [yr]": f"{baseline_years:g}",
                    "Cadence [d]": f"{cadence:g}",
                    r"$N_\mathrm{RV}$": int(n_rv),
                    r"Max RV accuracy [m s$^{-1}$]": requirement_for_group(group, param),
                }
            )
    requirements = pandas.DataFrame(requirement_rows)

    write_latex_table(
        requirements,
        "rv_accuracy_requirements",
        "Largest instrumental RV uncertainty in the currently fitted tested grid "
        f"that still recovers each parameter for at least {int(100 * SUCCESS_THRESHOLD)} "
        f"per cent of fitted systems under the {RECOVERY_THRESHOLD:g} relative-error "
        "criterion. A dash means no tested RV accuracy met the target.",
        "tab:rv-accuracy-requirements",
        column_format="lrrrc",
        environment="table*",
    )


def main():
    configure_logging()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Writing paper figures to %s", FIGURE_DIR)
    logger.info("Writing paper tables to %s", TABLE_DIR)

    results_path = result_path()
    logger.info("Using fitted results table %s", results_path)
    results = run_step("load fitted results", load_results, results_path)
    strategy_summary = run_step(
        "summarise strategy recovery",
        summarise_strategy_recovery,
        results,
    )
    strategy_summary_path = TABLE_DIR / "recovery_strategy_summary.csv"
    strategy_summary.to_csv(strategy_summary_path, index=False)
    logger.info("Wrote %s", strategy_summary_path.relative_to(PROJECT_ROOT))

    run_step(
        "mass recovery by strategy figure",
        plot_mass_recovery_by_strategy,
        strategy_summary,
    )
    run_step(
        "lens mass recovery by lens mass figure",
        plot_lens_mass_recovery_by_mass,
        results,
    )
    run_step(
        "lens mass class recovery by lens mass figure",
        plot_lens_mass_class_recovery_by_mass,
        results,
    )
    run_step(
        "lens mass relative error by lens mass figure",
        plot_lens_mass_relative_error_by_mass,
        results,
    )
    run_step(
        "lens mass relative error IQR by lens mass figure",
        plot_lens_mass_relative_error_iqr_by_mass,
        results,
    )
    run_step(
        "orbital period recovery by period figure",
        plot_orbital_period_recovery_by_period,
        results,
    )
    run_step(
        "parameter recovery curves figure",
        plot_parameter_recovery_curves,
        strategy_summary,
    )
    run_step(
        "prefit improvement by RV-count figure",
        plot_prefit_improvement_by_n_rv,
        results,
    )
    run_step("sample summary table", write_sample_summary, results)
    run_step("mass recovery table", write_mass_recovery_table, strategy_summary)
    run_step(
        "RV accuracy requirement table",
        write_rv_accuracy_requirement_table,
        strategy_summary,
    )

    logger.info("Paper output generation complete")
    print("\nWrote figures:")
    for path in sorted(FIGURE_DIR.glob("*")):
        print(" -", path.relative_to(PROJECT_ROOT))
    print("\nWrote tables:")
    for path in sorted(TABLE_DIR.glob("*")):
        print(" -", path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
