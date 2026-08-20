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
from matplotlib.colors import ListedColormap, LogNorm, Normalize
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"

RECOVERY_THRESHOLD = 0.10
SUCCESS_THRESHOLD = 0.80
LENS_MASS_RANGE_BINS = [0.0, 3.0, 5.0, 20.0, 50.0]
LENS_MASS_RANGE_LABELS = [
    r"$<3\,M_\odot$",
    r"$3$--$5\,M_\odot$",
    r"$5$--$20\,M_\odot$",
    r"$20$--$50\,M_\odot$",
]
ORBITAL_PERIOD_RANGE_LABELS = ["<1 yr", "1-10 yr", "10-100 yr", ">=100 yr"]
ALL_MASSES_LABEL = "All masses"
COMPACT_OBJECT_CLASS_BINS = LENS_MASS_RANGE_BINS
COMPACT_OBJECT_CLASS_LABELS = LENS_MASS_RANGE_LABELS

PARAMETERS = {
    "Macc": {
        "fitted": "fitted_Macc",
        "label": r"$M_\mathrm{lens}$",
        "table_label": r"$M_\mathrm{lens}$",
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

PREFIT_COMPARISON_STAGE_STYLES = {
    "Photometry only": {"color": "#CC79A7", "marker": None, "linestyle": "--"},
    "Photometry + RV": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "RV only": {"color": "#009E73", "marker": "s", "linestyle": "-."},
    "RV only, no peak time": {"color": "#D55E00", "marker": "^", "linestyle": ":"},
}

SURVEY_DETECTED_COUNTS = {
    # Wiktorowicz et al. (2021), Table 2, standard-IMF n_sl values.
    "ZTF": 4027.97,
    "LSST": 10284.91,
    "TESS": 123.28,
}

OBSERVABLE_SPACE_SAMPLE_SIZE = 10_000
OBSERVABLE_SPACE_RANDOM_SEED = 20260813


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


def lens_mass_range(masses):
    return pandas.cut(
        masses,
        bins=LENS_MASS_RANGE_BINS,
        labels=LENS_MASS_RANGE_LABELS,
        include_lowest=True,
        right=False,
    )


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


def rv_accuracy_tick_labels(values):
    return [f"{float(value):g}" for value in values]


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
        rvfit_column = f"rvfit_{spec['fitted']}"
        rv_no_prior_column = f"rvnoprior_{spec['fitted']}"
        if prefit_column in results.columns:
            parameters[param] = {**spec, "prefit": prefit_column}
            if rvfit_column in results.columns:
                parameters[param]["rvfit"] = rvfit_column
            if rv_no_prior_column in results.columns:
                parameters[param]["rvnoprior"] = rv_no_prior_column
    if not parameters:
        raise KeyError(
            "Fitted results table has no photometry-only prefit columns. "
            "Expected columns such as prefit_fitted_Macc."
        )
    return parameters


def prefit_joint_error_rows(results):
    parameters = prefit_comparison_parameters(results)
    warn_on_matching_joint_and_rv_only_columns(results, parameters)

    rows = []
    for param, spec in parameters.items():
        for fit_stage, value_column in (
            ("Photometry only", spec["prefit"]),
            ("Photometry + RV", spec["fitted"]),
            ("RV only", spec.get("rvfit")),
            ("RV only, no peak time", spec.get("rvnoprior")),
        ):
            if value_column is None:
                continue
            needed = list(dict.fromkeys([param, "Macc", value_column, "pristine_id", "n_rv"]))
            finite = results.dropna(subset=needed).copy()
            if finite.empty:
                continue

            finite[param] = numeric_scalar_series(finite[param])
            finite["Macc"] = numeric_scalar_series(finite["Macc"])
            finite[value_column] = numeric_scalar_series(finite[value_column])
            finite = finite[
                numpy.isfinite(finite[param])
                & numpy.isfinite(finite["Macc"])
                & numpy.isfinite(finite[value_column])
            ]
            if finite.empty:
                continue
            mass_range = lens_mass_range(finite["Macc"])
            errors = relative_error(finite[param], finite[value_column])
            observing_columns = {
                column: (
                    numeric_scalar_series(finite[column]).to_numpy()
                    if column in finite
                    else numpy.full(len(finite), numpy.nan)
                )
                for column in ("baseline", "cadence", "RV accuracy")
            }
            stage_rows = pandas.DataFrame(
                {
                    "parameter": param,
                    "parameter_label": spec["label"],
                    "fit_stage": fit_stage,
                    "lens_mass_range": mass_range.astype(str).to_numpy(),
                    **observing_columns,
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


def warn_on_matching_joint_and_rv_only_columns(results, parameters):
    for param, spec in parameters.items():
        for label, rv_column in (
            ("RV-only", spec.get("rvfit")),
            ("RV-only no-peak-time", spec.get("rvnoprior")),
        ):
            if rv_column is None:
                continue
            needed = [spec["fitted"], rv_column]
            finite = results.dropna(subset=needed).copy()
            if finite.empty:
                continue
            joint = numeric_scalar_series(finite[spec["fitted"]])
            rv_only = numeric_scalar_series(finite[rv_column])
            valid = numpy.isfinite(joint) & numpy.isfinite(rv_only)
            if valid.any() and numpy.allclose(joint[valid], rv_only[valid], equal_nan=True):
                logger.warning(
                    "%s column %s matches joint-fit column %s for all %d finite %s "
                    "row(s); the plotted series will overlap the photometry+RV "
                    "series.",
                    label,
                    rv_column,
                    spec["fitted"],
                    int(valid.sum()),
                    param,
                )


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


def summarise_prefit_joint_errors_by_lens_mass_range(error_rows):
    return (
        error_rows.groupby(
            [
                "parameter",
                "parameter_label",
                "lens_mass_range",
                "fit_stage",
                "n_rv",
            ],
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
        .sort_values(["parameter", "lens_mass_range", "fit_stage", "n_rv"])
    )


def summarise_prefit_joint_errors_by_observing_strategy(error_rows):
    """Summarise prefit comparisons within each cadence/baseline panel."""
    strategy_columns = [
        "parameter",
        "parameter_label",
        "baseline",
        "cadence",
        "RV accuracy",
        "fit_stage",
    ]
    return (
        error_rows.groupby(strategy_columns, dropna=False)
        .agg(
            median_relative_error=pandas.NamedAgg(
                column="relative_error", aggfunc="median"
            ),
            q25_relative_error=pandas.NamedAgg(
                column="relative_error",
                aggfunc=lambda values: numpy.nanpercentile(values, 25),
            ),
            q75_relative_error=pandas.NamedAgg(
                column="relative_error",
                aggfunc=lambda values: numpy.nanpercentile(values, 75),
            ),
            recovery_fraction=pandas.NamedAgg(column="recovered", aggfunc="mean"),
            n_rows=pandas.NamedAgg(column="relative_error", aggfunc="size"),
            n_systems=pandas.NamedAgg(column="pristine_id", aggfunc="nunique"),
        )
        .reset_index()
        .sort_values(strategy_columns)
    )


def attach_photometric_observables(results):
    """Attach source peak amplification and crossing time to fitted rows."""
    enriched = results.copy()
    if "source_amp_sl" not in enriched:
        metadata_paths = [
            PROJECT_ROOT / "data" / "pristine_rv" / "metadata.csv",
            Path("/scratch/anm1u24/self_lensing_radial_velocity/data/pristine_rv/metadata.csv"),
        ]
        metadata_path = next((path for path in metadata_paths if path.exists()), None)
        if metadata_path is None:
            raise FileNotFoundError(
                "Could not find pristine photometry metadata needed for peak amplification"
            )
        metadata = pandas.read_csv(
            metadata_path,
            usecols=["simulation_id", "source_amp_sl"],
        ).rename(columns={"simulation_id": "pristine_id"})
        metadata["pristine_id"] = metadata["pristine_id"].astype(str)
        enriched["pristine_id"] = enriched["pristine_id"].astype(str)
        enriched = enriched.merge(metadata, on="pristine_id", how="left")
    enriched["peak_amplification"] = numeric_scalar_series(enriched["source_amp_sl"])
    crossing_column = (
        "prefit_fitted_tau_eff"
        if "prefit_fitted_tau_eff" in enriched
        else "tau_eff"
    )
    enriched["crossing_time_days"] = numeric_scalar_series(enriched[crossing_column])
    return enriched


def prefit_relative_error_improvement_rows(results):
    """Return the change in relative error from photometry-only to joint fits."""
    results = attach_photometric_observables(results)
    parameters = prefit_comparison_parameters(results)
    rows = []
    for param, spec in parameters.items():
        needed = [
            param,
            spec["prefit"],
            spec["fitted"],
            "baseline",
            "cadence",
            "RV accuracy",
            "pristine_id",
        ]
        finite = results.dropna(subset=needed).copy()
        if finite.empty:
            continue
        for column in (
            param,
            spec["prefit"],
            spec["fitted"],
            "baseline",
            "cadence",
            "RV accuracy",
        ):
            finite[column] = numeric_scalar_series(finite[column])
        finite = finite[
            numpy.isfinite(finite[param])
            & numpy.isfinite(finite[spec["prefit"]])
            & numpy.isfinite(finite[spec["fitted"]])
            & numpy.isfinite(finite["baseline"])
            & numpy.isfinite(finite["cadence"])
            & numpy.isfinite(finite["RV accuracy"])
        ]
        if finite.empty:
            continue
        prefit_error = relative_error(finite[param], finite[spec["prefit"]])
        joint_error = relative_error(finite[param], finite[spec["fitted"]])
        rows.append(
            pandas.DataFrame(
                {
                    "parameter": param,
                    "parameter_label": spec["label"],
                    "photometry_only_fitted_value": finite[spec["prefit"]].to_numpy(),
                    "baseline": finite["baseline"].to_numpy(),
                    "cadence": finite["cadence"].to_numpy(),
                    "RV accuracy": finite["RV accuracy"].to_numpy(),
                    "peak_amplification": finite["peak_amplification"].to_numpy(),
                    "crossing_time_days": finite["crossing_time_days"].to_numpy(),
                    "pristine_id": finite["pristine_id"].astype(str).to_numpy(),
                    "prefit_relative_error": prefit_error,
                    "joint_relative_error": joint_error,
                    "relative_error_improvement": prefit_error - joint_error,
                }
            )
        )
    if not rows:
        raise ValueError("No finite prefit and joint-fit values are available")
    return pandas.concat(rows, ignore_index=True)


def plot_prefit_relative_error_improvement_scatter_grids(
    results,
    color_column="RV accuracy",
    color_label=r"RV accuracy [m s$^{-1}$]",
    filename_suffix="",
):
    """Plot prefit-to-joint improvement against photometry-only fit values."""
    plot_data = prefit_relative_error_improvement_rows(results)
    plot_params = [
        param for param in ("Macc", "Porb", "a") if param in set(plot_data["parameter"])
    ]
    if not plot_params:
        raise ValueError("No configured parameters are available for improvement plots")

    for param in plot_params:
        param_data = plot_data[plot_data["parameter"] == param].copy()
        filename = (
            f"prefit_relative_error_improvement_by_cadence_baseline_{param}"
            f"{filename_suffix}"
        )
        colored_data = param_data[
            numpy.isfinite(param_data[color_column]) & (param_data[color_column] > 0)
        ].copy()
        if colored_data.empty:
            logger.warning(
                "Skipping %s because %s has no positive values",
                filename,
                color_column,
            )
            continue
        colored_data.to_csv(TABLE_DIR / f"{filename}.csv", index=False)
        baselines = numpy.sort(param_data["baseline"].unique())
        cadences = numpy.sort(param_data["cadence"].unique())
        color_values = colored_data[color_column].to_numpy(dtype=float)
        norm = (
            LogNorm(vmin=color_values.min(), vmax=color_values.max())
            if color_values.min() < color_values.max()
            else Normalize(vmin=color_values.min(), vmax=color_values.max() + 1.0)
        )

        fig, axes = pyplot.subplots(
            len(baselines),
            len(cadences),
            figsize=(8.2, 6.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = numpy.asarray(axes, dtype=object).reshape(len(baselines), len(cadences))
        scatter = None
        for row_index, baseline in enumerate(baselines):
            for col_index, cadence in enumerate(cadences):
                axis = axes[row_index, col_index]
                panel_data = colored_data[
                    (colored_data["baseline"] == baseline)
                    & (colored_data["cadence"] == cadence)
                    & (colored_data["photometry_only_fitted_value"] > 0)
                ]
                scatter = axis.scatter(
                    panel_data["photometry_only_fitted_value"],
                    panel_data["relative_error_improvement"],
                    c=panel_data[color_column],
                    cmap="viridis",
                    norm=norm,
                    s=9,
                    alpha=0.55,
                    linewidths=0,
                    rasterized=True,
                )
                axis.axhline(0, color="0.25", linewidth=0.7, linestyle="--")
                if len(panel_data) >= 10:
                    seaborn.kdeplot(
                        data=panel_data,
                        x="photometry_only_fitted_value",
                        y="relative_error_improvement",
                        log_scale=(True, False),
                        levels=5,
                        thresh=0.06,
                        color="black",
                        linewidths=0.6,
                        alpha=0.6,
                        warn_singular=False,
                        ax=axis,
                        zorder=3,
                    )
                if row_index == 0:
                    axis.set_title(f"Cadence {cadence:g} d")
                if col_index == 0:
                    axis.set_ylabel(
                        f"Baseline {baseline / 365:g} yr\nRelative-error improvement"
                    )
                else:
                    axis.set_ylabel("")
                if row_index == len(baselines) - 1:
                    label = param_data["parameter_label"].iloc[0]
                    unit = PARAMETERS[param]["unit"]
                    axis.set_xlabel(
                        f"Photometry-only fitted {label}"
                        + (f" [{unit}]" if unit else "")
                    )
                else:
                    axis.set_xlabel("")
                axis.set_xscale("log")
                axis.set_yscale("symlog", linthresh=0.01)
                axis.grid(True, which="both", alpha=0.22)

        if scatter is not None:
            fig.colorbar(
                scatter,
                ax=axes,
                location="right",
                shrink=0.78,
                label=color_label,
            )
        for suffix in ("pdf", "png"):
            fig.savefig(FIGURE_DIR / f"{filename}.{suffix}", bbox_inches="tight")
        pyplot.close(fig)


def plot_prefit_relative_error_improvement_scatter_grids_by_photometric_observable(
    results,
):
    """Write scatter-grid variants coloured by pre-RV photometric observables."""
    for color_column, color_label, filename_suffix in (
        (
            "peak_amplification",
            "Peak amplification",
            "_peak_amplification",
        ),
        (
            "crossing_time_days",
            "Crossing time [d]",
            "_crossing_time",
        ),
    ):
        plot_prefit_relative_error_improvement_scatter_grids(
            results,
            color_column=color_column,
            color_label=color_label,
            filename_suffix=filename_suffix,
        )


def plot_photometry_followup_recommendation_maps(results):
    """Map the least demanding RV strategy reaching 90% of attainable gain."""
    results = attach_photometric_observables(results)
    parameters = prefit_comparison_parameters(results)
    feature_columns = ["peak_amplification", "crossing_time_days"]
    plot_params = [param for param in ("Macc", "Porb", "a") if param in parameters]

    for param in plot_params:
        spec = parameters[param]
        needed = [
            param,
            spec["prefit"],
            spec["fitted"],
            *feature_columns,
            "baseline",
            "cadence",
            "RV accuracy",
            "n_rv",
        ]
        data = results.dropna(subset=needed).copy()
        for column in needed:
            if column in data and column != param:
                data[column] = numeric_scalar_series(data[column])
        data = data[
            numpy.isfinite(data[param])
            & numpy.isfinite(data[spec["prefit"]])
            & numpy.isfinite(data[spec["fitted"]])
            & (data["peak_amplification"] > 0)
            & (data["crossing_time_days"] > 0)
        ].copy()
        if data.empty:
            continue
        data["prefit_relative_error"] = relative_error(data[param], data[spec["prefit"]])
        data["joint_relative_error"] = relative_error(data[param], data[spec["fitted"]])
        data["peak_bin"] = pandas.qcut(data["peak_amplification"], q=4, duplicates="drop")
        data["crossing_bin"] = pandas.qcut(data["crossing_time_days"], q=4, duplicates="drop")
        data = data.dropna(subset=["peak_bin", "crossing_bin"])
        data["peak_bin"] = data["peak_bin"].astype(str)
        data["crossing_bin"] = data["crossing_bin"].astype(str)
        group_columns = ["peak_bin", "crossing_bin", "baseline", "cadence", "RV accuracy", "n_rv"]
        strategy = data.groupby(group_columns, observed=True).agg(
            prefit_error=("prefit_relative_error", "median"),
            joint_error=("joint_relative_error", "median"),
            n_systems=("pristine_id", "nunique"),
        ).reset_index()
        bin_columns = ["peak_bin", "crossing_bin"]
        strategy["best_joint_error"] = strategy.groupby(bin_columns)["joint_error"].transform("min")
        strategy["photometry_error"] = strategy.groupby(bin_columns)["prefit_error"].transform("median")
        gain = strategy["photometry_error"] - strategy["best_joint_error"]
        strategy["attainable_improvement_fraction"] = numpy.where(
            gain > 0,
            (strategy["photometry_error"] - strategy["joint_error"]) / gain,
            numpy.nan,
        )
        sufficient = strategy[strategy["attainable_improvement_fraction"] >= 0.9].copy()
        sufficient = sufficient.sort_values(
            ["peak_bin", "crossing_bin", "n_rv", "baseline", "RV accuracy"],
            ascending=[True, True, True, True, False],
        )
        recommendation = sufficient.groupby(bin_columns, observed=True).first().reset_index()
        if recommendation.empty:
            logger.warning("No 90%%-plateau recommendation is available for %s", param)
            continue
        filename = f"photometry_followup_recommendation_{param}"
        recommendation.to_csv(TABLE_DIR / f"{filename}.csv", index=False)
        peak_labels = list(data["peak_bin"].drop_duplicates())
        crossing_labels = list(data["crossing_bin"].drop_duplicates())
        panels = [
            ("cadence", "Cadence [d]"),
            ("baseline", "Baseline [yr]"),
            ("RV accuracy", r"RV accuracy [m s$^{-1}$]"),
        ]
        fig, axes = pyplot.subplots(1, 3, figsize=(10.2, 3.7), sharey=True, constrained_layout=True)
        for axis, (column, title) in zip(axes, panels):
            values = recommendation.pivot(index="crossing_bin", columns="peak_bin", values=column)
            values = values.reindex(index=crossing_labels, columns=peak_labels)
            display = values / 365.0 if column == "baseline" else values
            annotation = display.map(lambda value: "" if pandas.isna(value) else f"{value:g}")
            seaborn.heatmap(
                display,
                ax=axis,
                cmap=ListedColormap(seaborn.color_palette("viridis", as_cmap=False)),
                annot=annotation,
                fmt="",
                cbar=False,
                linewidths=0.5,
                linecolor="white",
            )
            axis.set_title(title)
            axis.set_xlabel("Photometric peak-amplification quartile")
            axis.tick_params(axis="x", rotation=35)
        axes[0].set_ylabel("Photometry-only crossing-time quartile")
        for axis in axes[1:]:
            axis.set_ylabel("")
        fig.suptitle(
            f"Least demanding strategy reaching 90% of attainable {PARAMETERS[param]['label']} improvement",
            fontsize=10,
        )
        for suffix in ("pdf", "png"):
            fig.savefig(FIGURE_DIR / f"{filename}.{suffix}", bbox_inches="tight")
        pyplot.close(fig)


def plot_prefit_improvement_by_n_rv(results):
    error_rows = prefit_joint_error_rows(results)
    summary = summarise_prefit_joint_errors(error_rows)
    # The CSV summary is not currently included in main.tex.
    # summary.to_csv(TABLE_DIR / "prefit_improvement_by_n_rv.csv", index=False)

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

    for axis, param in zip(axes, plot_params):
        param_data = summary[summary["parameter"] == param]
        for stage, style in PREFIT_COMPARISON_STAGE_STYLES.items():
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
        for stage, style in PREFIT_COMPARISON_STAGE_STYLES.items()
    ]
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


def plot_prefit_improvement_by_lens_mass_range(results):
    error_rows = prefit_joint_error_rows(results)
    summary = summarise_prefit_joint_errors_by_lens_mass_range(error_rows)
    # The CSV summary is not currently included in main.tex.
    # summary.to_csv(
    #     TABLE_DIR / "prefit_improvement_by_lens_mass_range.csv",
    #     index=False,
    # )

    plot_params = [
        param for param in ("Macc", "a", "Porb", "b") if param in set(summary["parameter"])
    ]
    if not plot_params:
        raise ValueError("No configured parameters are available for prefit comparison")

    mass_ranges = [
        label
        for label in LENS_MASS_RANGE_LABELS
        if label in set(summary["lens_mass_range"])
    ]
    if not mass_ranges:
        raise ValueError("No configured lens-mass ranges are available for prefit comparison")

    fig, axes = pyplot.subplots(
        len(plot_params),
        len(mass_ranges),
        figsize=(2.4 * len(mass_ranges), 1.9 * len(plot_params)),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    if len(plot_params) == 1 and len(mass_ranges) == 1:
        axes = numpy.asarray([[axes]])
    elif len(plot_params) == 1:
        axes = numpy.asarray([axes])
    elif len(mass_ranges) == 1:
        axes = numpy.asarray([[axis] for axis in axes])

    for row_index, param in enumerate(plot_params):
        for col_index, mass_range_label in enumerate(mass_ranges):
            axis = axes[row_index, col_index]
            panel_data = summary[
                (summary["parameter"] == param)
                & (summary["lens_mass_range"] == mass_range_label)
            ]
            for stage, style in PREFIT_COMPARISON_STAGE_STYLES.items():
                stage_data = panel_data[
                    panel_data["fit_stage"] == stage
                ].sort_values("n_rv")
                if stage_data.empty:
                    continue
                x = stage_data["n_rv"].to_numpy(dtype=float)
                median = stage_data["median_relative_error"].to_numpy(dtype=float)
                q25 = stage_data["q25_relative_error"].to_numpy(dtype=float)
                q75 = stage_data["q75_relative_error"].to_numpy(dtype=float)
                axis.plot(
                    x,
                    median,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.1,
                    markersize=3.5,
                )
                axis.fill_between(
                    x,
                    q25,
                    q75,
                    color=style["color"],
                    alpha=0.14,
                    linewidth=0,
                )

            if row_index == 0:
                axis.set_title(mass_range_label)
            if col_index == 0:
                label = panel_data["parameter_label"].iloc[0] if not panel_data.empty else param
                axis.set_ylabel(f"{label}\nRelative error")
            else:
                axis.set_ylabel("")
            if row_index == len(plot_params) - 1:
                axis.set_xlabel(r"$N_\mathrm{RV}$")
            else:
                axis.set_xlabel("")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", axis="both", alpha=0.22)

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
        for stage, style in PREFIT_COMPARISON_STAGE_STYLES.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"prefit_improvement_by_lens_mass_range.{suffix}",
            bbox_inches="tight",
        )
    pyplot.close(fig)


def plot_prefit_improvement_by_cadence_and_baseline(results):
    """Write one prefit-comparison grid per physical parameter.

    Each panel fixes the observing cadence and baseline, so RV accuracy is the
    varying quantity on the horizontal axis.  This avoids using ``n_rv`` as an
    axis within a panel, where it is necessarily constant.
    """
    error_rows = prefit_joint_error_rows(results)
    summary = summarise_prefit_joint_errors_by_observing_strategy(error_rows)
    available_parameters = set(summary["parameter"])
    plot_params = [
        param
        for param in ("Macc", "a", "Porb", "b")
        if param in available_parameters
    ]
    if not plot_params:
        raise ValueError("No configured parameters are available for prefit comparison")

    for param in plot_params:
        param_data = summary[summary["parameter"] == param].copy()
        # This figure and its CSV sidecar are not currently included in main.tex.
        # param_data.to_csv(
        #     TABLE_DIR / f"prefit_improvement_by_cadence_baseline_{param}.csv",
        #     index=False,
        # )
        baselines = numpy.sort(param_data["baseline"].dropna().unique())
        cadences = numpy.sort(param_data["cadence"].dropna().unique())
        if len(baselines) == 0 or len(cadences) == 0:
            continue

        fig, axes = pyplot.subplots(
            len(baselines),
            len(cadences),
            figsize=(2.45 * len(cadences), 1.85 * len(baselines)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = numpy.asarray(axes, dtype=object).reshape(len(baselines), len(cadences))

        for row_index, baseline in enumerate(baselines):
            for col_index, cadence in enumerate(cadences):
                axis = axes[row_index, col_index]
                panel_data = param_data[
                    (param_data["baseline"] == baseline)
                    & (param_data["cadence"] == cadence)
                ]
                for stage, style in PREFIT_COMPARISON_STAGE_STYLES.items():
                    stage_data = panel_data[
                        panel_data["fit_stage"] == stage
                    ].sort_values("RV accuracy")
                    if stage_data.empty:
                        continue
                    x = stage_data["RV accuracy"].to_numpy(dtype=float)
                    median = stage_data["median_relative_error"].to_numpy(dtype=float)
                    q25 = stage_data["q25_relative_error"].to_numpy(dtype=float)
                    q75 = stage_data["q75_relative_error"].to_numpy(dtype=float)
                    axis.plot(
                        x,
                        median,
                        color=style["color"],
                        marker=style["marker"],
                        linestyle=style["linestyle"],
                        linewidth=1.1,
                        markersize=3.2,
                    )
                    axis.fill_between(
                        x,
                        q25,
                        q75,
                        color=style["color"],
                        alpha=0.14,
                        linewidth=0,
                    )

                if row_index == 0:
                    axis.set_title(f"Cadence {cadence:g} d")
                if col_index == 0:
                    axis.set_ylabel(
                        f"Baseline {baseline / 365:g} yr\nRelative error"
                    )
                else:
                    axis.set_ylabel("")
                if row_index == len(baselines) - 1:
                    axis.set_xlabel(r"RV accuracy [m s$^{-1}$]")
                else:
                    axis.set_xlabel("")
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.grid(True, which="both", axis="both", alpha=0.22)

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
            for stage, style in PREFIT_COMPARISON_STAGE_STYLES.items()
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.08),
            frameon=False,
        )
        filename = f"prefit_improvement_by_cadence_baseline_{param}"
        for suffix in ("pdf", "png"):
            fig.savefig(FIGURE_DIR / f"{filename}.{suffix}", bbox_inches="tight")
        pyplot.close(fig)


def prefit_compact_object_class_rows(results):
    parameters = prefit_comparison_parameters(results)
    warn_on_matching_joint_and_rv_only_columns(results, parameters)
    mass_spec = parameters.get("Macc")
    if mass_spec is None:
        raise KeyError(
            "Fitted results table has no photometry-only prefit mass column. "
            "Expected prefit_fitted_Macc."
        )

    needed = ["Macc", "pristine_id", "n_rv"]
    finite = results.dropna(subset=needed).copy()
    if finite.empty:
        raise ValueError(
            "No finite paired truth and fitted mass values "
            "were available in the fitted results table."
        )

    finite["Macc"] = numeric_scalar_series(finite["Macc"])
    finite = finite[numpy.isfinite(finite["Macc"])].copy()
    if finite.empty:
        raise ValueError(
            "No finite truth masses were available in the fitted results table."
        )

    fit_stages = (
        ("Photometry only", mass_spec["prefit"]),
        ("Photometry + RV", mass_spec["fitted"]),
        ("RV only", mass_spec.get("rvfit")),
        ("RV only, no peak time", mass_spec.get("rvnoprior")),
    )
    rows = []
    for fit_stage, value_column in fit_stages:
        if value_column is None:
            continue
        stage_data = finite.dropna(subset=[value_column]).copy()
        if stage_data.empty:
            continue
        stage_data[value_column] = numeric_scalar_series(stage_data[value_column])
        stage_data = stage_data[numpy.isfinite(stage_data[value_column])].copy()
        if stage_data.empty:
            continue

        true_class = compact_object_class(stage_data["Macc"])
        fitted_class = compact_object_class(stage_data[value_column])
        mass_range = lens_mass_range(stage_data["Macc"])
        rows.append(
            pandas.DataFrame(
                {
                    "fit_stage": fit_stage,
                    "lens_mass_range": mass_range.astype(str).to_numpy(),
                    "n_rv": stage_data["n_rv"].astype(int).to_numpy(),
                    "pristine_id": stage_data["pristine_id"].astype(str).to_numpy(),
                    "correct_class": (
                        true_class.notna()
                        & fitted_class.notna()
                        & (true_class == fitted_class)
                    ).to_numpy(),
                }
            )
        )

    if not rows:
        raise ValueError(
            "No finite photometry-only, photometry+RV, or RV-only fitted mass values "
            "were available in the fitted results table."
        )
    return pandas.concat(rows, ignore_index=True)


def summarise_prefit_compact_object_class_rows(class_rows):
    return (
        class_rows.groupby(
            ["lens_mass_range", "fit_stage", "n_rv"],
            dropna=False,
        )
        .agg(
            correct_class_fraction=pandas.NamedAgg(
                column="correct_class", aggfunc="mean"
            ),
            n_rows=pandas.NamedAgg(column="correct_class", aggfunc="size"),
            n_systems=pandas.NamedAgg(column="pristine_id", aggfunc="nunique"),
        )
        .reset_index()
        .sort_values(["lens_mass_range", "fit_stage", "n_rv"])
    )


def plot_self_lensing_observable_space():
    import jax
    import jax.numpy as jnp
    from selflensing.batch import make_batched_light_curve_generator
    from selflensing.lightcurves import PARAMETER_ORDER
    from selflensing.orbits import (
        AU_M,
        DAY_S,
        M_SUN_KG,
        R_SUN_M,
        einstein_radius_m,
        orbital_period_days,
    )

    jax.config.update("jax_enable_x64", True)

    n_samples = OBSERVABLE_SPACE_SAMPLE_SIZE
    key = jax.random.key(OBSERVABLE_SPACE_RANDOM_SEED)
    keys = jax.random.split(key, 5)

    def uniform(draw_key, low, high):
        return jax.random.uniform(draw_key, (n_samples,), minval=low, maxval=high)

    def log_uniform(draw_key, low, high):
        return 10 ** uniform(draw_key, numpy.log10(low), numpy.log10(high))

    companion_mass_msun = 1.0
    companion_radius_rsun = 1.0
    companion_teff_k = 5800.0

    semi_major_axis_au = log_uniform(keys[0], 0.3, 100.0)
    lens_mass_msun = log_uniform(keys[1], 3.0, 50.0)
    companion_mass = jnp.full(n_samples, companion_mass_msun)
    companion_radius = jnp.full(n_samples, companion_radius_rsun)
    companion_teff = jnp.full(n_samples, companion_teff_k)
    eccentricity = uniform(keys[2], 0.0, 0.6)
    periapsis_phase = uniform(keys[3], 0.0, 2.0 * numpy.pi)
    impact_fraction = uniform(keys[4], 0.0, 0.95)

    semi_major_axis_m = semi_major_axis_au * AU_M
    lens_mass_kg = lens_mass_msun * M_SUN_KG
    companion_mass_kg = companion_mass * M_SUN_KG
    companion_radius_m = companion_radius * R_SUN_M

    edge_on_einstein_radius_m = einstein_radius_m(
        semi_major_axis_m,
        jnp.ones(n_samples),
        lens_mass_kg,
        eccentricity,
        periapsis_phase,
    )
    maximum_impact = 1.0 + companion_radius_m / edge_on_einstein_radius_m
    impact_einstein_radii = impact_fraction * maximum_impact
    cos_inclination = jnp.clip(
        impact_einstein_radii * edge_on_einstein_radius_m / semi_major_axis_m,
        0.0,
        0.2,
    )
    inclination = jnp.arccos(cos_inclination)

    param_matrix = jnp.stack(
        [
            {
                "a": semi_major_axis_au,
                "i": inclination,
                "Macc": lens_mass_msun,
                "Mcomp": companion_mass,
                "Rcomp": companion_radius,
                "Teff": companion_teff,
                "e": eccentricity,
                "periapsis_phase": periapsis_phase,
            }[name]
            for name in PARAMETER_ORDER
        ],
        axis=-1,
    )

    sin_inclination = jnp.sin(inclination)
    orbital_period_days_value = orbital_period_days(
        semi_major_axis_m,
        lens_mass_kg + companion_mass_kg,
    )
    orbital_period_seconds = orbital_period_days_value * DAY_S
    einstein_radius_m_value = einstein_radius_m(
        semi_major_axis_m,
        sin_inclination,
        lens_mass_kg,
        eccentricity,
        periapsis_phase,
    )
    impact_parameter = jnp.abs(
        semi_major_axis_m * jnp.cos(inclination) / einstein_radius_m_value
    )
    maximum_impact = 1.0 + companion_radius_m / einstein_radius_m_value
    crossing_time_days = (
        (orbital_period_seconds * (einstein_radius_m_value + companion_radius_m))
        / (jnp.pi * semi_major_axis_m * sin_inclination)
        * jnp.sqrt(
            jnp.clip(1.0 - (impact_parameter / maximum_impact) ** 2, 0.0, 1.0)
        )
    ) / DAY_S

    generate = make_batched_light_curve_generator(
        jnp.array([0.0]),
        peak_times=0.0,
        sl=True,
        ev=False,
        db=False,
        device=jax.devices()[0],
    )
    peak_flux = numpy.asarray(generate(param_matrix).block_until_ready())[:, 0]
    peak_amplification_ppm = 1e6 * (peak_flux - 1.0)
    crossing_time_hr = numpy.asarray(crossing_time_days) * 24.0
    orbital_period_yr = numpy.asarray(orbital_period_days_value) / 365.25
    lens_mass = numpy.asarray(lens_mass_msun)
    inclination = numpy.asarray(inclination)

    valid = (
        numpy.isfinite(peak_amplification_ppm)
        & numpy.isfinite(crossing_time_hr)
        & numpy.isfinite(orbital_period_yr)
        & (peak_amplification_ppm > 1.0)
        & (crossing_time_hr > 0.0)
        & (orbital_period_yr > 0.0)
    )
    if not valid.any():
        raise ValueError("No valid self-lensing observable-space samples were generated")

    summary = pandas.DataFrame(
        [
            {
                "sample_size": n_samples,
                "valid_rows": int(valid.sum()),
                "random_seed": OBSERVABLE_SPACE_RANDOM_SEED,
                "fixed_Mcomp_Msun": companion_mass_msun,
                "fixed_Rcomp_Rsun": companion_radius_rsun,
                "fixed_Teff_K": companion_teff_k,
                "median_peak_amplification_ppm": numpy.nanmedian(
                    peak_amplification_ppm[valid]
                ),
                "median_crossing_time_hr": numpy.nanmedian(crossing_time_hr[valid]),
            }
        ]
    )
    # The CSV summary is not currently included in main.tex.
    # summary.to_csv(TABLE_DIR / "self_lensing_observable_space_summary.csv", index=False)

    color_by = [
        (r"$M_\mathrm{lens}$ [$M_\odot$]", lens_mass[valid], None),
        (
            r"$P_\mathrm{orb}$ [yr]",
            orbital_period_yr[valid],
            LogNorm(
                vmin=numpy.nanmin(orbital_period_yr[valid]),
                vmax=numpy.nanmax(orbital_period_yr[valid]),
            ),
        ),
        (r"$i$ [rad]", inclination[valid], None),
    ]

    fig, axes = pyplot.subplots(
        1,
        3,
        figsize=(7.2, 2.45),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for axis, (label, values, norm) in zip(axes, color_by):
        scatter = axis.scatter(
            crossing_time_hr[valid],
            peak_amplification_ppm[valid],
            c=values,
            s=4,
            cmap="viridis",
            norm=norm,
            alpha=0.72,
            linewidths=0,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Crossing time [hr]")
        axis.grid(True, which="both", axis="both", alpha=0.22)
        fig.colorbar(scatter, ax=axis, label=label)
    axes[0].set_ylabel("Peak amplification [ppm]")

    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"self_lensing_observable_space.{suffix}",
            bbox_inches="tight",
        )
    pyplot.close(fig)


def plot_prefit_compact_object_class_by_lens_mass_range(results):
    class_rows = prefit_compact_object_class_rows(results)
    summary = summarise_prefit_compact_object_class_rows(class_rows)
    # The CSV summary is not currently included in main.tex.
    # summary.to_csv(
    #     TABLE_DIR / "prefit_compact_object_class_by_lens_mass_range.csv",
    #     index=False,
    # )

    mass_ranges = [
        label
        for label in LENS_MASS_RANGE_LABELS
        if label in set(summary["lens_mass_range"])
    ]
    if not mass_ranges:
        raise ValueError("No configured lens-mass ranges are available for class comparison")

    fig, axes = pyplot.subplots(
        1,
        len(mass_ranges),
        figsize=(2.4 * len(mass_ranges), 2.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(mass_ranges) == 1:
        axes = [axes]

    for axis, mass_range_label in zip(axes, mass_ranges):
        panel_data = summary[summary["lens_mass_range"] == mass_range_label]
        for stage, style in PREFIT_COMPARISON_STAGE_STYLES.items():
            stage_data = panel_data[
                panel_data["fit_stage"] == stage
            ].sort_values("n_rv")
            if stage_data.empty:
                continue
            axis.plot(
                stage_data["n_rv"].to_numpy(dtype=float),
                100.0 * stage_data["correct_class_fraction"].to_numpy(dtype=float),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.2,
                markersize=4,
            )

        axis.set_title(mass_range_label)
        axis.set_xscale("log")
        axis.set_ylim(-3, 103)
        axis.axhline(
            100.0 * SUCCESS_THRESHOLD,
            color="0.35",
            linestyle=":",
            linewidth=0.8,
        )
        axis.grid(True, which="both", axis="both", alpha=0.22)
        axis.set_xlabel(r"$N_\mathrm{RV}$")
    axes[0].set_ylabel("Correct compact-object class [per cent]")

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
        for stage, style in PREFIT_COMPARISON_STAGE_STYLES.items()
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle=":",
            linewidth=0.9,
            label=f"{int(100 * SUCCESS_THRESHOLD)} per cent",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"prefit_compact_object_class_by_lens_mass_range.{suffix}",
            bbox_inches="tight",
        )
    pyplot.close(fig)


def percent_matrix(table, value_column, baseline):
    subset = table[table["baseline"] == baseline]
    matrix = subset.pivot(index="cadence", columns="RV accuracy", values=value_column)
    matrix = matrix.sort_index().sort_index(axis=1)
    return 100.0 * matrix


def colorbar_vmax(values, minimum=100.0):
    """Return a colour-bar maximum that does not clip finite plotted values."""
    finite_values = numpy.asarray(values, dtype=float)
    finite_values = finite_values[numpy.isfinite(finite_values)]
    if finite_values.size == 0:
        return minimum
    return max(minimum, float(finite_values.max()))


def plot_population_weighted_survey_followup_heatmaps():
    survey_counts_path = TABLE_DIR / "population_weighted_followup_survey_counts.csv"
    if not survey_counts_path.exists():
        logger.warning(
            "Skipping population-weighted survey follow-up heatmaps because %s "
            "does not exist. Run src/5_population_weighted_followup.py first.",
            survey_counts_path.relative_to(PROJECT_ROOT),
        )
        return

    plot_population_weighted_period_heatmaps(
        pandas.read_csv(survey_counts_path),
        metrics=[
            ("classified_fraction", "class", "Recovered [per cent]", "viridis", ".0f", 100),
        ],
        filename_prefix="population_weighted_rv_followup_heatmaps_counts",
    )


def plot_population_weighted_relative_error_heatmaps():
    survey_counts_path = TABLE_DIR / "population_weighted_followup_survey_counts.csv"
    if not survey_counts_path.exists():
        logger.warning(
            "Skipping population-weighted relative-error heatmaps because %s "
            "does not exist. Run src/5_population_weighted_followup.py first.",
            survey_counts_path.relative_to(PROJECT_ROOT),
        )
        return

    survey_counts = pandas.read_csv(survey_counts_path)
    metrics = [
        (
            "mass_relative_error_median",
            "macc",
            r"$M_\mathrm{lens}$",
            r"Median $|\Delta M_\mathrm{lens}|/M_\mathrm{lens}$ [per cent]",
        ),
        (
            "period_relative_error_median",
            "porb",
            r"$P_\mathrm{orb}$",
            r"Median $|\Delta P_\mathrm{orb}|/P_\mathrm{orb}$ [per cent]",
        ),
    ]
    missing = [
        value_column
        for value_column, _, _, _ in metrics
        if value_column not in survey_counts.columns
    ]
    if missing:
        logger.warning(
            "Skipping population-weighted relative-error heatmaps because %s "
            "does not contain columns: %s",
            survey_counts_path.relative_to(PROJECT_ROOT),
            ", ".join(missing),
        )
        return

    plot_population_weighted_period_heatmaps(
        survey_counts,
        metrics=[
            (value_column, slug, colorbar_label, "viridis_r", ".1f", None)
            for value_column, slug, _, colorbar_label in metrics
        ],
        filename_prefix="population_weighted_rv_followup_heatmaps_relative_error",
    )


def plot_population_weighted_period_heatmaps(survey_counts, metrics, filename_prefix):
    """Plot weighted metrics by orbital period in the recovery-grid layout."""
    required_columns = {"lens_mass_range", "period_range"}
    missing = sorted(required_columns.difference(survey_counts.columns))
    if missing:
        raise KeyError(
            "Population-weighted heatmaps require period-bin rows. Re-run "
            "src/5_population_weighted_followup.py; missing columns: "
            f"{missing}"
        )

    reference_survey = next(iter(SURVEY_DETECTED_COUNTS))
    data = survey_counts[
        (survey_counts["survey"] == reference_survey)
        & (survey_counts["lens_mass_range"] == ALL_MASSES_LABEL)
        & survey_counts["period_range"].isin(ORBITAL_PERIOD_RANGE_LABELS)
    ].copy()
    if data.empty:
        raise ValueError(
            "Population-weighted heatmaps require rows with lens_mass_range='All masses'. "
            "Re-run src/5_population_weighted_followup.py."
        )

    baselines = numpy.sort(data["baseline"].unique())
    cadences = numpy.sort(data["cadence"].unique())
    rv_values = numpy.sort(data["RV accuracy"].unique())
    period_ranges = ORBITAL_PERIOD_RANGE_LABELS

    for value_column, slug, colorbar_label, cmap, fmt, fixed_vmax in metrics:
        vmax = fixed_vmax or colorbar_vmax(100.0 * data[value_column])
        fig, axes = pyplot.subplots(
            len(baselines),
            len(cadences),
            figsize=(8.2, 6.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = numpy.asarray(axes, dtype=object).reshape(len(baselines), len(cadences))
        last_mesh = None
        for row_index, baseline in enumerate(baselines):
            for col_index, cadence in enumerate(cadences):
                axis = axes[row_index, col_index]
                subset = data[
                    (data["baseline"] == baseline) & (data["cadence"] == cadence)
                ]
                matrix = subset.pivot(
                    index="RV accuracy",
                    columns="period_range",
                    values=value_column,
                ).reindex(index=rv_values, columns=period_ranges)
                heatmap = seaborn.heatmap(
                    100.0 * matrix,
                    ax=axis,
                    vmin=0,
                    vmax=vmax,
                    cmap=cmap,
                    annot=True,
                    annot_kws={"fontsize": 6.0},
                    fmt=fmt,
                    cbar=False,
                    linewidths=0.35,
                    linecolor="white",
                )
                last_mesh = heatmap.collections[0]
                if row_index == 0:
                    axis.set_title(f"Cadence {cadence:g} d")
                if col_index == 0:
                    axis.set_ylabel(
                        f"Baseline {baseline / 365:g} yr\\n"
                        + r"RV accuracy [m s$^{-1}$]"
                    )
                else:
                    axis.set_ylabel("")
                if row_index == len(baselines) - 1:
                    axis.set_xlabel("True orbital period range")
                    axis.tick_params(axis="x", rotation=35)
                else:
                    axis.set_xlabel("")
                    axis.tick_params(axis="x", labelbottom=False)
        if last_mesh is not None:
            fig.colorbar(
                last_mesh,
                ax=axes,
                location="right",
                shrink=0.78,
                label=colorbar_label,
            )
        for suffix in ("pdf", "png"):
            fig.savefig(
                FIGURE_DIR / f"{filename_prefix}_{slug}.{suffix}",
                bbox_inches="tight",
            )
        pyplot.close(fig)


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
    finite = binned_metric_rows(
        results,
        value_column=value_column,
        metric_column=metric_column,
        unit=unit,
        n_bins=n_bins,
        bins=bins,
        labels=labels,
        label_formatter=label_formatter,
    )

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


def binned_metric_rows(
    results,
    value_column,
    metric_column,
    unit,
    n_bins=4,
    bins=None,
    labels=None,
    label_formatter=range_label,
):
    """Return finite metric rows annotated with the true-value bin."""
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
    return finite


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
    # This CSV sidecar is not included in main.tex.
    # table.to_csv(TABLE_DIR / f"{filename}.csv", index=False)

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
    vmax=None,
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
    # This CSV sidecar is not included in main.tex.
    # table.to_csv(TABLE_DIR / f"{filename}.csv", index=False)

    if percent_scale and vmax is not None:
        vmax = colorbar_vmax(100.0 * table[metric_name], minimum=vmax)

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
                vmax=vmax,
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


def plot_relative_error_distribution_by_truth_range(
    results,
    value_column,
    metric_column,
    unit,
    filename,
    x_label,
    bins=None,
    labels=None,
    label_formatter=range_label,
):
    """Replace each relative-error heatmap cell with a 1D histogram."""
    plot_data = binned_metric_rows(
        results,
        value_column=value_column,
        metric_column=metric_column,
        unit=unit,
        bins=bins,
        labels=labels,
        label_formatter=label_formatter,
    )
    # This CSV sidecar is not included in main.tex.
    # plot_data.to_csv(TABLE_DIR / f"{filename}.csv", index=False)

    baselines = numpy.sort(plot_data["baseline"].dropna().unique())
    cadences = numpy.sort(plot_data["cadence"].dropna().unique())
    bin_labels = (
        list(labels)
        if labels is not None
        else list(plot_data["_value_bin_label"].drop_duplicates())
    )
    rv_accuracies = numpy.sort(plot_data["RV accuracy"].dropna().unique())
    positive_data = plot_data[plot_data[metric_column] > 0].copy()
    if positive_data.empty:
        raise ValueError(f"No positive relative errors available for {metric_column}")

    log_errors = numpy.log10(positive_data[metric_column].to_numpy(dtype=float))
    log_min = numpy.floor(log_errors.min())
    log_max = numpy.ceil(log_errors.max())
    if log_min == log_max:
        log_min -= 0.5
        log_max += 0.5
    histogram_edges = numpy.linspace(log_min, log_max, 11)

    fig, axes = pyplot.subplots(
        len(baselines),
        len(cadences),
        figsize=(8.2, 6.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = numpy.asarray(axes, dtype=object).reshape(len(baselines), len(cadences))

    for row_index, baseline in enumerate(baselines):
        for col_index, cadence in enumerate(cadences):
            axis = axes[row_index, col_index]
            panel_data = positive_data[
                (positive_data["baseline"] == baseline)
                & (positive_data["cadence"] == cadence)
            ]

            # Make the parent axes identical to the heatmap grid. Each inset
            # occupies the corresponding heatmap cell and contains its own
            # normalised, log-relative-error histogram.
            axis.set_xlim(0, len(bin_labels))
            axis.set_ylim(len(rv_accuracies), 0)
            axis.set_xticks(numpy.arange(len(bin_labels)) + 0.5, bin_labels)
            axis.set_yticks(
                numpy.arange(len(rv_accuracies)) + 0.5,
                rv_accuracy_tick_labels(rv_accuracies),
            )
            axis.set_aspect("equal")
            axis.set_facecolor("white")
            axis.set_xticks(numpy.arange(len(bin_labels) + 1), minor=True)
            axis.set_yticks(numpy.arange(len(rv_accuracies) + 1), minor=True)
            axis.grid(which="minor", color="0.72", linewidth=0.65)
            axis.tick_params(which="minor", bottom=False, left=False)

            for y_index, accuracy in enumerate(rv_accuracies):
                for x_index, value_bin_label in enumerate(bin_labels):
                    cell_data = panel_data[
                        (panel_data["RV accuracy"] == accuracy)
                        & (panel_data["_value_bin_label"] == value_bin_label)
                    ][metric_column].to_numpy(dtype=float)
                    cell_data = cell_data[numpy.isfinite(cell_data) & (cell_data > 0)]
                    if cell_data.size == 0:
                        continue
                    inset = axis.inset_axes(
                        [
                            (x_index + 0.06) / len(bin_labels),
                            1.0 - (y_index + 0.94) / len(rv_accuracies),
                            0.88 / len(bin_labels),
                            0.88 / len(rv_accuracies),
                        ]
                    )
                    counts, _ = numpy.histogram(numpy.log10(cell_data), bins=histogram_edges)
                    frequencies = counts / counts.sum()
                    inset.bar(
                        histogram_edges[:-1],
                        frequencies,
                        width=numpy.diff(histogram_edges),
                        align="edge",
                        color="#0072B2",
                        edgecolor="white",
                        linewidth=0.18,
                    )
                    inset.set_xlim(log_min, log_max)
                    inset.set_ylim(0, max(0.35, frequencies.max() * 1.08))
                    inset.set_axis_off()

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

    fig.text(
        0.5,
        0.01,
        "Within each cell: relative-error histogram (logarithmic bins; error increases left to right)",
        ha="center",
        va="bottom",
        fontsize=8,
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


def compact_object_class_confusion(results):
    classified = pandas.DataFrame(
        {
            "true_class": compact_object_class(results["Macc"]),
            "fitted_class": compact_object_class(results["fitted_Macc"]),
        }
    ).dropna()

    counts = pandas.crosstab(
        classified["fitted_class"],
        classified["true_class"],
    ).reindex(
        index=COMPACT_OBJECT_CLASS_LABELS,
        columns=COMPACT_OBJECT_CLASS_LABELS,
        fill_value=0,
    )
    row_totals = counts.sum(axis=1).replace(0, numpy.nan)
    purity = counts.div(row_totals, axis=0)
    return counts, purity


def write_compact_object_class_confusion_table(results):
    counts, purity = compact_object_class_confusion(results)
    table = pandas.DataFrame(
        {
            "Fitted class": counts.index,
            "N fitted": counts.sum(axis=1).astype(int).to_numpy(),
            **{
                f"True {label} [per cent]": (100.0 * purity[label]).round(1).to_numpy()
                for label in COMPACT_OBJECT_CLASS_LABELS
            },
        }
    )

    write_latex_table(
        table,
        "compact_object_class_confusion",
        "Compact-object class purity for the currently fitted subset. Rows are "
        "defined by the class inferred from the fitted lens mass, and columns "
        "give the percentage of those rows whose true lens mass lies in each "
        "class.",
        "tab:compact-object-class-confusion",
        column_format="lrrrrr",
        environment="table*",
    )


def plot_compact_object_class_confusion(results):
    counts, purity = compact_object_class_confusion(results)
    annotations = pandas.DataFrame(
        "",
        index=purity.index,
        columns=purity.columns,
    )
    for row_label in purity.index:
        for column_label in purity.columns:
            value = purity.loc[row_label, column_label]
            count = counts.loc[row_label, column_label]
            annotations.loc[row_label, column_label] = (
                "" if pandas.isna(value) else f"{100.0 * value:.1f}\n({int(count)})"
            )

    fig, axis = pyplot.subplots(figsize=(4.2, 3.4), constrained_layout=True)
    seaborn.heatmap(
        100.0 * purity,
        annot=annotations,
        fmt="",
        cmap="viridis",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "True class fraction [per cent]"},
        linewidths=0.4,
        linecolor="white",
        ax=axis,
    )
    axis.set_xlabel("True compact-object class")
    axis.set_ylabel("Fitted compact-object class")
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)

    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"compact_object_class_confusion.{suffix}",
            bbox_inches="tight",
        )
    pyplot.close(fig)


def write_population_weighted_survey_count_tables():
    survey_counts_path = TABLE_DIR / "population_weighted_followup_survey_counts.csv"
    if not survey_counts_path.exists():
        logger.warning(
            "Skipping population-weighted survey count tables because %s "
            "does not exist. Run src/5_population_weighted_followup.py first.",
            survey_counts_path.relative_to(PROJECT_ROOT),
        )
        return

    survey_counts = pandas.read_csv(survey_counts_path)
    metrics = [
        ("classified_fraction", "classified_survey_count", "class", "compact-object class"),
        ("measured_mass_fraction", "measured_mass_survey_count", "macc", r"$M_\mathrm{lens}$"),
        ("measured_period_fraction", "measured_period_survey_count", "porb", r"$P_\mathrm{orb}$"),
    ]
    strategy_columns = ["baseline_years", "cadence", "RV accuracy"]
    reference_survey = next(iter(SURVEY_DETECTED_COUNTS))
    rv_column = r"RV accuracy [m s$^{-1}$]"
    summary_rows = []

    for fraction_metric, count_metric, slug, label in metrics:
        reference = survey_counts[survey_counts["survey"] == reference_survey].copy()
        table = reference[strategy_columns + [fraction_metric]].rename(
            columns={
                "baseline_years": "Baseline [yr]",
                "cadence": "Cadence [d]",
                "RV accuracy": rv_column,
                fraction_metric: "Recovered [per cent]",
            }
        )
        for survey in SURVEY_DETECTED_COUNTS:
            survey_table = survey_counts[survey_counts["survey"] == survey]
            counts = survey_table[strategy_columns + [count_metric]].rename(
                columns={count_metric: survey}
            )
            table = table.merge(
                counts,
                left_on=["Baseline [yr]", "Cadence [d]", rv_column],
                right_on=strategy_columns,
                how="left",
            )
            table = table.drop(columns=strategy_columns)

        table = table.sort_values(
            ["Baseline [yr]", "Cadence [d]", rv_column]
        )
        table["Recovered [per cent]"] = (100.0 * table["Recovered [per cent]"]).round(1)
        for survey in SURVEY_DETECTED_COUNTS:
            table[survey] = table[survey].round(1)

        csv_path = TABLE_DIR / f"population_weighted_followup_counts_{slug}.csv"
        table.to_csv(csv_path, index=False)
        for survey in SURVEY_DETECTED_COUNTS:
            minimum = table[survey].min()
            maximum = table[survey].max()
            mean = table[survey].mean()
            summary_rows.append(
                {
                    "parameter": slug,
                    "label": label,
                    "survey": survey,
                    "minimum": minimum,
                    "maximum": maximum,
                    "mean": mean,
                }
            )

        lines = [
            r"\begin{table*}",
            r"    \centering",
            r"    \tiny",
            r"    \renewcommand{\arraystretch}{0.64}",
            r"    \setlength{\tabcolsep}{2pt}",
            r"    \setlength{\arrayrulewidth}{0.25pt}",
            (
                "    "
                + rf"\caption{{Predicted survey recovery rates for {label}. "
                + "Recovered percentages are survey-independent emulator outputs; "
                + "ZTF, LSST, and TESS columns give the corresponding expected "
                + "absolute recovered counts after scaling by each survey's "
                + r"standard-IMF detectable yield.}"
            ),
            rf"    \label{{tab:population-weighted-followup-counts-{slug}}}",
            r"    \resizebox{\textwidth}{!}{%",
            r"    \begin{tabular}{rrrrrrr}",
            r"        \hline",
            (
                r"        Baseline [yr] & Cadence [d] & RV accuracy [m s$^{-1}$] "
                r"& Recovered [per cent] & ZTF & LSST & TESS \\"
            ),
            r"        \hline",
        ]
        grouped_cells = {}
        for column_index, column in enumerate(
            ["Baseline [yr]", "Cadence [d]", rv_column]
        ):
            group_columns = ["Baseline [yr]", "Cadence [d]", rv_column][
                : column_index + 1
            ]
            for _, group in table.groupby(group_columns, sort=False):
                row_indices = list(group.index)
                row_span = len(row_indices)
                value = group.iloc[0][column]
                if row_span > 1:
                    grouped_cells[(row_indices[0], column)] = (
                        rf"\slrvmultirow{{{row_span}}}{{{value:g}}}"
                    )
                    for row_index in row_indices[1:]:
                        grouped_cells[(row_index, column)] = ""
                else:
                    grouped_cells[(row_indices[0], column)] = f"{value:g}"

        ordered_rows = list(table.iterrows())
        for ordered_index, (row_index, row) in enumerate(ordered_rows):
            lines.append(
                "        "
                + f"{grouped_cells[(row_index, 'Baseline [yr]')]} & "
                + f"{grouped_cells[(row_index, 'Cadence [d]')]} & "
                + f"{grouped_cells[(row_index, rv_column)]} & "
                + f"{row['Recovered [per cent]']:.1f} & "
                + f"{row['ZTF']:.1f} & "
                + f"{row['LSST']:.1f} & "
                + f"{row['TESS']:.1f} \\\\"
            )
            if ordered_index == len(ordered_rows) - 1:
                continue

            _, next_row = ordered_rows[ordered_index + 1]
            if row["Baseline [yr]"] != next_row["Baseline [yr]"]:
                lines.append(r"        \noalign{\vskip 1.5pt}\hline\noalign{\vskip 1.5pt}")
            elif row["Cadence [d]"] != next_row["Cadence [d]"]:
                lines.append(r"        \noalign{\vskip 1pt}\cline{2-7}\noalign{\vskip 2.5pt}")
        lines.extend(
            [
                r"        \hline",
                r"    \end{tabular}",
                r"    }",
                r"    \setlength{\arrayrulewidth}{0.4pt}",
                r"    \renewcommand{\arraystretch}{1}",
                r"\end{table*}",
                "",
            ]
        )
        tex_path = TABLE_DIR / f"population_weighted_followup_counts_{slug}.tex"
        tex_path.write_text("\n".join(lines))

    summary_path = TABLE_DIR / "population_weighted_followup_counts_summary.csv"
    summary = pandas.DataFrame(summary_rows)
    summary.to_csv(summary_path, index=False)
    summary_lines = [
        r"\begin{table}",
        r"    \centering",
        r"    \caption{Summary of predicted survey detection rates across the RV follow-up strategies shown in Figs.~\ref{fig:population-weighted-rv-followup-heatmaps-counts-class}--\ref{fig:population-weighted-rv-followup-heatmaps-counts-porb}. Full strategy-by-strategy counts are given in Tables~\ref{tab:population-weighted-followup-counts-class}, \ref{tab:population-weighted-followup-counts-macc}, and \ref{tab:population-weighted-followup-counts-porb}.}",
        r"    \label{tab:population-weighted-followup-counts-summary}",
        r"    \begin{tabular}{llrr}",
        r"        \hline",
        r"        Parameter & Survey & Range & Mean \\",
        r"        \hline",
    ]
    for _, row in summary.iterrows():
        summary_lines.append(
            "        "
            + rf"{row['label']} & {row['survey']} & "
            + f"{row['minimum']:.1f}--{row['maximum']:.1f} & "
            + f"{row['mean']:.1f} \\\\"
        )
    summary_lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    summary_tex_path = TABLE_DIR / "population_weighted_followup_counts_summary.tex"
    summary_tex_path.write_text("\n".join(summary_lines))


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
        colorbar_label=r"Median $|\Delta M_\mathrm{lens}|/M_\mathrm{lens}$ [per cent]",
        bins=[0.0, 3.0, 5.0, 20.0, 50.0],
        labels=[
            r"$<3\,M_\odot$",
            r"$3$--$5\,M_\odot$",
            r"$5$--$20\,M_\odot$",
            r"$20$--$50\,M_\odot$",
        ],
        percent_scale=True,
        vmax=100,
    )


def plot_lens_mass_class_recovery_by_period(results):
    """Plot compact-object class recovery grouped by true orbital period."""
    classified = results.copy()
    true_class = compact_object_class(classified["Macc"])
    fitted_class = compact_object_class(classified["fitted_Macc"])
    classified["Macc_compact_object_class_recovered"] = (
        true_class.notna() & fitted_class.notna() & (true_class == fitted_class)
    )

    plot_recovery_by_truth_range(
        classified,
        param="Macc",
        value_column="Porb",
        unit="d",
        filename="lens_mass_class_recovery_by_period",
        x_label=r"True orbital period range",
        label_formatter=period_range_label,
        recovered_column="Macc_compact_object_class_recovered",
        colorbar_label="Correct compact-object class [per cent]",
    )


def plot_lens_mass_relative_error_by_period(results):
    """Plot median lens-mass relative error grouped by true orbital period."""
    plot_metric_by_truth_range(
        results,
        value_column="Porb",
        metric_column="Macc_relative_error",
        metric_name="median_relative_error",
        unit="d",
        filename="lens_mass_relative_error_by_period",
        x_label=r"True orbital period range",
        colorbar_label=r"Median $|\Delta M_\mathrm{lens}|/M_\mathrm{lens}$ [per cent]",
        label_formatter=period_range_label,
        percent_scale=True,
        vmax=100,
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
            r"IQR of $|\Delta M_\mathrm{lens}|/M_\mathrm{lens}$ [per cent]"
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
        vmax=100,
    )


def plot_lens_mass_relative_error_distribution_by_mass(results):
    plot_relative_error_distribution_by_truth_range(
        results,
        value_column="Macc",
        metric_column="Macc_relative_error",
        unit=r"$M_\odot$",
        filename="lens_mass_relative_error_distribution_by_lens_mass",
        x_label=r"True lens mass range",
        bins=[0.0, 3.0, 5.0, 20.0, 50.0],
        labels=[
            r"$<3\,M_\odot$",
            r"$3$--$5\,M_\odot$",
            r"$5$--$20\,M_\odot$",
            r"$20$--$50\,M_\odot$",
        ],
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


def plot_orbital_period_relative_error_by_period(results):
    plot_metric_by_truth_range(
        results,
        value_column="Porb",
        metric_column="Porb_relative_error",
        metric_name="median_relative_error",
        unit="d",
        filename="orbital_period_relative_error_by_period",
        x_label=r"True orbital period range",
        colorbar_label=r"Median $|\Delta P_\mathrm{orb}|/P_\mathrm{orb}$ [per cent]",
        label_formatter=period_range_label,
        percent_scale=True,
        vmax=100,
    )


def plot_orbital_period_relative_error_distribution_by_period(results):
    plot_relative_error_distribution_by_truth_range(
        results,
        value_column="Porb",
        metric_column="Porb_relative_error",
        unit="d",
        filename="orbital_period_relative_error_distribution_by_period",
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
        rf"$|\Delta M_\mathrm{{lens}}|/M_\mathrm{{lens}} \leq {RECOVERY_THRESHOLD:g}$.",
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

    run_step(
        "self-lensing observable-space figure",
        plot_self_lensing_observable_space,
    )

    results_path = result_path()
    logger.info("Using fitted results table %s", results_path)
    results = run_step("load fitted results", load_results, results_path)
    # This also adds the relative-error columns needed by active recovery figures.
    run_step("calculate recovery metrics", summarise_strategy_recovery, results)
    # The strategy-summary CSV is not currently included in main.tex.
    # strategy_summary_path = TABLE_DIR / "recovery_strategy_summary.csv"
    # logger.info("Wrote %s", strategy_summary_path.relative_to(PROJECT_ROOT))

    # Not currently included in main.tex.
    # run_step(
    #     "mass recovery by strategy figure",
    #     plot_mass_recovery_by_strategy,
    #     strategy_summary,
    # )
    # Not currently included in main.tex.
    # run_step(
    #     "lens mass recovery by lens mass figure",
    #     plot_lens_mass_recovery_by_mass,
    #     results,
    # )
    run_step(
        "lens mass class recovery by lens mass figure",
        plot_lens_mass_class_recovery_by_mass,
        results,
    )
    # Not currently included in main.tex.
    # run_step(
    #     "compact-object class confusion figure",
    #     plot_compact_object_class_confusion,
    #     results,
    # )
    run_step(
        "lens mass relative error by lens mass figure",
        plot_lens_mass_relative_error_by_mass,
        results,
    )
    run_step(
        "lens mass class recovery by orbital period figure",
        plot_lens_mass_class_recovery_by_period,
        results,
    )
    run_step(
        "lens mass relative error by orbital period figure",
        plot_lens_mass_relative_error_by_period,
        results,
    )
    # Not currently included in main.tex.
    # run_step(
    #     "lens mass relative-error distribution by lens mass figure",
    #     plot_lens_mass_relative_error_distribution_by_mass,
    #     results,
    # )
    # run_step(
    #     "lens mass relative error IQR by lens mass figure",
    #     plot_lens_mass_relative_error_iqr_by_mass,
    #     results,
    # )
    # Not currently included in main.tex.
    # run_step(
    #     "orbital period recovery by period figure",
    #     plot_orbital_period_recovery_by_period,
    #     results,
    # )
    run_step(
        "orbital period relative error by period figure",
        plot_orbital_period_relative_error_by_period,
        results,
    )
    # Not currently included in main.tex.
    # run_step(
    #     "orbital period relative-error distribution by period figure",
    #     plot_orbital_period_relative_error_distribution_by_period,
    #     results,
    # )
    # Not currently included in main.tex.
    # run_step(
    #     "parameter recovery curves figure",
    #     plot_parameter_recovery_curves,
    #     strategy_summary,
    # )
    run_step(
        "prefit improvement by RV-count figure",
        plot_prefit_improvement_by_n_rv,
        results,
    )
    run_step(
        "prefit improvement by lens-mass range figure",
        plot_prefit_improvement_by_lens_mass_range,
        results,
    )
    # Not currently included in main.tex.
    # run_step(
    #     "prefit improvement by cadence and baseline figures",
    #     plot_prefit_improvement_by_cadence_and_baseline,
    #     results,
    # )
    # Not currently included in main.tex.
    # run_step(
    #     "prefit relative-error improvement scatter grids",
    #     plot_prefit_relative_error_improvement_scatter_grids,
    #     results,
    # )
    # Not currently included in main.tex.
    # run_step(
    #     "prefit relative-error improvement scatter grids by photometric observable",
    #     plot_prefit_relative_error_improvement_scatter_grids_by_photometric_observable,
    #     results,
    # )
    # Not currently included in main.tex.
    # run_step(
    #     "photometry-informed RV follow-up recommendation maps",
    #     plot_photometry_followup_recommendation_maps,
    #     results,
    # )
    run_step(
        "prefit compact-object class by lens-mass range figure",
        plot_prefit_compact_object_class_by_lens_mass_range,
        results,
    )
    run_step(
        "population-weighted survey follow-up heatmaps",
        plot_population_weighted_survey_followup_heatmaps,
    )
    run_step(
        "population-weighted relative-error heatmaps",
        plot_population_weighted_relative_error_heatmaps,
    )
    # Tables are commented out in main.tex.
    # run_step(
    #     "population-weighted survey follow-up count tables",
    #     write_population_weighted_survey_count_tables,
    # )
    # run_step("sample summary table", write_sample_summary, results)
    # run_step("mass recovery table", write_mass_recovery_table, strategy_summary)
    # run_step(
    #     "compact-object class confusion table",
    #     write_compact_object_class_confusion_table,
    #     results,
    # )
    # run_step(
    #     "RV accuracy requirement table",
    #     write_rv_accuracy_requirement_table,
    #     strategy_summary,
    # )

    logger.info("Paper output generation complete")
    print("\nWrote figures:")
    for path in sorted(FIGURE_DIR.glob("*")):
        print(" -", path.relative_to(PROJECT_ROOT))
    print("\nWrote tables:")
    for path in sorted(TABLE_DIR.glob("*")):
        print(" -", path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
