"""Generate paper-ready recovery figures and tables from fitted simulations.

The fit campaign may still be running. Outputs from this script are a snapshot
of the fitted subset present in fitted_timeseries.gz at execution time.
Re-run after the fit completes to refresh the paper outputs.
"""

from pathlib import Path
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy
import pandas
import seaborn
from matplotlib import pyplot


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


def load_results(path):
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
        results[column] = pandas.to_numeric(results[column], errors="coerce")

    results = results.dropna(subset=["baseline", "cadence", "RV accuracy"]).copy()
    results["baseline_years"] = results["baseline"] / 365.0
    results["n_rv"] = numpy.ceil(results["baseline"] / results["cadence"]).astype(int)
    results["strategy"] = (
        results["baseline_years"].map(lambda value: f"{value:g} yr")
        + ", "
        + results["cadence"].map(lambda value: f"{value:g} d")
    )
    return results


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

    fig.suptitle(
        r"Lens mass recovery "
        r"($|\Delta M_\mathrm{acc}|/M_\mathrm{acc} \leq 0.1$)",
        y=1.08,
    )
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


def plot_recovery_by_truth_range(
    results,
    param,
    value_column,
    unit,
    filename,
    title,
    x_label,
    bins=None,
    labels=None,
    label_formatter=range_label,
):
    table = binned_recovery_table(
        results,
        value_column=value_column,
        recovered_column=f"{param}_recovered",
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
            label="Recovered systems [per cent]",
        )
    fig.suptitle(title, y=1.02)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURE_DIR / f"{filename}.{suffix}", bbox_inches="tight")
    pyplot.close(fig)


def plot_lens_mass_recovery_by_mass(results):
    plot_recovery_by_truth_range(
        results,
        param="Macc",
        value_column="Macc",
        unit=r"$M_\odot$",
        filename="lens_mass_recovery_by_lens_mass",
        title=(
            r"Lens mass recovery by true lens mass "
            r"($|\Delta M_\mathrm{acc}|/M_\mathrm{acc} \leq 0.1$)"
        ),
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
        title=(
            r"Orbital period recovery by true orbital period "
            r"($|\Delta P_\mathrm{orb}|/P_\mathrm{orb} \leq 0.1$)"
        ),
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
    figure.fig.subplots_adjust(top=0.88)
    figure.fig.suptitle(r"Parameter recovery for $|\Delta x|/x \leq 0.1$", y=0.99)
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
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results(result_path())
    strategy_summary = summarise_strategy_recovery(results)
    strategy_summary.to_csv(TABLE_DIR / "recovery_strategy_summary.csv", index=False)

    plot_mass_recovery_by_strategy(strategy_summary)
    plot_lens_mass_recovery_by_mass(results)
    plot_orbital_period_recovery_by_period(results)
    plot_parameter_recovery_curves(strategy_summary)
    write_sample_summary(results)
    write_mass_recovery_table(strategy_summary)
    write_rv_accuracy_requirement_table(strategy_summary)

    print("Wrote figures:")
    for path in sorted(FIGURE_DIR.glob("*")):
        print(" -", path.relative_to(PROJECT_ROOT))
    print("\nWrote tables:")
    for path in sorted(TABLE_DIR.glob("*")):
        print(" -", path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
