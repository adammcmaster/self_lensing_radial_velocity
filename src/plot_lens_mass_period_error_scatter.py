"""Create a binned true-mass/true-period recovery-error figure."""

from pathlib import Path
import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy
import pandas
from matplotlib import pyplot
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(
    "/scratch/anm1u24/self_lensing_radial_velocity/data/fitted_timeseries.gz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "paper" / "figures" / "lens_mass_period_relative_error_2d_histogram.png"


def scalar_number(value):
    """Convert scalar-like values in the fitted table to floats."""
    try:
        array = numpy.asarray(value)
        if array.size == 1:
            return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        pass
    return numpy.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = pandas.read_pickle(args.results)
    required = {"Macc", "Porb", "fitted_Macc"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    truth_mass = results["Macc"].map(scalar_number).to_numpy()
    truth_period = results["Porb"].map(scalar_number).to_numpy()
    fitted_mass = results["fitted_Macc"].map(scalar_number).to_numpy()
    relative_error = numpy.abs(fitted_mass - truth_mass) / numpy.abs(truth_mass)
    valid = (
        numpy.isfinite(truth_mass)
        & numpy.isfinite(truth_period)
        & numpy.isfinite(relative_error)
        & (truth_mass > 0)
        & (truth_period > 0)
    )
    if not valid.any():
        raise ValueError("No finite, positive fitted results are available to plot.")

    # Bin in log space because both axes span wide dynamic ranges.  Each cell's
    # colour is the median mass error of systems in that cell, retaining the
    # requested error encoding while avoiding overplotting.
    bins = 35
    mass_edges = numpy.geomspace(truth_mass[valid].min(), truth_mass[valid].max(), bins + 1)
    period_edges = numpy.geomspace(
        truth_period[valid].min(), truth_period[valid].max(), bins + 1
    )
    mass_bin = numpy.digitize(truth_mass[valid], mass_edges) - 1
    period_bin = numpy.digitize(truth_period[valid], period_edges) - 1
    mass_bin = numpy.clip(mass_bin, 0, bins - 1)
    period_bin = numpy.clip(period_bin, 0, bins - 1)
    median_error = numpy.full((bins, bins), numpy.nan)
    for x_index in range(bins):
        for y_index in range(bins):
            in_bin = (mass_bin == x_index) & (period_bin == y_index)
            if in_bin.any():
                median_error[y_index, x_index] = numpy.median(relative_error[valid][in_bin])

    positive_errors = median_error[numpy.isfinite(median_error) & (median_error > 0)]
    colour_floor = positive_errors.min() if positive_errors.size else 1e-12
    plotted_error = numpy.where(median_error == 0, colour_floor, median_error)
    colour_ceiling = max(numpy.nanmax(plotted_error), colour_floor * 10)

    figure, axis = pyplot.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    histogram = axis.pcolormesh(
        mass_edges,
        period_edges,
        plotted_error,
        cmap="viridis_r",
        norm=LogNorm(vmin=colour_floor, vmax=colour_ceiling),
        shading="auto",
        rasterized=True,
    )

    # Estimate density in log coordinates, then transform the contour grid
    # back to the physical coordinates used by the plot axes.
    log_mass = numpy.log10(truth_mass[valid])
    log_period = numpy.log10(truth_period[valid])
    if numpy.unique(log_mass).size > 1 and numpy.unique(log_period).size > 1:
        kde = gaussian_kde(numpy.vstack((log_mass, log_period)))
        kde_mass, kde_period = numpy.meshgrid(
            numpy.linspace(log_mass.min(), log_mass.max(), 160),
            numpy.linspace(log_period.min(), log_period.max(), 160),
        )
        density = kde(numpy.vstack((kde_mass.ravel(), kde_period.ravel()))).reshape(
            kde_mass.shape
        )
        density_levels = density.max() * numpy.array([0.1, 0.25, 0.4, 0.55, 0.7])
        axis.contour(
            10**kde_mass,
            10**kde_period,
            density,
            levels=density_levels,
            colors="white",
            linewidths=0.8,
            alpha=0.9,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"True lens mass, $M_{\mathrm{lens}}\;[M_\odot]$")
    axis.set_ylabel(r"True orbital period, $P_{\mathrm{orb}}\;[\mathrm{d}]$")
    axis.set_title(
        "Median lens-mass recovery error across the simulated parameter space"
    )
    colourbar = figure.colorbar(histogram, ax=axis, pad=0.02)
    colourbar.set_label(
        r"Median relative error in fitted lens mass, "
        r"$|M_{\mathrm{fit}}-M_{\mathrm{true}}|/M_{\mathrm{true}}$"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)
    figure.savefig(args.output.with_suffix(".pdf"))
    print(f"Plotted {valid.sum():,} fitted systems.")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
