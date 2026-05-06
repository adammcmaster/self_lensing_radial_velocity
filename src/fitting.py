import logging

from astropy import units
from astropy.time import Time

from matplotlib import pyplot

import corner
import numpy
import seaborn

from selflensing.generators import LightcurveGenerator, RVGenerator
from selflensing.models import SelfLensingSystem

from lensing_jax import install_selflensing_jax_lensing_patch
from utils import DEFAULTS, DATA_PATH

logger = logging.getLogger(__name__)


TIMESERIES_TABLE_PATH = DATA_PATH / "fitted_timeseries.gz"

install_selflensing_jax_lensing_patch()


def plot_fit(row):
    logger.debug(
        "Plotting fit for realistic_idx=%s realistic_id=%s",
        row.get("realistic_idx"),
        row.get("realistic_id"),
    )
    sls = SelfLensingSystem(
        a=row["fitted_a"] * units.au,
        i=row["fitted_i"] * units.rad,
        Macc=row["fitted_Macc"] * units.M_sun,
        e=row["fitted_e"] * units.dimensionless_unscaled,
        periapsis_phase=row["fitted_periapsis_phase"] * units.rad,
        Mcomp=DEFAULTS["Mcomp"],
        Rcomp=DEFAULTS["Rcomp"],
        Teff=DEFAULTS["Teff"],
    )
    peak_time = Time(row["peak_time"], format="jd")
    rvg = RVGenerator(
        sls, times=Time(row["rv_times"], format="jd"), peak_time=peak_time
    )
    lcg = LightcurveGenerator(
        sls,
        times=Time(row["lc_times"], format="jd"),
        peak_time=peak_time,
        lensing_method="IntegratedWittMaoJax",
    )
    model_lc = numpy.array(lcg.flux_array) - 1.0

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

    # Top subplot: RV
    seaborn.scatterplot(
        x=row["rv_times"], y=row["rv"], color="blue", ax=rv_axis
    )  # Simulated data
    rv_axis.errorbar(
        row["rv_times"],
        row["rv"],
        yerr=row["RV accuracy"],
        fmt="none",
        ecolor="gray",
        alpha=0.7,
    )
    seaborn.lineplot(
        {"times": row["rv_times"], "rv": rvg.rv},
        x="times",
        y="rv",
        color="red",
        ax=rv_axis,
    )  # Model
    rv_axis.set_title("Radial Velocity")
    rv_axis.set_xlabel("RV Times")
    rv_axis.set_ylabel("RV")

    # Lower subplot: Lightcurve
    seaborn.scatterplot(
        x=row["lc_times"], y=row["lc"], color="green", ax=lc_axis
    )  # Simulated data
    seaborn.lineplot(
        {"times": row["lc_times"], "lc": model_lc},
        x="times",
        y="lc",
        color="red",
        ax=lc_axis,
    )  # Model
    lc_axis.set_title("Lightcurve")
    lc_axis.set_xlabel("LC Times")
    lc_axis.set_ylabel("LC")

    lc_times = numpy.array(row["lc_times"])
    lc = numpy.array(row["lc"])
    fit_window_start = row["peak_time"] - row["tau_eff"]
    fit_window_end = row["peak_time"] + row["tau_eff"]
    inset_lc_mask = (lc_times >= fit_window_start) & (
        lc_times <= fit_window_end
    )
    # Light-curve fitting window
    seaborn.scatterplot(
        x=lc_times[inset_lc_mask],
        y=lc[inset_lc_mask],
        color="green",
        ax=lc_window_axis,
    )  # Simulated data
    seaborn.lineplot(
        {"times": lc_times[inset_lc_mask], "lc": model_lc[inset_lc_mask]},
        x="times",
        y="lc",
        color="red",
        ax=lc_window_axis,
    )  # Model
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

    # Calculate residuals
    rv_residuals = numpy.array(row["rv"]) - numpy.array(rvg.rv)
    lc_residuals = lc - model_lc

    # Plot RV residuals
    seaborn.scatterplot(
        x=row["rv_times"], y=rv_residuals, color="blue", ax=rv_resid_axis
    )
    rv_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    rv_resid_axis.set_ylabel("RV Residuals")

    # Plot full LC residuals
    seaborn.scatterplot(
        x=row["lc_times"], y=lc_residuals, color="green", ax=lc_full_resid_axis
    )
    lc_full_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    lc_full_resid_axis.set_ylabel("LC Residuals")

    # Plot fitting-window LC residuals
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

    # Create a multiline block of text underneath the subplots
    b_max = row.get("b_max", numpy.nan)
    textstr = (
        f"$a$={row['a']:.2f} AU ({row['fitted_a']:.2f} AU)\n"
        f"$M_\\text{{BH}}$={row['Macc']:.2f} M$_\\odot$ ({row['fitted_Macc']:.2f} M$_\\odot$)\n"
        f"$P_\\text{{orb}}$={row['Porb']:.2f} d ({row['fitted_porb']:.2f} d)\n"
        f"$\\tau_\\text{{eff}}$={row['tau_eff']:.2f} d ({row['fitted_tau_eff']:.2f} d)\n"
        f"$i$={row['i']:.2f} ({row['fitted_i']:.2f})\n"
        f"$b$={row['b']:.2f} ({row['fitted_b']:.2f})\n"
        f"$b_\\text{{max}}$={b_max:.2f} ({row['fitted_b_max']:.2f})\n"
        f"$e$={row['e']:.2f} ({row['fitted_e']:.2f})\n"
        f"$\\varphi$={row['periapsis_phase']:.2f} rad ({row['fitted_periapsis_phase']:.2f} rad)\n"
        f"Baseline={row['baseline']:.2f} d\n"
        f"Cadence={row['cadence']:.2f} d"
    )
    # Place the text to the right of the subplots
    # Add extra whitespace to the right so the text doesn't overlap the plots
    pyplot.subplots_adjust(right=0.79, hspace=0.08)
    fig.text(0.8, 0.5, textstr, ha="left", va="center", fontsize=14, wrap=True)
    return fig


def plot_emcee_diagnostics(sampler, param_names=None, axes_scale="linear", truths=None):
    """
    Plots walker traces and corner plot for an emcee sampler.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        The sampler object after running MCMC.
    param_names : list of str, optional
        Names of parameters for labeling axes.
    """

    samples = sampler.get_chain()
    logger.debug("Plotting emcee diagnostics for sampler chain shape=%s", samples.shape)
    ndim = samples.shape[2]

    fig_walkers, axes = pyplot.subplots(ndim, 1, figsize=(10, 2.5 * ndim), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], color="black", alpha=0.2)
        if param_names is not None:
            ax.set_ylabel(param_names[i])
        if truths is not None:
            ax.axhline(truths[i], color="blue", linewidth=1)
        if type(axes_scale) == str:
            ax.set_yscale(axes_scale)
        else:
            ax.set_yscale(axes_scale[i])
    axes[-1].set_xlabel("Step Number")
    pyplot.tight_layout()

    flat_samples = sampler.get_chain(discard=0, thin=1, flat=True)
    fig_corner = corner.corner(
        flat_samples,
        labels=param_names,
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 12},
        truths=truths,
        axes_scale=axes_scale,
    )
    return fig_walkers, fig_corner
