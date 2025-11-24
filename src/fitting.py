from astropy import units
from astropy.time import Time

from matplotlib import pyplot, patches

import corner
import numpy
import seaborn

from selflensing.generators import LightcurveGenerator, RVGenerator
from selflensing.models import SelfLensingSystem

from utils import DEFAULTS, DATA_PATH


TIMESERIES_TABLE_PATH = DATA_PATH / "fitted_timeseries.gz"


def plot_fit(row):
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
        sls, times=Time(row["lc_times"], format="jd"), peak_time=peak_time
    )

    fig, ((rv_axis, lc_axis), (rv_resid_axis, lc_resid_axis)) = pyplot.subplots(
        2, 2, figsize=(14, 6), sharex="col", gridspec_kw={"height_ratios": (4, 1)}
    )

    # Left subplot: RV
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

    # Right subplot: Lightcurve
    seaborn.scatterplot(
        x=row["lc_times"], y=row["lc"], color="green", ax=lc_axis
    )  # Simulated data
    seaborn.lineplot(
        {"times": row["lc_times"], "lc": lcg.flux_array},
        x="times",
        y="lc",
        color="red",
        ax=lc_axis,
    )  # Model
    lc_axis.set_title("Lightcurve")
    lc_axis.set_xlabel("LC Times")
    lc_axis.set_ylabel("LC")

    # Draw inset of lensing event
    lc_xlim = lc_axis.get_xlim()
    lc_ylim = lc_axis.get_ylim()
    width = (lc_xlim[1] - lc_xlim[0]) / 2.4
    height = (lc_ylim[1] - lc_ylim[0]) / 2.2
    square = patches.Rectangle(
        (lc_xlim[1] - width, lc_ylim[1] - height),
        width,
        height,
        color="white",
        alpha=0.9,
        zorder=2,
    )
    lc_axis.add_patch(square)

    inset = lc_axis.inset_axes([0.7, 0.7, 0.25, 0.25])
    lc_times = numpy.array(row["lc_times"])
    WIDTH = 10  # How many crossing times to show either side of the peak
    inset_lc_mask = (lc_times >= (row["peak_time"] - row["tau_eff"] * WIDTH)) & (
        lc_times <= (row["peak_time"] + row["tau_eff"] * WIDTH)
    )
    seaborn.scatterplot(
        x=lc_times[inset_lc_mask], y=row["lc"][inset_lc_mask], color="green", ax=inset
    )  # Simulated data
    seaborn.lineplot(
        {"times": lc_times[inset_lc_mask], "lc": lcg.flux_array[inset_lc_mask]},
        x="times",
        y="lc",
        color="red",
        ax=inset,
    )  # Model
    inset.set_xlabel("")
    inset.set_ylabel("")

    # Calculate residuals
    rv_residuals = numpy.array(row["rv"]) - numpy.array(rvg.rv)
    lc_residuals = numpy.array(row["lc"]) - numpy.array(lcg.flux_array)

    # Plot RV residuals
    seaborn.scatterplot(
        x=row["rv_times"], y=rv_residuals, color="blue", ax=rv_resid_axis
    )
    rv_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    rv_resid_axis.set_ylabel("RV Residuals")

    # Plot LC residuals
    seaborn.scatterplot(
        x=row["lc_times"], y=lc_residuals, color="green", ax=lc_resid_axis
    )
    lc_resid_axis.axhline(0, color="black", linestyle="--", linewidth=1)
    lc_resid_axis.set_ylabel("LC Residuals")

    # Create a multiline block of text underneath the subplots
    textstr = (
        f"$a$={row['a']:.2f} AU ({row['fitted_a']:.2f} AU)\n"
        f"$M_\\text{{BH}}$={row['Macc']:.2f} M$_\\odot$ ({row['fitted_Macc']:.2f} M$_\\odot$)\n"
        f"$P_\\text{{orb}}$={row['Porb']:.2f} d ({row['fitted_porb']:.2f} d)\n"
        f"$\\tau_\\text{{eff}}$={row['tau_eff']:.2f} d ({row['fitted_tau_eff']:.2f} d)\n"
        f"$i$={row['i']:.2f} ({row['fitted_i']:.2f})\n"
        f"$b$={row['b']:.2f} ({row['fitted_b']:.2f})\n"
        f"$b_\\text{{max}}$={row['b_max']:.2f} ({row['fitted_b_max']:.2f})\n"
        f"$e$={row['e']:.2f} ({row['fitted_e']:.2f})\n"
        f"$\\varphi$={row['periapsis_phase']:.2f} rad ({row['fitted_periapsis_phase']:.2f} rad)\n"
        f"Baseline={row['baseline']:.2f} d\n"
        f"Cadence={row['cadence']:.2f} d"
    )
    # Place the text to the right of the subplots
    # Add extra whitespace to the right so the text doesn't overlap the plots
    pyplot.subplots_adjust(right=0.79, hspace=0.0)
    fig.text(0.8, 0.5, textstr, ha="left", va="center", fontsize=14, wrap=True)
    pyplot.subplots_adjust(hspace=0.0)
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
