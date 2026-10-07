"""Jointly refit a long-period event and plot photometry, RV and posterior.

Run from the repository root with:
    .venv/bin/python src/plot_example_joint_fit.py --device gpu
The fit calls stage 3's run_joint_batch directly, using its production settings,
photometry binning, likelihood, saved prefit state and memory bounds. --reuse
redraws only compatible stage-3 archives. Plot models run on CPU, one sample
and at most 128 times per call, independently of the fitting device.
"""

import argparse
import importlib
import json
import os
from pathlib import Path

_device_parser = argparse.ArgumentParser(add_help=False)
_device_parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
_device_args, _ = _device_parser.parse_known_args()
os.environ["JAX_PLATFORMS"] = "cuda,cpu" if _device_args.device == "gpu" else "cpu"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

from prefit import load_lightcurve, PREFIT_PARAMETER_NAMES
from selflensing.lightcurves import radial_velocity_m_per_s, light_curve_flux
from selflensing.fitting.numeric import parameter_row
from rv_timeseries import metadata_rows_from_rv_path, load_rv_timeseries_curve

ROOT = Path(__file__).resolve().parents[1]


FIT_SETUP = "stage3_joint_v1"


def production_pipeline(device):
    """Import stage 3 without running its campaign or taking its results lock."""
    os.environ["SLRV_JAX_PLATFORMS"] = os.environ["JAX_PLATFORMS"]
    os.environ.setdefault("SLRV_JAX_COMPILATION_CACHE_DIR", "/tmp/slrv-example-jax-cache")
    pipeline = importlib.import_module("3_targeted_joint_fit")
    pipeline.FIT_DEVICE = "gpu:0" if device == "gpu" else "cpu:0"
    pipeline.FIT_SAVE_SAMPLERS = True  # Only retain more outputs; same sampler/RNG.
    if not pipeline.FIT_USE_LC:
        raise ValueError("SLRV_FIT_USE_LC must be enabled for a joint example")
    return pipeline


def fit_example(pipeline, row, rv_row, rv):
    """Use the campaign's exact batch fitting entry point for one RV variant."""
    combined = row.to_dict()
    combined.update(rv_row)
    rows = [(0, pd.Series(combined))]
    initial = pipeline.saved_prefit_walker_state(rows[0][1])
    if initial.shape != (pipeline.FIT_NWALKERS, 5):
        raise ValueError("Saved prefit walker count differs from production FIT_NWALKERS")
    _, _, samplers, final = pipeline.run_joint_batch(
        rows, {str(rv_row["rv_id"]): rv}, initial[None, :, :])
    sampler = samplers[0]
    return sampler, initial, np.asarray(final.coords)


def bounded_models(samples, times, row, *, radial=False):
    """Match the numeric production forward model, with fixed memory bounds.

    Evaluate on CPU, never vmap all posterior samples over the lensing model.
    Padding time chunks avoids recompiling for each final partial chunk.
    """
    samples = np.asarray(samples, dtype=float)
    times = np.asarray(times, dtype=float)
    output = np.empty((len(samples), len(times)))
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        @jax.jit
        def evaluate(p, t):
            params = parameter_row(p, row.Mcomp, row.Rcomp)
            if radial:
                return radial_velocity_m_per_s(t, params, row.peak_time)
            return light_curve_flux(t, params, row.peak_time) - 1.0

        for i, sample in enumerate(samples):
            for start in range(0, len(times), 128):
                current = times[start:start + 128]
                padded = np.pad(current, (0, 128 - len(current)), mode="edge")
                values = np.asarray(evaluate(jnp.asarray(sample), jnp.asarray(padded)))
                output[i, start:start + len(current)] = values[:len(current)]
    return output


def rv_models(samples, times, row):
    return bounded_models(samples, times, row, radial=True)


def model_curves(samples, times, row):
    return bounded_models(samples, times, row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--simulation-id", default="simulation_000050")
    parser.add_argument("--nobs", type=int, default=10)
    parser.add_argument("--rv-accuracy", type=float, default=10.0)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper/figures/example_joint_fit")
    args = parser.parse_args()
    devices = jax.devices("gpu" if args.device == "gpu" else "cpu")
    jax.config.update("jax_default_device", devices[0])
    pipeline = production_pipeline(args.device)
    table = pd.read_pickle(ROOT / "data/photometry_prefits.gz")
    matches = table[table.simulation_id == args.simulation_id]
    if len(matches) != 1:
        raise ValueError("Select a unique simulation ID")
    row = matches.iloc[0]
    rv_rows = metadata_rows_from_rv_path(ROOT / "data/rv" / f"{row.simulation_id}.fits.gz")
    rv_matches = [r for r in rv_rows if r["nobs"] == args.nobs
                  and np.isclose(r["RV accuracy"], args.rv_accuracy)]
    if len(rv_matches) != 1:
        raise ValueError("Select a unique saved RV schedule using --nobs and --rv-accuracy")
    rv_row = rv_matches[0]
    rv = load_rv_timeseries_curve(rv_row)
    raw = load_lightcurve(row)
    baseline = np.ptp(raw["times"])
    if row.Porb <= baseline or row.tau_eff < 2:
        raise ValueError("Example must have P > baseline and crossing time >= 2 days")
    if not raw["times"].min() + row.tau_eff < row.peak_time < raw["times"].max() - row.tau_eff:
        raise ValueError("Select an event fully inside the observing baseline")
    production_lc = pipeline.valid_binned_full_lightcurve(rv, row)
    curve = {"times": production_lc["lc_times"], "flux": production_lc["lc"],
             "err": production_lc["lc_err"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = args.output.with_suffix(".npz")
    if not args.reuse:
        print(f"Stage-3 joint fit: {pipeline.FIT_NWALKERS} walkers, "
              f"{pipeline.FIT_NSTEPS} steps, {pipeline.FIT_BURN} burn-in, "
              f"proposal fraction={pipeline.FIT_PROPOSAL_SCALE_FRACTION}, "
              f"walker blocks={pipeline.FIT_WALKERS_PER_CHUNK}", flush=True)
        sampler, initialized, final = fit_example(pipeline, row, rv_row, rv)
        acceptance = (np.asarray(sampler.acceptance) if sampler.acceptance is not None
                      else np.empty((0,)))
        if archive.exists():
            with np.load(archive) as previous:
                compatible = ("fit_setup" in previous
                              and str(previous["fit_setup"]) == FIT_SETUP)
            if not compatible:
                index = 1
                backup = archive.with_name(f"{archive.stem}_custom_fit_{index}.npz")
                while backup.exists():
                    index += 1
                    backup = archive.with_name(f"{archive.stem}_custom_fit_{index}.npz")
                archive.replace(backup)
                print(f"Preserved previous custom-fit archive: {backup}", flush=True)
        np.savez_compressed(
            archive, fit_setup=FIT_SETUP,
            chain=sampler.get_chain(), log_prob=sampler.get_log_prob(),
            acceptance=acceptance,
            flat_chain=sampler.get_chain(flat=True, discard=pipeline.FIT_BURN),
            flat_log_prob=sampler.get_log_prob(flat=True, discard=pipeline.FIT_BURN),
            initial_walker_coords=initialized, final_walker_coords=final,
            chain_start_step=sampler.chain_start_step,
            parameter_names=PREFIT_PARAMETER_NAMES, simulation_id=row.simulation_id,
            steps=pipeline.FIT_NSTEPS, burn=pipeline.FIT_BURN,
            seed=0, walkers=pipeline.FIT_NWALKERS,
            proposal_scale_fraction=pipeline.FIT_PROPOSAL_SCALE_FRACTION,
            walkers_per_chunk=pipeline.FIT_WALKERS_PER_CHUNK,
            progress_chunk_size=pipeline.FIT_PROGRESS_CHUNK_SIZE,
            times=curve["times"], flux=curve["flux"], err=curve["err"],
            rv_id=rv_row["rv_id"], rv_times=rv["rv_times"],
            rv=rv["rv"], rv_err=rv["rv_err"],
        )
    with np.load(archive) as saved:
        if "fit_setup" not in saved or str(saved["fit_setup"]) != FIT_SETUP:
            raise ValueError("This archive used the old custom fit. Rerun without --reuse "
                             "to obtain the production stage-3 setup.")
        if str(saved["simulation_id"]) != row.simulation_id:
            raise ValueError("Saved archive belongs to a different simulation")
        if str(saved["rv_id"]) != rv_row["rv_id"]:
            raise ValueError("Saved archive belongs to a different RV schedule")
        samples, logp = saved["flat_chain"], saved["flat_log_prob"]
        curve = {k: saved[k] for k in ("times", "flux", "err")}
        settings = {k: int(saved[k]) for k in ("steps", "burn", "seed", "walkers")}
        acceptance = saved["acceptance"][settings["burn"]:]
        acceptance_fraction = float(np.mean(acceptance)) if acceptance.size else None
        proposal_fraction = float(saved["proposal_scale_fraction"])
        rv = {k: saved[k] for k in ("rv_times", "rv", "rv_err")}
    # Fitting uses the chosen device; all subsequent forward-model plots use CPU.
    jax.config.update("jax_default_device", jax.devices("cpu")[0])
    finite = np.isfinite(logp) & np.all(np.isfinite(samples), axis=1)
    samples, logp = samples[finite], logp[finite]
    if not len(samples):
        raise ValueError("No finite posterior samples")
    best = samples[np.argmax(logp)]
    grid = np.linspace(raw["times"].min(), raw["times"].max(), 700)
    print("Rendering models on CPU in bounded time/sample blocks", flush=True)
    best_curve = model_curves(best[None, :], grid, row)[0]
    best_at_data = model_curves(best[None, :], curve["times"], row)[0]
    best_rv_data = rv_models(best[None, :], rv["rv_times"], row)[0]
    # Angular departure from edge-on makes the inclination readable. Phase retains
    # its full [0, 360) range; no samples are silently removed at the wrap.
    displayed = samples.copy()
    displayed[:, [1, 4]] = np.rad2deg(displayed[:, [1, 4]])
    truth = np.array([row.a, np.rad2deg(row.i), row.Macc, row.e,
                      np.rad2deg(row.periapsis_phase)])
    # A delta from edge-on avoids unreadable Matplotlib offset notation.
    displayed[:, 1] = (90 - displayed[:, 1]) * 3600
    truth[1] = (90 - truth[1]) * 3600
    best_displayed = displayed[np.argmax(logp)]
    rng = np.random.default_rng(settings["seed"])
    subset = rng.choice(len(displayed), min(20000, len(displayed)), replace=False)
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    fig = plt.figure(figsize=(16, 9))
    # Reposition corner's axes on the combined canvas, preserving vector output.
    axes = [fig.add_axes([0, 0, .1, .1]) for _ in range(25)]
    limits = []
    for k in range(5):
        lo = min(displayed[:, k].min(), truth[k])
        hi = max(displayed[:, k].max(), truth[k])
        pad = max((hi - lo) * .03, 1e-8)
        limits.append((lo - pad, hi + pad))
    corner.corner(displayed[subset], fig=fig, bins=35, color="#245b78", range=limits,
                  labels=[r"$a$ (au)", "$90^\\circ-i$\n(arcsec)", r"$M_{\rm lens}$ ($M_\odot$)",
                          r"$e$", r"$\omega$ (deg)"],
                  truths=truth, truth_color="#b44d32",
                  plot_datapoints=False, fill_contours=True,
                  levels=(0.68, 0.95), max_n_ticks=3,
                  label_kwargs={"fontsize": 12})
    corner.overplot_lines(fig, best_displayed, color="#36845b", ls="--", lw=1.3)
    corner.overplot_points(fig, best_displayed[None, :], color="#36845b",
                          marker="o", ms=4, zorder=5)
    for index, ax in enumerate(axes):
        iy, ix = divmod(index, 5)
        ax.set_position([.55 + ix * .083, .12 + (4 - iy) * .1574, .079, .1504])
        ax.tick_params(labelsize=9)
        if ix == 0 and iy > 0:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
    left = fig.add_gridspec(2, 1, left=.075, right=.455, bottom=.12, top=.90,
                            hspace=.42)
    photometry_panels = left[0].subgridspec(2, 1, height_ratios=[3, 1], hspace=.06)
    rv_panels = left[1].subgridspec(2, 1, height_ratios=[3, 1], hspace=.06)
    ax = fig.add_subplot(photometry_panels[0])
    t = grid - row.peak_time
    ax.errorbar(curve["times"] - row.peak_time, curve["flux"] * 1e3,
                yerr=curve["err"] * 1e3, fmt=".", ms=4, color=".35",
                alpha=.75, elinewidth=.6, label="Simulated photometry (6 h bins)")
    ax.plot(t, best_curve * 1e3, color="#b44d32", lw=1.6,
            label="Highest-posterior model")
    ax.set(ylabel="Relative flux excess\n(ppt)", xlim=(t.min(), t.max()))
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper right", fontsize=10, frameon=False)
    fig.text(.265, .925,
             f"Single self-lensing event: $P_{{\\rm orb}}={row.Porb:.0f}$ d, "
             f"$\\tau_{{\\rm eff}}={row.tau_eff:.2f}$ d",
             ha="center", va="baseline", fontsize=11)
    residual = fig.add_subplot(photometry_panels[1], sharex=ax)
    residual.errorbar(curve["times"] - row.peak_time,
                      (curve["flux"] - best_at_data) * 1e3,
                      yerr=curve["err"] * 1e3, fmt=".", ms=4,
                      color=".35", elinewidth=.6)
    residual.axhline(0, color="#b44d32", lw=1)
    residual.set(xlabel=r"Time from lensing peak, $t-t_0$ (d)", ylabel="Residual\n(ppt)")
    residual.yaxis.label.set_size(10)
    rv_grid = np.linspace(rv["rv_times"].min(), rv["rv_times"].max(), 700)
    best_rv = rv_models(best[None, :], rv_grid, row)[0] / 1000
    rv_ax = fig.add_subplot(rv_panels[0])
    years = (rv_grid - row.peak_time) / 365.25
    rv_ax.plot(years, best_rv, color="#b44d32", lw=1.6)
    rv_ax.errorbar((rv["rv_times"] - row.peak_time) / 365.25,
                  rv["rv"] / 1000, yerr=rv["rv_err"] / 1000,
                  fmt="o", ms=4, color=".25", elinewidth=1)
    rv_ax.set_ylabel("Radial velocity\n(km s$^{-1}$)")
    rv_ax.tick_params(labelbottom=False)
    rv_residual = fig.add_subplot(rv_panels[1], sharex=rv_ax)
    rv_residual.errorbar((rv["rv_times"] - row.peak_time) / 365.25,
                         rv["rv"] - best_rv_data, yerr=rv["rv_err"],
                         fmt="o", ms=3, color=".25", elinewidth=.8)
    rv_residual.axhline(0, color="#b44d32", lw=1)
    rv_residual.set(xlabel=r"Time from lensing peak, $t-t_0$ (yr)",
                    ylabel="Residual\n(m s$^{-1}$)")
    rv_residual.yaxis.label.set_size(10)
    fig.text(.7555, .925, "Joint photometry–RV post-burn-in samples",
             ha="center", va="baseline", fontsize=11)
    fig.legend(handles=[Line2D([], [], color="#b44d32", marker="s", lw=1.5,
                               markersize=5, label="Simulated true values"),
                        Line2D([], [], color="#36845b", marker="o", ls="--", lw=1.3,
                               markersize=5, label="Highest-posterior model")],
               loc="upper center", bbox_to_anchor=(.80, .84),
               frameon=False, fontsize=11)
    for suffix in (".pdf", ".png"):
        fig.savefig(args.output.with_suffix(suffix), dpi=200)
    plt.close(fig)
    metadata = dict(simulation_id=row.simulation_id, period_days=float(row.Porb),
                    crossing_time_days=float(row.tau_eff), baseline_days=float(baseline),
                    finite_samples=len(samples),
                    best_parameters=dict(zip(PREFIT_PARAMETER_NAMES, best.tolist())),
                    best_log_prob=float(logp.max()),
                    chi_squared=float(np.sum(((curve["flux"] - best_at_data) / curve["err"])**2)),
                    fitted_points=len(curve["times"]),
                    posterior_quality="production_random_walk",
                    device=args.device,
                    acceptance_fraction=acceptance_fraction,
                    fit_setup=FIT_SETUP, proposal_scale_fraction=proposal_fraction,
                    photometry_shading=False, rv_shading=False,
                    truth_parameters=dict(zip(PREFIT_PARAMETER_NAMES,
                        [float(row[k]) for k in PREFIT_PARAMETER_NAMES])), **settings)
    metadata.update(rv_id=rv_row["rv_id"], nobs=args.nobs,
                    rv_accuracy_m_per_s=args.rv_accuracy,
                    rv_chi_squared=float(np.sum(((rv["rv"] - rv_models(
                        best[None, :], rv["rv_times"], row)[0]) / rv["rv_err"])**2)))
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
