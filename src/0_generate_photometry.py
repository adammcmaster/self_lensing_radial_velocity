"""Generate the local v2.0 TESS-like self-lensing photometry library.

The deterministic signal and the GP noise are generated through the current
batched selflensing API.  This keeps the simulation forward model identical to
the model used by the GPU prefit, avoiding the historical model-version drift.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# The TESS GP's lensing callback executes on the host.  Keep a CPU backend
# available alongside CUDA so that callback can be placed correctly.
os.environ["JAX_PLATFORMS"] = os.environ.get("SLRV_JAX_PLATFORMS", "cuda,cpu")

import jax
import jax.numpy as jnp
import numpy
from astropy import constants, units
from astropy.time import Time
from tqdm import tqdm

from photometry import PHOTOMETRY_SIMULATION_PATH
from selflensing.batch import (
    make_batched_light_curve_generator,
    batched_gp_light_curve_jit,
)
from selflensing.gp import TESSGP
from selflensing.gp import FluxFinderSensitivity as flux_finder
from selflensing.gp.PhysicalRelations import effective_temperature_K, stellar_radius_Rsun
from selflensing.gp import PhysicalRelations as physical_relations
from selflensing.gp import PriorsSamplers as gp_priors
from selflensing.models import DEFAULT_LIMITS, SelfLensingSystem
from utils import configure_logging


PHOTOMETRY_SIMULATION_COUNT = int(
    os.environ.get("SLRV_PHOTOMETRY_SIMULATION_COUNT", 10_000)
)
PHOTOMETRY_GENERATION_BATCH_SIZE = int(
    # Stage 0 keeps no gradients or MCMC history, unlike the prefit.  It can
    # therefore use a substantially wider batch to amortise host/device
    # dispatch and make the GPU work large enough to be efficient.
    os.environ.get("SLRV_PHOTOMETRY_GENERATION_BATCH_SIZE", 64)
)
PHOTOMETRY_RANDOM_SEED = int(os.environ.get("SLRV_PHOTOMETRY_RANDOM_SEED", 20260825))
TESS_CADENCE_SECONDS = 600.0
TESS_SECTOR_DAYS = 27.0
PHOTOMETRY_PEAK_JD = float(os.environ.get("SLRV_PHOTOMETRY_PEAK_JD", 2460000.0))
DETECTION_SIGMA = float(os.environ.get("SLRV_PHOTOMETRY_DETECTION_SIGMA", 5.0))
GP_KERNEL_TYPE = "GS"
PHOTOMETRY_WRITE_WORKERS = int(os.environ.get("SLRV_PHOTOMETRY_WRITE_WORKERS", 2))


def _log_uniform(rng, low, high, size):
    return 10.0 ** rng.uniform(numpy.log10(low), numpy.log10(high), size=size)


def _eccentricity_log_uniform(rng, size):
    """Sample e logarithmically after shifting its zero lower bound by one."""
    upper = DEFAULT_LIMITS["e"][1]
    return 10.0 ** rng.uniform(0.0, numpy.log10(1.0 + upper), size=size) - 1.0


def sample_parameter_rows(rng, size):
    """Draw paper-distribution parameters in the public selflensing units."""
    return {
        "a": _log_uniform(rng, *DEFAULT_LIMITS["a"], size),
        "i": rng.uniform(*DEFAULT_LIMITS["i"], size=size),
        # Lens mass is deliberately log-uniform in v2.0.
        "Macc": _log_uniform(rng, *DEFAULT_LIMITS["Macc"], size),
        "Mcomp": rng.uniform(*DEFAULT_LIMITS["Mcomp"], size=size),
        "e": _eccentricity_log_uniform(rng, size),
        "periapsis_phase": rng.uniform(*DEFAULT_LIMITS["periapsis_phase"], size=size),
        "d": rng.uniform(*DEFAULT_LIMITS["d"], size=size),
    }


def viable_parameter_mask(parameter_rows):
    """Vectorised equivalent of ``not system.rlof and system.sl_possible``.

    Constructing a ``SelfLensingSystem`` involves many scalar Astropy/unxt
    quantity operations.  The batched forward model only needs its numeric
    parameter matrix, so apply the exact geometry tests to arrays first and
    reserve object construction for the small subset ultimately written out.
    """
    a_m = parameter_rows["a"] * constants.au.value
    macc_kg = parameter_rows["Macc"] * constants.M_sun.value
    mcomp_kg = parameter_rows["Mcomp"] * constants.M_sun.value
    rcomp_m = parameter_rows["Mcomp"] ** 0.8 * constants.R_sun.value
    eccentricity = parameter_rows["e"]
    phase = parameter_rows["periapsis_phase"]
    inclination = parameter_rows["i"]

    # These expressions match SelfLensingSystem.rE(phase=0), b/b_max,
    # and r_roche (whose separation is evaluated at periapsis).
    conjunction_separation = a_m * (1.0 - eccentricity**2) / (
        1.0 + eccentricity * numpy.cos(phase)
    )
    einstein_radius = numpy.sqrt(
        4.0
        * constants.G.value
        * macc_kg
        * conjunction_separation
        * numpy.sin(inclination)
        / constants.c.value**2
    )
    self_lensing = a_m * numpy.abs(numpy.cos(inclination)) <= einstein_radius + rcomp_m

    q = mcomp_kg / macc_kg
    q_two_thirds = q ** (2.0 / 3.0)
    roche_factor = 0.49 * q_two_thirds / (
        0.6 * q_two_thirds + numpy.log(1.0 + q ** (1.0 / 3.0))
    )
    roche_radius = a_m * (1.0 - eccentricity) * roche_factor
    return self_lensing & (roche_radius > rcomp_m)


def select_rows(parameter_rows, mask):
    return {name: values[mask] for name, values in parameter_rows.items()}


def parameter_matrix(parameter_rows):
    mcomp = parameter_rows["Mcomp"]
    stellar_radius = mcomp**0.8
    stellar_temperature = (
        (mcomp**3.5 * constants.L_sun.value)
        / (
            4.0
            * numpy.pi
            * constants.sigma_sb.value
            * (stellar_radius * constants.R_sun.value) ** 2
        )
    ) ** 0.25
    return numpy.column_stack(
        (
            parameter_rows["a"],
            parameter_rows["i"],
            parameter_rows["Macc"],
            mcomp,
            stellar_radius,
            stellar_temperature,
            parameter_rows["e"],
            parameter_rows["periapsis_phase"],
        )
    )


def system_from_rows(parameter_rows, index):
    """Construct a quantity-aware system only for a retained output row."""
    mcomp = parameter_rows["Mcomp"][index]
    return SelfLensingSystem(
        a=parameter_rows["a"][index] * units.au,
        i=parameter_rows["i"][index] * units.rad,
        Macc=parameter_rows["Macc"][index] * units.M_sun,
        Mcomp=mcomp * units.M_sun,
        Rcomp=stellar_radius_Rsun(mcomp) * units.R_sun,
        Teff=effective_temperature_K(mcomp) * units.K,
        e=parameter_rows["e"][index] * units.dimensionless_unscaled,
        periapsis_phase=parameter_rows["periapsis_phase"][index] * units.rad,
        d=parameter_rows["d"][index] * units.kpc,
    )


def make_tess_gp_input_generator():
    """Build a cached, batched equivalent of ``TESSGP(..., kernel_type='GS')``.

    ``TESSGP`` is convenient for individual systems but calls
    ``conv_mass_to_mag`` for each one.  That function reloads the instrument
    response and noise files every time.  Stage 0 has thousands of systems, so
    retain one ``FluxModel`` and vectorise the small amount of TESS-GP setup.
    The returned values retain TESSGP's count-space conventions exactly.
    """
    flux_model = flux_finder.FluxModel()

    @jax.jit
    def _generate(masses, distances):
        magnitudes, fractional_noise = jax.vmap(flux_model.mass_to_mag)(masses, distances)
        log_count_baselines = jnp.log10(
            flux_finder.counts_per_cadence_from_mag(
                magnitudes, cadence_s=TESS_CADENCE_SECONDS
            )
        )
        count_baselines = 10.0**log_count_baselines
        log_granulation_frequency = jnp.log10(
            jax.vmap(physical_relations.granulation_frequency_relation)(masses)
        )
        log_granulation_sigma = jax.vmap(
            lambda log_mu: gp_priors.expected_log_sigma(log_mu, "G")
        )(log_count_baselines)
        log_seismic_frequency = jnp.log10(
            jax.vmap(physical_relations.asteroseismic_frequency_relation)(masses)
        )
        log_seismic_sigma = jax.vmap(
            lambda log_mu: gp_priors.expected_log_sigma(log_mu, "S")
        )(log_count_baselines)
        parameters = jnp.stack(
            [
                log_granulation_frequency,
                log_granulation_sigma,
                log_seismic_frequency,
                jnp.full_like(log_count_baselines, TESSGP.Q_s),
                log_seismic_sigma,
            ],
            axis=1,
        )
        return parameters, count_baselines, fractional_noise * count_baselines

    def generate(parameter_rows):
        masses = jnp.asarray(parameter_rows["Mcomp"])
        distances = jnp.asarray(parameter_rows["d"])
        return tuple(numpy.asarray(value) for value in jax.device_get(_generate(masses, distances)))

    return generate


def _header_values(system, source_amplification):
    return {
        "A": float(system.a.value),
        "I": float(system.i.value),
        "MBH": float(system.Macc.value),
        "RSTAR": float(system.Rcomp.value),
        "MSTAR": float(system.Mcomp.value),
        "TEFF": float(system.Teff.value),
        "E": float(system.e.value),
        "PERI": float(system.periapsis_phase.value),
        "D": float(system.d.value),
        "P": float(system.porb.value),
        "TAU_EFF": float(system.tau_eff.value),
        "TAU_SL": float(system.tau_sl.value),
        "PEAK_T": PHOTOMETRY_PEAK_JD,
        "AMP_SL": float(source_amplification),
        "SIMVER": "2.0",
        "GPKERNEL": GP_KERNEL_TYPE,
    }


def write_lightcurve(path, times, flux, flux_err, header_values):
    """Atomically write one FITS light curve without exposing partial files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from astropy.io import fits
    from astropy.table import Table

    table = Table()
    table["time"] = Time(times, format="jd", scale="utc")
    table["flux"] = numpy.asarray(flux, dtype=float)
    table["flux_err"] = numpy.asarray(flux_err, dtype=float)
    temporary_path = path.with_name(f"{path.name}.new.fits.gz")
    hdu = fits.BinTableHDU(table)
    for key, value in header_values.items():
        hdu.header[key] = value
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(temporary_path, overwrite=True)
    os.replace(temporary_path, path)


def existing_simulation_indices(path):
    indices = set()
    for candidate in path.glob("simulation_*.fits.gz"):
        try:
            indices.add(int(candidate.name.removeprefix("simulation_").removesuffix(".fits.gz")))
        except ValueError:
            continue
    return indices


def main():
    configure_logging()
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Stage 0 requires a JAX GPU; set SLRV_JAX_PLATFORMS for its backend.")
    if (
        PHOTOMETRY_SIMULATION_COUNT < 1
        or PHOTOMETRY_GENERATION_BATCH_SIZE < 1
        or PHOTOMETRY_WRITE_WORKERS < 1
    ):
        raise ValueError("Simulation count, GPU batch size, and writer count must be positive")

    output_path = PHOTOMETRY_SIMULATION_PATH
    output_path.mkdir(parents=True, exist_ok=True)
    completed = existing_simulation_indices(output_path)
    expected_indices = set(range(PHOTOMETRY_SIMULATION_COUNT))
    completed = completed & expected_indices
    if completed == expected_indices:
        print(f"Local v2.0 photometry library is already complete at {output_path}", flush=True)
        return

    cadence_days = TESS_CADENCE_SECONDS / 86400.0
    relative_times = numpy.arange(0.0, TESS_SECTOR_DAYS, cadence_days) - TESS_SECTOR_DAYS / 2.0
    absolute_times = PHOTOMETRY_PEAK_JD + relative_times
    device = next(device for device in jax.devices() if device.platform == "gpu")
    device_relative_times = jax.device_put(jnp.asarray(relative_times), device)
    deterministic = make_batched_light_curve_generator(
        relative_times, peak_times=0.0, device=device
    )
    lensing_only = make_batched_light_curve_generator(
        relative_times, peak_times=0.0, ev=False, db=False, device=device
    )
    tess_gp_inputs = make_tess_gp_input_generator()
    rng = numpy.random.default_rng(PHOTOMETRY_RANDOM_SEED)
    output_index = 0
    attempts = 0
    timings = {
        "system construction": 0.0,
        "deterministic model": 0.0,
        "TESS GP setup": 0.0,
        "GP sampling": 0.0,
        "host filtering/submission": 0.0,
        "writer backpressure": 0.0,
    }
    progress = tqdm(total=PHOTOMETRY_SIMULATION_COUNT, initial=len(completed), desc="Photometry simulations", unit="system")
    pending_writes = []

    def reap_writes(*, block=False):
        """Surface writer failures and update progress after atomic completion."""
        nonlocal pending_writes
        started = perf_counter()
        remaining = []
        for future in pending_writes:
            if block or future.done():
                future.result()
                progress.update(1)
            else:
                remaining.append(future)
        pending_writes = remaining
        timings["writer backpressure"] += perf_counter() - started

    # Threads avoid forking an already-initialised CUDA/JAX process.  Astropy's
    # FITS compression performs its expensive work in native code, so writers
    # can run concurrently with the GPU generation loop.
    with ThreadPoolExecutor(max_workers=PHOTOMETRY_WRITE_WORKERS) as writers:
        try:
            while output_index < PHOTOMETRY_SIMULATION_COUNT:
                started = perf_counter()
                candidates = sample_parameter_rows(rng, 4 * PHOTOMETRY_GENERATION_BATCH_SIZE)
                viable_rows = select_rows(candidates, viable_parameter_mask(candidates))
                timings["system construction"] += perf_counter() - started
                attempts += len(candidates["a"])
                for start in range(0, len(viable_rows["a"]), PHOTOMETRY_GENERATION_BATCH_SIZE):
                    batch = {
                        name: values[start : start + PHOTOMETRY_GENERATION_BATCH_SIZE]
                        for name, values in viable_rows.items()
                    }
                    if not len(batch["a"]):
                        continue
                    started = perf_counter()
                    params = parameter_matrix(batch)
                    mean_flux = deterministic(params)
                    sl_flux = lensing_only(params)
                    # Generator calls are asynchronous.  Attribute their real
                    # execution time here rather than to a later host transfer.
                    jax.block_until_ready(mean_flux)
                    jax.block_until_ready(sl_flux)
                    timings["deterministic model"] += perf_counter() - started

                    started = perf_counter()
                    gp_params, count_baselines, noise_measure = tess_gp_inputs(batch)
                    timings["TESS GP setup"] += perf_counter() - started

                    started = perf_counter()
                    # Use the module-level JIT directly.  Constructing a new
                    # generator here creates a distinct JIT function object
                    # every batch, which prevents JAX from reusing its compiled
                    # GPU executable even when the shapes are identical.
                    noisy_counts = batched_gp_light_curve_jit(
                        jax.random.PRNGKey(PHOTOMETRY_RANDOM_SEED + attempts + start),
                        device_relative_times,
                        mean_flux * jax.device_put(count_baselines[:, None], device),
                        jax.device_put(gp_params, device),
                        kernel_type=GP_KERNEL_TYPE,
                        noise_measure=jax.device_put(noise_measure, device),
                        times_unit="day",
                    )
                    noisy_counts = numpy.asarray(noisy_counts)
                    timings["GP sampling"] += perf_counter() - started

                    started = perf_counter()
                    noisy_flux = noisy_counts / count_baselines[:, None]
                    fractional_errors = noise_measure / count_baselines
                    source_amplitudes = numpy.asarray(sl_flux).max(axis=1) - 1.0
                    detectable = source_amplitudes >= DETECTION_SIGMA * fractional_errors
                    for index, (flux, flux_err, amplitude, is_detectable) in enumerate(
                        zip(noisy_flux, fractional_errors, source_amplitudes, detectable)
                    ):
                        if not bool(is_detectable):
                            continue
                        path = output_path / f"simulation_{output_index:06d}.fits.gz"
                        if output_index not in completed:
                            system = system_from_rows(batch, index)
                            pending_writes.append(
                                writers.submit(
                                    write_lightcurve,
                                    path,
                                    absolute_times,
                                    flux,
                                    numpy.full_like(flux, flux_err),
                                    _header_values(system, amplitude),
                                )
                            )
                        output_index += 1
                        if output_index >= PHOTOMETRY_SIMULATION_COUNT:
                            break
                    if output_index >= PHOTOMETRY_SIMULATION_COUNT:
                        break
                    timings["host filtering/submission"] += perf_counter() - started
                    # Keep a bounded in-memory queue while letting the GPU start
                    # the next batch before FITS compression catches up.
                    if len(pending_writes) >= 8 * PHOTOMETRY_WRITE_WORKERS:
                        reap_writes(block=True)
                    else:
                        reap_writes()
        finally:
            reap_writes(block=True)
            progress.close()
    print(
        f"Wrote {PHOTOMETRY_SIMULATION_COUNT} local v2.0 simulations to {output_path} "
        f"after drawing {attempts} candidates.",
        flush=True,
    )
    timing_summary = ", ".join(
        f"{name}: {seconds:.1f}s" for name, seconds in timings.items()
    )
    print(f"Stage 0 timing breakdown — {timing_summary}", flush=True)


if __name__ == "__main__":
    main()
