"""Report convergence and reference-comparison diagnostics for stage-1 chains.

The input is a directory of ``*.npz`` posterior files produced by
``1_prefit_and_schedule.py``.  Optionally supply a longer-run reference
directory for a direct check that a faster benchmark has not moved the
posterior relevant to scheduling.
"""

import argparse
from pathlib import Path

import numpy
import pandas


def _load(path, burn):
    with numpy.load(path) as data:
        chain = numpy.asarray(data["chain"], dtype=float)
        names = tuple(str(name) for name in data["parameter_names"])
        warmup_acceptance = (
            numpy.asarray(data["warmup_acceptance"], dtype=float)
            if "warmup_acceptance" in data.files
            else numpy.asarray([])
        )
        proposal_scale = (
            numpy.asarray(data["proposal_scale"], dtype=float)
            if "proposal_scale" in data.files
            else numpy.asarray([])
        )
    if chain.ndim != 3:
        raise ValueError(f"{path} has an invalid chain shape {chain.shape}")
    retained = chain[burn:]
    if len(retained) < 4:
        raise ValueError(f"{path} has fewer than four post-burn-in steps")
    return retained, names, warmup_acceptance, proposal_scale


def split_rhat(chain):
    """Gelman--Rubin split-R-hat for a ``(steps, walkers, parameters)`` chain."""
    steps, walkers, _ = chain.shape
    half = steps // 2
    split = numpy.concatenate((chain[:half], chain[-half:]), axis=1)
    # Reorder to (chains, draws, parameters).
    split = numpy.moveaxis(split, 1, 0)
    means = split.mean(axis=1)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half * means.var(axis=0, ddof=1)
    variance = ((half - 1) / half) * within + between / half
    return numpy.sqrt(numpy.divide(variance, within, out=numpy.full_like(variance, numpy.inf), where=within > 0))


def approximate_ess(chain):
    """Conservative autocorrelation ESS using the initial-positive sequence."""
    draws, walkers, parameters = chain.shape
    centred = chain - chain.mean(axis=0, keepdims=True)
    variance = numpy.mean(centred * centred, axis=(0, 1))
    tau = numpy.ones(parameters)
    # Average autocorrelation over walkers. Stop at the first non-positive
    # lag per parameter, which intentionally errs on the conservative side.
    active = variance > 0
    for lag in range(1, draws):
        acov = numpy.mean(centred[:-lag] * centred[lag:], axis=(0, 1))
        rho = numpy.divide(acov, variance, out=numpy.zeros_like(acov), where=variance > 0)
        valid = active & (rho > 0)
        if not numpy.any(valid):
            break
        tau[valid] += 2.0 * rho[valid]
        active &= rho > 0
    return draws * walkers / tau


def acceptance_proxy(chain):
    """Acceptance rate inferred from position changes in a random-walk chain."""
    moved = numpy.any(numpy.diff(chain, axis=0) != 0, axis=-1)
    return float(numpy.mean(moved))


def diagnostics(path, burn):
    chain, names, warmup_acceptance, proposal_scale = _load(path, burn)
    flat = chain.reshape((-1, chain.shape[-1]))
    rhat = split_rhat(chain)
    ess = approximate_ess(chain)
    output = pandas.DataFrame(
        {
            "parameter": names,
            "rhat": rhat,
            "ess_approx": ess,
            "acceptance_proxy": acceptance_proxy(chain),
            "median": numpy.median(flat, axis=0),
            "low_16": numpy.percentile(flat, 16, axis=0),
            "high_84": numpy.percentile(flat, 84, axis=0),
        }
    )
    if warmup_acceptance.size:
        output["final_warmup_acceptance"] = float(warmup_acceptance[-1])
    if proposal_scale.size == len(names):
        output["adapted_proposal_scale"] = proposal_scale
    return output


def compare(candidate, reference, burn, reference_burn):
    candidate_diagnostics = diagnostics(candidate, burn).set_index("parameter")
    reference_diagnostics = diagnostics(reference, reference_burn).set_index("parameter")
    shared = candidate_diagnostics.index.intersection(reference_diagnostics.index)
    output = candidate_diagnostics.loc[shared].join(
        reference_diagnostics.loc[shared], lsuffix="_candidate", rsuffix="_reference"
    )
    reference_width = output["high_84_reference"] - output["low_16_reference"]
    output["median_shift_in_reference_68_width"] = numpy.divide(
        output["median_candidate"] - output["median_reference"],
        reference_width,
        out=numpy.full(len(output), numpy.nan),
        where=reference_width > 0,
    )
    return output.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("posterior_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--burn", type=int, default=100)
    parser.add_argument(
        "--reference-burn",
        type=int,
        help="Discard count for the reference chain (defaults to --burn).",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = sorted(args.posterior_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No posterior files in {args.posterior_dir}")
    rows = []
    for path in paths:
        if args.reference_dir:
            reference = args.reference_dir / path.name
            if not reference.exists():
                continue
            result = compare(
                path,
                reference,
                args.burn,
                args.reference_burn if args.reference_burn is not None else args.burn,
            )
        else:
            result = diagnostics(path, args.burn)
        result.insert(0, "simulation_id", path.stem)
        rows.append(result)
    report = pandas.concat(rows, ignore_index=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    summary_columns = [
        column
        for column in (
            "rhat",
            "ess_approx",
            "acceptance_proxy",
            "final_warmup_acceptance",
            "adapted_proposal_scale",
            "rhat_candidate",
            "ess_approx_candidate",
            "acceptance_proxy_candidate",
            "rhat_reference",
            "ess_approx_reference",
            "acceptance_proxy_reference",
            "median_shift_in_reference_68_width",
        )
        if column in report
    ]
    print(report.groupby("parameter")[summary_columns].agg(["median", "max", "min"]).to_string())
    if args.output:
        print(f"Saved detailed diagnostics to {args.output}")


if __name__ == "__main__":
    main()
