"""
Submit realistic-timeseries fitting jobs as a SLURM array.

The submitter writes small CSV batch manifests and an sbatch script under
data/slurm/. Each array task fits one manifest and writes its own result shard,
avoiding concurrent writes to the main fitted_timeseries.gz table.
"""

import argparse
import math
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import pandas

from realistic import load_realistic_timeseries
from utils import DATA_PATH


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMIT_ROOT = DATA_PATH / "slurm"
DEFAULT_FITTED_TABLE_PATH = DATA_PATH / "fitted_timeseries.gz"


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch realistic timeseries fits and submit them to SLURM."
    )
    parser.add_argument(
        "--submit-root",
        type=Path,
        default=DEFAULT_SUBMIT_ROOT,
        help="Directory where batch manifests, logs, and result shards are written.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Name for this submit directory. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--partition",
        default="a100",
        help="SLURM partition/queue to submit to.",
    )
    parser.add_argument(
        "--account",
        default=None,
        help="Optional SLURM account/project name.",
    )
    parser.add_argument(
        "--time",
        default="24:00:00",
        help="Wall time per array task.",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=positive_int,
        default=4,
        help="CPU cores per array task.",
    )
    parser.add_argument(
        "--nodes",
        type=positive_int,
        default=1,
        help="SLURM nodes per array task.",
    )
    parser.add_argument(
        "--mem",
        default="32G",
        help="Memory per array task.",
    )
    parser.add_argument(
        "--gres",
        default="gpu:a100:1",
        help="SLURM GRES request. Use an empty string to omit it.",
    )
    parser.add_argument(
        "--target-rows-per-job",
        type=positive_int,
        default=10,
        help="Preferred number of realistic rows per array task.",
    )
    parser.add_argument(
        "--max-jobs",
        type=positive_int,
        default=128,
        help="Upper bound on array size.",
    )
    parser.add_argument(
        "--sample-size",
        type=positive_int,
        default=None,
        help="Optional maximum number of currently-unfitted rows to submit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --sample-size is smaller than the available set.",
    )
    parser.add_argument(
        "--fit-max-workers",
        type=positive_int,
        default=2,
        help="SLRV_FIT_MAX_WORKERS inside each array task.",
    )
    parser.add_argument(
        "--fit-nsteps",
        type=positive_int,
        default=None,
        help="Optional SLRV_FIT_NSTEPS override for submitted jobs.",
    )
    parser.add_argument(
        "--fit-nwalkers",
        type=positive_int,
        default=None,
        help="Optional SLRV_FIT_NWALKERS override for submitted jobs.",
    )
    parser.add_argument(
        "--fit-burn",
        type=positive_int,
        default=None,
        help="Optional SLRV_FIT_BURN override for submitted jobs.",
    )
    parser.add_argument(
        "--plain-python",
        action="store_true",
        help="Run the fitter via plain python instead of 'uv run python'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests and the sbatch script but do not call sbatch.",
    )
    return parser.parse_args()


def load_existing_fitted_ids(
    path=DEFAULT_FITTED_TABLE_PATH,
    submit_root=DEFAULT_SUBMIT_ROOT,
):
    fitted_ids = set()
    path = Path(path)
    if path.exists():
        fitted = pandas.read_pickle(path)
        if "realistic_id" in fitted.columns:
            fitted_ids.update(fitted["realistic_id"].dropna().astype(str))

    shard_paths = []
    shard_dir = path.parent / "fit_shards"
    if shard_dir.exists():
        shard_paths.extend(sorted(shard_dir.glob("*.gz")))
    submit_root = Path(submit_root)
    if submit_root.exists():
        shard_paths.extend(sorted(submit_root.glob("*/fit_shards/*.gz")))

    for shard_path in shard_paths:
        fitted = pandas.read_pickle(shard_path)
        if "realistic_id" in fitted.columns:
            fitted_ids.update(fitted["realistic_id"].dropna().astype(str))
    return fitted_ids


def split_batches(table, target_rows_per_job, max_jobs):
    if table.empty:
        return []

    preferred_jobs = math.ceil(len(table) / target_rows_per_job)
    n_jobs = min(max_jobs, preferred_jobs)
    rows_per_job = math.ceil(len(table) / n_jobs)
    return [
        table.iloc[start : start + rows_per_job].copy()
        for start in range(0, len(table), rows_per_job)
    ]


def write_batches(batches, batch_dir):
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths = []
    for index, batch in enumerate(batches):
        path = batch_dir / f"batch_{index:04d}.csv"
        batch[["realistic_id", "realistic_path"]].to_csv(path, index=False)
        manifest_paths.append(path)
    return manifest_paths


def sbatch_header(args, n_batches, log_dir):
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=slrv-fit",
        f"#SBATCH --partition={args.partition}",
        f"#SBATCH --array=0-{n_batches - 1}",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --output={log_dir}/%A_%a.out",
        f"#SBATCH --error={log_dir}/%A_%a.err",
    ]
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    if args.gres:
        lines.append(f"#SBATCH --gres={args.gres}")
    return lines


def write_sbatch_script(args, submit_dir, n_batches):
    log_dir = submit_dir / "logs"
    shard_dir = submit_dir / "fit_shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    python_command = "python" if args.plain_python else "uv run python"
    lines = sbatch_header(args, n_batches, log_dir)
    lines += [
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "",
        "export SLRV_REQUIRE_GPU=1",
        "export SLRV_FIT_DEVICE=gpu:0",
        f"export SLRV_FIT_MAX_WORKERS={args.fit_max_workers}",
        f"export SLRV_PLOT_MAX_WORKERS={args.fit_max_workers}",
        f"export SLRV_FIT_CHUNK_SIZE={max(1, 2 * args.fit_max_workers)}",
        "export SLRV_MP_START_METHOD=spawn",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export VECLIB_MAXIMUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "",
        "BATCH_FILE=$(printf '%s/batches/batch_%04d.csv' "
        f"{shlex.quote(str(submit_dir))} \"$SLURM_ARRAY_TASK_ID\")",
        "RESULT_FILE=$(printf '%s/fit_shards/fitted_%04d.gz' "
        f"{shlex.quote(str(submit_dir))} \"$SLURM_ARRAY_TASK_ID\")",
        "export SLRV_FIT_BATCH_FILE=\"$BATCH_FILE\"",
        "export SLRV_FIT_OUTPUT_PATH=\"$RESULT_FILE\"",
    ]
    if args.fit_nsteps is not None:
        lines.append(f"export SLRV_FIT_NSTEPS={args.fit_nsteps}")
    if args.fit_nwalkers is not None:
        lines.append(f"export SLRV_FIT_NWALKERS={args.fit_nwalkers}")
    if args.fit_burn is not None:
        lines.append(f"export SLRV_FIT_BURN={args.fit_burn}")
    lines += [
        "",
        f"{python_command} src/3_fit_realistic_timeseries.py",
        "",
    ]

    script_path = submit_dir / "run_array.sbatch"
    script_path.write_text("\n".join(lines))
    return script_path


def submit(script_path):
    completed = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"sbatch failed for {script_path}: {message}")
    return completed.stdout.strip()


def main():
    args = parse_args()
    submit_name = args.name or datetime.now().strftime("realistic_fit_%Y%m%d_%H%M%S")
    submit_dir = (args.submit_root / submit_name).resolve()
    batch_dir = submit_dir / "batches"

    table = load_realistic_timeseries()
    fitted_ids = load_existing_fitted_ids(submit_root=args.submit_root)
    if fitted_ids:
        table = table[~table["realistic_id"].astype(str).isin(fitted_ids)].copy()
    table.sort_values(["realistic_path", "realistic_id"], inplace=True)

    if args.sample_size is not None and len(table) > args.sample_size:
        table = table.sample(args.sample_size, random_state=args.seed).sort_values(
            ["realistic_path", "realistic_id"]
        )

    batches = split_batches(table, args.target_rows_per_job, args.max_jobs)
    if not batches:
        print("No unfitted realistic timeseries rows found; nothing to submit.")
        return

    manifest_paths = write_batches(batches, batch_dir)
    script_path = write_sbatch_script(args, submit_dir, len(batches))

    print(f"Prepared {len(table)} rows in {len(manifest_paths)} SLURM array tasks.")
    print(f"Submit directory: {submit_dir}")
    print(f"SBATCH script: {script_path}")
    if args.dry_run:
        print("Dry run only; not submitted.")
        return

    print(submit(script_path))


if __name__ == "__main__":
    main()
