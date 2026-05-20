"""
Submit realistic-timeseries generation jobs as a SLURM array.

Each array task owns a batch of pristine simulation IDs. That keeps writes to
grouped realistic FITS files disjoint, because each pristine simulation maps to
one realistic output file.
"""

import argparse
import math
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import pandas

from pristine import PRISTINE_METADATA_PATH, load_pristine_timeseries
from realistic import PARAMS, load_realistic_timeseries
from utils import DATA_PATH


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMIT_ROOT = DATA_PATH / "slurm"


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch realistic-timeseries generation and submit it to SLURM."
    )
    parser.add_argument(
        "--submit-root",
        type=Path,
        default=DEFAULT_SUBMIT_ROOT,
        help="Directory where batch manifests, logs, and metadata shards are written.",
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
        default="12:00:00",
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
        default="16G",
        help="Memory per array task.",
    )
    parser.add_argument(
        "--gres",
        default="",
        help="Optional SLURM GRES request, for example gpu:a100:1.",
    )
    parser.add_argument(
        "--target-pristine-per-job",
        type=positive_int,
        default=4,
        help="Preferred number of pristine simulations per array task.",
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
        help="Optional maximum number of pristine simulations to submit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --sample-size is smaller than the available set.",
    )
    parser.add_argument(
        "--param-max-workers",
        type=positive_int,
        default=1,
        help="SLRV_GENERATE_PARAM_MAX_WORKERS inside each array task.",
    )
    parser.add_argument(
        "--realistic-max-workers",
        type=positive_int,
        default=4,
        help="SLRV_GENERATE_REALISTIC_MAX_WORKERS inside each array task.",
    )
    parser.add_argument(
        "--plain-python",
        action="store_true",
        help="Run the generator via plain python instead of 'uv run python'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests and the sbatch script but do not call sbatch.",
    )
    return parser.parse_args()


def missing_count_for_pristine(pristine_id, existing_keys):
    count = 0
    for baseline in PARAMS["campaign_baselines"]:
        for cadence in PARAMS["cadences"]:
            for rv_accuracy in PARAMS["rv_accuracies"]:
                key = (
                    str(pristine_id),
                    int(cadence),
                    f"{float(rv_accuracy):.17g}",
                    int(baseline),
                )
                if key not in existing_keys:
                    count += 1
    return count


def load_pristine_index():
    if PRISTINE_METADATA_PATH.exists():
        return pandas.read_csv(PRISTINE_METADATA_PATH)
    return load_pristine_timeseries()


def load_existing_realistic_keys():
    table = load_realistic_timeseries()
    if table.empty:
        return set()
    return {
        (
            str(row["pristine_id"]),
            int(row["cadence"]),
            f"{float(row['RV accuracy']):.17g}",
            int(row["baseline"]),
        )
        for _, row in table.iterrows()
    }


def pristine_with_missing_realistic_rows(pristine):
    existing_keys = load_existing_realistic_keys()
    rows = []
    for _, row in pristine.iterrows():
        missing_count = missing_count_for_pristine(row["simulation_id"], existing_keys)
        if missing_count:
            rows.append(
                {
                    "simulation_id": row["simulation_id"],
                    "rv_path": row["rv_path"],
                    "missing_realistic_count": missing_count,
                }
            )
    return pandas.DataFrame(rows)


def split_batches(table, target_pristine_per_job, max_jobs):
    if table.empty:
        return []

    preferred_jobs = math.ceil(len(table) / target_pristine_per_job)
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
        batch.to_csv(path, index=False)
        manifest_paths.append(path)
    return manifest_paths


def sbatch_header(args, n_batches, log_dir):
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=slrv-gen-real",
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
    metadata_dir = submit_dir / "metadata_shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    python_command = "python" if args.plain_python else "uv run python"
    lines = sbatch_header(args, n_batches, log_dir)
    lines += [
        "",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "",
        "export SLRV_SKIP_PRISTINE_GENERATION=1",
        f"export SLRV_GENERATE_PARAM_MAX_WORKERS={args.param_max_workers}",
        f"export SLRV_GENERATE_REALISTIC_MAX_WORKERS={args.realistic_max_workers}",
        "export SLRV_MP_START_METHOD=spawn",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export VECLIB_MAXIMUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "",
        "BATCH_FILE=$(printf '%s/batches/batch_%04d.csv' "
        f"{shlex.quote(str(submit_dir))} \"$SLURM_ARRAY_TASK_ID\")",
        "METADATA_FILE=$(printf '%s/metadata_shards/metadata_%04d.csv' "
        f"{shlex.quote(str(submit_dir))} \"$SLURM_ARRAY_TASK_ID\")",
        "export SLRV_GENERATE_BATCH_FILE=\"$BATCH_FILE\"",
        "export SLRV_GENERATE_METADATA_OUTPUT_PATH=\"$METADATA_FILE\"",
        "",
        f"{python_command} src/2_generate_realistic_timeseries.py",
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
    submit_name = args.name or datetime.now().strftime(
        "realistic_generate_%Y%m%d_%H%M%S"
    )
    submit_dir = (args.submit_root / submit_name).resolve()
    batch_dir = submit_dir / "batches"

    pristine = load_pristine_index()
    table = pristine_with_missing_realistic_rows(pristine)
    table.sort_values("simulation_id", inplace=True)

    if args.sample_size is not None and len(table) > args.sample_size:
        table = table.sample(args.sample_size, random_state=args.seed).sort_values(
            "simulation_id"
        )

    batches = split_batches(table, args.target_pristine_per_job, args.max_jobs)
    if not batches:
        print("No missing realistic timeseries combinations found; nothing to submit.")
        return

    manifest_paths = write_batches(batches, batch_dir)
    script_path = write_sbatch_script(args, submit_dir, len(batches))

    total_realistic = int(table["missing_realistic_count"].sum())
    print(
        f"Prepared {len(table)} pristine simulations "
        f"({total_realistic} realistic timeseries) "
        f"in {len(manifest_paths)} SLURM array tasks."
    )
    print(f"Submit directory: {submit_dir}")
    print(f"SBATCH script: {script_path}")
    if args.dry_run:
        print("Dry run only; not submitted.")
        return

    print(submit(script_path))


if __name__ == "__main__":
    main()
