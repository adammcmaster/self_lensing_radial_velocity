"""
Reduce SLURM fit result shards into the main fitted-timeseries table.

This reads per-array-task outputs written by submit_realistic_fit_slurm.py,
combines them with the existing fitted_timeseries.gz table, deduplicates by
realistic_id/realistic_idx, and writes the reduced table atomically.
"""

import argparse
import os
from pathlib import Path

import pandas

from utils import DATA_PATH


DEFAULT_SUBMIT_ROOT = DATA_PATH / "slurm"
DEFAULT_OUTPUT_PATH = DATA_PATH / "fitted_timeseries.gz"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge SLURM fit shards into the fitted-timeseries table."
    )
    parser.add_argument(
        "submit_dirs",
        nargs="*",
        type=Path,
        help=(
            "Specific SLURM submit directories to reduce. "
            "Defaults to every directory under --submit-root."
        ),
    )
    parser.add_argument(
        "--submit-root",
        type=Path,
        default=DEFAULT_SUBMIT_ROOT,
        help="Root directory containing timestamped SLURM submit directories.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Fitted-timeseries table to update.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Write only the shard rows instead of merging with the existing table.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the reduction summary without writing the output table.",
    )
    return parser.parse_args()


def result_checkpoint_key(row):
    if "realistic_id" in row and not pandas.isna(row["realistic_id"]):
        return f"id:{row['realistic_id']}"
    if "realistic_idx" in row and not pandas.isna(row["realistic_idx"]):
        return f"idx:{row['realistic_idx']}"
    return f"row:{row.name}"


def discover_submit_dirs(args):
    if args.submit_dirs:
        return [path.resolve() for path in args.submit_dirs]

    submit_root = args.submit_root.resolve()
    if not submit_root.exists():
        return []
    return sorted(path for path in submit_root.iterdir() if path.is_dir())


def shard_paths_for_submit_dirs(submit_dirs):
    shard_paths = []
    for submit_dir in submit_dirs:
        shard_paths.extend(sorted((submit_dir / "fit_shards").glob("*.gz")))
    return shard_paths


def read_table(path):
    table = pandas.read_pickle(path)
    if isinstance(table, pandas.Series):
        table = table.to_frame().T
    return table


def read_existing_table(path):
    if not path.exists():
        return pandas.DataFrame()
    return read_table(path)


def read_shards(shard_paths):
    tables = []
    skipped = []
    for path in shard_paths:
        try:
            table = read_table(path)
        except (EOFError, OSError, ValueError, pandas.errors.EmptyDataError) as error:
            skipped.append((path, error))
            continue
        if not table.empty:
            tables.append(table)
    if not tables:
        return pandas.DataFrame(), skipped
    return pandas.concat(tables, ignore_index=True, sort=False), skipped


def deduplicate(table):
    if table.empty:
        return table
    table = table.copy().reset_index(drop=True)
    if "realistic_id" in table.columns or "realistic_idx" in table.columns:
        table["_fit_key"] = table.apply(result_checkpoint_key, axis=1)
        table.drop_duplicates(subset="_fit_key", keep="last", inplace=True)
        table.drop(columns="_fit_key", inplace=True)
    table.reset_index(drop=True, inplace=True)
    return table


def write_table_atomically(table, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(f"{path}.new.gz")
    table.to_pickle(tmp_path)
    os.replace(tmp_path, path)


def main():
    args = parse_args()
    submit_dirs = discover_submit_dirs(args)
    shard_paths = shard_paths_for_submit_dirs(submit_dirs)
    shard_rows, skipped = read_shards(shard_paths)
    existing = (
        pandas.DataFrame() if args.replace else read_existing_table(args.output_path)
    )

    if shard_rows.empty and existing.empty:
        print("No fit shards or existing fitted table found; nothing to reduce.")
        return

    if existing.empty:
        combined = shard_rows
    elif shard_rows.empty:
        combined = existing
    else:
        combined = pandas.concat([existing, shard_rows], ignore_index=True, sort=False)

    reduced = deduplicate(combined)

    print(f"Submit directories scanned: {len(submit_dirs)}")
    print(f"Shard files found: {len(shard_paths)}")
    print(f"Shard rows loaded: {len(shard_rows)}")
    print(f"Existing rows loaded: {0 if args.replace else len(existing)}")
    print(f"Reduced rows: {len(reduced)}")
    print(f"Output path: {args.output_path.resolve()}")
    if skipped:
        print(f"Skipped unreadable shards: {len(skipped)}")
        for path, error in skipped[:10]:
            print(f"  {path}: {error}")
        if len(skipped) > 10:
            print(f"  ... {len(skipped) - 10} more")

    if args.dry_run:
        print("Dry run only; not writing output table.")
        return

    write_table_atomically(reduced, args.output_path)
    print("Wrote reduced fitted-timeseries table.")


if __name__ == "__main__":
    main()
