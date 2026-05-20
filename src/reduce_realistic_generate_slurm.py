"""
Reduce SLURM realistic-generation metadata shards into realistic/metadata.csv.
"""

import argparse
from pathlib import Path

import pandas

from realistic import (
    REALISTIC_METADATA_PATH,
    load_realistic_timeseries,
    update_realistic_metadata,
)
from utils import DATA_PATH


DEFAULT_SUBMIT_ROOT = DATA_PATH / "slurm"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge SLURM realistic-generation metadata shards."
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
        "--from-fits",
        action="store_true",
        help="Rebuild/update metadata by scanning realistic FITS files too.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the reduction summary without writing metadata.",
    )
    return parser.parse_args()


def discover_submit_dirs(args):
    if args.submit_dirs:
        return [path.resolve() for path in args.submit_dirs]

    submit_root = args.submit_root.resolve()
    if not submit_root.exists():
        return []
    return sorted(path for path in submit_root.iterdir() if path.is_dir())


def metadata_shard_paths(submit_dirs):
    paths = []
    for submit_dir in submit_dirs:
        paths.extend(sorted((submit_dir / "metadata_shards").glob("*.csv")))
    return paths


def read_metadata_shards(paths):
    tables = []
    skipped = []
    for path in paths:
        try:
            table = pandas.read_csv(path)
        except (EOFError, OSError, ValueError, pandas.errors.EmptyDataError) as error:
            skipped.append((path, error))
            continue
        if not table.empty:
            tables.append(table)
    if not tables:
        return pandas.DataFrame(), skipped
    return pandas.concat(tables, ignore_index=True, sort=False), skipped


def main():
    args = parse_args()
    submit_dirs = discover_submit_dirs(args)
    shard_paths = metadata_shard_paths(submit_dirs)
    shard_rows, skipped = read_metadata_shards(shard_paths)

    if args.from_fits:
        current = load_realistic_timeseries()
    else:
        current = (
            pandas.read_csv(REALISTIC_METADATA_PATH)
            if REALISTIC_METADATA_PATH.exists()
            else pandas.DataFrame()
        )

    print(f"Submit directories scanned: {len(submit_dirs)}")
    print(f"Metadata shard files found: {len(shard_paths)}")
    print(f"Metadata shard rows loaded: {len(shard_rows)}")
    print(f"Existing metadata rows loaded: {len(current)}")
    print(f"Metadata path: {REALISTIC_METADATA_PATH}")
    if skipped:
        print(f"Skipped unreadable metadata shards: {len(skipped)}")
        for path, error in skipped[:10]:
            print(f"  {path}: {error}")
        if len(skipped) > 10:
            print(f"  ... {len(skipped) - 10} more")

    if args.dry_run:
        print("Dry run only; not writing metadata.")
        return

    if shard_rows.empty:
        table = (
            load_realistic_timeseries()
            if args.from_fits
            else update_realistic_metadata([])
        )
    else:
        table = update_realistic_metadata(shard_rows.to_dict("records"))
    print(f"Wrote realistic metadata with {len(table)} rows.")


if __name__ == "__main__":
    main()
