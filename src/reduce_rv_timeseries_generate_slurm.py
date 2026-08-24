"""
Reduce SLURM rv-generation metadata shards into rv/metadata.csv.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas
from tqdm import tqdm

from rv_timeseries import (
    RV_TIMESERIES_METADATA_PATH,
    _normalise_metadata_table,
    _read_metadata_cache,
    _write_metadata_cache,
    load_rv_timeseries,
    update_rv_metadata,
)
from utils import DATA_PATH


DEFAULT_SUBMIT_ROOT = DATA_PATH / "slurm"
DEFAULT_WORKERS = min(8, os.cpu_count() or 1)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge SLURM rv-generation metadata shards."
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
        help="Rebuild/update metadata by scanning rv FITS files too.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes used to read metadata shard files.",
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


def read_metadata_shard(path):
    try:
        table = pandas.read_csv(path)
    except (EOFError, OSError, ValueError, pandas.errors.EmptyDataError) as error:
        return None, str(path), str(error)
    if table.empty:
        return None, str(path), None
    return table, str(path), None


def read_metadata_shards(paths, workers=DEFAULT_WORKERS):
    tables = []
    skipped = []
    if not paths:
        return pandas.DataFrame(), skipped

    if workers == 1 or len(paths) == 1:
        results = map(read_metadata_shard, paths)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(read_metadata_shard, paths)
            for table, path, error in tqdm(
                results,
                total=len(paths),
                desc="Reading metadata shards",
            ):
                if error is not None:
                    skipped.append((path, error))
                elif table is not None:
                    tables.append(table)
            if not tables:
                return pandas.DataFrame(), skipped
            return pandas.concat(tables, ignore_index=True, sort=False), skipped

    for table, path, error in tqdm(
        results,
        total=len(paths),
        desc="Reading metadata shards",
    ):
        if error is not None:
            skipped.append((path, error))
        elif table is not None:
            tables.append(table)

    if not tables:
        return pandas.DataFrame(), skipped
    return pandas.concat(tables, ignore_index=True, sort=False), skipped


def update_rv_metadata_with_progress(shard_rows):
    with tqdm(total=4, desc="Reducing metadata") as progress:
        progress.set_postfix_str("reading cache")
        cached = _normalise_metadata_table(_read_metadata_cache())
        progress.update()

        progress.set_postfix_str("combining rows")
        if cached.empty:
            table = shard_rows.copy()
        elif shard_rows.empty:
            table = cached
        else:
            table = pandas.concat([cached, shard_rows], ignore_index=True, sort=False)
        progress.update()

        progress.set_postfix_str("deduplicating")
        table = _normalise_metadata_table(table)
        progress.update()

        progress.set_postfix_str("writing csv")
        _write_metadata_cache(table)
        progress.update()

    return table


def main():
    args = parse_args()
    submit_dirs = discover_submit_dirs(args)
    shard_paths = metadata_shard_paths(submit_dirs)
    shard_rows, skipped = read_metadata_shards(shard_paths, workers=args.workers)

    if args.from_fits:
        current = load_rv_timeseries()
    else:
        current = (
            pandas.read_csv(RV_TIMESERIES_METADATA_PATH)
            if RV_TIMESERIES_METADATA_PATH.exists()
            else pandas.DataFrame()
        )

    print(f"Submit directories scanned: {len(submit_dirs)}")
    print(f"Metadata shard files found: {len(shard_paths)}")
    print(f"Metadata shard read workers: {args.workers}")
    print(f"Metadata shard rows loaded: {len(shard_rows)}")
    print(f"Existing metadata rows loaded: {len(current)}")
    print(f"Metadata path: {RV_TIMESERIES_METADATA_PATH}")
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
            load_rv_timeseries()
            if args.from_fits
            else update_rv_metadata([])
        )
    else:
        table = update_rv_metadata_with_progress(shard_rows)
    print(f"Wrote rv metadata with {len(table)} rows.")


if __name__ == "__main__":
    main()
