import logging
import os
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from time import perf_counter

import numpy
import pandas
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from photometry import load_photometry_timeseries
from utils import DATA_PATH

logger = logging.getLogger(__name__)


PARAMS = {
    "nobs": tuple(range(3, 11)),
    "campaign_baselines": numpy.array([5 * 365.25]),
    "rv_accuracies": [10**n for n in (-1, 0, 1, 2, 3)],
}
PARAM_GRID = tuple(
    (
        int(nobs),
        float(rv_accuracy),
        int(baseline),
    )
    for baseline in PARAMS["campaign_baselines"]
    for nobs in PARAMS["nobs"]
    for rv_accuracy in PARAMS["rv_accuracies"]
)
PARAM_NOBS = numpy.fromiter(
    (nobs for nobs, _, _ in PARAM_GRID),
    dtype=int,
    count=len(PARAM_GRID),
)
PARAM_RV_ACCURACIES = numpy.fromiter(
    (rv_accuracy for _, rv_accuracy, _ in PARAM_GRID),
    dtype=float,
    count=len(PARAM_GRID),
)
PARAM_BASELINES = numpy.fromiter(
    (baseline for _, _, baseline in PARAM_GRID),
    dtype=int,
    count=len(PARAM_GRID),
)
PARAM_ID_SUFFIXES = tuple(
    f"|{nobs}|{rv_accuracy:.17g}|{baseline}".encode("utf-8")
    for nobs, rv_accuracy, baseline in PARAM_GRID
)

RV_TIMESERIES_PATH = (DATA_PATH / "rv").resolve()
RV_TIMESERIES_METADATA_PATH = RV_TIMESERIES_PATH / "metadata.csv"
RV_TIMESERIES_METADATA_PICKLE_PATH = RV_TIMESERIES_METADATA_PATH.with_suffix(".pkl")
RV_TIMESERIES_ID_NAMESPACE = uuid.UUID("1d34c442-474c-5d23-83ff-c121e5ed4a73")
RV_TIMESERIES_ID_NAMESPACE_BYTES = RV_TIMESERIES_ID_NAMESPACE.bytes
PHOTOMETRY_HDU_NAME = "PHOTOMETRY"
RV_TIMESERIES_METADATA_MAX_WORKERS = int(
    os.environ.get("SLRV_RV_TIMESERIES_METADATA_MAX_WORKERS", min(8, os.cpu_count() or 1))
)
RV_TIMESERIES_METADATA_COLUMNS = (
    "rv_id",
    "rv_path",
    "system_id",
    "RV accuracy",
    "nobs",
    "baseline",
)

PHOTOMETRY_SHARED_COLUMNS = (
    "source_rv_path",
    "photometry_path",
    "a",
    "e",
    "periapsis_phase",
    "Macc",
    "Porb",
    "tau_eff",
    "Mcomp",
    "Rcomp",
    "Teff",
    "b",
    "i",
    "peak_time",
)

METADATA_COLUMNS = (
    RV_TIMESERIES_METADATA_COLUMNS[:3]
    + PHOTOMETRY_SHARED_COLUMNS
    + RV_TIMESERIES_METADATA_COLUMNS[3:]
)


def rv_paths(path=RV_TIMESERIES_PATH):
    paths = sorted(Path(path).glob("*.fits.gz"))
    logger.debug("Found %d rv timeseries paths in %s", len(paths), path)
    return paths


def _rv_hdu_name(index):
    return f"TS{index:05d}"


def rv_key_from_values(system_id, nobs, rv_accuracy, baseline):
    return (
        str(system_id),
        int(nobs),
        f"{float(rv_accuracy):.17g}",
        int(baseline),
    )


def _uuid5_from_name_bytes(name):
    digest = sha1(RV_TIMESERIES_ID_NAMESPACE_BYTES + name).digest()
    return str(uuid.UUID(bytes=digest[:16], version=5))


def rv_id_from_values(system_id, nobs, rv_accuracy, baseline):
    key = rv_key_from_values(system_id, nobs, rv_accuracy, baseline)
    return _uuid5_from_name_bytes("|".join(map(str, key)).encode("utf-8"))


def rv_key(row):
    return rv_key_from_values(
        row["system_id"],
        row.get("nobs", row.get("nobs")),
        row["RV accuracy"],
        row["baseline"],
    )


def rv_id_from_params(photometry, nobs, rv_accuracy, baseline):
    return rv_id_from_values(
        photometry["simulation_id"],
        nobs,
        rv_accuracy,
        baseline,
    )


def rv_path_from_params(photometry, nobs, rv_accuracy, baseline):
    path = rv_path_from_photometry(photometry)
    return path


def rv_path_from_photometry(photometry):
    path = RV_TIMESERIES_PATH / f"{photometry['simulation_id']}.fits.gz"
    return path


def system_id_from_rv_path(path):
    return Path(path).name.removesuffix(".fits.gz")


def existing_rv_keys(table=None):
    if table is None:
        table = load_rv_timeseries()
    if table.empty:
        logger.debug("No existing rv keys because metadata table is empty")
        return set()
    keys = {rv_key(row) for _, row in table.iterrows()}
    logger.debug("Loaded %d existing rv keys", len(keys))
    return keys


def _empty_metadata_table():
    return pandas.DataFrame(columns=RV_TIMESERIES_METADATA_COLUMNS)


def _empty_expanded_metadata_table():
    return pandas.DataFrame(columns=METADATA_COLUMNS)


def _metadata_pickle_path(path=RV_TIMESERIES_METADATA_PATH):
    return Path(path).with_suffix(".pkl")


def _metadata_pickle_is_fresh(csv_path, pickle_path):
    return (
        pickle_path.exists() and pickle_path.stat().st_mtime >= csv_path.stat().st_mtime
    )


def _metadata_csv_has_storage_schema(path=RV_TIMESERIES_METADATA_PATH):
    path = Path(path)
    if not path.exists():
        return True
    return list(pandas.read_csv(path, nrows=0).columns) == list(
        RV_TIMESERIES_METADATA_COLUMNS
    )


def _metadata_table_has_storage_schema(table):
    return list(table.columns) == list(RV_TIMESERIES_METADATA_COLUMNS)


def _write_metadata_pickle(table, path=RV_TIMESERIES_METADATA_PATH, normalise=True):
    if normalise:
        table = _normalise_metadata_table(table)
    pickle_path = _metadata_pickle_path(path)
    pickle_path.parent.mkdir(exist_ok=True, parents=True)
    logger.debug("Writing %d rv metadata rows to %s", len(table), pickle_path)
    table.to_pickle(pickle_path)


def _read_metadata_cache(path=RV_TIMESERIES_METADATA_PATH):
    path = Path(path)
    if not path.exists():
        logger.debug("RV metadata cache does not exist at %s", path)
        return _empty_metadata_table()

    pickle_path = _metadata_pickle_path(path)
    if _metadata_pickle_is_fresh(path, pickle_path):
        table = pandas.read_pickle(pickle_path)
        logger.debug("Read %d rv metadata rows from %s", len(table), pickle_path)
        return table

    table = pandas.read_csv(path)
    logger.debug("Read %d rv metadata rows from %s", len(table), path)
    _write_metadata_pickle(table, path)
    return table


def _write_metadata_cache(table, path=RV_TIMESERIES_METADATA_PATH, normalise=True):
    path = Path(path)
    if normalise:
        table = _normalise_metadata_table(table)
    else:
        table = table.reindex(columns=RV_TIMESERIES_METADATA_COLUMNS)
    path.parent.mkdir(exist_ok=True, parents=True)
    logger.debug("Writing %d rv metadata rows to %s", len(table), path)
    table.to_csv(path, index=False)
    _write_metadata_pickle(table, path, normalise=False)


def _normalise_metadata_table(table):
    if table.empty:
        logger.debug("Normalising empty rv metadata table")
        return _empty_metadata_table()

    before = len(table)
    table = table.reindex(columns=RV_TIMESERIES_METADATA_COLUMNS)
    paths = table["rv_path"].dropna().unique()
    path_map = {path: str(Path(path).resolve()) for path in paths}
    table["rv_path"] = table["rv_path"].map(path_map)
    if table["rv_id"].duplicated().any():
        table.drop_duplicates(subset="rv_id", keep="last", inplace=True)
    table.sort_values(
        ["system_id", "nobs", "RV accuracy", "baseline"], inplace=True
    )
    table.reset_index(drop=True, inplace=True)
    logger.debug(
        "Normalised rv metadata table from %d to %d rows", before, len(table)
    )
    return table


def _expanded_metadata_from_photometry(rv, photometry):
    if rv.empty:
        return _empty_expanded_metadata_table()

    photometry = photometry.rename(
        columns={
            "simulation_id": "system_id",
            "rv_path": "source_rv_path",
        }
    )
    photometry_columns = ("system_id",) + PHOTOMETRY_SHARED_COLUMNS
    photometry = photometry.reindex(columns=photometry_columns)
    table = rv.merge(
        photometry,
        on="system_id",
        how="left",
        validate="many_to_one",
    )
    table = table.reindex(columns=METADATA_COLUMNS)
    missing_photometry = table["a"].isna() if "a" in table else []
    if numpy.any(missing_photometry):
        # Scheduled RV simulations no longer require dense photometry RV files.
        # Their immutable truth metadata lives with the photometry-prefit table.
        from fitting import PREFIT_RESULTS_PATH

        if PREFIT_RESULTS_PATH.exists():
            prefits = pandas.read_pickle(PREFIT_RESULTS_PATH).rename(
                columns={"simulation_id": "system_id"}
            )
            prefit_columns = ("system_id",) + tuple(
                column for column in PHOTOMETRY_SHARED_COLUMNS if column in prefits
            )
            replacement = rv.merge(
                prefits.reindex(columns=prefit_columns), on="system_id", how="left"
            ).reindex(columns=METADATA_COLUMNS)
            table.loc[missing_photometry, :] = replacement.loc[missing_photometry, :]
            missing_photometry = table["a"].isna()
    if numpy.any(missing_photometry):
        missing_ids = sorted(
            table.loc[missing_photometry, "system_id"].dropna().unique()
        )
        raise KeyError(
            "Could not expand rv metadata because photometry metadata is missing "
            f"for system_id values: {missing_ids[:10]}"
        )
    return table


def _expand_metadata_table(table, normalise=True):
    rv = _normalise_metadata_table(table) if normalise else table
    if rv.empty:
        return _empty_expanded_metadata_table()
    photometry = load_photometry_timeseries()
    return _expanded_metadata_from_photometry(rv, photometry)


def _expand_metadata_table_with_photometry(table, photometry, normalise=True):
    rv = _normalise_metadata_table(table) if normalise else table
    if rv.empty:
        return _empty_expanded_metadata_table()
    return _expanded_metadata_from_photometry(rv, photometry)


def _to_float(value):
    if hasattr(value, "to_value"):
        return float(value.to_value())
    return float(value)


def _representative_error(value):
    if hasattr(value, "to_value"):
        value = value.to_value()
    array = numpy.asarray(value, dtype=float)
    if array.ndim == 0:
        return float(array)
    finite = array[numpy.isfinite(array)]
    if len(finite) == 0:
        return numpy.nan
    return float(numpy.median(finite))


def _metadata_from_header(header, path):
    return {
        "rv_id": header["RV_ID"],
        "rv_path": str(path.resolve()),
        "system_id": header["SYS_ID"],
        "RV accuracy": header["RV_ACC"],
        "nobs": header.get("NOBS", header.get("NOBS")),
        "baseline": header["BASELINE"],
    }


def metadata_rows_from_rv_path(path):
    path = Path(path)
    logger.debug("Loading rv metadata rows from %s", path)
    with fits.open(path) as hdul:
        rows = [
            _metadata_from_header(hdu.header, path)
            for hdu in hdul[1:]
            if "RV_ID" in hdu.header
        ]
    logger.debug("Loaded %d rv metadata rows from %s", len(rows), path)
    return rows


def metadata_rows_from_rv_paths(paths):
    paths = list(paths)
    if not paths:
        return []
    if RV_TIMESERIES_METADATA_MAX_WORKERS == 1 or len(paths) == 1:
        return [
            row for path in paths for row in metadata_rows_from_rv_path(path)
        ]

    max_workers = min(RV_TIMESERIES_METADATA_MAX_WORKERS, len(paths))
    logger.debug(
        "Loading rv metadata from %d FITS files with %d threads",
        len(paths),
        max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        grouped_rows = tqdm(
            executor.map(metadata_rows_from_rv_path, paths),
            total=len(paths),
            desc="Reading rv metadata",
        )
        return [row for rows in grouped_rows for row in rows]


def metadata_rows_from_params_for_photometry(photometry):
    rows = []
    for nobs, rv_accuracy, baseline in PARAM_GRID:
        rows.append(
            metadata_row_from_params(
                rv_id=rv_id_from_params(
                    photometry,
                    nobs,
                    rv_accuracy,
                    baseline,
                ),
                rv_path=rv_path_from_photometry(photometry),
                photometry=photometry,
                rv_accuracy=rv_accuracy,
                nobs=nobs,
                baseline=baseline,
            )
        )
    return rows


def metadata_table_from_params_for_paths(paths, photometry):
    paths = list(paths)
    if not paths:
        return _empty_metadata_table()

    start_time = perf_counter()
    system_ids = set(photometry["simulation_id"].astype(str))
    missing_ids = sorted(
        system_id_from_rv_path(path)
        for path in paths
        if system_id_from_rv_path(path) not in system_ids
    )
    if missing_ids:
        raise KeyError(
            "Cannot derive rv metadata because photometry metadata is missing "
            f"for simulation_id values: {missing_ids[:10]}"
        )

    n_per_path = len(PARAM_GRID)
    total_rows = len(paths) * n_per_path
    logger.debug(
        "Building %d deterministic rv metadata rows for %d paths",
        total_rows,
        len(paths),
    )
    data = {
        "rv_id": [],
        "rv_path": [],
        "system_id": [],
        "RV accuracy": numpy.tile(PARAM_RV_ACCURACIES, len(paths)),
        "nobs": numpy.tile(PARAM_NOBS, len(paths)),
        "baseline": numpy.tile(PARAM_BASELINES, len(paths)),
    }

    for path in paths:
        system_id = system_id_from_rv_path(path)
        path = Path(path).resolve()
        system_id_bytes = str(system_id).encode("utf-8")
        data["rv_id"].extend(
            _uuid5_from_name_bytes(system_id_bytes + suffix)
            for suffix in PARAM_ID_SUFFIXES
        )
        data["rv_path"].extend([str(path)] * n_per_path)
        data["system_id"].extend([system_id] * n_per_path)

    table = pandas.DataFrame(data, columns=RV_TIMESERIES_METADATA_COLUMNS)
    logger.debug(
        "Built %d deterministic rv metadata rows in %.1f s",
        len(table),
        perf_counter() - start_time,
    )
    return table


def metadata_rows_from_params_for_paths(paths, photometry):
    return metadata_table_from_params_for_paths(paths, photometry).to_dict("records")


def metadata_row_from_params(
    rv_id,
    rv_path,
    photometry,
    rv_accuracy,
    nobs,
    baseline,
):
    return {
        "rv_id": rv_id,
        "rv_path": str(Path(rv_path).resolve()),
        "system_id": photometry["simulation_id"],
        "source_rv_path": photometry.get("rv_path", ""),
        "photometry_path": photometry["photometry_path"],
        "a": photometry["a"],
        "e": photometry["e"],
        "periapsis_phase": photometry["periapsis_phase"],
        "Macc": photometry["Macc"],
        "Porb": photometry["Porb"],
        "tau_eff": photometry["tau_eff"],
        "Mcomp": photometry["Mcomp"],
        "Rcomp": photometry["Rcomp"],
        "Teff": photometry["Teff"],
        "b": photometry.get("b", numpy.nan),
        "i": photometry["i"],
        "peak_time": photometry["peak_time"],
        "RV accuracy": rv_accuracy,
        "nobs": nobs,
        "baseline": baseline,
    }


def _metadata_header(row, rv_err):
    return {
        "RV_ID": row["rv_id"],
        "SYS_ID": row["system_id"],
        "SRC_RV": row["source_rv_path"],
        "PHOTPATH": row["photometry_path"],
        "A": float(row["a"]),
        "E": float(row["e"]),
        "PERI": float(row["periapsis_phase"]),
        "MBH": float(row["Macc"]),
        "P": float(row["Porb"]),
        "TAU_EFF": float(row["tau_eff"]),
        "MSTAR": float(row["Mcomp"]),
        "RSTAR": float(row["Rcomp"]),
        "TEFF": float(row["Teff"]),
        # Astropy's table metadata serialiser accepts this custom attribute as
        # text; readers can convert the dimensionless value back to float.
        "B": str(float(row["b"])),
        "I": float(row["i"]),
        "PEAK_T": float(row["peak_time"]),
        "RV_ACC": float(row["RV accuracy"]),
        "NOBS": int(row["nobs"]),
        "BASELINE": float(row["baseline"]),
        "RV_ERR": _representative_error(rv_err),
    }


def _padded_table_column(values, length):
    array = numpy.full(length, numpy.nan, dtype=float)
    values = numpy.asarray(values, dtype=float)
    array[: len(values)] = values
    return array


def _error_values(error, length):
    values = numpy.asarray(error, dtype=float)
    if values.ndim == 0:
        return numpy.full(length, float(values))
    if len(values) != length:
        raise ValueError(f"Expected {length} error values, got {len(values)}")
    return values


def _rv_data_table(rv_times, rv, rv_err):
    """Build the RV columns stored for one parameter set."""
    n_rows = len(rv)
    table = Table()
    table["rv_time"] = _padded_table_column(rv_times, n_rows)
    table["rv_time"].unit = "d"
    table["rv"] = _padded_table_column(rv, n_rows)
    table["rv"].unit = "m / s"
    table["rv_err"] = _padded_table_column(_error_values(rv_err, n_rows), n_rows)
    table["rv_err"].unit = "m / s"
    return table


def _rv_curve_table(row, rv_times, rv, rv_err):
    """Build the compact RV-only table stored for one parameter set."""
    table = _rv_data_table(rv_times, rv, rv_err)
    table.meta.update(_metadata_header(row, rv_err))
    return table


def _photometry_table(lc_times, lc, lc_err):
    """Build the single photometry table shared by all RV variants in a file."""
    n_rows = len(lc)
    table = Table()
    table["lc_time"] = _padded_table_column(lc_times, n_rows)
    table["lc_time"].unit = "d"
    table["lc"] = _padded_table_column(lc, n_rows)
    table["lc_err"] = _padded_table_column(_error_values(lc_err, n_rows), n_rows)
    return table


def _shared_photometry_from_hdul(hdul):
    for hdu in hdul[1:]:
        if hdu.name == PHOTOMETRY_HDU_NAME and hdu.data is not None:
            data = hdu.data
            names = set(data.names or ())
            if {"lc_time", "lc", "lc_err"} <= names:
                lc_mask = numpy.isfinite(numpy.asarray(data["lc_time"], dtype=float))
                return {
                    "lc_times": numpy.asarray(data["lc_time"], dtype=float)[lc_mask],
                    "lc": numpy.asarray(data["lc"], dtype=float)[lc_mask],
                    "lc_err": numpy.asarray(data["lc_err"], dtype=float)[lc_mask],
                }
    raise ValueError("RV FITS file has no shared PHOTOMETRY extension")


def _rv_hdu_index(hdul, rv_id):
    for i, hdu in enumerate(hdul[1:], start=1):
        if hdu.header.get("RV_ID") == rv_id:
            return i
    return None


def _curve_from_hdu(hdu, shared_photometry):
    data = hdu.data
    rv_mask = numpy.isfinite(numpy.asarray(data["rv_time"], dtype=float))
    curve = {
        "rv_times": numpy.asarray(data["rv_time"], dtype=float)[rv_mask],
        "rv": numpy.asarray(data["rv"], dtype=float)[rv_mask],
        "rv_err": numpy.asarray(data["rv_err"], dtype=float)[rv_mask],
    }
    curve.update(shared_photometry)
    return curve


def write_rv_timeseries_file(curves, path):
    path = Path(path)
    if not curves:
        return

    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists():
        with fits.open(path) as hdul:
            hdus = fits.HDUList(hdul)
            _shared_photometry_from_hdul(hdus)
            existing_ids = {
                hdu.header.get("RV_ID") for hdu in hdus[1:] if "RV_ID" in hdu.header
            }
            next_index = len(existing_ids) + 1
            for curve in curves:
                row = curve["row"]
                if row["rv_id"] in existing_ids:
                    logger.debug(
                        "Skipping existing rv timeseries %s in %s",
                        row["rv_id"],
                        path,
                    )
                    continue
                table = _rv_curve_table(
                    row,
                    curve["rv_times"],
                    curve["rv"],
                    curve["rv_err"],
                )
                hdu = fits.table_to_hdu(table)
                hdu.name = _rv_hdu_name(next_index)
                next_index += 1
                existing_ids.add(row["rv_id"])
                hdus.append(hdu)
            _write_fits_atomically(hdus, path)
        return

    photometry_hdu = fits.table_to_hdu(
        _photometry_table(
            curves[0]["lc_times"],
            curves[0]["lc"],
            curves[0]["lc_err"],
        )
    )
    photometry_hdu.name = PHOTOMETRY_HDU_NAME
    hdus = fits.HDUList([fits.PrimaryHDU(), photometry_hdu])
    for index, curve in enumerate(curves, start=1):
        table = _rv_curve_table(
            curve["row"],
            curve["rv_times"],
            curve["rv"],
            curve["rv_err"],
        )
        hdu = fits.table_to_hdu(table)
        hdu.name = _rv_hdu_name(index)
        hdus.append(hdu)
    _write_fits_atomically(hdus, path)


def _write_fits_atomically(hdus, path):
    """Replace an RV FITS file only after the complete new file is written."""
    path = Path(path)
    temporary_path = path.with_name(f".{path.stem}.new{path.suffix}")
    try:
        hdus.writeto(temporary_path, overwrite=True)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_rv_timeseries_curve(row_or_path):
    if isinstance(row_or_path, (pandas.Series, dict)):
        path = row_or_path["rv_path"]
        rv_id = row_or_path.get("rv_id")
    else:
        path = row_or_path
        rv_id = None

    logger.debug("Loading rv timeseries curve from %s", path)
    with fits.open(path) as hdul:
        shared_photometry = _shared_photometry_from_hdul(hdul)
        hdu_index = None
        if rv_id is not None:
            hdu_index = _rv_hdu_index(hdul, rv_id)
        else:
            rv_hdu_indices = [
                index
                for index, hdu in enumerate(hdul[1:], start=1)
                if "RV_ID" in hdu.header
            ]
            if len(rv_hdu_indices) == 1:
                hdu_index = rv_hdu_indices[0]
        if hdu_index is None:
            raise KeyError(f"Could not find rv_id={rv_id} in {path}")

        curve = _curve_from_hdu(hdul[hdu_index], shared_photometry)
        logger.debug(
            "Loaded rv curve with %d RV samples and %d LC samples",
            len(curve["rv"]),
            len(curve["lc"]),
        )
        return curve


def load_rv_timeseries_curves(rows):
    rows = list(rows)
    if not rows:
        return {}

    paths = {str(row["rv_path"]) for _, row in rows}
    if len(paths) != 1:
        raise ValueError(
            f"Expected rows from one rv FITS file, got {len(paths)} paths"
        )

    path = next(iter(paths))
    rv_ids = {str(row["rv_id"]) for _, row in rows}
    logger.debug(
        "Loading %d rv curves from grouped FITS file %s",
        len(rv_ids),
        path,
    )
    with fits.open(path) as hdul:
        shared_photometry = _shared_photometry_from_hdul(hdul)
        curves = {}
        for hdu in hdul[1:]:
            rv_id = hdu.header.get("RV_ID")
            if rv_id in rv_ids:
                curves[rv_id] = _curve_from_hdu(hdu, shared_photometry)

    missing_ids = rv_ids - set(curves)
    if missing_ids:
        raise KeyError(f"Could not find rv_ids={sorted(missing_ids)} in {path}")
    return curves


def update_rv_metadata(rows):
    if not rows:
        logger.debug("No new rv metadata rows supplied; reloading metadata")
        return load_rv_timeseries()

    logger.debug("Updating rv metadata with %d new rows", len(rows))
    cached = _normalise_metadata_table(_read_metadata_cache())
    new_table = pandas.DataFrame(rows)
    if cached.empty:
        table = new_table
    else:
        table = pandas.concat([cached, new_table], ignore_index=True)

    table = _normalise_metadata_table(table)
    _write_metadata_cache(table)
    return _expand_metadata_table(table, normalise=False)


def load_rv_timeseries():
    rv_path_set = {str(path) for path in rv_paths()}
    logger.debug(
        "Loading rv metadata for %d FITS files", len(rv_path_set)
    )
    photometry = load_photometry_timeseries()
    raw_cached = _read_metadata_cache()
    cache_needs_rewrite = not _metadata_csv_has_storage_schema() or (
        not raw_cached.empty and not _metadata_table_has_storage_schema(raw_cached)
    )
    if not cache_needs_rewrite and _metadata_table_has_storage_schema(raw_cached):
        cached = raw_cached
    else:
        cached = _normalise_metadata_table(raw_cached)
    cached_len = len(cached)
    if not cached.empty:
        cached = cached[cached["rv_path"].isin(rv_path_set)].copy()
    removed_stale_rows = len(cached) != cached_len

    cached_paths = set(cached["rv_path"]) if not cached.empty else set()
    missing_paths = sorted(rv_path_set - cached_paths)
    logger.debug("RV metadata cache is missing %d paths", len(missing_paths))

    if missing_paths:
        logger.debug(
            "Deriving rv metadata rows for %d paths from deterministic IDs",
            len(missing_paths),
        )
        new_table = metadata_table_from_params_for_paths(missing_paths, photometry)
        if cached.empty:
            table = new_table
        else:
            logger.debug(
                "Combining %d cached rows with %d newly-derived rows",
                len(cached),
                len(new_table),
            )
            table = pandas.concat([cached, new_table], ignore_index=True)
        logger.debug("Normalising %d rv metadata rows", len(table))
        table = _normalise_metadata_table(table)
    else:
        table = cached

    if missing_paths or removed_stale_rows or cache_needs_rewrite:
        _write_metadata_cache(table, normalise=False)

    expanded = _expand_metadata_table_with_photometry(table, photometry, normalise=False)
    logger.debug("Returning %d rv metadata rows", len(expanded))
    return expanded
