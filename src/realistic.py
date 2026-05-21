import logging
from pathlib import Path
import uuid

import numpy
import pandas
from astropy.io import fits
from astropy.table import Table

from pristine import load_pristine_timeseries
from utils import DATA_PATH

logger = logging.getLogger(__name__)


PARAMS = {
    "cadences": [30, 150, 300],
    "campaign_baselines": 365 * numpy.array([1, 3, 5]),
    "rv_accuracies": numpy.logspace(-1, 3, 50),
}

REALISTIC_PATH = (DATA_PATH / "realistic").resolve()
REALISTIC_PATH.mkdir(exist_ok=True, parents=True)
REALISTIC_METADATA_PATH = REALISTIC_PATH / "metadata.csv"
REALISTIC_METADATA_PICKLE_PATH = REALISTIC_METADATA_PATH.with_suffix(".pkl")
REALISTIC_ID_NAMESPACE = uuid.UUID("1d34c442-474c-5d23-83ff-c121e5ed4a73")

REALISTIC_METADATA_COLUMNS = (
    "realistic_id",
    "realistic_path",
    "pristine_idx",
    "pristine_id",
    "RV accuracy",
    "cadence",
    "baseline",
    "rv_err",
    "lc_err",
)

PRISTINE_SHARED_COLUMNS = (
    "pristine_rv_path",
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
    REALISTIC_METADATA_COLUMNS[:4]
    + PRISTINE_SHARED_COLUMNS
    + REALISTIC_METADATA_COLUMNS[4:]
)


def realistic_paths(path=REALISTIC_PATH):
    paths = sorted(Path(path).glob("*.fits.gz"))
    logger.debug("Found %d realistic timeseries paths in %s", len(paths), path)
    return paths


def _realistic_hdu_name(index):
    return f"TS{index:05d}"


def realistic_key_from_values(pristine_id, cadence, rv_accuracy, baseline):
    logger.debug(
        "Building realistic key for pristine_id=%s cadence=%s rv_accuracy=%s baseline=%s",
        pristine_id,
        cadence,
        rv_accuracy,
        baseline,
    )
    return (
        str(pristine_id),
        int(cadence),
        f"{float(rv_accuracy):.17g}",
        int(baseline),
    )


def realistic_key(row):
    return realistic_key_from_values(
        row["pristine_id"],
        row["cadence"],
        row["RV accuracy"],
        row["baseline"],
    )


def realistic_id_from_params(pristine, cadence, rv_accuracy, baseline):
    key = realistic_key_from_values(
        pristine["simulation_id"],
        cadence,
        rv_accuracy,
        baseline,
    )
    realistic_id = str(uuid.uuid5(REALISTIC_ID_NAMESPACE, "|".join(map(str, key))))
    logger.debug("Generated realistic_id=%s for key=%s", realistic_id, key)
    return realistic_id


def realistic_path_from_params(pristine, cadence, rv_accuracy, baseline):
    path = realistic_path_from_pristine(pristine)
    logger.debug("Resolved realistic path %s", path)
    return path


def realistic_path_from_pristine(pristine):
    path = REALISTIC_PATH / f"{pristine['simulation_id']}.fits.gz"
    logger.debug("Resolved grouped realistic path %s", path)
    return path


def existing_realistic_keys(table=None):
    if table is None:
        table = load_realistic_timeseries()
    if table.empty:
        logger.debug("No existing realistic keys because metadata table is empty")
        return set()
    keys = {realistic_key(row) for _, row in table.iterrows()}
    logger.debug("Loaded %d existing realistic keys", len(keys))
    return keys


def _empty_metadata_table():
    return pandas.DataFrame(columns=REALISTIC_METADATA_COLUMNS)


def _empty_expanded_metadata_table():
    return pandas.DataFrame(columns=METADATA_COLUMNS)


def _metadata_pickle_path(path=REALISTIC_METADATA_PATH):
    return Path(path).with_suffix(".pkl")


def _metadata_pickle_is_fresh(csv_path, pickle_path):
    return pickle_path.exists() and pickle_path.stat().st_mtime >= csv_path.stat().st_mtime


def _metadata_csv_has_storage_schema(path=REALISTIC_METADATA_PATH):
    path = Path(path)
    if not path.exists():
        return True
    return list(pandas.read_csv(path, nrows=0).columns) == list(
        REALISTIC_METADATA_COLUMNS
    )


def _write_metadata_pickle(table, path=REALISTIC_METADATA_PATH):
    table = _normalise_metadata_table(table)
    pickle_path = _metadata_pickle_path(path)
    pickle_path.parent.mkdir(exist_ok=True, parents=True)
    logger.debug("Writing %d realistic metadata rows to %s", len(table), pickle_path)
    table.to_pickle(pickle_path)


def _read_metadata_cache(path=REALISTIC_METADATA_PATH):
    path = Path(path)
    if not path.exists():
        logger.debug("Realistic metadata cache does not exist at %s", path)
        return _empty_metadata_table()

    pickle_path = _metadata_pickle_path(path)
    if _metadata_pickle_is_fresh(path, pickle_path):
        table = pandas.read_pickle(pickle_path)
        logger.debug("Read %d realistic metadata rows from %s", len(table), pickle_path)
        return table

    table = pandas.read_csv(path)
    logger.debug("Read %d realistic metadata rows from %s", len(table), path)
    _write_metadata_pickle(table, path)
    return table


def _write_metadata_cache(table, path=REALISTIC_METADATA_PATH):
    path = Path(path)
    table = _normalise_metadata_table(table)
    path.parent.mkdir(exist_ok=True, parents=True)
    logger.debug("Writing %d realistic metadata rows to %s", len(table), path)
    table.to_csv(path, index=False)
    _write_metadata_pickle(table, path)


def _normalise_metadata_table(table):
    if table.empty:
        logger.debug("Normalising empty realistic metadata table")
        return _empty_metadata_table()

    before = len(table)
    table = table.reindex(columns=REALISTIC_METADATA_COLUMNS)
    table["realistic_path"] = table["realistic_path"].map(
        lambda p: str(Path(p).resolve())
    )
    table.drop_duplicates(subset="realistic_id", keep="last", inplace=True)
    table.sort_values(
        ["pristine_id", "cadence", "RV accuracy", "baseline"], inplace=True
    )
    table.reset_index(drop=True, inplace=True)
    logger.debug(
        "Normalised realistic metadata table from %d to %d rows", before, len(table)
    )
    return table


def _expanded_metadata_from_pristine(realistic, pristine):
    if realistic.empty:
        return _empty_expanded_metadata_table()

    pristine = pristine.rename(
        columns={
            "simulation_id": "pristine_id",
            "rv_path": "pristine_rv_path",
        }
    )
    pristine_columns = ("pristine_id",) + PRISTINE_SHARED_COLUMNS
    pristine = pristine.reindex(columns=pristine_columns)
    table = realistic.merge(
        pristine,
        on="pristine_id",
        how="left",
        validate="many_to_one",
    )
    table = table.reindex(columns=METADATA_COLUMNS)
    missing_pristine = table["a"].isna() if "a" in table else []
    if numpy.any(missing_pristine):
        missing_ids = sorted(table.loc[missing_pristine, "pristine_id"].dropna().unique())
        raise KeyError(
            "Could not expand realistic metadata because pristine metadata is missing "
            f"for pristine_id values: {missing_ids[:10]}"
        )
    return table


def _expand_metadata_table(table):
    realistic = _normalise_metadata_table(table)
    if realistic.empty:
        return _empty_expanded_metadata_table()
    pristine = load_pristine_timeseries()
    return _expanded_metadata_from_pristine(realistic, pristine)


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
        "realistic_id": header["REAL_ID"],
        "realistic_path": str(path.resolve()),
        "pristine_idx": header["PRIS_IDX"],
        "pristine_id": header["PRIS_ID"],
        "RV accuracy": header["RV_ACC"],
        "cadence": header["CADENCE"],
        "baseline": header["BASELINE"],
        "rv_err": header["RV_ERR"],
        "lc_err": header["LC_ERR"],
    }


def metadata_rows_from_realistic_path(path):
    path = Path(path)
    logger.debug("Loading realistic metadata rows from %s", path)
    with fits.open(path) as hdul:
        rows = [
            _metadata_from_header(hdu.header, path)
            for hdu in hdul[1:]
            if "REAL_ID" in hdu.header
        ]
    logger.debug("Loaded %d realistic metadata rows from %s", len(rows), path)
    return rows


def metadata_row_from_params(
    realistic_id,
    realistic_path,
    pristine_idx,
    pristine,
    rv_accuracy,
    cadence,
    baseline,
    rv_err,
    lc_err,
):
    logger.debug(
        "Creating realistic metadata row id=%s path=%s pristine_idx=%s cadence=%s baseline=%s rv_accuracy=%s",
        realistic_id,
        realistic_path,
        pristine_idx,
        cadence,
        baseline,
        rv_accuracy,
    )
    return {
        "realistic_id": realistic_id,
        "realistic_path": str(Path(realistic_path).resolve()),
        "pristine_idx": pristine_idx,
        "pristine_id": pristine["simulation_id"],
        "pristine_rv_path": pristine["rv_path"],
        "photometry_path": pristine["photometry_path"],
        "a": pristine["a"],
        "e": pristine["e"],
        "periapsis_phase": pristine["periapsis_phase"],
        "Macc": pristine["Macc"],
        "Porb": pristine["Porb"],
        "tau_eff": pristine["tau_eff"],
        "Mcomp": pristine["Mcomp"],
        "Rcomp": pristine["Rcomp"],
        "Teff": pristine["Teff"],
        "b": pristine["b"],
        "i": pristine["i"],
        "peak_time": pristine["peak_time"],
        "RV accuracy": rv_accuracy,
        "cadence": cadence,
        "baseline": baseline,
        "rv_err": rv_err,
        "lc_err": _representative_error(lc_err),
    }


def _metadata_header(row):
    return {
        "REAL_ID": row["realistic_id"],
        "PRIS_IDX": int(row["pristine_idx"]),
        "PRIS_ID": row["pristine_id"],
        "PRIS_RV": row["pristine_rv_path"],
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
        "B": float(row["b"]),
        "I": float(row["i"]),
        "PEAK_T": float(row["peak_time"]),
        "RV_ACC": float(row["RV accuracy"]),
        "CADENCE": float(row["cadence"]),
        "BASELINE": float(row["baseline"]),
        "RV_ERR": _to_float(row["rv_err"]),
        "LC_ERR": float(row["lc_err"]),
    }


def _curve_table(row, rv_times, rv, rv_err, lc_times, lc, lc_err):
    n_rows = max(len(rv), len(lc))

    def padded(values):
        array = numpy.full(n_rows, numpy.nan, dtype=float)
        values = numpy.asarray(values, dtype=float)
        array[: len(values)] = values
        return array

    def error_values(error, length):
        values = numpy.asarray(error, dtype=float)
        if values.ndim == 0:
            return numpy.full(length, float(values))
        if len(values) != length:
            raise ValueError(f"Expected {length} error values, got {len(values)}")
        return values

    table = Table()
    table["rv_time"] = padded(rv_times)
    table["rv_time"].unit = "d"
    table["rv"] = padded(rv)
    table["rv"].unit = "m / s"
    table["rv_err"] = padded(error_values(rv_err, len(rv)))
    table["rv_err"].unit = "m / s"
    table["lc_time"] = padded(lc_times)
    table["lc_time"].unit = "d"
    table["lc"] = padded(lc)
    table["lc_err"] = padded(error_values(lc_err, len(lc)))
    table.meta.update(_metadata_header(row))
    return table


def _realistic_hdu_index(hdul, realistic_id):
    for i, hdu in enumerate(hdul[1:], start=1):
        if hdu.header.get("REAL_ID") == realistic_id:
            return i
    return None


def _curve_from_hdu(hdu):
    data = hdu.data
    rv_mask = numpy.isfinite(numpy.asarray(data["rv_time"], dtype=float))
    lc_mask = numpy.isfinite(numpy.asarray(data["lc_time"], dtype=float))
    return {
        "rv_times": numpy.asarray(data["rv_time"], dtype=float)[rv_mask],
        "rv": numpy.asarray(data["rv"], dtype=float)[rv_mask],
        "rv_err": numpy.asarray(data["rv_err"], dtype=float)[rv_mask],
        "lc_times": numpy.asarray(data["lc_time"], dtype=float)[lc_mask],
        "lc": numpy.asarray(data["lc"], dtype=float)[lc_mask],
        "lc_err": numpy.asarray(data["lc_err"], dtype=float)[lc_mask],
    }


def write_realistic_timeseries_file(curves, path):
    path = Path(path)
    if not curves:
        return

    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists():
        with fits.open(path) as hdul:
            hdus = fits.HDUList(hdul)
            existing_ids = {
                hdu.header.get("REAL_ID")
                for hdu in hdus[1:]
                if "REAL_ID" in hdu.header
            }
            next_index = len(hdus)
            for curve in curves:
                row = curve["row"]
                if row["realistic_id"] in existing_ids:
                    logger.debug(
                        "Skipping existing realistic timeseries %s in %s",
                        row["realistic_id"],
                        path,
                    )
                    continue
                table = _curve_table(
                    row,
                    curve["rv_times"],
                    curve["rv"],
                    curve["rv_err"],
                    curve["lc_times"],
                    curve["lc"],
                    curve["lc_err"],
                )
                hdu = fits.table_to_hdu(table)
                hdu.name = _realistic_hdu_name(next_index)
                next_index += 1
                existing_ids.add(row["realistic_id"])
                hdus.append(hdu)
            hdus.writeto(path, overwrite=True)
        return

    hdus = fits.HDUList([fits.PrimaryHDU()])
    for index, curve in enumerate(curves, start=1):
        table = _curve_table(
            curve["row"],
            curve["rv_times"],
            curve["rv"],
            curve["rv_err"],
            curve["lc_times"],
            curve["lc"],
            curve["lc_err"],
        )
        hdu = fits.table_to_hdu(table)
        hdu.name = _realistic_hdu_name(index)
        hdus.append(hdu)
    hdus.writeto(path)


def load_realistic_timeseries_curve(row_or_path):
    if isinstance(row_or_path, (pandas.Series, dict)):
        path = row_or_path["realistic_path"]
        realistic_id = row_or_path.get("realistic_id")
    else:
        path = row_or_path
        realistic_id = None

    logger.debug("Loading realistic timeseries curve from %s", path)
    with fits.open(path) as hdul:
        hdu_index = None
        if realistic_id is not None:
            hdu_index = _realistic_hdu_index(hdul, realistic_id)
        elif len(hdul) == 2:
            hdu_index = 1
        if hdu_index is None:
            raise KeyError(f"Could not find realistic_id={realistic_id} in {path}")

        curve = _curve_from_hdu(hdul[hdu_index])
        logger.debug(
            "Loaded realistic curve with %d RV samples and %d LC samples",
            len(curve["rv"]),
            len(curve["lc"]),
        )
        return curve


def load_realistic_timeseries_curves(rows):
    rows = list(rows)
    if not rows:
        return {}

    paths = {str(row["realistic_path"]) for _, row in rows}
    if len(paths) != 1:
        raise ValueError(
            f"Expected rows from one realistic FITS file, got {len(paths)} paths"
        )

    path = next(iter(paths))
    realistic_ids = {str(row["realistic_id"]) for _, row in rows}
    logger.debug(
        "Loading %d realistic curves from grouped FITS file %s",
        len(realistic_ids),
        path,
    )
    with fits.open(path) as hdul:
        curves = {}
        for hdu in hdul[1:]:
            realistic_id = hdu.header.get("REAL_ID")
            if realistic_id in realistic_ids:
                curves[realistic_id] = _curve_from_hdu(hdu)

    missing_ids = realistic_ids - set(curves)
    if missing_ids:
        raise KeyError(f"Could not find realistic_ids={sorted(missing_ids)} in {path}")
    return curves


def update_realistic_metadata(rows):
    if not rows:
        logger.debug("No new realistic metadata rows supplied; reloading metadata")
        return load_realistic_timeseries()

    logger.debug("Updating realistic metadata with %d new rows", len(rows))
    cached = _normalise_metadata_table(_read_metadata_cache())
    new_table = pandas.DataFrame(rows)
    if cached.empty:
        table = new_table
    else:
        table = pandas.concat([cached, new_table], ignore_index=True)

    table = _normalise_metadata_table(table)
    _write_metadata_cache(table)
    return _expand_metadata_table(table)


def load_realistic_timeseries():
    realistic_path_set = {str(path) for path in realistic_paths()}
    logger.debug(
        "Loading realistic metadata for %d FITS files", len(realistic_path_set)
    )
    raw_cached = _read_metadata_cache()
    cache_needs_rewrite = not _metadata_csv_has_storage_schema() or (
        not raw_cached.empty
        and list(raw_cached.columns) != list(REALISTIC_METADATA_COLUMNS)
    )
    cached = _normalise_metadata_table(raw_cached)
    cached_len = len(cached)
    if not cached.empty:
        cached = cached[cached["realistic_path"].isin(realistic_path_set)].copy()
    removed_stale_rows = len(cached) != cached_len

    cached_paths = set(cached["realistic_path"]) if not cached.empty else set()
    missing_paths = sorted(realistic_path_set - cached_paths)
    logger.debug("Realistic metadata cache is missing %d paths", len(missing_paths))

    if missing_paths:
        new_rows = [
            row
            for path in missing_paths
            for row in metadata_rows_from_realistic_path(path)
        ]
        new_table = pandas.DataFrame(new_rows)
        if cached.empty:
            cached = new_table
        else:
            cached = pandas.concat([cached, new_table], ignore_index=True)

    table = _normalise_metadata_table(cached)
    if missing_paths or removed_stale_rows or cache_needs_rewrite:
        _write_metadata_cache(table)

    expanded = _expand_metadata_table(table)
    logger.debug("Returning %d realistic metadata rows", len(expanded))
    return expanded
