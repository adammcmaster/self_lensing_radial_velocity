from pathlib import Path
import uuid

import numpy
import pandas
from astropy.io import fits
from astropy.table import Table

from utils import DATA_PATH


PARAMS = {
    "cadences": [30, 150, 300],
    "campaign_baselines": 365 * numpy.array([1, 3, 5]),
    "rv_accuracies": numpy.logspace(-1, 3, 50),
}

REALISTIC_PATH = (DATA_PATH / "realistic").resolve()
REALISTIC_PATH.mkdir(exist_ok=True, parents=True)
REALISTIC_METADATA_PATH = REALISTIC_PATH / "metadata.csv"
REALISTIC_ID_NAMESPACE = uuid.UUID("1d34c442-474c-5d23-83ff-c121e5ed4a73")

METADATA_COLUMNS = (
    "realistic_id",
    "realistic_path",
    "pristine_idx",
    "pristine_id",
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
    "RV accuracy",
    "cadence",
    "baseline",
    "rv_err",
    "lc_err",
)


def realistic_paths(path=REALISTIC_PATH):
    return sorted(Path(path).glob("*.fits.gz"))


def realistic_key_from_values(pristine_id, cadence, rv_accuracy, baseline):
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
    return str(uuid.uuid5(REALISTIC_ID_NAMESPACE, "|".join(map(str, key))))


def realistic_path_from_params(pristine, cadence, rv_accuracy, baseline):
    realistic_id = realistic_id_from_params(pristine, cadence, rv_accuracy, baseline)
    return REALISTIC_PATH / f"{realistic_id}.fits.gz"


def existing_realistic_keys(table=None):
    if table is None:
        table = load_realistic_timeseries()
    if table.empty:
        return set()
    return {realistic_key(row) for _, row in table.iterrows()}


def _empty_metadata_table():
    return pandas.DataFrame(columns=METADATA_COLUMNS)


def _read_metadata_cache(path=REALISTIC_METADATA_PATH):
    if not path.exists():
        return _empty_metadata_table()
    return pandas.read_csv(path)


def _write_metadata_cache(table, path=REALISTIC_METADATA_PATH):
    path.parent.mkdir(exist_ok=True, parents=True)
    table.to_csv(path, index=False)


def _normalise_metadata_table(table):
    if table.empty:
        return _empty_metadata_table()

    table = table.reindex(columns=METADATA_COLUMNS)
    table["realistic_path"] = table["realistic_path"].map(
        lambda p: str(Path(p).resolve())
    )
    table.drop_duplicates(subset="realistic_path", keep="last", inplace=True)
    table.sort_values("realistic_path", inplace=True)
    table.reset_index(drop=True, inplace=True)
    return table


def _metadata_from_header(header, path):
    return {
        "realistic_id": header["REAL_ID"],
        "realistic_path": str(path.resolve()),
        "pristine_idx": header["PRIS_IDX"],
        "pristine_id": header["PRIS_ID"],
        "pristine_rv_path": header["PRIS_RV"],
        "photometry_path": header["PHOTPATH"],
        "a": header["A"],
        "e": header["E"],
        "periapsis_phase": header["PERI"],
        "Macc": header["MBH"],
        "Porb": header["P"],
        "tau_eff": header["TAU_EFF"],
        "Mcomp": header["MSTAR"],
        "Rcomp": header["RSTAR"],
        "Teff": header["TEFF"],
        "b": header["B"],
        "i": header["I"],
        "peak_time": header["PEAK_T"],
        "RV accuracy": header["RV_ACC"],
        "cadence": header["CADENCE"],
        "baseline": header["BASELINE"],
        "rv_err": header["RV_ERR"],
        "lc_err": header["LC_ERR"],
    }


def metadata_row_from_realistic_path(path):
    path = Path(path)
    with fits.open(path) as hdul:
        return _metadata_from_header(hdul[1].header, path)


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
        "lc_err": lc_err,
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
        "RV_ERR": float(row["rv_err"]),
        "LC_ERR": float(row["lc_err"]),
    }


def write_realistic_timeseries(
    row,
    rv_times,
    rv,
    rv_err,
    lc_times,
    lc,
    lc_err,
):
    path = Path(row["realistic_path"])
    if path.exists():
        return

    path.parent.mkdir(exist_ok=True, parents=True)

    rv_table = Table()
    rv_table["time"] = rv_times
    rv_table["time"].unit = "d"
    rv_table["rv"] = rv
    rv_table["rv"].unit = "m / s"
    rv_table["rv_err"] = numpy.full(len(rv), rv_err)
    rv_table["rv_err"].unit = "m / s"
    rv_table.meta.update(_metadata_header(row))

    lc_table = Table()
    lc_table["time"] = lc_times
    lc_table["time"].unit = "d"
    lc_table["lc"] = lc
    lc_table["lc_err"] = numpy.full(len(lc), lc_err)
    lc_table.meta.update(_metadata_header(row))

    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.table_to_hdu(rv_table),
            fits.table_to_hdu(lc_table),
        ]
    )
    hdul[1].name = "RV"
    hdul[2].name = "LC"
    hdul.writeto(path)


def load_realistic_timeseries_curve(row_or_path):
    if isinstance(row_or_path, (pandas.Series, dict)):
        path = row_or_path["realistic_path"]
    else:
        path = row_or_path

    with fits.open(path) as hdul:
        rv_data = hdul["RV"].data
        lc_data = hdul["LC"].data
        return {
            "rv_times": numpy.asarray(rv_data["time"], dtype=float),
            "rv": numpy.asarray(rv_data["rv"], dtype=float),
            "rv_err": numpy.asarray(rv_data["rv_err"], dtype=float),
            "lc_times": numpy.asarray(lc_data["time"], dtype=float),
            "lc": numpy.asarray(lc_data["lc"], dtype=float),
            "lc_err": numpy.asarray(lc_data["lc_err"], dtype=float),
        }


def update_realistic_metadata(rows):
    if not rows:
        return load_realistic_timeseries()

    cached = _normalise_metadata_table(_read_metadata_cache())
    new_table = pandas.DataFrame(rows)
    if cached.empty:
        table = new_table
    else:
        table = pandas.concat([cached, new_table], ignore_index=True)

    table = _normalise_metadata_table(table)
    _write_metadata_cache(table)
    return table


def load_realistic_timeseries():
    realistic_path_set = {str(path) for path in realistic_paths()}
    cached = _normalise_metadata_table(_read_metadata_cache())
    if not cached.empty:
        cached = cached[cached["realistic_path"].isin(realistic_path_set)].copy()

    cached_paths = set(cached["realistic_path"]) if not cached.empty else set()
    missing_paths = [
        path for path in realistic_path_set if path not in cached_paths
    ]

    if missing_paths:
        new_rows = [metadata_row_from_realistic_path(path) for path in missing_paths]
        new_table = pandas.DataFrame(new_rows)
        if cached.empty:
            cached = new_table
        else:
            cached = pandas.concat([cached, new_table], ignore_index=True)

    table = _normalise_metadata_table(cached)
    if missing_paths or len(table) != len(realistic_path_set):
        _write_metadata_cache(table)

    return table
