from pathlib import Path
import warnings

warnings.filterwarnings("ignore", module="erfa.core")

import numpy
import pandas
from astropy import units
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from tqdm.contrib.concurrent import process_map

from selflensing.generators import RVGenerator
from selflensing.models import SelfLensingSystem

from utils import DATA_PATH, default_rv_times

PHOTOMETRY_SIMULATION_PATH = Path.home() / "Code/lc-filter/data/simulations/1.0/lensing"
PRISTINE_RV_PATH = DATA_PATH / "pristine_rv"
PRISTINE_RV_PATH.mkdir(exist_ok=True, parents=True)
PRISTINE_METADATA_PATH = PRISTINE_RV_PATH / "metadata.csv"

PARAMETER_KEYS = (
    "A",
    "I",
    "MBH",
    "RSTAR",
    "MSTAR",
    "TEFF",
    "E",
    "PERI",
    "D",
    "P",
    "B",
    "TAU_EFF",
    "TAU_SL",
    "PEAK_T",
    "AMP_SL",
    "AMP_EL",
    "AMP_DB",
    "MEAN_CV",
    "VIS",
)

METADATA_COLUMNS = (
    "simulation_id",
    "photometry_path",
    "rv_path",
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
    "source_Porb",
    "source_tau_eff",
    "source_b",
    "source_distance",
    "source_amp_sl",
    "source_amp_el",
    "source_amp_db",
    "source_mean_cv",
    "source_visibility",
)


def photometry_simulation_paths(path=PHOTOMETRY_SIMULATION_PATH):
    return sorted(Path(path).glob("*.fits.gz"))


def pristine_rv_paths(path=PRISTINE_RV_PATH):
    return sorted(Path(path).glob("*.fits.gz"))


def _metadata_from_header(header):
    return {key: header[key] for key in PARAMETER_KEYS if key in header}


def _jd_from_fits_time(time_column):
    time = numpy.asarray(time_column)
    if time.ndim == 2:
        return time.sum(axis=1)
    return time.astype(float)


def _self_lensing_system_from_metadata(metadata):
    return SelfLensingSystem(
        a=metadata["A"] * units.au,
        i=metadata["I"] * units.rad,
        Macc=metadata["MBH"] * units.M_sun,
        e=metadata["E"] * units.dimensionless_unscaled,
        periapsis_phase=metadata["PERI"] * units.rad,
        Mcomp=metadata["MSTAR"] * units.M_sun,
        Rcomp=metadata["RSTAR"] * units.R_sun,
        Teff=metadata["TEFF"] * units.K,
    )


def _write_rv_curve(path, metadata, rvg, source_path):
    if path.exists():
        return

    table = Table()
    table["time"] = rvg.times.jd
    table["time"].unit = "d"
    table["rv"] = rvg.rv.to_value(units.m / units.s)
    table["rv"].unit = "m / s"

    table.meta.update(metadata)
    table.meta["SOURCE"] = source_path.name
    table.meta["RV_UNIT"] = "m / s"
    table.meta["TIMESYS"] = "UTC"
    table.meta["JDREF"] = 0.0

    path.parent.mkdir(exist_ok=True, parents=True)
    table.write(path)


def _photometry_path_from_rv_header(header):
    source = header.get("SOURCE")
    if source is None:
        return None
    return PHOTOMETRY_SIMULATION_PATH / source


def _metadata_row(metadata, rv_path, photometry_path=None):
    return {
        "simulation_id": Path(rv_path).stem.removesuffix(".fits"),
        "photometry_path": str(photometry_path) if photometry_path else None,
        "rv_path": str(rv_path),
        "a": metadata["A"],
        "e": metadata["E"],
        "periapsis_phase": metadata["PERI"],
        "Macc": metadata["MBH"],
        "Porb": metadata["P"],
        "tau_eff": metadata["TAU_EFF"],
        "Mcomp": metadata["MSTAR"],
        "Rcomp": metadata["RSTAR"],
        "Teff": metadata["TEFF"],
        "b": metadata["B"],
        "i": metadata["I"],
        "peak_time": metadata["PEAK_T"],
        "source_Porb": metadata.get("P"),
        "source_tau_eff": metadata.get("TAU_EFF"),
        "source_b": metadata.get("B"),
        "source_distance": metadata.get("D"),
        "source_amp_sl": metadata.get("AMP_SL"),
        "source_amp_el": metadata.get("AMP_EL"),
        "source_amp_db": metadata.get("AMP_DB"),
        "source_mean_cv": metadata.get("MEAN_CV"),
        "source_visibility": metadata.get("VIS"),
    }


def metadata_row_from_rv_path(rv_path):
    rv_path = Path(rv_path)
    with fits.open(rv_path) as hdul:
        header = hdul[1].header
        metadata = _metadata_from_header(header)
        photometry_path = _photometry_path_from_rv_header(header)
    return _metadata_row(metadata, rv_path, photometry_path)


def _empty_metadata_table():
    return pandas.DataFrame(columns=METADATA_COLUMNS)


def _read_metadata_cache(path=PRISTINE_METADATA_PATH):
    if not path.exists():
        return _empty_metadata_table()
    return pandas.read_csv(path)


def _write_metadata_cache(table, path=PRISTINE_METADATA_PATH):
    path.parent.mkdir(exist_ok=True, parents=True)
    table.to_csv(path, index=False)


def _normalise_metadata_table(table):
    if table.empty:
        return _empty_metadata_table()

    table = table.reindex(columns=METADATA_COLUMNS)
    table["rv_path"] = table["rv_path"].map(lambda p: str(Path(p)))
    table.drop_duplicates(subset="rv_path", keep="last", inplace=True)
    table.sort_values("rv_path", inplace=True)
    table.reset_index(drop=True, inplace=True)
    return table


def load_pristine_rv_curve(row_or_path):
    """Load one pristine RV curve from its canonical FITS file."""
    rv_path = (
        row_or_path["rv_path"]
        if isinstance(row_or_path, pandas.Series)
        else row_or_path
    )
    with fits.open(rv_path) as hdul:
        data = hdul[1].data
        return {
            "rv_times": numpy.asarray(data["time"], dtype=float),
            "rv": numpy.asarray(data["rv"], dtype=float),
        }


def load_pristine_lightcurve(row_or_path):
    """Load the source photometric light curve for one pristine RV curve."""
    if isinstance(row_or_path, pandas.Series):
        photometry_path = row_or_path["photometry_path"]
    else:
        with fits.open(row_or_path) as hdul:
            photometry_path = _photometry_path_from_rv_header(hdul[1].header)

    if photometry_path is None:
        raise ValueError("No source photometry path is recorded for this RV file")

    with fits.open(photometry_path) as hdul:
        data = hdul[1].data
        # Shift flux to mean=0
        flux = numpy.asarray(data["flux"], dtype=float)
        flux -= numpy.mean(flux)
        return {
            "lc_times": _jd_from_fits_time(data["time"]),
            "lc": flux,
            "lc_err": numpy.asarray(data["flux_err"], dtype=float),
        }


def calc_rv(photometry_path, save_curve=True):
    photometry_path = Path(photometry_path)
    rv_path = PRISTINE_RV_PATH / photometry_path.name
    if save_curve and rv_path.exists():
        return metadata_row_from_rv_path(rv_path)

    with fits.open(photometry_path) as hdul:
        header = hdul[1].header
        metadata = _metadata_from_header(header)

    sls = _self_lensing_system_from_metadata(metadata)
    peak_time = Time(metadata["PEAK_T"], format="jd")
    rvg = RVGenerator(
        sls,
        times=default_rv_times,
        peak_time=peak_time,
    )

    if save_curve:
        _write_rv_curve(rv_path, metadata, rvg, photometry_path)

    return _metadata_row(metadata, rv_path, photometry_path)


def write_pristine_rv_curve(photometry_path):
    calc_rv(photometry_path, save_curve=True)


def generate_pristine_timeseries(desc=None, photometry_path=PHOTOMETRY_SIMULATION_PATH):
    all_paths = photometry_simulation_paths(photometry_path)
    if not all_paths:
        raise FileNotFoundError(
            f"No photometric simulations found in {photometry_path}"
        )

    paths = [path for path in all_paths if not (PRISTINE_RV_PATH / path.name).exists()]
    if not paths:
        return

    process_map(
        write_pristine_rv_curve,
        paths,
        desc=desc,
    )


def load_pristine_timeseries():
    """Return metadata and paths for pristine RV files, without loading curve arrays."""
    rv_paths = [str(path) for path in pristine_rv_paths()]
    rv_path_set = set(rv_paths)

    cached = _normalise_metadata_table(_read_metadata_cache())
    if not cached.empty:
        cached = cached[cached["rv_path"].isin(rv_path_set)].copy()

    cached_paths = set(cached["rv_path"]) if not cached.empty else set()
    missing_paths = [path for path in rv_paths if path not in cached_paths]

    if missing_paths:
        new_rows = [metadata_row_from_rv_path(path) for path in missing_paths]
        new_table = pandas.DataFrame(new_rows)
        if cached.empty:
            cached = new_table
        else:
            cached = pandas.concat([cached, new_table], ignore_index=True)

    table = _normalise_metadata_table(cached)
    if missing_paths or len(table) != len(rv_paths):
        _write_metadata_cache(table)

    return table
