import logging
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

logger = logging.getLogger(__name__)

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
    paths = sorted(Path(path).glob("*.fits.gz"))
    logger.debug("Found %d photometry simulation paths in %s", len(paths), path)
    return paths


def pristine_rv_paths(path=PRISTINE_RV_PATH):
    paths = sorted(Path(path).glob("*.fits.gz"))
    logger.debug("Found %d pristine RV paths in %s", len(paths), path)
    return paths


def _metadata_from_header(header):
    metadata = {key: header[key] for key in PARAMETER_KEYS if key in header}
    logger.debug("Loaded %d metadata values from FITS header", len(metadata))
    return metadata


def _jd_from_fits_time(time_column):
    time = numpy.asarray(time_column)
    if time.ndim == 2:
        logger.debug("Converting 2D FITS time column with shape %s to JD", time.shape)
        return time.sum(axis=1)
    logger.debug("Converting 1D FITS time column with shape %s to float JD", time.shape)
    return time.astype(float)


def _self_lensing_system_from_metadata(metadata):
    logger.debug(
        "Building SelfLensingSystem from metadata: A=%s, I=%s, MBH=%s, E=%s, PERI=%s",
        metadata.get("A"),
        metadata.get("I"),
        metadata.get("MBH"),
        metadata.get("E"),
        metadata.get("PERI"),
    )
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
        logger.debug("Skipping existing pristine RV curve at %s", path)
        return

    logger.debug("Writing pristine RV curve to %s from %s", path, source_path)
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
        logger.debug("RV FITS header has no SOURCE value")
        return None
    path = PHOTOMETRY_SIMULATION_PATH / source
    logger.debug("Resolved photometry source %s to %s", source, path)
    return path


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
    logger.debug("Loading pristine metadata row from %s", rv_path)
    with fits.open(rv_path) as hdul:
        header = hdul[1].header
        metadata = _metadata_from_header(header)
        photometry_path = _photometry_path_from_rv_header(header)
    return _metadata_row(metadata, rv_path, photometry_path)


def _empty_metadata_table():
    return pandas.DataFrame(columns=METADATA_COLUMNS)


def _read_metadata_cache(path=PRISTINE_METADATA_PATH):
    if not path.exists():
        logger.debug("Pristine metadata cache does not exist at %s", path)
        return _empty_metadata_table()
    table = pandas.read_csv(path)
    logger.debug("Read %d pristine metadata rows from %s", len(table), path)
    return table


def _write_metadata_cache(table, path=PRISTINE_METADATA_PATH):
    path.parent.mkdir(exist_ok=True, parents=True)
    logger.debug("Writing %d pristine metadata rows to %s", len(table), path)
    table.to_csv(path, index=False)


def _normalise_metadata_table(table):
    if table.empty:
        logger.debug("Normalising empty pristine metadata table")
        return _empty_metadata_table()

    before = len(table)
    table = table.reindex(columns=METADATA_COLUMNS)
    table["rv_path"] = table["rv_path"].map(lambda p: str(Path(p)))
    table.drop_duplicates(subset="rv_path", keep="last", inplace=True)
    table.sort_values("rv_path", inplace=True)
    table.reset_index(drop=True, inplace=True)
    logger.debug("Normalised pristine metadata table from %d to %d rows", before, len(table))
    return table


def load_pristine_rv_curve(row_or_path):
    """Load one pristine RV curve from its canonical FITS file."""
    rv_path = (
        row_or_path["rv_path"]
        if isinstance(row_or_path, pandas.Series)
        else row_or_path
    )
    logger.debug("Loading pristine RV curve from %s", rv_path)
    with fits.open(rv_path) as hdul:
        data = hdul[1].data
        logger.debug("Loaded pristine RV curve with %d samples", len(data))
        return {
            "rv_times": numpy.asarray(data["time"], dtype=float),
            "rv": numpy.asarray(data["rv"], dtype=float),
        }


def load_pristine_lightcurve(row_or_path):
    """Load the source photometric light curve for one pristine RV curve."""
    if isinstance(row_or_path, pandas.Series):
        photometry_path = row_or_path["photometry_path"]
        logger.debug("Loading pristine lightcurve from metadata path %s", photometry_path)
    else:
        logger.debug("Resolving pristine lightcurve path from RV file %s", row_or_path)
        with fits.open(row_or_path) as hdul:
            photometry_path = _photometry_path_from_rv_header(hdul[1].header)

    if photometry_path is None:
        logger.debug("Cannot load pristine lightcurve because no photometry path is recorded")
        raise ValueError("No source photometry path is recorded for this RV file")

    logger.debug("Opening pristine photometry FITS at %s", photometry_path)
    with fits.open(photometry_path) as hdul:
        data = hdul[1].data
        # Shift flux to mean=0
        flux = numpy.asarray(data["flux"], dtype=float)
        original_mean = numpy.mean(flux)
        flux -= numpy.mean(flux)
        logger.debug(
            "Loaded lightcurve with %d samples and shifted flux mean from %.6g to %.6g",
            len(flux),
            original_mean,
            numpy.mean(flux),
        )
        return {
            "lc_times": _jd_from_fits_time(data["time"]),
            "lc": flux,
            "lc_err": numpy.asarray(data["flux_err"], dtype=float),
        }


def calc_rv(photometry_path, save_curve=True):
    photometry_path = Path(photometry_path)
    rv_path = PRISTINE_RV_PATH / photometry_path.name
    logger.debug("Calculating pristine RV for %s; save_curve=%s", photometry_path, save_curve)
    if save_curve and rv_path.exists():
        logger.debug("Pristine RV already exists at %s; loading metadata", rv_path)
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

    logger.debug("Finished pristine RV calculation for %s", photometry_path)
    return _metadata_row(metadata, rv_path, photometry_path)


def write_pristine_rv_curve(photometry_path):
    calc_rv(photometry_path, save_curve=True)


def generate_pristine_timeseries(desc=None, photometry_path=PHOTOMETRY_SIMULATION_PATH):
    logger.debug("Generating pristine timeseries from %s", photometry_path)
    all_paths = photometry_simulation_paths(photometry_path)
    if not all_paths:
        raise FileNotFoundError(
            f"No photometric simulations found in {photometry_path}"
        )

    paths = [path for path in all_paths if not (PRISTINE_RV_PATH / path.name).exists()]
    logger.debug(
        "Pristine generation has %d total source paths and %d missing RV curves",
        len(all_paths),
        len(paths),
    )
    if not paths:
        logger.debug("No pristine RV curves need to be generated")
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
    logger.debug("Loading pristine timeseries metadata for %d RV files", len(rv_paths))

    cached = _normalise_metadata_table(_read_metadata_cache())
    if not cached.empty:
        cached = cached[cached["rv_path"].isin(rv_path_set)].copy()

    cached_paths = set(cached["rv_path"]) if not cached.empty else set()
    missing_paths = [path for path in rv_paths if path not in cached_paths]
    logger.debug("Pristine metadata cache is missing %d paths", len(missing_paths))

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

    logger.debug("Returning %d pristine metadata rows", len(table))
    return table
