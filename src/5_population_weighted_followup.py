"""Population-weighted RV follow-up summaries for the paper.

This script assumes the photometric self-lensing candidate population already
exists. It transfers the recovery response measured from the fitted RV follow-up
grid onto the population-synthesis table using nearest neighbours in physical
parameter space, then writes paper-ready tables and figures.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy
import pandas
import seaborn
from matplotlib import pyplot
from matplotlib.lines import Line2D

try:
    from fastparquet import ParquetFile
except ImportError:  # pragma: no cover - dependency is declared by project.
    ParquetFile = None

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only in stripped envs.
    cKDTree = None

from utils import DATA_PATH, configure_logging


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd()
FIGURE_DIR = PROJECT_ROOT / "paper" / "figures"
TABLE_DIR = PROJECT_ROOT / "paper" / "tables"
POPULATION_PATH = DATA_PATH / "SW_std_sldata.pkl.gz"
POPULATION_CACHE_DIR = DATA_PATH / "population_cache"
POPULATION_CACHE_PATH = POPULATION_CACHE_DIR / "followup_population_cache.parquet"
DETECTED_SAMPLE_CACHE_DIR = PROJECT_ROOT / "paper" / "tables" / "cache"
DETECTED_SAMPLE_CACHE_TEMPLATE = "detected_followup_sample_{sample_size}.parquet"
FITTED_PATH = DATA_PATH / "fitted_timeseries.gz"
PRISTINE_METADATA_PATH = DATA_PATH / "pristine_rv" / "metadata.csv"

RECOVERY_THRESHOLD = 0.10
DEFAULT_K = 25
TRANSFER_CHUNK_SIZE = int(os.environ.get("SLRV_POPULATION_TRANSFER_CHUNK_SIZE", 10000))
DEFAULT_DETECTED_SAMPLE_SIZE = int(
    os.environ.get("SLRV_DETECTED_POPULATION_SAMPLE_SIZE", 10000)
)
R_SUN_PER_AU = 215.032
EPSILON = 1.0e-12
LENS_MASS_BINS = [0.0, 3.0, 5.0, 20.0, 50.0, numpy.inf]
LENS_MASS_LABELS = [
    r"$<3\,M_\odot$",
    r"$3$--$5\,M_\odot$",
    r"$5$--$20\,M_\odot$",
    r"$20$--$50\,M_\odot$",
    r"$\geq50\,M_\odot$",
]
PERIOD_BINS = [0.0, 365.25, 3652.5, 36525.0, numpy.inf]
PERIOD_LABELS = ["<1 yr", "1-10 yr", "10-100 yr", ">=100 yr"]
PRECISION_TIER_LABELS = {
    0.1: "0.1 m/s",
    1.0: "1 m/s",
    10.0: "10 m/s",
    100.0: "100 m/s",
    1000.0: "1 km/s",
}
SURVEY_DETECTED_COUNTS = {
    # Wiktorowicz et al. (2021), Table 2, standard-IMF n_sl values.
    "ZTF": 4027.97,
    "LSST": 10284.91,
    "TESS": 123.28,
}

COLUMN_ALIASES = {
    "Macc": [
        "Macc",
        "macc",
        "M_lens",
        "m_lens",
        "lens_mass",
        "Mco",
        "M_CO",
        "m_CO",
        "Mrem",
        "M_rem",
        "mass_lens",
    ],
    "Porb": [
        "Porb",
        "porb",
        "P_orb",
        "p_orb",
        "orbital_period",
        "period",
        "P",
        "P_days",
        "porb_days",
    ],
    "a": [
        "a",
        "sma",
        "semi_major_axis",
        "semi-major_axis",
        "semi_major",
        "A",
    ],
    "e": ["e", "ecc", "eccentricity"],
    "b": ["b", "impact_parameter", "impact", "u0"],
    "i": ["i", "inclination", "inc"],
    "weight": [
        "weight",
        "weights",
        "pop_weight",
        "population_weight",
        "yield",
        "N",
        "n",
        "number",
        "rate",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate population-weighted RV follow-up paper outputs."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Alias for --detected-sample-size.",
    )
    parser.add_argument(
        "--detected-sample-size",
        type=int,
        default=DEFAULT_DETECTED_SAMPLE_SIZE,
        help=(
            "Number of rows to draw from the weighted detected-system "
            "distribution. The sample is scaled to the expected detectable "
            "system count."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of nearest pristine simulations used for response transfer.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the legacy all-population cache from the gzipped pickle.",
    )
    parser.add_argument(
        "--all-population-cache",
        action="store_true",
        help=(
            "Aggregate the full cached population response. This is much slower "
            "than the default detected-system sample and is not the paper "
            "default."
        ),
    )
    return parser.parse_args()


def scalar_number(value):
    if value is None:
        return numpy.nan
    try:
        array = numpy.asarray(value)
        if array.shape == ():
            return float(array)
        if array.size == 1:
            return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return numpy.nan


def numeric_scalar_series(values):
    return pandas.to_numeric(values.map(scalar_number), errors="coerce")


def first_existing_column(table, aliases):
    columns = {column.lower(): column for column in table.columns}
    for alias in aliases:
        if alias in table.columns:
            return alias
        if alias.lower() in columns:
            return columns[alias.lower()]
    return None


def mapped_columns(table, required):
    mapping = {}
    missing = []
    for name in required:
        column = first_existing_column(table, COLUMN_ALIASES[name])
        if column is None:
            missing.append(name)
        else:
            mapping[name] = column
    if missing:
        available = ", ".join(map(str, table.columns[:200]))
        raise KeyError(
            "Could not map population columns for "
            f"{', '.join(missing)}. Available columns include: {available}"
        )
    return mapping


def read_population_source():
    if not POPULATION_PATH.exists():
        raise FileNotFoundError(f"Population table not found at {POPULATION_PATH}")
    logger.info("Reading population table from %s", POPULATION_PATH)
    population = pandas.read_pickle(POPULATION_PATH)
    if not isinstance(population, pandas.DataFrame):
        raise TypeError(
            "Expected the population pickle to contain a pandas DataFrame, "
            f"got {type(population)!r}"
        )
    logger.info("Loaded population table with shape %s", population.shape)
    return population


def period_from_a_days(a_au, total_mass_msun):
    years = numpy.sqrt(numpy.asarray(a_au, dtype=float) ** 3 / total_mass_msun)
    return 365.25 * years


def normalise_population_table(population):
    mapping = mapped_columns(population, ["Macc"])
    optional_mapping = {
        name: first_existing_column(population, aliases)
        for name, aliases in COLUMN_ALIASES.items()
        if name not in mapping
    }
    logger.info(
        "Population column mapping: %s",
        {
            **mapping,
            **{
                name: column
                for name, column in optional_mapping.items()
                if column is not None
            },
        },
    )

    normalised = pandas.DataFrame()
    normalised["population_id"] = numpy.arange(len(population), dtype=numpy.int64)
    normalised["Macc"] = numeric_scalar_series(population[mapping["Macc"]]).to_numpy()

    for name in ("Porb", "a", "e", "b", "i"):
        column = optional_mapping.get(name)
        if column is not None:
            normalised[name] = numeric_scalar_series(population[column]).to_numpy()

    if "a" in normalised:
        finite_a = normalised["a"][numpy.isfinite(normalised["a"])]
        if not finite_a.empty and finite_a.median() > 500.0:
            logger.info("Converting population semi-major axis from R_sun to AU")
            normalised["a"] = normalised["a"] / R_SUN_PER_AU

    if "Porb" not in normalised and "a" in normalised:
        total_mass = normalised["Macc"].copy()
        companion_column = first_existing_column(
            population,
            ["Mcomp", "mcomp", "M_star", "m_star", "companion_mass", "star_mass"],
        )
        if companion_column is not None:
            total_mass = total_mass + numeric_scalar_series(population[companion_column]).to_numpy()
        normalised["Porb"] = period_from_a_days(normalised["a"], total_mass)

    if "e" not in normalised:
        normalised["e"] = 0.0
    if "b" not in normalised and "i" in normalised:
        normalised["b"] = numpy.abs(numpy.cos(normalised["i"].astype(float)))

    weight_column = optional_mapping.get("weight")
    if weight_column is None:
        normalised["weight"] = 1.0
        normalised["weight_source"] = "unit"
    else:
        normalised["weight"] = numeric_scalar_series(population[weight_column]).to_numpy()
        normalised["weight_source"] = weight_column

    normalised = normalised.replace([numpy.inf, -numpy.inf], numpy.nan)
    normalised = normalised.dropna(subset=["Macc", "Porb", "e", "weight"]).copy()
    normalised = normalised[normalised["weight"] > 0].copy()
    normalised["weight"] = normalised["weight"].astype(float)
    normalised["lens_mass_range"] = pandas.cut(
        normalised["Macc"],
        bins=LENS_MASS_BINS,
        labels=LENS_MASS_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)
    normalised["period_range"] = pandas.cut(
        normalised["Porb"],
        bins=PERIOD_BINS,
        labels=PERIOD_LABELS,
        include_lowest=True,
        right=False,
    ).astype(str)

    required = ["Macc", "Porb", "e", "weight"]
    missing = [column for column in required if column not in normalised]
    if missing:
        raise KeyError(f"Normalised population table is missing {missing}")
    if normalised.empty:
        raise ValueError("Normalised population table is empty")
    return normalised


def load_population(rebuild_cache=False, sample_size=None):
    if POPULATION_CACHE_PATH.exists() and not rebuild_cache:
        logger.info("Reading cached population table from %s", POPULATION_CACHE_PATH)
        population = pandas.read_parquet(POPULATION_CACHE_PATH)
    else:
        raw_population = read_population_source()
        if sample_size is not None and sample_size < len(raw_population):
            raw_population = raw_population.sample(sample_size, random_state=42)
            logger.info(
                "Using deterministic raw population sample of %d rows before cache build",
                sample_size,
            )
        population = normalise_population_table(raw_population)
        if sample_size is None:
            POPULATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Writing population cache to %s", POPULATION_CACHE_PATH)
            population.to_parquet(POPULATION_CACHE_PATH, index=False)
        else:
            logger.info("Skipping persistent cache write for sample-mode run")

    if sample_size is not None and sample_size < len(population):
        population = population.sample(sample_size, random_state=42).reset_index(drop=True)
        logger.info("Using deterministic sample of %d population rows", sample_size)
    return population


def load_detected_population(sample_size):
    detected_cache_path = None
    if sample_size is not None:
        detected_cache_path = DETECTED_SAMPLE_CACHE_DIR / DETECTED_SAMPLE_CACHE_TEMPLATE.format(
            sample_size=sample_size,
        )
    if detected_cache_path is not None and detected_cache_path.exists():
        logger.info("Reading cached detected population sample from %s", detected_cache_path)
        return pandas.read_parquet(detected_cache_path)

    raw_population = read_population_source()
    required_columns = {"visible", "weight"}
    missing = sorted(required_columns.difference(raw_population.columns))
    if missing:
        raise KeyError(
            "Population table is missing detectable-system columns: "
            f"{missing}. Available columns: {list(raw_population.columns)}"
        )

    detected = raw_population[
        raw_population["visible"].eq(1)
        & raw_population["weight"].gt(0)
    ].copy()
    expected_detected_count = float(detected["weight"].sum())
    logger.info(
        "Identified %d detectable population rows representing %.3f expected systems",
        len(detected),
        expected_detected_count,
    )
    if detected.empty or expected_detected_count <= 0:
        raise ValueError("No positive-weight detectable population rows were found")

    if sample_size is not None and sample_size < len(detected):
        detected = detected.sample(
            n=sample_size,
            replace=True,
            weights="weight",
            random_state=42,
        )
        logger.info(
            "Using weighted detected-population sample of %d rows",
            sample_size,
        )

    population = normalise_population_table(detected)
    population["weight"] = expected_detected_count / len(population)
    population["weight_source"] = "detected_weighted_sample_scaled"
    if detected_cache_path is not None:
        DETECTED_SAMPLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Writing detected population sample cache to %s", detected_cache_path)
        population.to_parquet(detected_cache_path, index=False)
    return population


def relative_error(truth, fitted):
    truth = numpy.asarray(truth, dtype=float)
    fitted = numpy.asarray(fitted, dtype=float)
    scale = numpy.abs(truth).copy()
    scale[~numpy.isfinite(scale) | (scale == 0)] = 1.0
    return numpy.abs(fitted - truth) / scale


def compact_object_class(masses):
    return pandas.cut(
        masses,
        bins=LENS_MASS_BINS[:-1],
        labels=LENS_MASS_LABELS[:-1],
        include_lowest=True,
        right=False,
    )


def load_fitted_results():
    if not FITTED_PATH.exists():
        raise FileNotFoundError(f"Fitted results not found at {FITTED_PATH}")
    results = pandas.read_pickle(FITTED_PATH).copy()
    required = {
        "pristine_id",
        "baseline",
        "cadence",
        "RV accuracy",
        "Macc",
        "Porb",
        "fitted_Macc",
        "fitted_porb",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise KeyError(f"Fitted results table is missing columns: {missing}")
    for column in required.difference({"pristine_id"}):
        results[column] = numeric_scalar_series(results[column])
    results = results.dropna(subset=list(required)).copy()
    results["classified"] = (
        compact_object_class(results["Macc"]).notna()
        & compact_object_class(results["fitted_Macc"]).notna()
        & (compact_object_class(results["Macc"]) == compact_object_class(results["fitted_Macc"]))
    )
    results["measured_mass"] = (
        relative_error(results["Macc"], results["fitted_Macc"]) <= RECOVERY_THRESHOLD
    )
    results["measured_period"] = (
        relative_error(results["Porb"], results["fitted_porb"]) <= RECOVERY_THRESHOLD
    )
    results["measured_mass_and_period"] = (
        results["measured_mass"] & results["measured_period"]
    )
    return results


def load_pristine_metadata():
    if not PRISTINE_METADATA_PATH.exists():
        raise FileNotFoundError(f"Pristine metadata not found at {PRISTINE_METADATA_PATH}")
    metadata = pandas.read_csv(PRISTINE_METADATA_PATH)
    required = {"simulation_id", "Macc", "Porb", "e"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise KeyError(f"Pristine metadata table is missing columns: {missing}")
    for column in ("Macc", "Porb", "e", "b", "i"):
        if column in metadata:
            metadata[column] = numeric_scalar_series(metadata[column])
    return metadata


def feature_table(table):
    features = pandas.DataFrame(index=table.index)
    features["log_Macc"] = numpy.log10(table["Macc"].astype(float))
    features["log_Porb"] = numpy.log10(table["Porb"].astype(float))
    features["e"] = table["e"].astype(float)
    if "b" in table.columns and table["b"].notna().any():
        features["b"] = table["b"].astype(float)
    elif "i" in table.columns and table["i"].notna().any():
        features["b"] = numpy.abs(numpy.cos(table["i"].astype(float)))
    else:
        features["b"] = 0.0
    finite = numpy.isfinite(features).all(axis=1)
    return features[finite], finite


def scaled_feature_arrays(pristine, population):
    pristine_features, pristine_mask = feature_table(pristine)
    population_features, population_mask = feature_table(population)
    columns = ["log_Macc", "log_Porb", "e", "b"]
    scale_source = pandas.concat(
        [pristine_features[columns], population_features[columns]], ignore_index=True
    )
    center = scale_source.median()
    scale = scale_source.quantile(0.84) - scale_source.quantile(0.16)
    scale = scale.replace(0, numpy.nan).fillna(scale_source.std()).replace(0, 1.0)
    return (
        ((pristine_features[columns] - center) / scale).to_numpy(dtype=float),
        ((population_features[columns] - center) / scale).to_numpy(dtype=float),
        pristine.loc[pristine_mask].copy(),
        population.loc[population_mask].copy(),
    )


def feature_scaler(reference):
    features, mask = feature_table(reference)
    columns = ["log_Macc", "log_Porb", "e", "b"]
    center = features[columns].median()
    scale = features[columns].quantile(0.84) - features[columns].quantile(0.16)
    scale = scale.replace(0, numpy.nan).fillna(features[columns].std()).replace(0, 1.0)
    return center, scale, reference.loc[mask].copy()


def apply_feature_scaler(table, center, scale):
    features, mask = feature_table(table)
    columns = ["log_Macc", "log_Porb", "e", "b"]
    scaled = ((features[columns] - center) / scale).to_numpy(dtype=float)
    return scaled, table.loc[mask].copy()


def nearest_neighbour_weights(pristine_features, population_features, k):
    k = min(k, len(pristine_features))
    if cKDTree is not None:
        tree = cKDTree(pristine_features)
        distances, indices = tree.query(population_features, k=k, workers=-1)
    else:
        logger.warning("scipy is unavailable; using slower chunked nearest neighbours")
        indices = []
        distances = []
        for start in range(0, len(population_features), 2048):
            chunk = population_features[start : start + 2048]
            diff = chunk[:, None, :] - pristine_features[None, :, :]
            dist = numpy.sqrt(numpy.sum(diff * diff, axis=2))
            idx = numpy.argpartition(dist, kth=k - 1, axis=1)[:, :k]
            row = numpy.arange(len(chunk))[:, None]
            order = numpy.argsort(dist[row, idx], axis=1)
            indices.append(idx[row, order])
            distances.append(dist[row, idx][row, order])
        indices = numpy.vstack(indices)
        distances = numpy.vstack(distances)

    if k == 1:
        indices = indices[:, None]
        distances = distances[:, None]
    weights = 1.0 / (distances + EPSILON)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return indices, weights


def response_by_strategy(results):
    response = (
        results.groupby(["baseline", "cadence", "RV accuracy", "pristine_id"], dropna=False)
        .agg(
            classified=("classified", "mean"),
            measured_mass=("measured_mass", "mean"),
            measured_period=("measured_period", "mean"),
            measured_mass_and_period=("measured_mass_and_period", "mean"),
        )
        .reset_index()
    )
    return response


def transferred_rows(population, pristine, response, neighbour_indices, neighbour_weights):
    pristine_ids = pristine["simulation_id"].astype(str).to_numpy()
    neighbour_ids = pristine_ids[neighbour_indices]
    strategies = response[["baseline", "cadence", "RV accuracy"]].drop_duplicates()
    rows = []
    for _, strategy in strategies.sort_values(["baseline", "cadence", "RV accuracy"]).iterrows():
        strategy_response = response[
            (response["baseline"] == strategy["baseline"])
            & (response["cadence"] == strategy["cadence"])
            & (response["RV accuracy"] == strategy["RV accuracy"])
        ].set_index("pristine_id")

        metric_arrays = {}
        coverage = None
        for metric in (
            "classified",
            "measured_mass",
            "measured_period",
            "measured_mass_and_period",
        ):
            metric_by_id = strategy_response[metric].reindex(pristine_ids).to_numpy(dtype=float)
            valid = numpy.isfinite(metric_by_id[neighbour_indices])
            valid_weight = numpy.sum(neighbour_weights * valid, axis=1)
            if coverage is None:
                coverage = valid_weight
            values = numpy.where(valid, metric_by_id[neighbour_indices], 0.0)
            with numpy.errstate(invalid="ignore", divide="ignore"):
                metric_arrays[metric] = numpy.sum(
                    values * neighbour_weights,
                    axis=1,
                ) / valid_weight

        rows.append(
            pandas.DataFrame(
                {
                    "population_id": population["population_id"].to_numpy(),
                    "weight": population["weight"].to_numpy(dtype=float),
                    "lens_mass_range": population["lens_mass_range"].to_numpy(),
                    "period_range": population["period_range"].to_numpy(),
                    "baseline": float(strategy["baseline"]),
                    "baseline_years": float(strategy["baseline"]) / 365.0,
                    "cadence": float(strategy["cadence"]),
                    "n_rv": int(numpy.ceil(float(strategy["baseline"]) / float(strategy["cadence"]))),
                    "RV accuracy": float(strategy["RV accuracy"]),
                    "precision_tier": PRECISION_TIER_LABELS.get(
                        float(strategy["RV accuracy"]),
                        f"{float(strategy['RV accuracy']):g} m/s",
                    ),
                    "response_coverage": coverage,
                    "nearest_pristine_ids": [";".join(ids) for ids in neighbour_ids],
                    **metric_arrays,
                }
            )
        )
    transferred = pandas.concat(rows, ignore_index=True)
    probability_columns = [
        "classified",
        "measured_mass",
        "measured_period",
        "measured_mass_and_period",
    ]
    probabilities = transferred[probability_columns].to_numpy(dtype=float)
    if numpy.isfinite(probabilities).any() and (
        (numpy.nanmin(probabilities) < -1.0e-9)
        or (numpy.nanmax(probabilities) > 1.0 + 1.0e-9)
    ):
        raise ValueError("Transferred recovery probabilities are outside [0, 1]")
    if transferred["response_coverage"].lt(-1.0e-9).any() or transferred[
        "response_coverage"
    ].gt(1.0 + 1.0e-9).any():
        raise ValueError("Transferred response coverage is outside [0, 1]")
    if not numpy.isfinite(probabilities).any():
        raise ValueError("Transferred recovery probabilities contain non-finite values")
    transferred[probability_columns] = transferred[probability_columns].clip(0.0, 1.0)
    return transferred


def weighted_average(values, weights):
    values = numpy.asarray(values, dtype=float)
    weights = numpy.asarray(weights, dtype=float)
    total = numpy.sum(weights)
    if total <= 0:
        return numpy.nan
    return numpy.sum(values * weights) / total


def weighted_average_finite(values, weights):
    values = numpy.asarray(values, dtype=float)
    weights = numpy.asarray(weights, dtype=float)
    finite = numpy.isfinite(values) & numpy.isfinite(weights) & (weights > 0)
    total = numpy.sum(weights[finite])
    denominator = numpy.sum(weights[numpy.isfinite(weights) & (weights > 0)])
    if total <= 0:
        return numpy.nan, 0.0, numpy.nan
    coverage = total / denominator if denominator > 0 else numpy.nan
    return numpy.sum(values[finite] * weights[finite]) / total, total, coverage


def rv_accuracy_tick_labels(values):
    return [f"{float(value):g}" for value in values]


def summarise_transferred(transferred):
    metric_columns = [
        "classified",
        "measured_mass",
        "measured_period",
        "measured_mass_and_period",
    ]
    group_columns = [
        "baseline",
        "baseline_years",
        "cadence",
        "n_rv",
        "RV accuracy",
        "precision_tier",
    ]
    rows = []
    for keys, group in transferred.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_population_rows"] = group["population_id"].nunique()
        row["population_weight"] = group["weight"].sum()
        row["response_coverage_fraction"] = weighted_average(
            group["response_coverage"],
            group["weight"],
        )
        for metric in metric_columns:
            fraction, denominator_weight, _ = weighted_average_finite(
                group[metric],
                group["weight"],
            )
            row[f"{metric}_denominator_weight"] = denominator_weight
            row[f"{metric}_fraction"] = fraction
            row[f"{metric}_weighted_count"] = fraction * denominator_weight
        rows.append(row)
    summary = pandas.DataFrame(rows).sort_values(group_columns)
    return summary


def summarise_precision_tiers(transferred):
    metric_columns = [
        "classified",
        "measured_mass",
        "measured_period",
        "measured_mass_and_period",
    ]
    rows = []
    group_columns = ["precision_tier", "RV accuracy", "lens_mass_range", "period_range"]
    for keys, group in transferred.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["population_weight"] = group["weight"].sum()
        row["response_coverage_fraction"] = weighted_average(
            group["response_coverage"],
            group["weight"],
        )
        for metric in metric_columns:
            fraction, denominator_weight, _ = weighted_average_finite(
                group[metric],
                group["weight"],
            )
            row[f"{metric}_denominator_weight"] = denominator_weight
            row[f"{metric}_fraction"] = fraction
            row[f"{metric}_weighted_count"] = fraction * denominator_weight
        rows.append(row)
    return pandas.DataFrame(rows).sort_values(["RV accuracy", "lens_mass_range", "period_range"])


def aggregate_transferred(population, pristine, response, neighbour_indices, neighbour_weights):
    """Aggregate transferred recovery probabilities without materialising rows."""
    pristine_ids = pristine["simulation_id"].astype(str).to_numpy()
    strategies = response[["baseline", "cadence", "RV accuracy"]].drop_duplicates()
    metric_columns = [
        "classified",
        "measured_mass",
        "measured_period",
        "measured_mass_and_period",
    ]
    weights = population["weight"].to_numpy(dtype=float)
    population_weight = float(weights.sum())
    if population_weight <= 0:
        raise ValueError("Population weight must be positive")

    summary_rows = []
    tier_rows = []
    group_columns = ["lens_mass_range", "period_range"]
    population_groups = population[group_columns].copy()
    population_groups["_row"] = numpy.arange(len(population_groups))

    for _, strategy in strategies.sort_values(["baseline", "cadence", "RV accuracy"]).iterrows():
        strategy_response = response[
            (response["baseline"] == strategy["baseline"])
            & (response["cadence"] == strategy["cadence"])
            & (response["RV accuracy"] == strategy["RV accuracy"])
        ].set_index("pristine_id")

        metric_probabilities = {}
        response_coverage = None
        for metric in metric_columns:
            metric_by_id = strategy_response[metric].reindex(pristine_ids).to_numpy(dtype=float)
            neighbour_values = metric_by_id[neighbour_indices]
            valid = numpy.isfinite(neighbour_values)
            valid_weight = numpy.sum(neighbour_weights * valid, axis=1)
            if response_coverage is None:
                response_coverage = valid_weight
            with numpy.errstate(invalid="ignore", divide="ignore"):
                values = numpy.sum(
                    numpy.where(valid, neighbour_values, 0.0) * neighbour_weights,
                    axis=1,
                ) / valid_weight
            if numpy.isfinite(values).any() and (
                (numpy.nanmin(values) < -1.0e-9)
                or (numpy.nanmax(values) > 1.0 + 1.0e-9)
            ):
                raise ValueError(f"Transferred {metric} probabilities are outside [0, 1]")
            metric_probabilities[metric] = numpy.clip(values, 0.0, 1.0)
        if response_coverage is None or not numpy.isfinite(response_coverage).all():
            raise ValueError("Transferred response coverage contains non-finite values")
        if ((response_coverage < -1.0e-9) | (response_coverage > 1.0 + 1.0e-9)).any():
            raise ValueError("Transferred response coverage is outside [0, 1]")

        baseline = float(strategy["baseline"])
        cadence = float(strategy["cadence"])
        rv_accuracy = float(strategy["RV accuracy"])
        base_row = {
            "baseline": baseline,
            "baseline_years": baseline / 365.0,
            "cadence": cadence,
            "n_rv": int(numpy.ceil(baseline / cadence)),
            "RV accuracy": rv_accuracy,
            "precision_tier": PRECISION_TIER_LABELS.get(
                rv_accuracy,
                f"{rv_accuracy:g} m/s",
            ),
            "n_population_rows": len(population),
            "population_weight": population_weight,
            "response_coverage_fraction": weighted_average(response_coverage, weights),
        }
        summary_row = dict(base_row)
        for metric, values in metric_probabilities.items():
            fraction, denominator_weight, _ = weighted_average_finite(values, weights)
            summary_row[f"{metric}_denominator_weight"] = denominator_weight
            summary_row[f"{metric}_fraction"] = fraction
            summary_row[f"{metric}_weighted_count"] = fraction * denominator_weight
        summary_rows.append(summary_row)

        for keys, group in population_groups.groupby(group_columns, dropna=False):
            row_indices = group["_row"].to_numpy(dtype=int)
            group_weights = weights[row_indices]
            tier_row = {
                **base_row,
                "n_population_rows": len(row_indices),
                "lens_mass_range": keys[0],
                "period_range": keys[1],
                "population_weight": float(group_weights.sum()),
                "response_coverage_fraction": weighted_average(
                    response_coverage[row_indices],
                    group_weights,
                ),
            }
            for metric, values in metric_probabilities.items():
                fraction, denominator_weight, _ = weighted_average_finite(
                    values[row_indices],
                    group_weights,
                )
                tier_row[f"{metric}_denominator_weight"] = denominator_weight
                tier_row[f"{metric}_fraction"] = fraction
                tier_row[f"{metric}_weighted_count"] = (
                    fraction * denominator_weight
                )
            tier_rows.append(tier_row)

    summary = pandas.DataFrame(summary_rows).sort_values(
        ["baseline", "cadence", "RV accuracy"]
    )
    precision_tiers = pandas.DataFrame(tier_rows).sort_values(
        ["baseline", "cadence", "RV accuracy", "lens_mass_range", "period_range"]
    )
    return summary, precision_tiers


def response_arrays_by_strategy(response, pristine_ids):
    strategies = response[["baseline", "cadence", "RV accuracy"]].drop_duplicates()
    metric_columns = [
        "classified",
        "measured_mass",
        "measured_period",
        "measured_mass_and_period",
    ]
    strategy_rows = []
    for _, strategy in strategies.sort_values(["baseline", "cadence", "RV accuracy"]).iterrows():
        strategy_response = response[
            (response["baseline"] == strategy["baseline"])
            & (response["cadence"] == strategy["cadence"])
            & (response["RV accuracy"] == strategy["RV accuracy"])
        ].set_index("pristine_id")
        baseline = float(strategy["baseline"])
        cadence = float(strategy["cadence"])
        rv_accuracy = float(strategy["RV accuracy"])
        strategy_rows.append(
            {
                "baseline": baseline,
                "baseline_years": baseline / 365.0,
                "cadence": cadence,
                "n_rv": int(numpy.ceil(baseline / cadence)),
                "RV accuracy": rv_accuracy,
                "precision_tier": PRECISION_TIER_LABELS.get(
                    rv_accuracy,
                    f"{rv_accuracy:g} m/s",
                ),
                "metrics": {
                    metric: strategy_response[metric]
                    .reindex(pristine_ids)
                    .to_numpy(dtype=float)
                    for metric in metric_columns
                },
            }
        )
    return strategy_rows, metric_columns


def empty_accumulator_row(strategy):
    return {
        "baseline": strategy["baseline"],
        "baseline_years": strategy["baseline_years"],
        "cadence": strategy["cadence"],
        "n_rv": strategy["n_rv"],
        "RV accuracy": strategy["RV accuracy"],
        "precision_tier": strategy["precision_tier"],
        "n_population_rows": 0,
        "population_weight": 0.0,
        "response_coverage_weighted_sum": 0.0,
        "metric_weighted_sums": {},
        "metric_denominator_weights": {},
    }


def finalise_accumulators(summary_accumulators, tier_accumulators, metric_columns):
    summary_rows = []
    for row in summary_accumulators.values():
        output = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "metric_weighted_sums",
                "metric_denominator_weights",
                "response_coverage_weighted_sum",
            }
        }
        output["response_coverage_fraction"] = (
            row["response_coverage_weighted_sum"] / row["population_weight"]
            if row["population_weight"] > 0
            else numpy.nan
        )
        for metric in metric_columns:
            weighted_sum = row["metric_weighted_sums"].get(metric, 0.0)
            denominator_weight = row["metric_denominator_weights"].get(metric, 0.0)
            fraction = (
                weighted_sum / denominator_weight
                if denominator_weight > 0
                else numpy.nan
            )
            output[f"{metric}_denominator_weight"] = denominator_weight
            output[f"{metric}_fraction"] = fraction
            output[f"{metric}_weighted_count"] = weighted_sum
        summary_rows.append(output)

    tier_rows = []
    for row in tier_accumulators.values():
        output = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "metric_weighted_sums",
                "metric_denominator_weights",
                "response_coverage_weighted_sum",
            }
        }
        output["response_coverage_fraction"] = (
            row["response_coverage_weighted_sum"] / row["population_weight"]
            if row["population_weight"] > 0
            else numpy.nan
        )
        for metric in metric_columns:
            weighted_sum = row["metric_weighted_sums"].get(metric, 0.0)
            denominator_weight = row["metric_denominator_weights"].get(metric, 0.0)
            fraction = (
                weighted_sum / denominator_weight
                if denominator_weight > 0
                else numpy.nan
            )
            output[f"{metric}_denominator_weight"] = denominator_weight
            output[f"{metric}_fraction"] = fraction
            output[f"{metric}_weighted_count"] = weighted_sum
        tier_rows.append(output)

    summary = pandas.DataFrame(summary_rows).sort_values(
        ["baseline", "cadence", "RV accuracy"]
    )
    precision_tiers = pandas.DataFrame(tier_rows).sort_values(
        ["baseline", "cadence", "RV accuracy", "lens_mass_range", "period_range"]
    )
    return summary, precision_tiers


def iter_population_cache_batches(path):
    if ParquetFile is None:
        logger.warning("fastparquet is unavailable; reading population cache in one batch")
        yield pandas.read_parquet(path)
        return
    parquet = ParquetFile(path)
    for row_group in parquet.iter_row_groups():
        yield row_group


def aggregate_population_cache(pristine, response, k):
    if not POPULATION_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Population cache not found at {POPULATION_CACHE_PATH}. "
            "Run without --sample-size once to build it."
        )
    center, scale, pristine = feature_scaler(pristine)
    pristine_features, pristine = apply_feature_scaler(pristine, center, scale)
    pristine_ids = pristine["simulation_id"].astype(str).to_numpy()
    if cKDTree is None:
        tree = None
    else:
        tree = cKDTree(pristine_features)
    strategy_rows, metric_columns = response_arrays_by_strategy(response, pristine_ids)
    output_specs = []
    response_columns = []
    for strategy in strategy_rows:
        strategy_key = (
            strategy["baseline"],
            strategy["cadence"],
            strategy["RV accuracy"],
        )
        for metric in metric_columns:
            output_specs.append((strategy_key, strategy, metric))
            response_columns.append(strategy["metrics"][metric])
    response_matrix = numpy.column_stack(response_columns)
    summary_accumulators = {}
    tier_accumulators = {}

    for batch_index, batch in enumerate(iter_population_cache_batches(POPULATION_CACHE_PATH), start=1):
        batch_features, batch = apply_feature_scaler(batch, center, scale)
        if batch.empty:
            continue
        weights = batch["weight"].to_numpy(dtype=float)
        logger.info(
            "Aggregating population cache row group %d with %d rows",
            batch_index,
            len(batch),
        )

        group_frame = pandas.DataFrame(
            {
                "lens_mass_range": batch["lens_mass_range"].to_numpy(),
                "period_range": batch["period_range"].to_numpy(),
                "_row": numpy.arange(len(batch), dtype=int),
            }
        )
        grouped_rows = list(group_frame.groupby(["lens_mass_range", "period_range"], dropna=False))

        query_k = min(k, len(pristine_features))
        for start in range(0, len(batch), TRANSFER_CHUNK_SIZE):
            stop = min(start + TRANSFER_CHUNK_SIZE, len(batch))
            feature_chunk = batch_features[start:stop]
            weight_chunk = weights[start:stop]
            if tree is None:
                neighbour_indices, neighbour_weights = nearest_neighbour_weights(
                    pristine_features,
                    feature_chunk,
                    k,
                )
            else:
                distances, neighbour_indices = tree.query(
                    feature_chunk,
                    k=query_k,
                    workers=-1,
                )
                if query_k == 1:
                    neighbour_indices = neighbour_indices[:, None]
                    distances = distances[:, None]
                neighbour_weights = 1.0 / (distances + EPSILON)
                neighbour_weights = neighbour_weights / neighbour_weights.sum(axis=1, keepdims=True)

            neighbour_response = response_matrix[neighbour_indices]
            valid_response = numpy.isfinite(neighbour_response)
            coverage = numpy.sum(neighbour_weights[:, :, None] * valid_response, axis=1)
            with numpy.errstate(invalid="ignore", divide="ignore"):
                probabilities = numpy.sum(
                    numpy.where(valid_response, neighbour_response, 0.0)
                    * neighbour_weights[:, :, None],
                    axis=1,
                ) / coverage
            if numpy.isfinite(probabilities).any() and (
                (numpy.nanmin(probabilities) < -1.0e-9)
                or (numpy.nanmax(probabilities) > 1.0 + 1.0e-9)
            ):
                raise ValueError("Transferred probabilities are outside [0, 1]")
            probabilities = numpy.clip(probabilities, 0.0, 1.0)
            finite_probability = numpy.isfinite(probabilities)
            weighted_sums = numpy.nansum(probabilities * weight_chunk[:, None], axis=0)
            denominator_sums = finite_probability.T @ weight_chunk
            coverage_sums = coverage.T @ weight_chunk
            chunk_weight = float(weight_chunk.sum())

            for output_index, (strategy_key, strategy, metric) in enumerate(output_specs):
                summary_row = summary_accumulators.setdefault(
                    strategy_key,
                    empty_accumulator_row(strategy),
                )
                if output_index % len(metric_columns) == 0:
                    summary_row["n_population_rows"] += stop - start
                    summary_row["population_weight"] += chunk_weight
                    summary_row["response_coverage_weighted_sum"] += float(
                        coverage_sums[output_index]
                    )
                summary_row["metric_weighted_sums"][metric] = (
                    summary_row["metric_weighted_sums"].get(metric, 0.0)
                    + float(weighted_sums[output_index])
                )
                summary_row["metric_denominator_weights"][metric] = (
                    summary_row["metric_denominator_weights"].get(metric, 0.0)
                    + float(denominator_sums[output_index])
                )

            for (lens_mass_range, period_range), group in grouped_rows:
                row_indices = group["_row"].to_numpy(dtype=int)
                mask = (row_indices >= start) & (row_indices < stop)
                if not mask.any():
                    continue
                local_indices = row_indices[mask] - start
                group_weights = weight_chunk[local_indices]
                group_weight = float(group_weights.sum())
                group_probabilities = probabilities[local_indices]
                group_finite = numpy.isfinite(group_probabilities)
                group_weighted_sums = numpy.nansum(
                    group_probabilities * group_weights[:, None],
                    axis=0,
                )
                group_denominator_sums = group_finite.T @ group_weights
                group_coverage_sums = coverage[local_indices].T @ group_weights
                for output_index, (strategy_key, strategy, metric) in enumerate(output_specs):
                    tier_key = (
                        strategy["baseline"],
                        strategy["cadence"],
                        strategy["RV accuracy"],
                        lens_mass_range,
                        period_range,
                    )
                    tier_row = tier_accumulators.setdefault(
                        tier_key,
                        {
                            **empty_accumulator_row(strategy),
                            "lens_mass_range": lens_mass_range,
                            "period_range": period_range,
                        },
                    )
                    if output_index % len(metric_columns) == 0:
                        tier_row["n_population_rows"] += len(local_indices)
                        tier_row["population_weight"] += group_weight
                        tier_row["response_coverage_weighted_sum"] += float(
                            group_coverage_sums[output_index]
                        )
                    tier_row["metric_weighted_sums"][metric] = (
                        tier_row["metric_weighted_sums"].get(metric, 0.0)
                        + float(group_weighted_sums[output_index])
                    )
                    tier_row["metric_denominator_weights"][metric] = (
                        tier_row["metric_denominator_weights"].get(metric, 0.0)
                        + float(group_denominator_sums[output_index])
                    )

    return finalise_accumulators(
        summary_accumulators,
        tier_accumulators,
        metric_columns,
    )


def write_latex_summary(summary):
    table = summary.copy()
    for column in [
        "response_coverage_fraction",
        "classified_fraction",
        "measured_mass_fraction",
        "measured_period_fraction",
        "measured_mass_and_period_fraction",
    ]:
        table[column] = 100.0 * table[column]
    table = table.rename(
        columns={
            "baseline_years": "Baseline [yr]",
            "cadence": "Cadence [d]",
            "n_rv": r"$N_\mathrm{RV}$",
            "precision_tier": "RV precision",
            "response_coverage_fraction": "Coverage [per cent]",
            "classified_fraction": "Classified [per cent]",
            "measured_mass_fraction": r"$M_\mathrm{acc}$ [per cent]",
            "measured_period_fraction": r"$P_\mathrm{orb}$ [per cent]",
            "measured_mass_and_period_fraction": "Both [per cent]",
        }
    )
    table = table[
        [
            "Baseline [yr]",
            "Cadence [d]",
            r"$N_\mathrm{RV}$",
            "RV precision",
            "Coverage [per cent]",
            "Classified [per cent]",
            r"$M_\mathrm{acc}$ [per cent]",
            r"$P_\mathrm{orb}$ [per cent]",
            "Both [per cent]",
        ]
    ]
    headers = list(table.columns)
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        (
            r"\caption{Population-weighted RV follow-up recovery fractions for "
            r"an already photometrically detected self-lensing candidate sample.}"
        ),
        r"\label{tab:population-weighted-followup-summary}",
        r"\begin{tabular}{rrrrrrrrr}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, (float, numpy.floating)):
                values.append(f"{value:.1f}")
            elif isinstance(value, (int, numpy.integer)):
                values.append(str(value))
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    tex = "\n".join(lines)
    (TABLE_DIR / "population_weighted_followup_summary.tex").write_text(tex)


def summarise_bh_like_followup(precision_tiers):
    bh_labels = {
        r"$5$--$20\,M_\odot$",
        r"$20$--$50\,M_\odot$",
        r"$\geq50\,M_\odot$",
    }
    metric_columns = [
        "classified",
        "measured_mass",
        "measured_period",
        "measured_mass_and_period",
    ]
    bh_tiers = precision_tiers[precision_tiers["lens_mass_range"].isin(bh_labels)].copy()
    if bh_tiers.empty:
        logger.warning("No BH-like population rows were found for the BH summary")
        return pandas.DataFrame()

    rows = []
    group_columns = [
        "baseline",
        "baseline_years",
        "cadence",
        "n_rv",
        "RV accuracy",
        "precision_tier",
    ]
    for keys, group in bh_tiers.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row["n_population_rows"] = group["n_population_rows"].sum()
        row["population_weight"] = group["population_weight"].sum()
        row["response_coverage_fraction"] = weighted_average(
            group["response_coverage_fraction"],
            group["population_weight"],
        )
        for metric in metric_columns:
            denominator = group[f"{metric}_denominator_weight"].sum()
            weighted_count = group[f"{metric}_weighted_count"].sum()
            row[f"{metric}_denominator_weight"] = denominator
            row[f"{metric}_weighted_count"] = weighted_count
            row[f"{metric}_fraction"] = (
                weighted_count / denominator if denominator > 0 else numpy.nan
            )
        rows.append(row)
    return pandas.DataFrame(rows).sort_values(group_columns)


def write_latex_bh_summary(summary):
    if summary.empty:
        return
    table = summary.copy()
    for column in [
        "response_coverage_fraction",
        "classified_fraction",
        "measured_mass_fraction",
        "measured_period_fraction",
        "measured_mass_and_period_fraction",
    ]:
        table[column] = 100.0 * table[column]
    table = table.rename(
        columns={
            "baseline_years": "Baseline [yr]",
            "cadence": "Cadence [d]",
            "n_rv": r"$N_\mathrm{RV}$",
            "precision_tier": "RV precision",
            "population_weight": r"$N_\mathrm{BH-like}$",
            "response_coverage_fraction": "Coverage [per cent]",
            "classified_fraction": "Classified [per cent]",
            "measured_mass_fraction": r"$M_\mathrm{acc}$ [per cent]",
            "measured_period_fraction": r"$P_\mathrm{orb}$ [per cent]",
            "measured_mass_and_period_fraction": "Both [per cent]",
        }
    )
    table = table[
        [
            "Baseline [yr]",
            "Cadence [d]",
            r"$N_\mathrm{RV}$",
            "RV precision",
            r"$N_\mathrm{BH-like}$",
            "Coverage [per cent]",
            "Classified [per cent]",
            r"$M_\mathrm{acc}$ [per cent]",
            r"$P_\mathrm{orb}$ [per cent]",
            "Both [per cent]",
        ]
    ]
    headers = list(table.columns)
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        (
            r"\caption{Population-weighted RV follow-up recovery fractions for "
            r"the BH-like subset of already photometrically detected "
            r"self-lensing candidates.}"
        ),
        r"\label{tab:population-weighted-bh-followup-summary}",
        r"\begin{tabular}{rrrrrrrrrr}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    for _, row in table.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, (float, numpy.floating)):
                values.append(f"{value:.1f}")
            elif isinstance(value, (int, numpy.integer)):
                values.append(str(value))
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}", ""])
    (TABLE_DIR / "population_weighted_bh_followup_summary.tex").write_text(
        "\n".join(lines)
    )


def plot_followup_heatmaps(summary):
    metrics = [
        ("classified_fraction", "Correct class"),
        ("measured_mass_fraction", r"$M_\mathrm{acc}$"),
        ("measured_period_fraction", r"$P_\mathrm{orb}$"),
        ("measured_mass_and_period_fraction", "Mass + period"),
    ]
    baselines = numpy.sort(summary["baseline"].unique())
    fig, axes = pyplot.subplots(
        len(metrics),
        len(baselines),
        figsize=(8.8, 7.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    last_mesh = None
    for row_index, (metric, label) in enumerate(metrics):
        for col_index, baseline in enumerate(baselines):
            axis = axes[row_index, col_index]
            subset = summary[summary["baseline"] == baseline]
            matrix = subset.pivot(
                index="cadence",
                columns="RV accuracy",
                values=metric,
            ).sort_index().sort_index(axis=1)
            heatmap = seaborn.heatmap(
                100.0 * matrix,
                ax=axis,
                vmin=0,
                vmax=100,
                cmap="viridis",
                annot=True,
                annot_kws={"fontsize": 6.0},
                fmt=".0f",
                square=True,
                cbar=False,
            )
            last_mesh = heatmap.collections[0]
            if row_index == 0:
                axis.set_title(f"{baseline / 365:g} yr")
            if col_index == 0:
                axis.set_ylabel(f"{label}\nCadence [d]")
            else:
                axis.set_ylabel("")
            if row_index == len(metrics) - 1:
                axis.set_xlabel(r"RV accuracy [m s$^{-1}$]")
                axis.set_xticklabels(
                    rv_accuracy_tick_labels(matrix.columns),
                    rotation=40,
                    ha="right",
                )
            else:
                axis.set_xlabel("")
                axis.set_xticklabels([])
    if last_mesh is not None:
        fig.colorbar(last_mesh, ax=axes, location="right", shrink=0.82, label="Weighted recovery [per cent]")
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"population_weighted_rv_followup_heatmaps.{suffix}",
            bbox_inches="tight",
        )
    pyplot.close(fig)


def survey_scaled_followup_counts(summary):
    metric_columns = [
        "classified",
        "measured_mass",
        "measured_period",
        "measured_mass_and_period",
    ]
    rows = []
    for survey, detected_count in SURVEY_DETECTED_COUNTS.items():
        survey_rows = summary.copy()
        survey_rows.insert(0, "survey", survey)
        survey_rows.insert(1, "survey_detected_count", detected_count)
        for metric in metric_columns:
            survey_rows[f"{metric}_survey_count"] = (
                survey_rows[f"{metric}_fraction"] * detected_count
            )
        rows.append(survey_rows)
    return pandas.concat(rows, ignore_index=True)


def plot_mass_period_recovery(precision_tiers):
    rv_order = [1000.0, 100.0, 10.0, 1.0, 0.1]
    rv_labels = [PRECISION_TIER_LABELS[value] for value in rv_order]
    best_strategy = precision_tiers[
        (precision_tiers["baseline"] == precision_tiers["baseline"].max())
        & (precision_tiers["cadence"] == precision_tiers["cadence"].min())
    ].copy()
    if best_strategy.empty:
        raise ValueError("No data available for mass/period recovery plot")

    metrics = [
        ("measured_mass_fraction", r"$M_\mathrm{acc}$"),
        ("measured_period_fraction", r"$P_\mathrm{orb}$"),
    ]
    fig, axes = pyplot.subplots(
        len(metrics),
        len(PERIOD_LABELS),
        figsize=(7.2, 4.7),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    colors = seaborn.color_palette("colorblind", n_colors=len(LENS_MASS_LABELS))
    color_map = dict(zip(LENS_MASS_LABELS, colors))
    markers = ["o", "s", "^", "D", "v"]
    marker_map = dict(zip(LENS_MASS_LABELS, markers))
    x = numpy.arange(len(rv_order), dtype=float)

    for row_index, (metric, metric_label) in enumerate(metrics):
        for col_index, period_label in enumerate(PERIOD_LABELS):
            axis = axes[row_index, col_index]
            period_data = best_strategy[best_strategy["period_range"] == period_label]
            for lens_mass_label in LENS_MASS_LABELS:
                mass_data = period_data[
                    period_data["lens_mass_range"] == lens_mass_label
                ].set_index("RV accuracy")
                y = 100.0 * mass_data.reindex(rv_order)[metric].to_numpy(dtype=float)
                axis.plot(
                    x,
                    y,
                    color=color_map[lens_mass_label],
                    marker=marker_map[lens_mass_label],
                    linewidth=1.1,
                    markersize=3.0,
                )
            if row_index == 0:
                axis.set_title(period_label)
            if col_index == 0:
                axis.set_ylabel(f"{metric_label}\nrecovery [per cent]")
            axis.set_ylim(-3, 103)
            axis.set_xticks(x)
            if row_index == len(metrics) - 1:
                axis.set_xticklabels(rv_labels, rotation=35, ha="right")
                axis.set_xlabel(r"RV accuracy")
            else:
                axis.set_xticklabels([])
            axis.grid(True, axis="y", alpha=0.25)

    fig.subplots_adjust(top=0.80, bottom=0.18, left=0.08, right=0.99, wspace=0.12, hspace=0.16)
    mass_handles = [
        Line2D(
            [0],
            [0],
            color=color_map[label],
            marker=marker_map[label],
            linewidth=1.2,
            markersize=3.5,
            label=label,
        )
        for label in LENS_MASS_LABELS
    ]
    fig.legend(
        handles=mass_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURE_DIR / f"population_weighted_mass_period_recovery.{suffix}",
            bbox_inches="tight",
        )
    pyplot.close(fig)


def validate_weight_conservation(transferred):
    actual = transferred.groupby(
        ["baseline", "cadence", "RV accuracy"], dropna=False
    )["weight"].sum()
    if not numpy.allclose(actual.to_numpy(), actual.iloc[0]):
        raise ValueError("Population weights were not conserved through transfer")


def main():
    configure_logging()
    logger.setLevel(logging.INFO)
    args = parse_args()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    pristine = load_pristine_metadata()
    fitted = load_fitted_results()
    response = response_by_strategy(fitted)

    detected_sample_size = args.sample_size or args.detected_sample_size
    if args.all_population_cache:
        if args.rebuild_cache or not POPULATION_CACHE_PATH.exists():
            population = load_population(rebuild_cache=args.rebuild_cache)
            del population
        summary, precision_tiers = aggregate_population_cache(
            pristine,
            response,
            args.k,
        )
    else:
        population = load_detected_population(detected_sample_size)
        pristine_features, population_features, pristine, population = scaled_feature_arrays(
            pristine,
            population,
        )
        neighbour_indices, neighbour_weights = nearest_neighbour_weights(
            pristine_features,
            population_features,
            args.k,
        )
        summary, precision_tiers = aggregate_transferred(
            population,
            pristine,
            response,
            neighbour_indices,
            neighbour_weights,
        )
    summary.to_csv(TABLE_DIR / "population_weighted_followup_summary.csv", index=False)
    precision_tiers.to_csv(
        TABLE_DIR / "population_weighted_precision_tiers.csv",
        index=False,
    )
    bh_summary = summarise_bh_like_followup(precision_tiers)
    bh_summary.to_csv(
        TABLE_DIR / "population_weighted_bh_followup_summary.csv",
        index=False,
    )
    survey_counts = survey_scaled_followup_counts(summary)
    survey_counts.to_csv(
        TABLE_DIR / "population_weighted_followup_survey_counts.csv",
        index=False,
    )
    write_latex_summary(summary)
    write_latex_bh_summary(bh_summary)
    plot_mass_period_recovery(precision_tiers)
    logger.info("Wrote population-weighted follow-up tables and figures")


if __name__ == "__main__":
    main()
