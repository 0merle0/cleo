"""Utility functions for parsing and processing plate reader kinetics data.

This module provides helpers for working with data exported from BioTek Neo2
plate readers. The typical workflow is:

    1. Parse raw instrument files (mScarlett endpoint + 4-MU kinetics).
    2. Apply standard-curve corrections to convert fluorescence → concentration.
    3. Merge with plate-map metadata (well → variant name, sample type, etc.).
    4. Fit kinetics (linear regression on concentration vs. time).
    5. Normalize rates by expression proxy (mScarlett concentration).
    6. Aggregate replicates, apply QC filters, and export.

Functions are intentionally self-contained so they can be copied and adapted for
different instrument formats without depending on the rest of the CLEO codebase.
"""

import datetime
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# File-name metadata extraction
# ---------------------------------------------------------------------------

def get_sample_plate_info(filepath: str) -> tuple[int, int]:
    """Extract (sample_number, plate_number) from a file name.

    Expects the file name to contain substrings like ``sample0`` and ``plate3``.

    Args:
        filepath: Full path to the instrument data file.

    Returns:
        Tuple of (sample_number, plate_number) parsed from the file name.

    Raises:
        ValueError: If either substring is missing from the file name.
    """
    filename = os.path.basename(filepath)

    match = re.search(r"plate(\d+)", filename)
    if match:
        plate_number = int(match.group(1))
    else:
        raise ValueError(
            f"No plate number found in '{filename}'. "
            "Include 'plate#' in the file name."
        )

    match = re.search(r"sample(\d+)", filename)
    if match:
        sample_number = int(match.group(1))
    else:
        raise ValueError(
            f"No sample number found in '{filename}'. "
            "Include 'sample#' in the file name."
        )

    return sample_number, plate_number


# ---------------------------------------------------------------------------
# Raw instrument parsing
# ---------------------------------------------------------------------------

def parse_neo2_endpoint(
    filepath: str,
    start_row: int = 54,
    parse_name: bool = False,
) -> pd.DataFrame:
    """Parse a BioTek Neo2 TSV file containing a single endpoint read.

    The Neo2 exports a block of metadata rows followed by a single row of
    well-level measurements. This function extracts the measurement row,
    pivots it into tidy format (one row per well), and optionally parses
    plate/sample metadata from the file name.

    Args:
        filepath: Path to the ``.txt`` export file.
        start_row: 0-indexed row where the measurement data begins. The
            default (54) works for a typical mScarlett endpoint protocol;
            adjust if the instrument header differs.
        parse_name: If ``True``, extract date, plate number, and sample
            number from the file name and add them as columns.

    Returns:
        Tidy DataFrame with columns ``well``, ``value``, ``serial_number``,
        ``datetime``, and optionally ``plate_number``, ``sample_number``,
        ``date_collected``, ``source_file``.
    """
    if parse_name:
        filename = os.path.basename(filepath)
        date = filename[0:6]
        sample_number, plate_number = get_sample_plate_info(filepath)

    header = pd.read_csv(
        filepath, encoding="cp1252", delimiter="\t", nrows=20,
    )
    serial_number = header[header.iloc[:, 0] == "Reader Serial Number:"].iloc[:, 1].values[0]
    date_val = header[header.iloc[:, 0] == "Date"].iloc[:, 1].values[0]
    time_val = header[header.iloc[:, 0] == "Time"].iloc[:, 1].values[0]

    df = pd.read_csv(
        filepath, encoding="cp1252", delimiter="\t",
        skiprows=range(1, start_row), nrows=1,
    )
    df = df.melt(id_vars=["Well"], var_name="well", value_name="value")
    df.dropna(inplace=True)
    df.drop(columns=["Well"], inplace=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    time_str = f"{date_val} {time_val}"
    df["datetime"] = datetime.datetime.strptime(time_str, "%m/%d/%Y %I:%M:%S %p").timestamp()

    df["serial_number"] = serial_number
    if parse_name:
        df["source_file"] = filepath
        df["plate_number"] = plate_number
        df["sample_number"] = sample_number
        df["date_collected"] = date
        df["time_collected"] = time_val

    return df


def parse_neo2_kinetics(
    filepath: str,
    start_row: int = 58,
    parse_name: bool = False,
) -> pd.DataFrame:
    """Parse a BioTek Neo2 TSV file containing a kinetics (time-series) read.

    Similar to :func:`parse_neo2_endpoint`, but handles multiple time-point
    rows. Each row in the raw data corresponds to one time point; this
    function melts them into tidy format (one row per well × time point).

    Args:
        filepath: Path to the ``.txt`` export file.
        start_row: 0-indexed row where the kinetics block begins. Default
            (58) is typical for a 4-MU kinetics protocol; adjust if needed.
        parse_name: If ``True``, extract metadata from the file name.

    Returns:
        Tidy DataFrame with columns ``time`` (seconds), ``well``, ``value``,
        ``serial_number``, ``datetime``, and optionally ``plate_number``,
        ``sample_number``, ``date_collected``, ``source_file``.
    """
    if parse_name:
        filename = os.path.basename(filepath)
        date = filename[0:6]
        sample_number, plate_number = get_sample_plate_info(filepath)

    header = pd.read_csv(
        filepath, encoding="cp1252", delimiter="\t", nrows=20,
    )
    serial_number = header[header.iloc[:, 0] == "Reader Serial Number:"].iloc[:, 1].values[0]
    _date = header[header.iloc[:, 0] == "Date"].iloc[:, 1].values[0]
    _time = header[header.iloc[:, 0] == "Time"].iloc[:, 1].values[0]
    _date_time = f"{_date} {_time}"

    df = pd.read_csv(
        filepath, encoding="cp1252", delimiter="\t",
        skiprows=range(1, start_row),
    )
    df = df.drop(df.columns[1], axis=1)
    df.rename(columns={"Time": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.time
    df["time"] = df["time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

    df = df.melt(id_vars=["time"], var_name="well", value_name="value")
    df.dropna(inplace=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["datetime"] = datetime.datetime.strptime(_date_time, "%m/%d/%Y %I:%M:%S %p").timestamp()

    df["serial_number"] = serial_number
    if parse_name:
        df["source_file"] = filepath
        df["plate_number"] = plate_number
        df["sample_number"] = sample_number
        df["date_collected"] = date

    return df


# ---------------------------------------------------------------------------
# Plate-map helpers
# ---------------------------------------------------------------------------

def clean_mappings_df(df_mappings: pd.DataFrame) -> pd.DataFrame:
    """Coerce plate-map columns to numeric types.

    Plate-map CSVs sometimes contain mixed types in ``replicate_number``,
    ``sample_number``, and ``column``. This function forces them to numeric,
    replacing un-parseable values with ``NaN``.

    Args:
        df_mappings: Raw plate-map DataFrame.

    Returns:
        The same DataFrame with cleaned numeric columns.
    """
    for col in ("replicate_number", "sample_number", "column"):
        df_mappings[col] = pd.to_numeric(df_mappings[col], errors="coerce")
    return df_mappings


# ---------------------------------------------------------------------------
# Standard-curve application
# ---------------------------------------------------------------------------

def apply_mscarlett_standard_curve(
    df_mscarlett: pd.DataFrame,
    standard_curve_path: str,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Convert raw mScarlett fluorescence to micromolar concentration.

    Uses a linear standard curve (slope + intercept) stored in a JSON file,
    keyed by instrument serial number. Negative concentrations are clipped
    to ``epsilon`` since they indicate below-detection-limit readings.

    Args:
        df_mscarlett: DataFrame with ``value`` and ``serial_number`` columns.
        standard_curve_path: Path to the JSON file mapping serial numbers to
            ``{"mscarlett": {"slope": ..., "y_intercept": ...}}``.
        epsilon: Floor value for negative concentrations.

    Returns:
        Input DataFrame with a new ``mscarlett_um`` column (micromolar).
    """
    with open(standard_curve_path) as f:
        standard_curves = json.load(f)

    df_mscarlett["mscarlett_um"] = df_mscarlett.apply(
        lambda x: (
            x["value"] - standard_curves[str(x["serial_number"])]["mscarlett"]["y_intercept"]
        ) / standard_curves[str(x["serial_number"])]["mscarlett"]["slope"],
        axis=1,
    )

    if epsilon:
        df_mscarlett.loc[df_mscarlett["mscarlett_um"] < 0, "mscarlett_um"] = epsilon

    return df_mscarlett


def apply_4MU_standard_curve(
    df_kinetics: pd.DataFrame,
    standard_curve_path: str,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Convert raw 4-MU fluorescence to micromolar concentration.

    Analogous to :func:`apply_mscarlett_standard_curve` but for the 4-MU
    substrate channel used in PETase kinetics assays.

    Args:
        df_kinetics: DataFrame with ``value`` and ``serial_number`` columns.
        standard_curve_path: Path to the standard-curve JSON.
        epsilon: Floor value for negative concentrations.

    Returns:
        Input DataFrame with a new ``4MU_um`` column (micromolar).
    """
    with open(standard_curve_path) as f:
        standard_curves = json.load(f)

    df_kinetics["4MU_um"] = df_kinetics.apply(
        lambda x: (
            x["value"] - standard_curves[str(x["serial_number"])]["4MU"]["y_intercept"]
        ) / standard_curves[str(x["serial_number"])]["4MU"]["slope"],
        axis=1,
    )

    if epsilon:
        df_kinetics.loc[df_kinetics["4MU_um"] < 0, "4MU_um"] = epsilon

    return df_kinetics


# ---------------------------------------------------------------------------
# High-level parse + merge
# ---------------------------------------------------------------------------

def get_mscarlett_and_kinetics(
    mappings_data_files: list[str],
    raw_data_files: list[str],
    standard_curve_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse raw data files and merge with plate-map metadata.

    This is the main entry point for loading a set of plate-reader exports.
    It:

    1. Reads and concatenates plate-map CSVs.
    2. Handles duplicate variant names by appending ``:duplicate#`` suffixes.
    3. Parses each raw ``.txt`` file for both the mScarlett endpoint and
       4-MU kinetics blocks.
    4. Merges the parsed data with the plate map on
       ``(well, sample_number, plate_number)``.
    5. Applies standard-curve corrections to convert fluorescence to
       micromolar concentrations.

    Args:
        mappings_data_files: List of plate-map CSV paths.
        raw_data_files: List of raw instrument ``.txt`` file paths.
        standard_curve_path: Path to the standard-curve JSON.

    Returns:
        Tuple of (df_mscarlett, df_kinetics), both merged with plate-map
        metadata and standard-curve corrected.
    """
    mappings_df_list = []
    mscarlett_df_list = []
    kinetics_df_list = []

    for file in mappings_data_files:
        df_mappings = pd.read_csv(file)
        df_mappings = clean_mappings_df(df_mappings)
        mappings_df_list.append(df_mappings)

    df_mappings = pd.concat(mappings_df_list)

    # Rename duplicate variant names so each source well is unique
    df_temp = df_mappings[
        (df_mappings.replicate_number == 1) & (df_mappings.sample_type == "sample")
    ]
    duplicates = df_temp[df_temp.name.duplicated()].reset_index(drop=True).name.tolist()

    for duplicate_name in duplicates:
        source_wells = df_mappings.loc[df_mappings.name == duplicate_name][
            "source_dna_well"
        ].unique()
        for i in range(len(source_wells)):
            source_well_to_change = source_wells[i]
            new_name = f"{duplicate_name}:duplicate{i}"
            df_mappings.loc[
                (df_mappings.name == duplicate_name)
                & (df_mappings.source_dna_well == source_well_to_change),
                "name",
            ] = new_name

    for file in raw_data_files:
        df_mscarlett = parse_neo2_endpoint(file, start_row=51, parse_name=True)
        mscarlett_df_list.append(df_mscarlett)

        df_kinetics = parse_neo2_kinetics(file, start_row=55, parse_name=True)
        kinetics_df_list.append(df_kinetics)

    df_mscarlett = pd.concat(mscarlett_df_list)
    df_kinetics = pd.concat(kinetics_df_list)

    df_mscarlett = df_mscarlett.merge(
        df_mappings, on=["well", "sample_number", "plate_number"], how="left",
    )
    df_kinetics = df_kinetics.merge(
        df_mappings, on=["well", "sample_number", "plate_number"], how="left",
    )

    df_mscarlett = apply_mscarlett_standard_curve(
        df_mscarlett, standard_curve_path, epsilon=1e-8,
    )
    df_kinetics = apply_4MU_standard_curve(
        df_kinetics, standard_curve_path, epsilon=1e-8,
    )

    return df_mscarlett, df_kinetics


# ---------------------------------------------------------------------------
# Kinetics fitting
# ---------------------------------------------------------------------------

def get_least_squares_fit(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Compute a simple least-squares linear fit (slope, intercept).

    Uses the closed-form normal equation for a single-variable linear
    regression, which is faster than ``scipy.stats.linregress`` for
    small arrays.

    Args:
        x: Independent variable (e.g., timestamps in seconds).
        y: Dependent variable (e.g., 4-MU concentration in µM).

    Returns:
        Tuple of (slope, intercept).
    """
    N = x.shape[0]
    m = (N * (x * y).sum() - x.sum() * y.sum()) / (N * (x ** 2).sum() - x.sum() ** 2)
    b = (y.sum() - m * x.sum()) / N
    return m, b
