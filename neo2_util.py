import pandas as pd
import torch

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import sys
import seaborn as sns
import re
from pathlib import Path
import json

# from Bio import SeqIO
import scipy.stats as stat
from scipy.stats import linregress
import textwrap

##############################
# Utility functions and imports
##############################


def get_sample_plate_info(filepath):
    """
    Extracts the plate number and sample number from the given file path.

    The function expects the file name to contain the substrings "plate#" and "sample#",
    where # represents the respective numbers. If these substrings are not found,
    a message will be printed to inform the user.

    Parameters:
    filepath (str): The full path to the file.

    Returns:
    tuple: A tuple containing the plate number and sample number as integers.

    Raises:
    ValueError: If the plate number or sample number is not found in the file name.
    """

    filename = os.path.basename(filepath)

    # Use regex to find the substring "plate" followed by a number
    match = re.search(r"plate(\d+)", filename)

    # If a match is found, extract the value next to "plate"
    if match:
        plate_number = int(match.group(1))
    else:
        print(
            "No plate number found, be sure to include plate# in the file name where # is the plate number"
        )

    ## parse plate ID
    match = re.search(r"sample(\d+)", filename)

    # If a match is found, extract the value next to "sample"
    if match:
        sample_number = int(match.group(1))
    else:
        print(
            "No sample number found, be sure to include sample# in the file name where # is the sample number"
        )

    return sample_number, plate_number


def parse_neo2_endpoint(filepath, start_row=54, parse_name=False):
    """
    Parses a TSV file from Neo2 with endpoint data.

    Parameters:
    -----------
    filepath : str
        The path to the TSV file to be parsed.
    start_row : int, optional
        The row number to start parsing data from (default is 54).
    parse_name : bool, optional
        If True, parses additional information from the filename (default is False).

    Returns:
    --------
    pandas.DataFrame
        A tidy DataFrame containing the parsed data with the following columns:
        - 'well': The well identifier.
        - 'value': The numeric value for each well.
        - 'serial_number': The instrument serial number.
        - 'source_file' (if parse_name is True): The source file path.
        - 'plate_number' (if parse_name is True): The plate number extracted from the filename.
        - 'sample_number' (if parse_name is True): The sample number extracted from the filename.
        - 'date_collected' (if parse_name is True): The date collected extracted from the filename.

    Raises:
    -------
    FileNotFoundError
        If the file specified by `filepath` does not exist.
    ValueError
        If the file does not contain the expected data format.

    Notes:
    ------
    - The function expects the filename to contain the date in 'yymmdd' format,
      and the substrings 'plate#' and 'sample#' where # is a number.
    - The file is expected to be encoded in 'cp1252' and delimited by tabs.
    - The function reads the instrument serial number from the first 20 rows of the file.
    - The data is parsed starting from the specified `start_row` and is expected to have a 'Well' column.
    """

    if parse_name:
        # print("parsing the name of the file")
        filename = os.path.basename(filepath)

        # date as the first characters of the name in the yymmdd format
        date = filename[0:6]

        # Use regex to find the substring "plate" followed by a number
        match = re.search(r"plate(\d+)", filename)

        # If a match is found, extract the value next to "plate"
        if match:
            plate_number = int(match.group(1))
        else:
            print(
                "No plate number found, be sure to include plate# in the file name where # is the plate number"
            )

        ## parse plate ID
        match = re.search(r"sample(\d+)", filename)

        # If a match is found, extract the value next to "sample"
        if match:
            sample_number = int(match.group(1))
        else:
            print(
                "No sample number found, be sure to include sample# in the file name where # is the sample number"
            )

    ## parse instrument serial number
    df = pd.read_csv(
        filepath,
        encoding="cp1252",
        delimiter="\t",
        nrows=20,
    )
    serial_number = df[df.iloc[:, 0] == "Reader Serial Number:"]
    serial_number = serial_number.iloc[:, 1].values[0]

    ## parse data
    df = pd.read_csv(
        filepath,
        encoding="cp1252",
        delimiter="\t",
        skiprows=range(1, start_row),
        nrows=1,
    )

    # make dataframe tidy
    df = df.melt(id_vars=["Well"], var_name="well", value_name="value")
    df.dropna(inplace=True)

    # clean the naming
    df.drop(columns=["Well"], inplace=True)

    # ensure value is numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # add info
    df["serial_number"] = serial_number
    if parse_name:
        df["source_file"] = filepath
        df["plate_number"] = plate_number
        df["sample_number"] = sample_number
        df["date_collected"] = date

    return df


def parse_neo2_kinetics(filepath, start_row=58, parse_name=False):
    """
    Parse a TSV file from neo2 with kinetics data.

    This function reads a TSV file containing kinetics data, processes it, and returns a tidy DataFrame.
    Optionally, it can also parse metadata from the filename.

    Parameters:
    -----------
    filepath : str
        The path to the TSV file to be parsed.
    start_row : int, optional
        The row number to start parsing the kinetics data from (default is 58).
    parse_name : bool, optional
        If True, the function will parse metadata (date, plate number, sample number) from the filename (default is False).

    Returns:
    --------
    pandas.DataFrame
        A tidy DataFrame containing the parsed kinetics data and optional metadata.

    The DataFrame contains the following columns:
    - time: Time in seconds.
    - well: Well identifier.
    - value: Kinetics measurement value.
    - serial_number: Instrument serial number.
    - source_file: (Optional) The source file path.
    - plate_number: (Optional) Plate number extracted from the filename.
    - sample_number: (Optional) Sample number extracted from the filename.
    - date_collected: (Optional) Date collected extracted from the filename in yymmdd format.

    Notes:
    ------
    - The function expects the TSV file to be encoded in 'cp1252' and delimited by tabs.
    - The function will drop any rows with missing values in the 'value' column.
    - The function will convert the 'time' column from HH:MM:SS format to seconds.
    - The filename should include 'plate#' and 'sample#' for the function to extract plate and sample numbers when parse_name is True.
    """

    if parse_name:
        # print("parsing the name of the file")
        filename = os.path.basename(filepath)

        # date as the first characters of the name in the yymmdd format
        date = filename[0:6]

        # Use regex to find the substring "plate" followed by a number
        match = re.search(r"plate(\d+)", filename)

        # If a match is found, extract the value next to "plate"
        if match:
            plate_number = int(match.group(1))
        else:
            print(
                "No plate number found, be sure to include plate# in the file name where # is the plate number"
            )

        ## parse plate ID
        match = re.search(r"sample(\d+)", filename)

        # If a match is found, extract the value next to "sample"
        if match:
            sample_number = int(match.group(1))
        else:
            print(
                "No sample number found, be sure to include sample# in the file name where # is the sample number"
            )

    ## parse instrument serial number
    df = pd.read_csv(
        filepath,
        encoding="cp1252",
        delimiter="\t",
        nrows=20,
    )
    serial_number = df[df.iloc[:, 0] == "Reader Serial Number:"]
    serial_number = serial_number.iloc[:, 1].values[0]

    ## parse kinetics data
    df = pd.read_csv(
        filepath,
        encoding="cp1252",
        delimiter="\t",
        skiprows=range(1, start_row),
    )

    df = df.drop(df.columns[1], axis=1)
    df.rename(columns={"Time": "time"}, inplace=True)

    # time conversion
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce").dt.time

    # Convert the time column to seconds
    df["time"] = df["time"].apply(lambda x: x.hour * 3600 + x.minute * 60 + x.second)

    # make dataframe tidy
    df = df.melt(id_vars=["time"], var_name="well", value_name="value")
    df.dropna(inplace=True)

    # ensure value is numeric
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # add info
    df["serial_number"] = serial_number
    if parse_name:
        df["source_file"] = filepath
        df["plate_number"] = plate_number
        df["sample_number"] = sample_number
        df["date_collected"] = date

    return df


def clean_mappings_df(df_mappings):
    """
    Ensure the 'replicate_number', 'sample_number', and 'column' columns in the DataFrame are numeric values.

    This function converts the specified columns in the input DataFrame to numeric values. If any value cannot be
    converted, it will be set to NaN.

    Parameters
    ----------
    df_mappings : pandas.DataFrame
        A DataFrame containing the columns 'replicate_number', 'sample_number', and 'column' which need to be
        converted to numeric values.

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with 'replicate_number', 'sample_number', and 'column' columns converted to numeric values.

    Raises
    ------
    KeyError
        If any of the required columns ('replicate_number', 'sample_number', 'column') are not present in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {
    ...     'replicate_number': ['1', '2', 'three'],
    ...     'sample_number': ['4', 'five', '6'],
    ...     'column': ['7', '8', 'nine']
    ... }
    >>> df = pd.DataFrame(data)
    >>> clean_mappings_df(df)
       replicate_number  sample_number  column
    0               1.0            4.0     7.0
    1               2.0            NaN     8.0
    2               NaN            6.0     NaN
    """
    df_mappings["replicate_number"] = pd.to_numeric(
        df_mappings["replicate_number"], errors="coerce"
    )
    df_mappings["sample_number"] = pd.to_numeric(
        df_mappings["sample_number"], errors="coerce"
    )
    df_mappings["column"] = pd.to_numeric(df_mappings["column"], errors="coerce")

    return df_mappings


def fasta_to_dataframe(fasta_file):
    """
    Read a simple FASTA file (name and sequence only) into a pandas DataFrame.

    This function parses a FASTA file and converts it into a pandas DataFrame
    where each row corresponds to a sequence record. The DataFrame will have
    two columns: 'ID' for the sequence identifier and 'sequence' for the
    sequence itself.

    Parameters:
    fasta_file (str): Path to the input FASTA file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the sequence data with columns
                  'ID' and 'sequence'.

    Example:
    >>> df = fasta_to_dataframe("example.fasta")
    >>> print(df.head())
             ID                                           sequence
    0  seq1  ATGCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
    1  seq2  CGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
    """
    records = SeqIO.parse(fasta_file, "fasta")
    data = [(record.id, str(record.seq)) for record in records]
    df = pd.DataFrame(data, columns=["ID", "sequence"])
    return df


def apply_mscarlett_standard_curve(
    df_mscarlett, standard_curve_path, epsilon=1e-8, write_path=None
):
    """
    Apply a standard curve to mScarlett data to convert raw values to micromolar concentrations.

    Parameters:
    df_mscarlett (pd.DataFrame): DataFrame containing mScarlett data with a 'value' column and a 'serial_number' column.
    standard_curve_path (str): Path to the JSON file containing the standard curves.
    epsilon (float, optional): Small value to replace negative concentrations with. Default is 1e-8.
    write_path (str, optional): Path to save the modified DataFrame as a CSV file. If None, the DataFrame is not saved. Default is None.

    Returns:
    pd.DataFrame: DataFrame with an additional 'mscarlett_um' column containing the micromolar concentrations.
    """

    with open(standard_curve_path, "r") as f:
        standard_curves = json.load(f)

    # apply standard curve to mscarlett data
    df_mscarlett["mscarlett_um"] = df_mscarlett.apply(
        lambda x: (
            x["value"] - standard_curves[x["serial_number"]]["mscarlett"]["y_intercept"]
        )
        / standard_curves[x["serial_number"]]["mscarlett"]["slope"],
        axis=1,
    )

    # set any negative values to zero
    if epsilon:
        df_mscarlett.loc[df_mscarlett["mscarlett_um"] < 0, "mscarlett_um"] = 1e-8

    if write_path:
        df_mscarlett.to_csv(write_path)

    return df_mscarlett


def apply_4MU_standard_curve(
    df_kinetics, standard_curve_path, epsilon=1e-8, write_path=None
):
    """
    Apply a standard curve to convert raw fluorescence values to 4MU concentrations.

    This function reads a standard curve from a JSON file and uses it to convert
    raw fluorescence values in a DataFrame to 4MU concentrations. Any negative
    concentration values are set to a small positive value (epsilon). Optionally,
    the resulting DataFrame can be written to a CSV file.

    Parameters:
    df_kinetics (pd.DataFrame): DataFrame containing the raw fluorescence values.
        Must contain columns 'value' and 'serial_number'.
    standard_curve_path (str): Path to the JSON file containing the standard curves.
    epsilon (float, optional): Small positive value to replace negative concentrations.
        Default is 1e-8.
    write_path (str, optional): Path to write the resulting DataFrame to a CSV file.
        If None, the DataFrame is not written to a file. Default is None.

    Returns:
    pd.DataFrame: DataFrame with an additional column '4MU_um' containing the
        converted 4MU concentrations.
    """

    with open(standard_curve_path, "r") as f:
        standard_curves = json.load(f)

    # fit 4MU with the standard curve
    df_kinetics["4MU_um"] = df_kinetics.apply(
        lambda x: (
            x["value"] - standard_curves[x["serial_number"]]["4MU"]["y_intercept"]
        )
        / standard_curves[x["serial_number"]]["4MU"]["slope"],
        axis=1,
    )

    # set any negative values to zero
    if epsilon:
        df_kinetics.loc[df_kinetics["4MU_um"] < 0, "4MU_um"] = 1e-8

    if write_path:
        df_kinetics.to_csv(write_path)

    return df_kinetics


def fit_line(group, x_column="time", y_column="4MU_um"):
    """
    Fit a line to a grouped series in a dataframe using linear regression.

    Parameters:
    group (pd.DataFrame): The dataframe containing the grouped data.
    x_column (str): The name of the column to be used as the x variable. Default is "time".
    y_column (str): The name of the column to be used as the y variable. Default is "4MU_um".

    Returns:
    pd.Series: A series containing the slope, intercept, r_value, p_value, and std_err of the fitted line.

    The returned series contains the following keys:
    - slope: The slope of the fitted line.
    - intercept: The intercept of the fitted line.
    - r_value: The correlation coefficient.
    - p_value: The p-value for a hypothesis test whose null hypothesis is that the slope is zero.
    - std_err: The standard error of the estimated gradient.
    """
    slope, intercept, r_value, p_value, std_err = linregress(
        group[x_column], group[y_column]
    )
    return pd.Series(
        {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }
    )


def apply_tidy_kinetic_fit(
    df,
    fit_function=fit_line,
    time_column="time",
    y_column="4MU_um",
    time_range=(0, 1800),
    group_columns=[
        "well",
        "replicate_number",
        "sample_number",
        "name",
        "sample_type",
        "plate_number",
    ],
    clean=True,
):
    """
    Applies a specified fitting function to y data within a given time range.

    Parameters:
        df (pd.DataFrame): Input dataframe in tidy format containing time and y data.
        fit_function (callable): The function to apply for fitting, should accept `x_column` and `y_column` as arguments.
        time_column (str): Column name for the time data.
        y_column (str): Column name for the dependent variable data to be fitted.
        time_range (tuple): Time range (min, max) for filtering data before fitting.
        group_columns (list): List of column names to group by before applying the fit function.
        clean (bool): If True, cleans the resulting dataframe by renaming 'slope' to 'rate' and dropping 'r_value', 'p_value', and 'std_err' columns.

    Returns:
        pd.DataFrame: Dataframe with the fit results for each group, with columns as defined by the fit function.
    """
    # Filter the data based on the specified time range
    df_filtered = df[
        (df[time_column] >= time_range[0]) & (df[time_column] <= time_range[1])
    ]

    # Apply the fit
    df_fits = (
        df_filtered.groupby(group_columns)
        .apply(fit_function, x_column=time_column, y_column=y_column)
        .reset_index()
    )

    if clean:
        df_fits.rename(columns={"slope": "rate"}, inplace=True)
        df_fits.drop(columns=["r_value", "p_value", "std_err"], inplace=True)
    return df_fits


def apply_tidy_background_subtraction(
    df, value, grouping, background="negative_control", epsilon=None
):
    """
    Subtracts background values from specified measurements in a tidy dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe in tidy format, containing data with specified measurement values and sample types.
        value (str): Column name of the measurement values to subtract background from.
        grouping (list of str): List of column names to group by when calculating background values.
        background (str, optional): The sample type name representing the background condition (default is "negative_control").
        epsilon (float, optional): Minimum allowable value for background and background-subtracted values.
                                    If specified, values less than epsilon will be set to epsilon.

    Returns:
        pd.DataFrame: Dataframe with background-subtracted values in a new column named "{value}_minus_background".
                      If epsilon is provided, any background-subtracted values below epsilon will be set to epsilon.
    """

    df_background = (
        df[df["sample_type"] == background]
        .groupby(grouping)[value]
        .mean()
        .reset_index()
        .rename(columns={value: f"background_{value}"})
    )

    if epsilon:
        df_background.loc[
            df_background[f"background_{value}"] < 0, f"background_{value}"
        ] = epsilon

    df = df.merge(df_background, on=grouping, how="left")

    df[f"{value}_minus_background"] = df[value] - df[f"background_{value}"]

    if epsilon:
        # set any negative rates to zero
        df.loc[df[f"{value}_minus_background"] < 0, f"{value}_minus_background"] = (
            epsilon
        )

    return df


def normalize_rate_by_mscarlett(df_kinetics_fits, df_mscarlett):
    """
    Normalize the rate by mScarlett concentration.

    This function merges the kinetics fits DataFrame with the mScarlett DataFrame
    based on well, replicate number, sample number, and plate number. It then
    normalizes the 'rate_minus_background' by the 'mscarlett_um' concentration
    and adds a new column 'rate_minus_background_normalized' to the DataFrame.

    Parameters:
    df_kinetics_fits (pd.DataFrame): DataFrame containing kinetics fits data with
                                     columns 'well', 'replicate_number', 'sample_number',
                                     'plate_number', and 'rate_minus_background'.
    df_mscarlett (pd.DataFrame): DataFrame containing mScarlett concentration data with
                                 columns 'well', 'replicate_number', 'sample_number',
                                 'plate_number', and 'mscarlett_um'.

    Returns:
    pd.DataFrame: A DataFrame with the normalized rate added as a new column
                  'rate_minus_background_normalized'.
    """
    df_kinetics_fits = pd.merge(
        df_kinetics_fits,
        df_mscarlett[
            [
                "well",
                "replicate_number",
                "sample_number",
                "plate_number",
                "mscarlett_um",
            ]
        ],
        on=["well", "replicate_number", "sample_number", "plate_number"],
        how="left",
    )

    df_kinetics_fits["rate_minus_background_normalized"] = (
        df_kinetics_fits["rate_minus_background"] / df_kinetics_fits["mscarlett_um"]
    )

    return df_kinetics_fits


def calculate_max_4MU(df_kinetics):
    """
    Calculate the maximum 4MU concentration for each well, replicate, and sample.

    Parameters:
    df_kinetics (pandas.DataFrame): DataFrame containing kinetic data with columns
                                    'name', 'replicate_number', 'sample_number', 'plate_number', and '4MU_um'.
    df_kinetics_fits (pandas.DataFrame): DataFrame containing fitted kinetic data.

    Returns:
    pandas.DataFrame: DataFrame with the maximum 4MU concentration for each well, replicate, and sample.
                      The returned DataFrame contains columns 'well', 'replicate_number', 'sample_number',
                      'plate_number', and '4MU_um_max'.
    """
    # Calculate max 4MU
    df_max = (
        df_kinetics.groupby(["name", "replicate_number", "sample_number"])
        .max()
        .reset_index()
    )
    df_max.rename(columns={"4MU_um": "4MU_um_max"}, inplace=True)
    df_max = df_max[
        ["name", "replicate_number", "sample_number", "plate_number", "4MU_um_max"]
    ]

    return df_max


def calculate_mscarlett_stats(df_mscarlett, epsilon=1e-8):
    """
    Calculate statistics for mScarlett fluorescence measurements.

    This function groups the input DataFrame by 'name', 'sample_number', and 'plate_number',
    and calculates the mean and standard deviation of 'mscarlett_um' for each group. It then
    renames the resulting columns to 'mscarlett_mean' and 'mscarlett_std_dev', respectively.
    Any negative mean values are set to a small positive value (1e-8). Finally, it computes
    the coefficient of variation (CV) for mScarlett measurements.

    Parameters:
    df_mscarlett (pd.DataFrame): A DataFrame containing mScarlett fluorescence measurements
                                 with columns 'name', 'sample_number', 'plate_number', and 'mscarlett_um'.
    epsilon (float, optional): A small positive value to replace any negative mean values. Default is 1e-8.

    Returns:
    pd.DataFrame: A DataFrame with the calculated statistics, including 'mscarlett_mean',
                  'mscarlett_std_dev', and 'mscarlett_cv' for each group.
    """
    df_mscarlett_stats = (
        df_mscarlett.groupby(["name", "sample_number", "plate_number"])["mscarlett_um"]
        .agg(
            mean="mean",
            std_dev="std",
        )
        .reset_index()
    )
    df_mscarlett_stats.rename(
        columns={
            "mean": "mscarlett_mean",
            "std_dev": "mscarlett_std_dev",
        },
        inplace=True,
    )

    # set any negative values to zero
    if epsilon:
        df_mscarlett_stats.loc[
            df_mscarlett_stats["mscarlett_mean"] < 0, "mscarlett_mean"
        ] = epsilon

    df_mscarlett_stats["mscarlett_cv"] = df_mscarlett_stats["mscarlett_std_dev"] / abs(
        df_mscarlett_stats["mscarlett_mean"]
    )

    return df_mscarlett_stats


def calculate_kinetics_stats(df_kinetics_fits, df_4MU_max, epsilon=1e-8):
    """
    This function groups the input DataFrame by 'name', 'sample_number', and 'plate_number',
    Additionally, it calculates the coefficient of variation (CV) for the normalized rate measurements.

    Parameters:
                                        with columns 'name', 'sample_number', 'plate_number',
                                        and 'rate_minus_background_normalized'.
    df_4MU_max (pd.DataFrame): A DataFrame containing 4MU maximum measurements with columns
                                'name', 'sample_number', 'plate_number', and '4MU_um_max'.
    epsilon (float): A small value to replace any negative mean values. Default is 1e-8.

    Returns:
                    'rate_norm_std_dev', 'rate_norm_cv', and '4MU_um_max_mean' for each group.
    """

    df_kinetics_stats = (
        df_kinetics_fits.groupby(["name", "sample_number", "plate_number"])[
            "rate_minus_background_normalized"
        ]
        .agg(
            mean="mean",
            std_dev="std",
        )
        .reset_index()
    )
    df_kinetics_stats.rename(
        columns={
            "mean": "rate_norm_mean",
            "std_dev": "rate_norm_std_dev",
        },
        inplace=True,
    )

    # set any negative values to zero
    if epsilon:
        df_kinetics_stats.loc[
            df_kinetics_stats["rate_norm_mean"] < 0, "rate_norm_mean"
        ] = epsilon

    df_kinetics_stats["rate_norm_cv"] = df_kinetics_stats["rate_norm_std_dev"] / abs(
        df_kinetics_stats["rate_norm_mean"]
    )

    ###
    ###
    df_temp = (
        df_kinetics_fits.groupby(["name", "sample_number", "plate_number"])[
            "rate_minus_background"
        ]
        .agg(
            mean="mean",
            std_dev="std",
        )
        .reset_index()
    )
    df_temp.rename(
        columns={
            "mean": "rate_mean",
            "std_dev": "rate_std_dev",
        },
        inplace=True,
    )

    # set any negative values to zero
    if epsilon:
        df_temp.loc[df_temp["rate_mean"] < 0, "rate_mean"] = epsilon

    df_temp["rate_cv"] = df_temp["rate_std_dev"] / abs(df_temp["rate_mean"])

    df_kinetics_stats = pd.merge(
        df_kinetics_stats,
        df_temp,
        on=["name", "sample_number", "plate_number"],
        how="left",
    )

    ###
    # group by name, sample_number, and plate_number, then calculate the mean of 4MU_um_max
    df_temp = (
        df_4MU_max.groupby(["name", "sample_number", "plate_number"])["4MU_um_max"]
        .mean()
        .reset_index()
    )
    df_temp.rename(
        columns={
            "4MU_um_max": "4MU_um_max_mean",
        },
        inplace=True,
    )

    df_kinetics_stats = pd.merge(
        df_kinetics_stats,
        df_temp,
        on=["name", "sample_number", "plate_number"],
        how="left",
    )

    return df_kinetics_stats


def generate_kinetic_fit_data(df_kinetics, df_kinetics_fits):
    """
    Generate kinetic fit data by applying linear fits to time data for plotting.
    This function merges the provided kinetics data with the kinetics fits data
    based on specified columns, and then calculates the fitted 4MU concentration
    using the rate and intercept from the fits data.
    Parameters:
    df_kinetics (pd.DataFrame): DataFrame containing the kinetics data.
    df_kinetics_fits (pd.DataFrame): DataFrame containing the kinetics fits data.
    Returns:
    pd.DataFrame: A DataFrame with the original kinetics data and the fitted 4MU
                  concentration added as a new column.
    """

    # apply linear fits to time data for plotting
    df_kinetics_fitted = pd.merge(
        df_kinetics,
        df_kinetics_fits,
        on=["well", "replicate_number", "sample_number", "plate_number"],
        how="left",
    )

    df_kinetics_fitted["4MU_um_fit"] = (
        df_kinetics_fitted["rate"] * df_kinetics_fitted["time"]
        + df_kinetics_fitted["intercept"]
    )

    return df_kinetics_fitted


def apply_tidy_background_comparison(
    df, value, grouping, background="negative_control", stdev_from_background=3
):
    """
    Adds new column with background threshold for comarison.

    Parameters:
        df (pd.DataFrame): Input dataframe in tidy format, containing data with specified measurement values and sample types.
        value (str): Column name of the measurement values to subtract background from.
        grouping (list of str): List of column names to group by when calculating background values.
        background (str, optional): The sample type name representing the background condition (default is "negative_control").
        stdev_from_background (float, optional): Number of standard deviations away from the mean of the background to consider the cutoff.

    Returns:
        pd.DataFrame: Dataframe with background comparison values in a new column named "background_{value}_threshold".
                      If epsilon is provided, any background-subtracted values below epsilon will be set to epsilon.
    """

    df_background = (
        df[df["sample_type"] == background]
        .groupby(grouping)[value]
        .agg(
            mean="mean",
            std_dev="std",
        )
        .reset_index()
    )

    df_background[f"background_{value}_threshold"] = (
        df_background["mean"] + stdev_from_background * df_background["std_dev"]
    )

    df_background.drop(columns=["mean", "std_dev"], inplace=True)

    df = df.merge(df_background, on=grouping, how="left")

    return df


def apply_filters(
    df_kinetics_fits,
    df_kinetics_stats,
    df_mscarlett_stats,
    mscarlett_threshold=0.2,
    cv_threshold=1,
    stdev_from_background=3,
):
    """
    Apply various filters to the kinetics data and return aggregated statistics.

    Parameters:
    df_kinetics_fits (pd.DataFrame): DataFrame containing kinetics fits data.
    df_kinetics_stats (pd.DataFrame): DataFrame containing kinetics statistics.
    df_mscarlett_stats (pd.DataFrame): DataFrame containing mScarlett statistics.
    mscarlett_threshold (float, optional): Threshold for mScarlett in uM. Default is 0.2.
    cv_threshold (float, optional): Coefficient of variation threshold for rate normalization. Default is 1.
    stdev_from_background (int, optional): Number of standard deviations from background for rate threshold. Default is 3.

    Returns:
    pd.DataFrame: DataFrame containing aggregated statistics with applied filters.
    """

    # calculate rate threshold by plate

    df_kinetics_fits = apply_tidy_background_comparison(
        df_kinetics_fits,
        value="rate",
        grouping=["sample_number", "plate_number"],
        background="negative_control",
        stdev_from_background=stdev_from_background,
    )

    # label data with threshold pass fail

    df_kinetics_fits["mscarlett_threshold_pass"] = (
        df_kinetics_fits["mscarlett_um"] > mscarlett_threshold
    )

    df_kinetics_fits["rate_norm_threshold_pass"] = (
        df_kinetics_fits["rate"] > df_kinetics_fits["background_rate_threshold"]
    )

    df_kinetics_stats["rate_norm_cv_threshold_pass"] = (
        df_kinetics_stats["rate_norm_cv"] < cv_threshold
    )

    # mscarlett aggregated stats
    df_stats = df_mscarlett_stats

    # mscarlett threshold
    df_pivot = df_kinetics_fits.pivot(
        index=["name", "sample_number", "plate_number"],
        columns="replicate_number",
        values="mscarlett_threshold_pass",
    ).reset_index()

    df_pivot["mscarlett_threshold_pass"] = df_pivot[[1, 2, 3]].all(axis=1)

    df_stats = pd.merge(
        df_stats,
        df_pivot[["name", "sample_number", "plate_number", "mscarlett_threshold_pass"]],
        on=["name", "sample_number", "plate_number"],
        how="left",
    )

    # kinetics aggregated stats
    df_stats = pd.merge(
        df_stats,
        df_kinetics_stats,
        on=["name", "sample_number", "plate_number"],
        how="left",
    )

    # kinetics threshold
    df_pivot = df_kinetics_fits.pivot(
        index=["name", "sample_number", "plate_number"],
        columns="replicate_number",
        values="rate_norm_threshold_pass",
    ).reset_index()

    df_pivot["rate_norm_threshold_pass"] = df_pivot[[1, 2, 3]].all(axis=1)

    df_stats = pd.merge(
        df_stats,
        df_pivot[["name", "sample_number", "plate_number", "rate_norm_threshold_pass"]],
        on=["name", "sample_number", "plate_number"],
        how="left",
    )

    return df_stats


def build_complete_df(
    df_mappings, df_mscarlett, df_4MU_max, df_kinetics_fits, df_stats
):
    """
    Build a complete DataFrame by merging multiple input DataFrames.

    This function takes several DataFrames containing different types of data
    and merges them into a single comprehensive DataFrame. The merging is done
    based on common columns such as 'name', 'sample_number', and 'plate_number'.

    Parameters:
    df_mappings (pd.DataFrame): DataFrame containing mapping information including
        'name', 'sequence', 'source_dna_well', 'sample_number', 'plate_number', and 'sample_type'.
    df_mscarlett (pd.DataFrame): DataFrame containing mScarlett data including
        'serial_number', 'date_collected', 'source_file', 'name', 'sample_number', 'plate_number', and 'mscarlett_um'.
    df_4MU_max (pd.DataFrame): DataFrame containing data for 4MU_um_max including
        'name', 'sample_number', 'plate_number', 'replicate_number', and '4MU_um_max'.
    df_kinetics_fits (pd.DataFrame): DataFrame containing rate data including
        'name', 'sample_number', 'plate_number', 'replicate_number', and 'rate_minus_background_normalized'.
    df_stats (pd.DataFrame): DataFrame containing aggregate statistics including
        'name', 'sample_number', and 'plate_number'.

    Returns:
    pd.DataFrame: A merged DataFrame containing all the input data combined based on
        common columns.
    """

    df_export = df_mappings[
        [
            "name",
            "sequence",
            "source_dna_well",
            "sample_number",
            "plate_number",
            "sample_type",
        ]
    ].drop_duplicates()

    # merge replicate data
    df_pivot = df_mappings[
        ["name", "sample_number", "replicate_number", "plate_number", "well"]
    ].pivot(
        index=["name", "sample_number", "plate_number"],
        columns="replicate_number",
        values="well",
    )

    df_pivot.columns = [f"well_{int(col)}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_export = df_export.merge(
        df_pivot, on=["name", "sample_number", "plate_number"], how="left"
    )

    # merge date and plate reader info
    df_temp = df_mscarlett[
        [
            "source_file",
            "serial_number",
            "date_collected",
            "name",
            "sample_number",
            "plate_number",
        ]
    ].drop_duplicates()

    df_export = df_export.merge(
        df_temp,
        on=["name", "sample_number", "plate_number"],
        how="left",
    )

    # merge mscarlett individual data
    df_pivot = df_mscarlett.pivot(
        index=["name", "sample_number", "plate_number"],
        columns="replicate_number",
        values="mscarlett_um",
    )

    df_pivot.columns = [f"mscarlett_um_{int(col)}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_export = df_export.merge(
        df_pivot, on=["name", "sample_number", "plate_number"], how="left"
    )

    # merge kinetic data 4MU_um_max
    df_pivot = df_4MU_max[
        ["name", "sample_number", "plate_number", "replicate_number", "4MU_um_max"]
    ].pivot(
        index=["name", "sample_number", "plate_number"],
        columns="replicate_number",
        values="4MU_um_max",
    )

    df_pivot.columns = [f"4MU_um_max_{int(col)}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_export = df_export.merge(
        df_pivot, on=["name", "sample_number", "plate_number"], how="left"
    )

    # merge rate individual data
    df_pivot = df_kinetics_fits.pivot(
        index=["name", "sample_number", "plate_number"],
        columns="replicate_number",
        values="rate",
    )

    df_pivot.columns = [f"rate_{int(col)}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_export = df_export.merge(
        df_pivot, on=["name", "sample_number", "plate_number"], how="left"
    )

    # merge normalized and background subtracted rate individual data
    df_pivot = df_kinetics_fits.pivot(
        index=["name", "sample_number", "plate_number"],
        columns="replicate_number",
        values="rate_minus_background_normalized",
    )

    df_pivot.columns = [f"rate_norm_{int(col)}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_export = df_export.merge(
        df_pivot, on=["name", "sample_number", "plate_number"], how="left"
    )

    # merge aggregate data
    df_export = df_export.merge(
        df_stats, on=["name", "sample_number", "plate_number"], how="left"
    )

    return df_export


def parity_plot(
    df_plot,
    value,
    replicates,
    title,
    figure_name,
    figure_directory,
    write_figure=False,
    write_format="png",
    sample_number=None,
    plate_number=None,
):
    """
    Generates a parity plot for the given data frame and replicates.

    Parameters:
    df_plot (pd.DataFrame): DataFrame containing the data to plot.
    value (str): The column name of the values to be plotted.
    replicates (list): List of replicate numbers to be plotted.
    title (str): Title of the plot.
    figure_name (str): Name of the figure file to save.
    figure_directory (Path): Directory where the figure file will be saved.
    write_figure (bool, optional): If True, the figure will be saved to the specified directory. Default is False.
    sample_number (int, optional): Sample number to filter the data. Default is None.
    plate_number (int, optional): Plate number to filter the data. Default is None.
    write_format (str, optional): Format to save the figure. Default is "png".
    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the plot.
    axs (numpy.ndarray): Array of Axes objects containing the subplots.
    """
    if sample_number is not None and plate_number is not None:
        # get data for a single plate
        df_plot = df_plot[
            (df_plot.sample_number == sample_number)
            & (df_plot.plate_number == plate_number)
        ]

    fig, axs = plt.subplots(
        1, len(replicates), figsize=(5 * len(replicates), 5), sharex=True, sharey=True
    )

    for num, replicate in enumerate(replicates):
        x_replicate = replicates[num]
        y_replicate = replicates[num - 1]
        x_df = df_plot[df_plot["replicate_number"] == x_replicate].sort_values("name")
        y_df = df_plot[df_plot["replicate_number"] == y_replicate].sort_values("name")
        x = x_df[value]
        y = y_df[value]
        axs[num].scatter(x, y)
        # Suppresses automatic display in Jupyter
        axs[num].set_xlabel(f"replicate {x_replicate}")
        axs[num].set_ylabel(f"replicate {y_replicate}")

        parity = [
            min(plt.xlim()[0], plt.ylim()[0]),
            max(plt.xlim()[1], plt.ylim()[1]),
        ]
        axs[num].plot(parity, parity, "k-", lw=1)
        # Suppresses automatic display in Jupyter
        plt.xlim(parity)
        plt.ylim(parity)

    # Set a title for the figure with plate and sample numbers
    fig.suptitle(title)

    # Adjust layout for spacing
    plt.tight_layout()

    # Show the plot
    plt.show()

    figure_path = figure_directory / f"{figure_name}.{write_format}"

    if write_figure:
        fig.savefig(
            figure_path,
            format=write_format,
            dpi=300,
        )
    return fig, axs


def extract_well_position(well):
    row = well[0]  # Extract row letter (A-P)
    column = int(well[1:])  # Extract column number (1-24)
    return row, column


def plot_mscarlett_heatmap(
    df_plot,
    value,
    title,
    figure_name,
    figure_directory,
    sample_number,
    plate_number,
    write_figure=False,
    write_format="png",
):
    """
    Plots a heatmap for mScarlett data from a given DataFrame.

    Parameters:
    df_plot (pd.DataFrame): DataFrame containing the data to plot.
    value (str): Column name in df_plot containing the values to plot.
    title (str): Title of the heatmap.
    figure_name (str): Name of the figure file to save.
    figure_directory (Path): Directory where the figure will be saved.
    sample_number (int): Sample number to filter the data.
    plate_number (int): Plate number to filter the data.
    write_figure (bool, optional): Whether to save the figure to a file. Default is False.
    write_format (str, optional): Format to save the figure. Default is "png".

    Returns:
    fig (matplotlib.figure.Figure): The figure object containing the heatmap.
    ax (matplotlib.axes._subplots.AxesSubplot): The Axes object with the heatmap.
    """

    # get data for a single plate
    df_plot = df_plot[
        (df_plot.sample_number == sample_number)
        & (df_plot.plate_number == plate_number)
    ]

    # Create a mapping of rows (A-P) to numeric indices (0-15)
    row_map = {chr(i): i - 65 for i in range(65, 81)}  # A=0, B=1, ..., P=15
    df_plot["row_num"] = df_plot["row"].map(row_map)

    # Create a 2D grid for the heatmap
    heatmap_data = np.zeros((16, 24))  # 16 rows (A-P), 24 columns (1-24)
    for index, row in df_plot.iterrows():
        heatmap_data[row["row_num"], row["column"] - 1] = row[value]  # Fill in the data

    # Plot the heatmap
    fig = plt.figure(figsize=(12, 8))
    ax = sns.heatmap(
        heatmap_data,
        cmap="Reds",
        # annot=True,
        # fmt=".1f",
        cbar=True,
        cbar_kws={"shrink": 0.80},
        xticklabels=range(1, 25),
        yticklabels=[chr(i) for i in range(65, 81)],
        square=True,
    )

    # Label the color bar
    colorbar = ax.collections[0].colorbar
    colorbar.set_label("[mScarlett] (\u03BCM)", rotation=270, labelpad=20)

    # Draw vertical lines between columns 8 and 9, and 16 and 17
    ax.axvline(x=8, color="black", linestyle="-", linewidth=2)  # Between column 8 and 9
    ax.axvline(
        x=16, color="black", linestyle="-", linewidth=2
    )  # Between column 16 and 17

    ax.set_title(title)

    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.show()

    figure_path = figure_directory / f"{figure_name}.{write_format}"

    if write_figure:
        fig.savefig(
            figure_path,
            format=write_format,
            dpi=300,
        )
    return fig, ax


def plot_384_kinetics(
    df_plot,
    value,
    title,
    figure_name,
    figure_directory,
    write_figure,
    write_format,
    sample_number,
    plate_number,
):

    # get data for a single plate
    df_plot = df_plot[
        (df_plot.sample_number == sample_number)
        & (df_plot.plate_number == plate_number)
    ]

    # Sort the wells in row-major order
    df_plot["Well_Row"] = df_plot["well"].str[0]  # A-P
    df_plot["Well_Col"] = df_plot["well"].str[1:].astype(int)  # 1-24

    # Create subplots (16 rows, 24 columns)
    fig, axes = plt.subplots(16, 24, figsize=(12, 8), sharex=True, sharey=True)
    # fig.suptitle("Kinetic Traces for 384-Well Plate", fontsize=16)

    # For each well, plot the kinetic trace
    for i, row in enumerate("ABCDEFGHIJKLMNOP"):  # Well rows A-P
        for j in range(1, 25):  # Well columns 1-24
            ax = axes[i, j - 1]
            well = f"{row}{j}"

            # Select data for the specific well
            well_data = df_plot[df_plot["well"] == well]
            if not well_data.empty:
                ax.plot(well_data["time"], well_data[value], label=well)

            # Add row labels on the first column
            if j == 1:
                ax.set_ylabel(row, fontsize=10, rotation=0, labelpad=15, va="center")

            # Add column labels on the bottom row
            if i == 15:
                ax.set_xlabel(j, fontsize=10, rotation=0, labelpad=15, va="center")

            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()

    figure_path = figure_directory / f"{figure_name}.{write_format}"

    if write_figure:
        fig.savefig(
            figure_path,
            format=write_format,
            dpi=300,
        )

    return fig, axes


##############################
##############################


def get_least_squares_fit(x, y):
    """Compute least squares fit for Torch tensors x and y."""
    N = x.shape[0]
    m = (N * (x * y).sum() - (x.sum() * y.sum())) / (N * (x**2).sum() - (x.sum()) ** 2)
    b = (y.sum() - (m * x.sum())) / N
    return m, b


def get_standard_curve_params(serial_num):
    """Get standard curve parameters for a given serial number."""

    MAX_MSCARLETT_CONCENTRATION = 1
    MIN_MSCARLETT_CONCENTRATION = 0.01
    MAX_4MU_CONCENTRATION = 100
    MIN_4MU_CONCENTRATION = 0.3

    BASE_DIR = "/projects/ml/itopt/neo2_standards"
    SERIAL_NUM_TO_READER = {23062123: "trinity", 21062324: "morpheus"}
    _MSCARLETT_START_CONCENTRATION = 27
    _MSCARLETT_REPLICATE_COLS = [14, 15, 16]
    _4MU_START_CONCENTRATION = 300
    _4MU_REPLICATE_COLS = [17, 18, 19]

    # get standard curve for the right machine
    reader_name = SERIAL_NUM_TO_READER[serial_num]
    _mscalett_raw_data = pd.read_excel(
        f"{BASE_DIR}/240919_e186_mscarlett_standard_{reader_name}.xlsx"
    )
    _4mu_raw_data = pd.read_excel(
        f"{BASE_DIR}/240919_e186_4MU_standard_{reader_name}.xlsx"
    )

    # get dilution series set up for mscarlett
    _mscarlett_dilution_series = [_MSCARLETT_START_CONCENTRATION]
    _mscarlett_dilution_series += [
        _MSCARLETT_START_CONCENTRATION / (2 ** (i + 1)) for i in range(15)
    ]
    _mscarlett_dilution_series[-1] = 0

    # get dilution series set up for 4MU
    _4mu_dilution_series = [_4MU_START_CONCENTRATION]
    _4mu_dilution_series += [
        _4MU_START_CONCENTRATION / (2 ** (i + 1)) for i in range(15)
    ]
    _4mu_dilution_series[-1] = 0

    # parse just read data for mscarlett
    standard_mscarlett = {
        "concentration": [],
        "fluoresence": [],
    }

    # get starting row index
    for row_index, r in _mscalett_raw_data.iterrows():
        if r["Unnamed: 1"] == "A":
            break

    read_data_mscarlett = _mscalett_raw_data.iloc[row_index:, :]
    for c in _MSCARLETT_REPLICATE_COLS:
        tmp = read_data_mscarlett.iloc[:, c].tolist()
        for i, f in enumerate(tmp):
            standard_mscarlett["concentration"].append(_mscarlett_dilution_series[i])
            standard_mscarlett["fluoresence"].append(f)

    df_mscarlett = pd.DataFrame(standard_mscarlett)

    # parse just read data for 4MU
    standard_4mu = {
        "concentration": [],
        "fluoresence": [],
    }

    # get starting row index
    for row_index, r in _4mu_raw_data.iterrows():
        if r["Unnamed: 1"] == "A":
            break

    read_data_4mu = _4mu_raw_data.iloc[row_index:, :]
    for c in _4MU_REPLICATE_COLS:
        tmp = read_data_4mu.iloc[:, c].tolist()
        for i, f in enumerate(tmp):
            standard_4mu["concentration"].append(_4mu_dilution_series[i])
            standard_4mu["fluoresence"].append(f)

    df_4mu = pd.DataFrame(standard_4mu)

    # get fits in the specified range
    df_4mu_sub = df_4mu[
        (df_4mu.concentration > MIN_4MU_CONCENTRATION)
        & (df_4mu.concentration < MAX_4MU_CONCENTRATION)
    ]
    x_4mu = torch.tensor(df_4mu_sub.fluoresence.tolist())
    y_4mu = torch.tensor(df_4mu_sub.concentration.tolist())
    m_4mu, b_4mu = get_least_squares_fit(x_4mu, y_4mu)

    df_mscarlett_sub = df_mscarlett[
        (df_mscarlett.concentration > MIN_MSCARLETT_CONCENTRATION)
        & (df_mscarlett.concentration < MAX_MSCARLETT_CONCENTRATION)
    ]
    x_mscarlett = torch.tensor(df_mscarlett_sub.fluoresence.tolist())
    y_mscarlett = torch.tensor(df_mscarlett_sub.concentration.tolist())
    m_mscralett, b_mscarlett = get_least_squares_fit(x_mscarlett, y_mscarlett)

    params = {
        "mscarlett": (m_mscralett, b_mscarlett),
        "4MU": (m_4mu, b_4mu),
    }
    return params
