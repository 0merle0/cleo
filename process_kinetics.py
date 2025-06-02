###################################################################################################
###################################################################################################
# Import the necessary libraries and set up the working directory
###################################################################################################
###################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import sys
import seaborn as sns
from pathlib import Path
import scipy.stats as stat
from scipy.stats import linregress
import glob
import hydra


def find_named_parent(path: Path, parent_name: str):
    path = path.resolve()
    # Check if the current path itself matches
    if path.name == parent_name:
        return path

    for parent in path.parents:
        if parent.name == parent_name:
            return parent

    return None


working_directory_path = Path(os.path.abspath(""))
sys.path.append(working_directory_path)

current_script_path = Path(__file__).resolve()

if find_named_parent(working_directory_path, "cleo"):
    cleo_directory_path = find_named_parent(working_directory_path, "cleo")
    sys.path.append(cleo_directory_path)
elif find_named_parent(current_script_path, "cleo"):
    cleo_directory_path = find_named_parent(current_script_path, "cleo")
    sys.path.append(cleo_directory_path)
else:
    print("No cleo directory found.")

from neo2_util import *


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="process_kinetics",
)
def main(cfg):
    ### setup the working directory
    dirs_list = [
        "raw_data",
        "mappings",
        "processed_data",
        "graphs",
        "graphs/per_plate",
        "graphs/aggregated",
    ]
    directories = {}
    directories["working"] = working_directory_path
    for directory in dirs_list:
        directories[directory] = directories["working"] / directory
        directories[directory].mkdir(exist_ok=True)

    raw_data_files = glob.glob(str(directories["raw_data"] / "*.txt"))
    mappings_data_files = glob.glob(str(directories["mappings"] / "*.csv"))

    # Check if raw_data CSVs exist
    if not raw_data_files:
        input(
            "Please place raw_data and mappings files in the respective directories, then press Enter to continue..."
        )
        if not raw_data_files:
            input(
                "No raw data files found. "
                "Please place them in the 'raw_data' directory, then press Enter to continue..."
            )
            # Check again
            raw_data_files = glob.glob(str(directories["raw_data"] / "*.txt"))
            if not raw_data_files:
                raise FileNotFoundError(
                    "Still no raw data files found. Cannot proceed."
                )

        # Check if mapping CSVs exist
        if not mappings_data_files:
            input(
                "No mappings files found. "
                "Please place them in the 'mappings' directory, then press Enter to continue..."
            )
            # Check again
            mappings_data_files = glob.glob(str(directories["mappings"] / "*.csv"))
            if not mappings_data_files:
                raise FileNotFoundError(
                    "Still no mappings files found. Cannot proceed."
                )

    ###################################################################################################
    ###################################################################################################
    # Import the necessary libraries and set up the working directory
    ###################################################################################################
    ###################################################################################################

    standard_curve_path = cfg.standard_curve_path
    mscarlett_threshold = cfg.mscarlett_threshold
    cv_threshold = cfg.cv_threshold
    rate_stdev_from_background = cfg.rate_stdev_from_background
    fit_interval = (cfg.fit_time_interval_low, cfg.fit_time_interval_high)

    # standard_curve_path = (
    #     "/projects/ml/itopt/neo2_standards/241202_standard_curves_jake_mod.json"
    # )
    # mscarlett_threshold = 0.025  # uM mscarlett in enzyme reaction
    # cv_threshold = 1  #
    # rate_stdev_from_background = 3  # rate
    # fit_interval = (0, 900)  # seconds

    # Get all files in the directories["working"] directory
    raw_data_files = glob.glob(str(directories["raw_data"] / "*.txt"))
    mappings_data_files = glob.glob(str(directories["mappings"] / "*"))

    print(f"found {len(raw_data_files)} raw data files")
    print(f"found {len(mappings_data_files)} mapping files")

    mappings_df_list = []
    mscarlett_df_list = []
    kinetics_df_list = []

    for file in mappings_data_files:
        ## Import well mappings
        df_mappings = pd.read_csv(file)
        df_mappings = clean_mappings_df(df_mappings)
        mappings_df_list.append(df_mappings)

    df_mappings = pd.concat(mappings_df_list)

    ##############################
    ##############################
    # correct mappings files
    df_temp = df_mappings[
        (df_mappings.replicate_number == 1) & (df_mappings.sample_type == "sample")
    ]
    duplicates = df_temp[df_temp.name.duplicated()].reset_index(drop=True).name.tolist()
    print(f"found {len(duplicates)} duplicates in the mappings files")

    # samples = df_mappings[df_mappings.sample_type == "sample"].name.unique()

    duplicate_name_pairs = []
    duplicate_names = []

    for duplicate_name in duplicates:
        source_wells = df_mappings.loc[df_mappings.name == duplicate_name][
            "source_dna_well"
        ].unique()
        number_of_duplicates = len(source_wells)

        for i in range(0, number_of_duplicates):  # -1
            source_well_to_change = df_mappings.loc[df_mappings.name == duplicate_name][
                "source_dna_well"
            ].unique()[i]

            new_duplicate_name = duplicate_name + f":duplicate{i}"
            duplicate_name_pairs.append((duplicate_name, new_duplicate_name))
            duplicate_names.append(duplicate_name)
            duplicate_names.append(duplicate_name)
            duplicate_name_pairs.append((duplicate_name, duplicate_name))
            duplicate_names.append(new_duplicate_name)

            df_mappings.loc[
                (df_mappings.name == duplicate_name)
                & (df_mappings.source_dna_well == source_well_to_change),
                "name",
            ] = new_duplicate_name

    df_temp = df_mappings[
        (df_mappings.replicate_number == 1) & (df_mappings.sample_type == "sample")
    ]
    print(
        f"there are {len(df_temp[df_temp.name.duplicated()].reset_index(drop=True).name.tolist())} duplicates that weren't renamed"
    )
    ##############################
    ##############################

    for file in raw_data_files:
        ## Import mscarlett and merge with mappings
        df_mscarlett = parse_neo2_endpoint(file, start_row=54, parse_name=True)
        mscarlett_df_list.append(df_mscarlett)

        ## Import kinetics and merge with mappings
        df_kinetics = parse_neo2_kinetics(file, start_row=58, parse_name=True)
        kinetics_df_list.append(df_kinetics)

    ## Combine all dataframes
    df_mscarlett = pd.concat(mscarlett_df_list)
    df_kinetics = pd.concat(kinetics_df_list)

    ## Merge with mappings
    df_mscarlett = df_mscarlett.merge(
        df_mappings, on=["well", "sample_number", "plate_number"], how="left"
    )
    df_kinetics = df_kinetics.merge(
        df_mappings, on=["well", "sample_number", "plate_number"], how="left"
    )

    selected_columns = ["sample_number", "plate_number"]
    print(
        f"found mappings for {len(df_mappings[selected_columns].drop_duplicates())} total plates from {len(df_mappings.sample_number.unique())} samples"
    )
    print(
        f"found mscarlett for {len(df_mscarlett[selected_columns].drop_duplicates())} total plates from {len(df_mscarlett.sample_number.unique())} samples"
    )
    print(
        f"found kinetics for {len(df_kinetics[selected_columns].drop_duplicates())} total plates from {len(df_kinetics.sample_number.unique())} samples"
    )

    # check that all the plates have serial numbers
    print(f"found mscarlett serial numbers: {df_mscarlett['serial_number'].unique()}")
    print(f"found kinetics serial numbers: {df_kinetics['serial_number'].unique()}")

    ## Apply standard curves
    df_mscarlett = apply_mscarlett_standard_curve(
        df_mscarlett,
        standard_curve_path,
        epsilon=1e-8,
        write_path=None,
    )
    df_mscarlett.to_csv(directories["processed_data"] / f"1_mscarlett.csv")

    df_kinetics = apply_4MU_standard_curve(
        df_kinetics,
        standard_curve_path,
        epsilon=1e-8,
        write_path=None,
    )
    df_kinetics.to_csv(directories["processed_data"] / f"2_kinetics.csv")

    ## calculate max 4MU
    df_4MU_max = calculate_max_4MU(df_kinetics)

    ## fit line to 4MU data to determine rate
    df_kinetics_fits = apply_tidy_kinetic_fit(
        df_kinetics,
        fit_function=fit_line,
        time_column="time",
        y_column="4MU_um",
        time_range=fit_interval,
        group_columns=[
            "well",
            "replicate_number",
            "sample_number",
            "name",
            "sample_type",
            "plate_number",
        ],
        clean=True,
    )

    ## subract background rate
    df_kinetics_fits = apply_tidy_background_subtraction(
        df_kinetics_fits,
        value="rate",
        grouping=["sample_number", "plate_number"],
        epsilon=1e-8,
    )

    ## normalize by mscarlett
    df_kinetics_fits = normalize_rate_by_mscarlett(df_kinetics_fits, df_mscarlett)
    df_kinetics.to_csv(directories["processed_data"] / f"3_kinetics_fits.csv")

    ## calculate aggregated mscalett stats
    df_mscarlett_stats = calculate_mscarlett_stats(df_mscarlett, epsilon=1e-8)
    df_kinetics_stats = calculate_kinetics_stats(
        df_kinetics_fits, df_4MU_max, epsilon=1e-8
    )

    ## generate fitted data for plotting
    df_kinetics_fitted = generate_kinetic_fit_data(df_kinetics, df_kinetics_fits)
    df_kinetics_fitted.to_csv(directories["processed_data"] / f"4_kinetics_fitted.csv")

    ## apply filters
    ## mscarlett threshold applied to each replicate individually and all three must pass
    ## cv threshold applied to cv
    ## background stdev threshold applied to each replicate individually and all three must pass
    df_stats = apply_filters(
        df_kinetics_fits,
        df_kinetics_stats,
        df_mscarlett_stats,
        mscarlett_threshold=mscarlett_threshold,
        cv_threshold=cv_threshold,
        stdev_from_background=rate_stdev_from_background,
    )

    df_export = build_complete_df(
        df_mappings, df_mscarlett, df_4MU_max, df_kinetics_fits, df_stats
    )
    df_export.to_csv(directories["processed_data"] / f"5_complete_df.csv")

    # print out some filtering stats
    print(f"starting with {len(df_export)} samples")
    print(
        f"samples that pass the mscarlett filter: {len(df_export[df_export['mscarlett_threshold_pass'] == True])}"
    )
    print(
        f"samples that pass the rate filter: {len(df_export[df_export['rate_norm_threshold_pass'] == True])}"
    )
    print(
        f"samples that pass the cv filter: {len(df_export[df_export['rate_norm_cv_threshold_pass'] == True])}"
    )

    df_pass = df_export[df_export["mscarlett_threshold_pass"] == True]
    df_pass = df_pass[df_pass["rate_norm_cv_threshold_pass"] == True]
    print(f"samples that pass mscarlett and cv filter: {len(df_pass)}")

    df_pass = df_export[df_export["mscarlett_threshold_pass"] == True]
    df_pass = df_pass[df_pass["rate_norm_threshold_pass"] == True]
    df_pass = df_pass[df_pass["rate_norm_cv_threshold_pass"] == True]
    print(f"samples that pass mscarlett, rate, and cv filter: {len(df_pass)}")

    ##############################
    ##############################
    # plot the mscarlett distribution
    ##############################
    ##############################
    df_plot = df_mscarlett
    title = f""
    figure_name = f"mscarlett_distibution"
    figure_directory = directories["graphs/aggregated"]
    write_figure = True
    write_format = "png"

    # filter for samples where all three replicates pass the mscarlett threshold
    df_plot = df_plot[df_plot["sample_type"].isin(["sample", "positive_control"])]

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    sns.histplot(
        data=df_plot,
        x="mscarlett_um",
        hue="sample_type",
        multiple="layer",
        kde=False,
        stat="percent",  # Normalize to density
        common_norm=False,  # Normalize each hue group independently
        ax=axes,
    )
    plt.xlabel("[mScarlett] (μM)")
    plt.ylabel("% of Samples")
    plt.show()

    if write_figure:
        figure_path = figure_directory / f"{figure_name}.{write_format}"
        fig.savefig(figure_path, format=write_format)

    ##############################
    ##############################
    # plot the rate_norm distribution
    ##############################
    ##############################

    df_plot = df_kinetics_fits
    title = f""
    figure_name = f"rate_norm_distibution"
    figure_directory = directories["graphs/aggregated"]
    write_figure = True
    write_format = "png"

    # filter for samples where all three replicates pass the mscarlett threshold
    df_plot = df_plot[df_plot["mscarlett_um"] > mscarlett_threshold]
    df_plot = df_plot[df_plot["sample_type"].isin(["sample", "positive_control"])]

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    sns.histplot(
        data=df_plot,
        x="rate_minus_background_normalized",
        hue="sample_type",
        multiple="layer",
        kde=False,
        stat="percent",  # Normalize to density
        common_norm=False,  # Normalize each hue group independently
        ax=axes,
    )
    plt.xlabel("Normalized Rate (\u03BCM 4MU/s/\u03BCM mScarlett)")
    plt.ylabel("% of Samples")
    plt.yscale("log")
    plt.show()

    if write_figure:
        figure_path = figure_directory / f"{figure_name}.{write_format}"
        fig.savefig(figure_path, format=write_format)

    ##############################
    ##############################
    # plot the distribution of mscarlett by plate
    ##############################
    ##############################

    df_plot = df_export
    value = "mscarlett"
    title = f"mscarlett distributions"
    figure_name = f"mscarlett_plate_distibutions"
    figure_directory = directories["graphs/aggregated"]
    write_figure = True
    write_format = "png"

    # df_plot = df_plot[df_plot["mscarlett_threshold_pass"] == True]
    # df_plot = df_plot[df_plot["rate_norm_cv_threshold_pass"] == True]
    df_plot = df_plot[df_plot["sample_type"].isin(["sample", "positive_control"])]

    df_plot["sample_plate"] = (
        "sample_"
        + df_plot["sample_number"].astype(str)
        + "_plate_"
        + df_plot["plate_number"].astype(str)
    )

    df_plot = df_plot.sort_values(by=["sample_number", "plate_number"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(
        data=df_plot, x="sample_plate", y=f"{value}_mean", color="white", ax=axes[0]
    )
    sns.stripplot(
        data=df_plot,
        x="sample_plate",
        y=f"{value}_mean",
        size=4,
        color="black",
        alpha=0.25,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("[mscarlett] (\u03BCM mScarlett)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

    sns.boxplot(
        data=df_plot, x="sample_plate", y=f"{value}_cv", color="white", ax=axes[1]
    )
    sns.stripplot(  # stripplot
        data=df_plot,
        x="sample_plate",
        y=f"{value}_cv",
        size=4,
        color="black",
        alpha=0.25,
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Coefficient of Variation")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

    fig.suptitle(f"mscarlett distributions")

    plt.show()

    if write_figure:
        figure_path = figure_directory / f"{figure_name}.{write_format}"
        fig.savefig(figure_path, format=write_format)

    ##############################
    ##############################
    # plot the distribution of rate_norm by plate
    ##############################
    ##############################

    df_plot = df_export
    value = "rate_norm"
    title = f"Normalized rate distributions\nmscarlett threshold: {mscarlett_threshold} μM\nrate cv threshold: {cv_threshold}"
    figure_name = f"rate_norm_plate_distibutions"
    figure_directory = directories["graphs/aggregated"]
    write_figure = True
    write_format = "png"

    df_plot = df_export
    df_plot = df_plot[df_plot["mscarlett_threshold_pass"] == True]
    df_plot = df_plot[df_plot["rate_norm_cv_threshold_pass"] == True]
    df_plot = df_plot[df_plot["sample_type"].isin(["sample", "positive_control"])]

    df_plot["sample_plate"] = (
        "sample_"
        + df_plot["sample_number"].astype(str)
        + "_plate_"
        + df_plot["plate_number"].astype(str)
    )

    df_plot = df_plot.sort_values(by=["sample_number", "plate_number"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(
        data=df_plot, x="sample_plate", y=f"{value}_mean", color="white", ax=axes[0]
    )
    sns.stripplot(
        data=df_plot,
        x="sample_plate",
        y=f"{value}_mean",
        size=4,
        color="black",
        alpha=0.25,
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Normalized Rate Mean (\u03BCM 4MU/s/\u03BCM mScarlett)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)

    sns.boxplot(
        data=df_plot, x="sample_plate", y=f"{value}_cv", color="white", ax=axes[1]
    )
    sns.stripplot(
        data=df_plot,
        x="sample_plate",
        y=f"{value}_cv",
        size=4,
        color="black",
        alpha=0.25,
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Coefficient of Variation")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)

    fig.suptitle(title)

    plt.show()

    if write_figure:
        figure_path = figure_directory / f"{figure_name}.{write_format}"
        fig.savefig(figure_path, format=write_format)

    ##############################
    ##############################
    # parity plots of all aggregated data
    ##############################
    ##############################

    df_plot = df_mscarlett
    value = "mscarlett_um"
    replicates = df_plot.replicate_number.unique().tolist()
    title = f"[mscarlett] (\u03BCM) - Aggregated"
    figure_name = f"mcarlett_parity_aggregated"
    figure_directory = directories["graphs/aggregated"]
    write_figure = True
    write_format = "png"

    fig, axs = parity_plot(
        df_plot,
        value,
        replicates,
        title,
        figure_name,
        figure_directory,
        write_figure,
        write_format,
    )

    df_plot = df_kinetics_fits
    value = "rate"
    replicates = df_plot.replicate_number.unique().tolist()
    title = f"Rate (\u03BCM 4MU/s) - Aggregated"
    figure_name = f"rate_parity_aggregated"
    figure_directory = directories["graphs/aggregated"]
    write_figure = True
    write_format = "png"

    fig, axs = parity_plot(
        df_plot,
        value,
        replicates,
        title,
        figure_name,
        figure_directory,
        write_figure,
        write_format,
    )

    # filter for samples where all three replicates pass the mscarlett threshold
    df_plot = df_kinetics_fits
    df_filter = df_stats[df_stats.mscarlett_threshold_pass == True]
    df_plot = df_plot[df_plot["name"].isin(df_filter.name)]
    df_plot = df_plot[df_plot["sample_type"] == "sample"]

    value = "rate_minus_background_normalized"
    replicates = df_plot.replicate_number.unique().tolist()
    title = f"Filtered Normalized Rate (\u03BCM 4MU/s/\u03BCM mScarlett) - Aggregated"
    figure_name = f"rate_norm_parity_aggregated"
    figure_directory = directories["graphs/aggregated"]
    write_figure = True
    write_format = "png"

    fig, axs = parity_plot(
        df_plot,
        value,
        replicates,
        title,
        figure_name,
        figure_directory,
        write_figure,
        write_format,
    )

    ##############################
    ##############################
    # generate inidividual plots for each plate
    ##############################
    ##############################

    df = df_kinetics_fits

    # Get unique pairs of plate_number and sample_number
    unique_pairs = df[["sample_number", "plate_number"]].drop_duplicates()

    # Optionally, convert to a list of tuples if needed
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))

    for unique_pair in unique_pairs_list:
        df_plot = df_mscarlett
        value = "mscarlett_um"
        sample_number = unique_pair[0]
        plate_number = unique_pair[1]
        replicates = df_plot.replicate_number.unique().tolist()
        title = f"[mscarlett] (\u03BCM) - Sample {sample_number}, Plate {plate_number}"
        figure_name = f"mcarlett_parity_sample{sample_number}_plate{plate_number}"
        figure_directory = directories["graphs/per_plate"]
        write_figure = True
        write_format = "png"

        fig, axs = parity_plot(
            df_plot,
            value,
            replicates,
            title,
            figure_name,
            figure_directory,
            write_figure,
            write_format,
            sample_number,
            plate_number,
        )

        df_plot = df_kinetics_fits
        value = "rate"
        sample_number = unique_pair[0]
        plate_number = unique_pair[1]
        replicates = df_plot.replicate_number.unique().tolist()
        title = f"Rate (\u03BCM 4MU/s) - Sample {sample_number}, Plate {plate_number}"
        figure_name = f"rate_parity_sample{sample_number}_plate{plate_number}"
        figure_directory = directories["graphs/per_plate"]
        write_figure = True
        write_format = "png"

        fig, axs = parity_plot(
            df_plot,
            value,
            replicates,
            title,
            figure_name,
            figure_directory,
            write_figure,
            write_format,
            sample_number,
            plate_number,
        )

        # filter for samples where all three replicates pass the mscarlett threshold
        df_plot = df_kinetics_fits
        df_filter = df_stats[df_stats.mscarlett_threshold_pass == True]
        df_plot = df_plot[df_plot["name"].isin(df_filter.name)]
        df_plot = df_plot[df_plot["sample_type"] == "sample"]

        value = "rate_minus_background_normalized"
        sample_number = unique_pair[0]
        plate_number = unique_pair[1]
        replicates = df_plot.replicate_number.unique().tolist()
        title = f"Filtered Normalized Rate (\u03BCM 4MU/s/\u03BCM mScarlett) - Sample {sample_number}, Plate {plate_number}"
        figure_name = f"rate_norm_parity_sample{sample_number}_plate{plate_number}"
        figure_directory = directories["graphs/per_plate"]
        write_figure = True
        write_format = "png"

        fig, axs = parity_plot(
            df_plot,
            value,
            replicates,
            title,
            figure_name,
            figure_directory,
            write_figure,
            write_format,
            sample_number,
            plate_number,
        )

        df_plot = df_mscarlett
        value = "mscarlett_um"
        sample_number = unique_pair[0]
        plate_number = unique_pair[1]
        title = f"[mscarlett] (\u03BCM) - Sample {sample_number}, Plate {plate_number}"
        figure_name = f"mcarlett_heatmap_sample{sample_number}_plate{plate_number}"
        figure_directory = directories["graphs/per_plate"]
        write_figure = True
        write_format = "png"

        plot_mscarlett_heatmap(
            df_plot,
            value,
            title,
            figure_name,
            figure_directory,
            sample_number,
            plate_number,
            write_figure,
            write_format,
        )

        df_plot = df_kinetics
        value = "value"
        sample_number = unique_pair[0]
        plate_number = unique_pair[1]
        title = f"Kinetics - Sample {sample_number}, Plate {plate_number}"
        figure_name = f"kinetics_sample{sample_number}_plate{plate_number}"
        figure_directory = directories["graphs/per_plate"]
        write_figure = True
        write_format = "png"

        plot_384_kinetics(
            df_plot,
            value,
            title,
            figure_name,
            figure_directory,
            write_figure,
            write_format,
            sample_number,
            plate_number,
        )

    return None


if __name__ == "__main__":
    main()
