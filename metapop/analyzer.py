# This file is part of the metapop package.
# It contains post processing methods for static products.
import warnings

import numpy as np
import polars as pl

__all__ = [
    "create_filename",
    "trim_string_column",
    "add_daily_incidence_scenario",
    "add_week_column",
    "relative_difference",
    "get_table",
    "calculate_outbreak_summary",
    "create_intervention_summary_table",
]


def relative_difference(
    data: pl.DataFrame,
    col_name: str,
    group_values: list,
    group_col_name: str = "Scenario",
    identifier: str = None,
):
    """
    Function to bootstrap relative differences between two groups in a Polars DataFrame.
    Relative differences are calculated from the values in `col_name` for the two groups specified in `group_values`.
    The first element of `group_values` is considered the base group, and the second element is the comparison group.
    The relative difference is calculated pairwise for each possible combination of the two groups, which do not have to be the same length.
    The median relative difference and confidence interval of the relative difference distribution are then returned.

    Args:
        data (pl.DataFrame): The input DataFrame containing the data.
        col_name (str): The name of the column for which to compute relative differences.
        group_values (list): A list containing two unique values from the group column to compare.
        group_col_name (str): The name of the column that contains the group values. Defaults to "Scenario".
        identifier (str): The name of the column that contains the identifier for one-to-one mapping of pairwise comparisons
            - If the identifier is set to the default of None, then the complete pairwise comparisons are conducted.

    Returns:
        tuple: An ordered tuple containing the
            (0): lower bound (2.5th percentile),
            (1): median, and
            (2): upper bound (97.5th percentile) of the relative differences.
    """

    assert len(group_values) == 2

    # If specified, pairwise comparison by each group matched along an index identifier column value
    # For the function get_table, "replicate" is used to pair intervention and no intervention scenarios
    if identifier is not None:
        diff_data = (
            data.filter(pl.col(group_col_name).is_in(group_values))
            .pivot(group_col_name, index=identifier, values=col_name)
            .filter(
                pl.col(group_values[0]).is_not_null()
                & pl.col(group_values[1]).is_not_null()
            )  # trim to base values that have a matching comparison
            .with_columns(
                (
                    pl.lit(100)
                    * (pl.col(group_values[0]) - pl.col(group_values[1]))
                    /
                    # Scale by base group median to avoid divide by zero and avoid overweighting negative differences on small base values
                    (pl.median(group_values[0]))
                ).alias("reldiff")
            )
        )
        if diff_data.is_empty():
            raise ValueError(
                f"No matching comparisons based on column {identifier}. Check input data frame or try running with all pairwise comparisons instead."
            )

        lwr = diff_data.select(pl.quantile("reldiff", quantile=0.025)).item()
        median = diff_data.select(pl.median("reldiff")).item()
        upr = diff_data.select(pl.quantile("reldiff", quantile=0.975)).item()

    # Pairwise comparison bootstrapped across all possible combinations
    else:
        # Filter the data for the two groups
        base_df = data.filter(pl.col(group_col_name) == pl.lit(group_values[0]))
        compare_df = data.filter(pl.col(group_col_name) == pl.lit(group_values[1]))

        # Compute relative differences using broadcasting
        base_vals = base_df[col_name].to_numpy()
        compare_vals = compare_df[col_name].to_numpy()
        diffs = base_vals[:, None] - compare_vals

        # Scale to median instead of each base to avoid overweighting negative differences on small base values
        rel_diffs = [100 * diff / np.median(base_vals) for diff in diffs]

        # Flatten the array and compute statistics
        lwr = np.quantile(rel_diffs, 0.025)
        median = np.median(rel_diffs)
        upr = np.quantile(rel_diffs, 0.975)

    return (lwr, median, upr)


def get_table(combined_results, IHR, rng):
    """
    Calculate the hospitalization summary based on the given IHR.

    Args:
        combined_results (pl.DataFrame): The combined results DataFrame.
        IHR                     (float): The infection hospitalization rate.
        rng       (np.random.Generator): Random number generator for binomial sampling.

    Returns:
        pl.DataFrame: A DataFrame containing the hospitalization summary.
    """
    # calculate hospitalizations based on IHR
    scenario_order = {"No interventions": 0, "Interventions": 1}
    combined_results = combined_results.sort(
        [pl.col("Scenario").replace(scenario_order), pl.col("replicate")]
    )

    combined_results = combined_results.with_columns(
        pl.Series(
            name="Hospitalizations",
            values=rng.binomial(
                combined_results["Total"].to_numpy().astype("int32"), IHR
            ),
        )
    )
    scenarios = ["No interventions", "Interventions"]

    # Specify `identifier="replicate"` to make strictly one-to-one difference comparisons
    totalinf_reldiff = relative_difference(
        combined_results,
        col_name="Total",
        group_values=scenarios,
        identifier="replicate",
    )

    hosp_reldiff = relative_difference(
        combined_results,
        col_name="Hospitalizations",
        group_values=scenarios,
        identifier="replicate",
    )

    # median and 95% Prediction Interval for infections and hospitalizations
    summary_stats = (
        combined_results.group_by("Scenario")
        .agg(
            [
                # Infections
                pl.col("Total").median().alias("inf_median"),
                pl.col("Total").quantile(0.025).alias("inf_ci_low"),
                pl.col("Total").quantile(0.975).alias("inf_ci_high"),
                # Hospitalizations
                pl.col("Hospitalizations").median().alias("hosp_median"),
                pl.col("Hospitalizations").quantile(0.025).alias("hosp_ci_low"),
                pl.col("Hospitalizations").quantile(0.975).alias("hosp_ci_high"),
            ]
        )
        .sort("Scenario", descending=True)
    )

    ## build table for the app
    infections = pl.DataFrame(
        {
            "": ["Infections, median (95% prediction interval)"],
            "No interventions": [
                f"{summary_stats['inf_median'][0]:.0f} ({summary_stats['inf_ci_low'][0]:.0f} - {summary_stats['inf_ci_high'][0]:.0f})"
            ],
            "Interventions": [
                f"{summary_stats['inf_median'][1]:.0f} ({summary_stats['inf_ci_low'][1]:.0f} - {summary_stats['inf_ci_high'][1]:.0f})"
            ],
            "Bootstrapped relative difference (%)": [
                f"{totalinf_reldiff[1]:.0f}% ({totalinf_reldiff[0]:.0f} - {totalinf_reldiff[2]:.0f})"
            ],
        }
    )
    hospitalizations = pl.DataFrame(
        {
            "": ["Hospitalizations, median (95% prediction interval)"],
            "No interventions": [
                f"{summary_stats['hosp_median'][0]:.0f} ({summary_stats['hosp_ci_low'][0]:.0f} - {summary_stats['hosp_ci_high'][0]:.0f})"
            ],
            "Interventions": [
                f"{summary_stats['hosp_median'][1]:.0f} ({summary_stats['hosp_ci_low'][1]:.0f} - {summary_stats['hosp_ci_high'][1]:.0f})"
            ],
            "Bootstrapped relative difference (%)": [
                f"{hosp_reldiff[1]:.0f}% ({hosp_reldiff[0]:.0f} - {hosp_reldiff[2]:.0f})"
            ],
        }
    )  # join the two tables
    outbreak_summary = infections.vstack(hospitalizations)

    return outbreak_summary


def calculate_outbreak_summary(combined_results, threshold):
    """
    Calculate the outbreak summary based on the given threshold.

    Args:
        combined_results (pl.DataFrame): The combined results DataFrame.
        threshold                 (int): The threshold for filtering replicates.

    Returns:
        pl.DataFrame: A DataFrame containing the outbreak summary.
    """
    # Filter combined_results for replicates where Total >= threshold
    filtered_results = combined_results.filter(pl.col("Total") >= threshold)

    # Group by Scenario and count unique replicates
    outbreak_summary = filtered_results.group_by("Scenario").agg(
        pl.col("replicate").n_unique().alias("outbreaks")
    )

    # Ensure both scenarios are present in the summary
    scenarios = ["No interventions", "Interventions"]

    for scenario in scenarios:
        if scenario not in outbreak_summary["Scenario"].to_list():
            # Add missing scenario with outbreaks = 0
            outbreak_summary = outbreak_summary.vstack(
                pl.DataFrame({"Scenario": [scenario], "outbreaks": [0]}).with_columns(
                    pl.col("outbreaks").cast(
                        outbreak_summary.schema["outbreaks"]
                    )  # Match the type
                )
            )

    return outbreak_summary


def add_daily_incidence_scenario(
    results, groups=None, scenario_column="intervention_scenario"
):
    """
    Add daily incidence to the results DataFrame, grouping by scenario column.
    Vectorized version for better performance.

    Args:
        results (pl.DataFrame): The results DataFrame.
        groups (list, optional): List of group indices. If None, process all groups.
        scenario_column (str): Name of the scenario column to group by. Default: "intervention_scenario".
                              If empty string, only group by group. If groups is also empty, only group by replicate.

    Returns:
        pl.DataFrame: The updated results DataFrame with daily incidence added.
    """
    # Filter for specified groups first
    if groups:
        results = results.filter(pl.col("group").is_in(groups))

    # Build grouping columns dynamically
    grouping_cols = ["replicate"]
    sort_cols = ["replicate"]

    if scenario_column:  # If scenario_column is not empty
        grouping_cols.append(scenario_column)
        sort_cols.append(scenario_column)

    if groups is not None:  # If groups is specified (even if empty list)
        grouping_cols.append("group")
        sort_cols.append("group")

    # Always add time column for sorting
    sort_cols.append("t")

    # Calculate incidence using vectorized operations
    results_with_incidence = results.sort(sort_cols).with_columns(
        # Calculate incidence as difference from previous Y value within each group
        (pl.col("Y") - pl.col("Y").shift(1))
        .fill_null(pl.col("Y"))  # Fill first value with Y itself, not 0
        .over(grouping_cols)  # Group by dynamically built columns
        .alias("Incidence")
    )

    # Check for negative incidence values and warn if found
    negative_count = results_with_incidence.filter(pl.col("Incidence") < 0).height
    if negative_count > 0:
        warnings.warn(
            "Warning: negative incidence values detected. "
            "This may indicate Y value misspecified for model.",
            UserWarning,
        )

    return results_with_incidence


def create_intervention_summary_table(
    results,
    rng,
    scenario_order,
    IHR=0.15,
):
    """
    Create a summary table comparing intervention scenarios to the baseline "none" scenario
    across different baseline immunity levels.

    Args:
        results (pl.DataFrame): Results DataFrame with columns including intervention_scenario,
                               initial_vaccine_coverage, replicate, Y (cumulative infections)
        rng: Random number generator
        scenario_order (list): List of scenarios in the desired order for comparison.
        IHR (float): Infection hospitalization rate (default: 0.15)

    Returns:
        pl.DataFrame: Summary table with intervention comparisons by baseline immunity level
    """
    # Save the initial RNG state
    initial_rng_state = rng.bit_generator.state

    # Get unique scenarios (excluding "none" since it's our baseline)
    scenarios = results["intervention_scenario"].unique().to_list()
    available_scenarios = [s for s in scenarios if s != "none"]

    # Filter scenario_order to only include scenarios present in the data
    intervention_scenarios = [s for s in scenario_order if s in available_scenarios]

    # Add any scenarios in the data that weren't in our predefined order
    additional_scenarios = [s for s in available_scenarios if s not in scenario_order]
    intervention_scenarios.extend(additional_scenarios)

    summary_rows = []

    for scenario in intervention_scenarios:
        # Filter data for this scenario
        scenario_data = results.filter(
            (pl.col("intervention_scenario") == scenario)
            | (pl.col("intervention_scenario") == "none")
        )

        # Get immunity levels available for this scenario
        scenario_immunity_levels = (
            scenario_data["initial_vaccine_coverage"].unique().sort().to_list()
        )

        for immunity_level in scenario_immunity_levels:
            # Create combined dataset exactly like app does
            none_data = (
                results.filter(
                    (pl.col("intervention_scenario") == "none")
                    & (pl.col("initial_vaccine_coverage") == immunity_level)
                    & (pl.col("t") == pl.col("t").max())
                )
                .select(["replicate", "Y"])
                .rename({"Y": "Total"})
                .with_columns(pl.lit("No interventions").alias("Scenario"))
            )

            intervention_data = (
                results.filter(
                    (pl.col("intervention_scenario") == scenario)
                    & (pl.col("initial_vaccine_coverage") == immunity_level)
                    & (pl.col("t") == pl.col("t").max())
                )
                .select(["replicate", "Y"])
                .rename({"Y": "Total"})
                .with_columns(pl.lit("Interventions").alias("Scenario"))
            )

            if len(none_data) == 0 or len(intervention_data) == 0:
                continue

            # Combine data
            combined_results = pl.concat([none_data, intervention_data])

            # Reset RNG to initial state before each comparison
            rng.bit_generator.state = initial_rng_state

            # Use get_table() exactly like the app (single RNG call inside get_table)
            app_table = get_table(combined_results, IHR, rng)

            # Extract results from the table
            infections_row = app_table.row(0)
            hospitalizations_row = app_table.row(1)

            # Add single row with all outcomes as columns
            summary_rows.append(
                {
                    "scenario": scenario,
                    "baseline_immunity": immunity_level,
                    "infections_no_interventions": infections_row[
                        1
                    ],  # "No interventions" column
                    "infections_interventions": infections_row[
                        2
                    ],  # "Interventions" column
                    "infections_relative_difference_pct": infections_row[
                        3
                    ],  # "Bootstrapped relative difference (%)" column
                    "hospitalizations_no_interventions": hospitalizations_row[1],
                    "hospitalizations_interventions": hospitalizations_row[2],
                    "hospitalizations_relative_difference_pct": hospitalizations_row[3],
                }
            )

    # Create final DataFrame
    summary_table = pl.DataFrame(summary_rows)

    return summary_table


def create_filename(
    base_name: str, date: str = "", suffix: str = "", format: str = ".csv"
) -> str:
    """
    Create a filename with optional date and suffix.

    Args:
        base_name: Base filename without extension
        date: Optional date string to append
        suffix: Optional suffix to append
        format: File extension (default: ".csv")

    Returns:
        Complete filename string
    """
    filename = base_name
    if date:
        filename += f"_{date}"
    if suffix:
        filename += f"_{suffix}"
    filename += format
    return filename


def trim_string_column(
    df: pl.DataFrame, column_name: str, char_length: int = 0
) -> pl.DataFrame:
    """
    Remove specified number of characters from the end of a string column.

    Args:
        df: Polars DataFrame
        column_name: Name of the string column to trim
        char_length: Number of characters to remove from end (default: 2)

    Returns:
        DataFrame with trimmed string column
    """
    if char_length <= 0:
        return df

    return df.with_columns(
        pl.col(column_name)
        .str.slice(0, pl.col(column_name).str.len_chars() - char_length)
        .alias(column_name)
    )


def add_week_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a 'week' column based on the 't' (day) column using ceiling division. Note that 't' starts at 1 in this output.

    Args:
        df: Polars DataFrame with a 't' column containing day values

    Returns:
        DataFrame with added 'week' column (ceiling of t/7)
    """
    return df.with_columns((pl.col("t") / 7).ceil().cast(pl.Int64).alias("week"))
