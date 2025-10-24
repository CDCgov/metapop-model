import os

import numpy as np
import polars as pl

import metapop as mp


def get_threshold_values(user_cov_table):
    # apply to coverage table and return it
    user_cov_table = user_cov_table.with_columns(
        pl.col("threshold")
        .map_elements(convert_cutoff_text, return_dtype=pl.Int64)
        .alias("threshold_age")
    )
    return user_cov_table


def get_coverage_cutoffs(user_cov_table):
    cutoff_values = [0, 1] + user_cov_table["threshold_age"].to_list() + [100]
    return cutoff_values


def convert_cutoff_text(text_value):
    r = (
        text_value.replace(" ", "")
        .replace("<", "")
        .replace("+", "")
        .replace("kindergarten", "")
        .replace("(", "")
        .replace(")", "")
        .replace("years", "")
    )
    r = int(r)
    return r


def convert_pop_text(text_value):
    r = text_value.replace(" ", "").replace("<", "").replace("+", "")
    r = r.split("-")
    r = [int(i) for i in r]
    if "<" in text_value:
        min_age = 0
        max_age = r[0]
    elif "+" in text_value:
        min_age = r[0]
        max_age = 100
    else:
        min_age = r[0]
        max_age = r[1] + 1
    return min_age, max_age


def get_coverage_ranges(cutoff_values):
    coverage_range = [
        (cutoff_values[i], cutoff_values[i + 1]) for i in range(len(cutoff_values) - 1)
    ]
    return coverage_range


def get_threshold_label_from_coverage_range(coverage_range, vacc_table):
    max_age = coverage_range[1]
    closest_row_index = vacc_table.select(
        (pl.col("threshold_age") - max_age).abs().arg_min()
    ).item()
    closest_row = vacc_table.row(closest_row_index, named=True)
    return closest_row["threshold"]


# rename df to immunity_df
def add_threshold_text_to_df(df, vacc_table):
    # apply to dataframe and return it
    df = df.with_columns(
        (
            pl.col("coverage_range").map_elements(
                lambda x: get_threshold_label_from_coverage_range(x, vacc_table),
                return_dtype=pl.String,
            )
        ).alias("threshold_text")
    )
    return df


def add_pop_ranges_to_pop_table(user_pop_table):
    # apply to table and return it
    user_pop_table = user_pop_table.with_columns(
        pl.col("population")
        .map_elements(convert_pop_text, return_dtype=pl.List(pl.Int64))
        .alias("pop_range")
    )
    return user_pop_table


def get_pop_label_from_coverage(coverage_range, pop_table):
    max_age = coverage_range[1]
    closest_row_index = pop_table.select(
        (pl.col("pop_range").list.get(1) - max_age).abs().arg_min()
    ).item()
    closest_row = pop_table.row(closest_row_index, named=True)
    return closest_row["population"]


def add_pop_label_to_df(df, pop_table):
    # add pop_text column
    df = df.with_columns(
        (
            pl.col("coverage_range").map_elements(
                lambda x: get_pop_label_from_coverage(x, pop_table),
                return_dtype=pl.String,
            )
        ).alias("pop_text")
    )
    return df


def add_pop_range_to_df(df):
    # add pop_range column
    df = df.with_columns(
        (
            pl.col("pop_text").map_elements(
                convert_pop_text, return_dtype=pl.List(pl.Int64)
            )
        ).alias("pop_range")
    )
    return df


def add_pop_percentage_to_df(df, pop_table):
    # join on pop_table to get population values
    df = df.with_columns(
        (
            pl.col("pop_text").map_elements(
                lambda x: pop_table.filter(pl.col("population") == x)["percentage"][0],
                return_dtype=pl.Float64,
            )
        ).alias("pop_percentage")
    )
    return df


def add_threshold_coverage_to_df(df, vacc_table):
    # add threshold_coverage column
    # this is the coverage values corresponding to the threshold_text
    # and the coverage for the lower bound of the coverage range
    df = df.with_columns(
        (
            pl.col("threshold_text").map_elements(
                lambda x: vacc_table.filter(pl.col("threshold") == x)["coverage"][0],
                return_dtype=pl.Float64,
            )
        ).alias("threshold_coverage")
    )
    return df


def add_coverage_range_min_max_to_df(df):
    df = df.with_columns(
        pl.col("coverage_range").list.get(0).alias("coverage_range_min_age"),
    )
    df = df.with_columns(
        pl.col("coverage_range").list.get(1).alias("coverage_range_max_age"),
    )
    return df


def add_pop_fraction_in_coverage_range_to_df(df):
    # calculate coverage_range_length
    expected_df = df.with_columns(
        (pl.col("coverage_range_max_age") - pl.col("coverage_range_min_age")).alias(
            "coverage_range_length"
        ),
    )

    # calculate pop_range_length
    expected_df = expected_df.with_columns(
        (pl.col("pop_range").list.get(1) - pl.col("pop_range").list.get(0)).alias(
            "pop_range_length"
        ),
    )

    # calculate fraction of population covered by coverage range
    expected_df = expected_df.with_columns(
        (pl.col("coverage_range_length") / pl.col("pop_range_length")).alias(
            "fraction_population_in_coverage_range"
        ),
    )
    return expected_df


def build_initial_baseline_immunity_dataframe_from_user_inputs(
    pop_table,
    vacc_table,
):
    cutoff_values = get_coverage_cutoffs(vacc_table)

    coverage_range = get_coverage_ranges(cutoff_values)

    # start building dataframe
    df = pl.DataFrame(
        {
            "coverage_range": coverage_range,
        }
    )

    # add matching threshold_text column
    df = add_threshold_text_to_df(df, vacc_table)

    # add matching pop_text column
    df = add_pop_label_to_df(df, pop_table)

    # add matching pop_range column
    df = add_pop_range_to_df(df)

    # add matching population percentage column
    df = add_pop_percentage_to_df(df, pop_table)

    # add matching threshold_coverage column
    df = add_threshold_coverage_to_df(df, vacc_table)

    # add coverage_range_min_age and coverage_range_max_age columns
    df = add_coverage_range_min_max_to_df(df)

    # add fraction_population_covered column
    df = add_pop_fraction_in_coverage_range_to_df(df)

    # remove unneeded columns
    df = df.drop(
        ["coverage_range_length", "pop_range_length", "coverage_range", "pop_range"]
    )

    print("columns in df:", df.columns)
    return df


def create_dataframe_for_baseline_immunity_calculation(df):
    """"""
    # make a copy of df
    joined_df = df.clone()

    # join on itself to get the threshold_coverage for the upper bound of the coverage age range
    joined_df = joined_df.join(
        df[["coverage_range_min_age", "threshold_coverage"]],
        left_on="coverage_range_max_age",
        right_on="coverage_range_min_age",
        how="full",
        suffix="_right",
    )

    # remove row with null coverage_range
    joined_df = joined_df.filter(
        pl.col("coverage_range_min_age").is_not_null()
        & pl.col("coverage_range_max_age").is_not_null()
    )

    # fill in threshold_coverage_right with threshold_coverage where null
    # this is for the last row where there is no threshold_coverage in the
    # original dataframe for the upper bound
    joined_df = joined_df.with_columns(
        pl.when(pl.col("threshold_coverage_right").is_null())
        .then(pl.col("threshold_coverage"))
        .otherwise(pl.col("threshold_coverage_right"))
        .alias("threshold_coverage_right_filled"),
    )

    # add mid range coverage value for trapezoid rule
    joined_df = joined_df.with_columns(
        (
            (pl.col("threshold_coverage") + pl.col("threshold_coverage_right_filled"))
            / 2
        ).alias("threshold_coverage_mid"),
    )

    # drop unneeded columns
    joined_df = joined_df.drop(
        ["coverage_range_min_age_right", "threshold_coverage_right"]
    ).rename({"threshold_coverage_right_filled": "threshold_coverage_upper"})

    # reorder columns
    joined_df = joined_df.select(
        [
            "coverage_range_min_age",
            "coverage_range_max_age",
            "threshold_text",
            "pop_text",
            "fraction_population_in_coverage_range",
            "pop_percentage",
            "threshold_coverage",
            "threshold_coverage_upper",
            "threshold_coverage_mid",
        ]
    )

    # return the joined dataframe that can be used for calulating the baseline immunity from age based inputs
    return joined_df


def calculate_baseline_immunity_from_dataframe(df):
    df = df.with_columns(
        (
            pl.col("threshold_coverage_mid")
            * pl.col("pop_percentage")
            / 100
            * pl.col("fraction_population_in_coverage_range")
        ).alias("weighted_coverage")
    )

    immunity = np.round(
        df.select(pl.col("weighted_coverage")).sum()["weighted_coverage"][0], 0
    )
    return immunity


config_path = os.path.join(
    os.path.dirname(__file__), "..", "metapop", "app_assets", "one_pop_config.yaml"
)


parms = mp.read_parameters(config_path)

pop_table = mp.initialize_pop_table(parms)
vacc_table = mp.initialize_vacc_table(parms)


# get range mappings
pop_table = add_pop_ranges_to_pop_table(pop_table)
vacc_table = get_threshold_values(vacc_table)

print(pop_table)
print(vacc_table)


expected_df = build_initial_baseline_immunity_dataframe_from_user_inputs(
    vacc_table,
    pop_table,
)


print("Expected df:")
print(expected_df)


joined_df = create_dataframe_for_baseline_immunity_calculation(expected_df)

print("joined df for baseline immunity calculation:")
print(joined_df)


immunity = calculate_baseline_immunity_from_dataframe(joined_df)
print("Final baseline immunity value:", immunity)
