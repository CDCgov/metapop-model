import os

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


def add_threshold_coverage_to_df(df, vacc_table):
    # add threshold_coverage column
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
            "fraction_population_covered"
        ),
    )
    return expected_df


config_path = os.path.join(
    os.path.dirname(__file__), "..", "metapop", "app_assets", "one_pop_config.yaml"
)

# print(config_path)

parms = mp.read_parameters(config_path)

pop_table = mp.initialize_pop_table(parms)
pop_table = pop_table.with_columns(pl.Series("percentage", [100, 0, 0]))

vacc_table = mp.initialize_vacc_table(parms)

vacc_table = vacc_table.with_columns(pl.Series("coverage", [100, 100, 0, 0]))

print(pop_table)
print(vacc_table)

# get population range mappings
pop_table = add_pop_ranges_to_pop_table(pop_table)
print(pop_table)

# get_coverage_distribution_mapping(pop_table, vacc_table)
vacc_table = get_threshold_values(vacc_table)
cutoff_values = get_coverage_cutoffs(vacc_table)
print("Cutoff values:", cutoff_values)
print("Vacc table with numeric thresholds:")
print(vacc_table)

coverage_range = get_coverage_ranges(cutoff_values)
print("Coverage ranges:", coverage_range)

# threshold_text = []
# for i in coverage_range:
#     val = get_threshold_label_from_coverage_range(i, vacc_table)
#     # threshold_text.append(closest_row["threshold"])
#     threshold_text.append(val)

# print("Threshold text:", threshold_text)

# start building dataframe
df = pl.DataFrame(
    {
        "coverage_range": coverage_range,
    }
)

# add threshold_text column
# df = df.with_columns(
#     (
#         pl.col("coverage_range").map_elements(
#             lambda x: get_threshold_label_from_coverage_range(x, vacc_table),
#             return_dtype=pl.String,
#         )
#     ).alias("threshold_text")
# )
df = add_threshold_text_to_df(df, vacc_table)


# add pop_text column
# df = df.with_columns(
#     (
#         pl.col("coverage_range").map_elements(
#             lambda x: get_pop_label_from_coverage(x, pop_table),
#             return_dtype=pl.String,
#         )
#     ).alias("pop_text")
# )
df = add_pop_label_to_df(df, pop_table)


# add pop_range column
# df = df.with_columns(
#     (
#         pl.col("pop_text").map_elements(
#             convert_pop_text, return_dtype=pl.List(pl.Int64)
#         )
#     ).alias("pop_range")
# )
df = add_pop_range_to_df(df)


# add threshold_coverage column
# df = df.with_columns(
#     (
#         pl.col("threshold_text").map_elements(
#             lambda x: vacc_table.filter(pl.col("threshold") == x)["coverage"][0],
#             return_dtype=pl.Float64,
#         )
#     ).alias("threshold_coverage")
# )
df = add_threshold_coverage_to_df(df, vacc_table)

print("Real dataframe with threshold text:")
print(df)


# df = df.with_columns(
#     pl.col("coverage_range").list.get(0).alias("coverage_range_min_age"),
# )
# df = df.with_columns(
#     pl.col("coverage_range").list.get(1).alias("coverage_range_max_age"),
# )
df = add_coverage_range_min_max_to_df(df)

expected_df = df.clone()


# threshold_text = [
#     "2 years",
#     "2 years",
#     "5 years (kindergarten)",
#     "13 years",
#     "18+",
#     "18+",
# ]

# pop_ranges = [
#     (0, 5),
#     (0, 5),
#     (0, 5),
#     (5, 18),
#     (5, 18),
#     (18, 100),
# ]

# pop_text = [
#     "<5",
#     "<5",
#     "<5",
#     "5-17",
#     "5-17",
#     "18+",
# ]

# threshold_coverages = vacc_table["coverage"].to_list()
# threshold_coverages = [0.0, 0.0] + threshold_coverages

# expected_df = pl.DataFrame(
#     {
#         "coverage_range": coverage_range,
#         "threshold_text": threshold_text,
#         "pop_range": pop_ranges,
#         "pop_text": pop_text,
#         "threshold_coverage": threshold_coverages,
#     }
# )
# print(expected_df)


# expected_df = df.with_columns(
#     pl.col("coverage_range").list.get(0).alias("coverage_range_min_age"),
# )
# expected_df = expected_df.with_columns(
#     pl.col("coverage_range").list.get(1).alias("coverage_range_max_age"),
# )


# calculate coverage_range_length
# expected_df = expected_df.with_columns(
#     (pl.col("coverage_range_max_age") - pl.col("coverage_range_min_age")).alias(
#         "coverage_range_length"
#     ),
# )

# # calculate pop_range_length
# expected_df = expected_df.with_columns(
#     (pl.col("pop_range").list.get(1) - pl.col("pop_range").list.get(0)).alias(
#         "pop_range_length"
#     ),
# )

# # calculate fraction of population covered by coverage range
# expected_df = expected_df.with_columns(
#     (pl.col("coverage_range_length") / pl.col("pop_range_length")).alias(
#         "fraction_population_covered"
#     ),
# )
expected_df = add_pop_fraction_in_coverage_range_to_df(expected_df)


# # reorder columns
# expected_df = expected_df.select(
#     [
#         "coverage_range",
#         "threshold_text",
#         "pop_range",
#         "pop_text",
#         "pop_range_length",
#         "fraction_population_covered",
#         "coverage_range_min_age",
#         "coverage_range_max_age",
#         "coverage_range_length",
#         "threshold_coverage",
#     ]
# )

joined_df = expected_df.clone()
joined_df = joined_df.join(
    expected_df[["coverage_range_min_age", "threshold_coverage"]],
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
joined_df = joined_df.with_columns(
    pl.when(pl.col("threshold_coverage_right").is_null())
    .then(pl.col("threshold_coverage"))
    .otherwise(pl.col("threshold_coverage_right"))
    .alias("threshold_coverage_right_filled"),
)

# fill in mid coverage value
joined_df = joined_df.with_columns(
    (
        (pl.col("threshold_coverage") + pl.col("threshold_coverage_right_filled")) / 2
    ).alias("threshold_coverage_mid"),
)

# join on pop_table to get population values
joined_df = joined_df.join(
    pop_table,
    left_on="pop_text",
    right_on="population",
    how="left",
)


# drop unneeded columns
joined_df = joined_df.drop(
    ["coverage_range_min_age_right", "threshold_coverage_right"]
).rename({"threshold_coverage_right_filled": "threshold_coverage_upper"})


# print("expected df:", expected_df)

# print(
#     joined_df.select(
#         pl.col("*").exclude(
#             "pop_text",
#             "threshold_text",
#             "coverage_range",
#             "pop_range",
#             "pop_range_length",
#             "coverage_range_length",
#         )
#     )
# )
# print("Excluding a single column:")
# print("joined df:", joined_df.columns)

# now some weighted calculations using the dataframe

joined_df = joined_df.with_columns(
    (
        pl.col("threshold_coverage_mid")
        * pl.col("percentage")
        / 100
        * pl.col("fraction_population_covered")
    ).alias("weighted_coverage")
)

x = joined_df.select(pl.col("weighted_coverage")).sum()
print("weighted coverage sum:", x["weighted_coverage"][0])
