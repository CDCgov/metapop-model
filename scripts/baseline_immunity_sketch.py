import os

import polars as pl

import metapop as mp


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
    print(r)


def get_coverage_cutoffs(user_cov_table):
    text_cutoffs = user_cov_table["threshold"].to_list()
    cutoff_values = [""]
    print("Cutoff text values:", cutoff_values)

    for val in text_cutoffs:
        print(f"Processing cutoff value: {val}")

    # num_cutoffs = []
    # for i, row in user_cov_table.iterrows():
    #     threshold = row["threshold"]
    #     min_age = row["min_age"]
    #     cutoff_map[threshold] = min_age
    #     num_cutoffs.append(threshold)
    # return num_cutoffs, cutoff_map


def get_coverage_distribution_mapping(user_pop_table, user_cov_table):
    print("cutoffs: ", user_cov_table["threshold"].to_list())
    # num_cutoffs, cutoff_map = get_coverage_cutoffs(user_cov_table)
    # if 0 not in num_cutoffs:
    #     num_cutoffs = [0] + num_cutoffs
    # print(f"num_cutoffs: {num_cutoffs}")
    # if 100 not in num_cutoffs:
    #     num_cutoffs = num_cutoffs + [100]
    # print(f"num_cutoffs after adding 100: {num_cutoffs}")
    # df = pl.DataFrame(
    #     {
    #         "threshold": cutoff_map.keys(),
    #         "min_age": [cutoff_map[i] for i in cutoff_map.keys()],
    #     }
    # )
    # print("coverage map:", df)
    # return df, cutoff_map


config_path = os.path.join(
    os.path.dirname(__file__), "..", "metapop", "app_assets", "one_pop_config.yaml"
)

print(config_path)

parms = mp.read_parameters(config_path)

pop_table = mp.initialize_pop_table(parms)
pop_table = pop_table.with_columns(pl.Series("percentage", [100, 0, 0]))

vacc_table = mp.initialize_vacc_table(parms)

vacc_table = vacc_table.with_columns(pl.Series("coverage", [100, 100, 0, 0]))

print(pop_table)
print(vacc_table)

# get_coverage_distribution_mapping(pop_table, vacc_table)
get_coverage_cutoffs(vacc_table)


coverage_range = [
    (0, 1),
    (1, 2),
    (2, 5),
    (5, 13),
    (13, 18),
    (18, 100),
]

threshold_text = [
    "<2 years",
    "<2 years",
    "5 years (kindergarten)",
    "13 years",
    "18+ years",
    "18+ years",
]

pop_ranges = [
    (0, 5),
    (0, 5),
    (0, 5),
    (5, 18),
    (5, 18),
    (18, 100),
]

pop_text = [
    "<5",
    "<5",
    "<5",
    "5-17",
    "5-17",
    "18+",
]

threshold_coverages = vacc_table["coverage"].to_list()
threshold_coverages = [0.0, 0.0] + threshold_coverages  # + [threshold_coverages[-1]]
print("threshold coverages:", threshold_coverages)

expected_df = pl.DataFrame(
    {
        "coverage_range": coverage_range,
        "threshold_text": threshold_text,
        "pop_range": pop_ranges,
        "pop_text": pop_text,
        "threshold_coverage": threshold_coverages,
    }
)

expected_df = expected_df.with_columns(
    pl.col("coverage_range").list.get(0).alias("coverage_range_min_age"),
)
expected_df = expected_df.with_columns(
    pl.col("coverage_range").list.get(1).alias("coverage_range_max_age"),
)


# calculate coverage_range_length
expected_df = expected_df.with_columns(
    (pl.col("coverage_range_max_age") - pl.col("coverage_range_min_age"))
    # .over(pl.col("pop_range").list.get(1) - pl.col("pop_range").list.get(0))
    .alias("coverage_range_length"),
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

# reorder columns
expected_df = expected_df.select(
    [
        "coverage_range",
        "threshold_text",
        "pop_range",
        "pop_text",
        "pop_range_length",
        "fraction_population_covered",
        "coverage_range_min_age",
        "coverage_range_max_age",
        "coverage_range_length",
        "threshold_coverage",
    ]
)

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


print("expected df:", expected_df)

print(
    joined_df.select(
        pl.col("*").exclude(
            "pop_text",
            "threshold_text",
            "coverage_range",
            "pop_range",
            "pop_range_length",
            "coverage_range_length",
        )
    )
)
print("Excluding a single column:")
print("joined df:", joined_df.columns)

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
print("weighted coverage sum:", x)
