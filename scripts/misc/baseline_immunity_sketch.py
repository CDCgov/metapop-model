import os

import polars as pl

import metapop as mp

config_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "metapop",
    "app_assets",
    "one_pop_config.yaml",
)

parms = mp.read_parameters(config_path)

pop_table = mp.initialize_pop_table(parms)
vacc_table = mp.initialize_vacc_table(parms)

# scenario: all population under 5 years old age group, full coverage for all eligible age groups
pop_table = pop_table.with_columns(pl.Series("percentage", [100.0, 0.0, 0.0]))

vacc_table = vacc_table.with_columns(
    pl.Series("coverage", [100.0, 100.0, 100.0, 100.0])
)

# get range mappings
pop_table = mp.add_pop_ranges_to_pop_table(pop_table)
vacc_table = mp.add_threshold_values_to_vacc_table(vacc_table)

immunity_df = mp.build_initial_baseline_immunity_dataframe_from_user_inputs(
    pop_table,
    vacc_table,
)

print(
    "Scenario: all population under 5 years old age group, full coverage for all eligible age groups"
)
print("immunity df:")
print(
    immunity_df.select(
        [
            "coverage_range_min_age",
            "coverage_range_max_age",
            "threshold_text",
            "threshold_coverage",
        ]
    )
)


joined_df = mp.create_dataframe_for_baseline_immunity_calculation(immunity_df)

print("joined df for baseline immunity calculation:")
print(
    joined_df.select(
        [
            "coverage_range_min_age",
            "coverage_range_max_age",
            "threshold_text",
            "fraction_population_in_coverage_range",
            "pop_percentage",
            "threshold_coverage",
            "threshold_coverage_upper",
            "threshold_coverage_mid",
        ]
    )
)


immunity = mp.calculate_baseline_immunity_from_dataframe(joined_df)
print(f"Final baseline immunity value: {immunity * 100:.0f}%")


immunity_2 = mp.calculate_baseline_immunity_from_dataframe_2(joined_df)
print(f"Final baseline immunity value 2: {immunity_2 * 100:.0f}%\n")


# a few examples where the two calculations differ
# scenario 1: only full coverage at 2 years old, all population under 5 years old

pop_table = pop_table.with_columns(pl.Series("percentage", [100.0, 0.0, 0.0]))
vacc_table = vacc_table.with_columns(
    pl.when(pl.col("threshold") == "2 years")
    .then(100.0)
    .otherwise(0.0)
    .alias("coverage")
)

expected_df = mp.build_initial_baseline_immunity_dataframe_from_user_inputs(
    pop_table,
    vacc_table,
)

joined_df = mp.create_dataframe_for_baseline_immunity_calculation(expected_df)
immunity = mp.calculate_baseline_immunity_from_dataframe(joined_df)
immunity_2 = mp.calculate_baseline_immunity_from_dataframe_2(joined_df)

print(
    f"All population under 5, only coverage (100%) at 2 years old -> Immunity 1: {immunity*100:.2f}%, Immunity 2: {immunity_2*100:.2f}%"
)
