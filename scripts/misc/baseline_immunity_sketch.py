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

pop_table = pop_table.with_columns(pl.Series("percentage", [100.0, 0.0, 0.0]))

vacc_table = vacc_table.with_columns(
    pl.Series("coverage", [100.0, 100.0, 100.0, 100.0])
)

# get range mappings
pop_table = mp.add_pop_ranges_to_pop_table(pop_table)
vacc_table = mp.add_threshold_values_to_vacc_table(vacc_table)

print(pop_table)
print(vacc_table)


expected_df = mp.build_initial_baseline_immunity_dataframe_from_user_inputs(
    pop_table,
    vacc_table,
)


print("Expected df:")
print(
    expected_df.select(
        [
            "coverage_range_min_age",
            "coverage_range_max_age",
            "threshold_text",
            "threshold_coverage",
        ]
    )
)


joined_df = mp.create_dataframe_for_baseline_immunity_calculation(expected_df)

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


mp.calculate_baseline_immunity_from_dataframe_2(joined_df)
