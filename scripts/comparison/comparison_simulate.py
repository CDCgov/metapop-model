import os

import griddler
import griddler.griddle
import polars as pl
import polars.selectors as cs

from metapop import simulate_replicates

if __name__ == "__main__":
    # setup output directory
    os.makedirs("output", exist_ok=True)
    output_dir = "output/comparison"
    os.makedirs(output_dir, exist_ok=True)
    # Baseline + isolation -> 8
    parameter_sets = griddler.griddle.read("scripts/comparison/comparison_config.yaml")

    results_all = simulate_replicates(parameter_sets)

    results_all = results_all.with_columns(
        pl.col("initial_vaccine_coverage").list.get(0).alias("initial_vaccine_coverage")
    )

    results = results_all.select(
        cs.by_name(
            [
                "intervention_scenario",
                "total_vaccine_uptake_doses",
                "vaccine_uptake_start_day",
                "symptomatic_isolation_start_day",
                "pre_rash_isolation_start_day",
                "initial_vaccine_coverage",
                "t",
                "group",
                "S",
                "V",
                "E1",
                "E2",
                "I1",
                "I2",
                "R",
                "Y",
                "X",
                "replicate",
            ]
        )
    )

    results.write_csv(os.path.join(output_dir, "results.csv"))
