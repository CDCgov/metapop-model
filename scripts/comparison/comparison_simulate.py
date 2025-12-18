import os
from datetime import datetime

import griddler
import griddler.griddle
import numpy as np
import polars as pl
import polars.selectors as cs

from metapop import simulate_replicates
from metapop.analyzer import (
    add_daily_incidence_scenario,
    add_week_column,
    create_filename,
    create_intervention_summary_table,
    trim_string_column,
)
from metapop.app_helper import read_parameters
from metapop.helper import seed_from_string

if __name__ == "__main__":
    ## Setup experiment ##
    os.makedirs("output", exist_ok=True)
    output_dir = "output/comparison"
    os.makedirs(output_dir, exist_ok=True)

    config_file = "scripts/comparison/comparison_config.yaml"
    expt_name = "proportional_vax"
    run_date = datetime.now().strftime("%Y_%m_%d")
    parameter_sets = griddler.griddle.read(config_file)

    ## Run simulations ##
    results_all = simulate_replicates(parameter_sets)

    results_all = results_all.with_columns(
        pl.col("initial_vaccine_coverage").list.get(0).alias("initial_vaccine_coverage")
    )

    ## Post process simulation results for plotting and summarizing ##
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

    results = add_week_column(results)
    results = add_daily_incidence_scenario(results, groups=[0])
    results = trim_string_column(results, "intervention_scenario", char_length=2)

    sims_filename = create_filename(
        base_name="simulation_runs",
        date=run_date,
        suffix=expt_name,
        fmt=".csv",
    )

    results.write_csv(os.path.join(output_dir, sims_filename))

    ## Create summary tables ##
    parms = read_parameters(config_file)
    hosp_rng = np.random.default_rng(
        [parms["seed"], seed_from_string("hospitalizations")]
    )

    summary_table = create_intervention_summary_table(
        results,
        IHR=0.15,
        scenario_order=[
            "all_interventions",
            "all_interventions_delay",
            "isolation_and_quarantine",
            "isolation_only",
        ],
        rng=hosp_rng,
    )

    table_filename = create_filename(
        base_name="summary_table",
        date=run_date,
        suffix=expt_name,
        fmt=".csv",
    )

    summary_table.write_csv(os.path.join(output_dir, table_filename))
