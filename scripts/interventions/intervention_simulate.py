import os
import polars.selectors as cs
import griddler
import griddler.griddle
from metapop import simulate

if __name__ == "__main__":
    # setup output directory
    os.makedirs("output", exist_ok=True)
    output_dir = "output/interventions"
    os.makedirs(output_dir, exist_ok=True)

    parameter_sets = griddler.griddle.read(
        "scripts/interventions/intervention_config.yaml"
    )
    results_all = griddler.run_squash(griddler.replicated(simulate), parameter_sets)
    results = results_all.select(
        cs.by_name(
            [
                "total_vaccine_uptake_doses",
                "symptomatic_isolation_start_day",
                "pre_rash_isolation_start_day",
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
