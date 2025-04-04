import os
import polars.selectors as cs
import griddler
import griddler.griddle
from metapop.model import *

if __name__ == "__main__":
    # setup output directory
    os.makedirs("output", exist_ok=True)
    output_dir = "output/onepop"
    os.makedirs(output_dir, exist_ok=True)

    parameter_sets = griddler.griddle.read("scripts/onepop/onepop_config.yaml")
    results_all = griddler.run_squash(griddler.replicated(simulate), parameter_sets)
    results = results_all.select(cs.by_name(['initial_coverage_scenario', 'total_vaccine_uptake_doses', 't', 'group', 'S', 'V', 'E1', 'E2', 'I1', 'I2', 'R', 'Y', 'X', 'replicate']))
    results.write_csv(os.path.join(output_dir, "results.csv"))
