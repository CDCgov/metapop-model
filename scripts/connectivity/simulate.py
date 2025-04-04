import os
import polars.selectors as cs
import griddler
import griddler.griddle
from metapop.model import *

if __name__ == "__main__":
    # setup output directory
    os.makedirs("output", exist_ok=True)
    output_dir = "output/connectivity"
    os.makedirs(output_dir, exist_ok=True)

    parameter_sets = griddler.griddle.read("scripts/connectivity/config.yaml")
    results_all = griddler.run_squash(griddler.replicated(simulate), parameter_sets)
    results = results_all.select(cs.by_name(['initial_coverage_scenario', 'k_21', 't', 'group', 'S', 'V', 'E1', 'E2', 'I1', 'I2', 'R', 'Y', 'X', 'replicate']))
    results.write_csv("output/connectivity/results.csv")
