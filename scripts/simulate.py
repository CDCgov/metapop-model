import numpy as np
import polars as pl
import polars.selectors as cs
import griddler
import griddler.griddle
from metapop import SEIRModel
from metapop.helper import *

def simulate(parms):

    #### set beta matrix based on desired R0 and connectivity scenario ###
    parms = set_beta_parameter(parms)
    r0_base = get_r0(parms)
    parms['beta_factor'] = parms["desired_r0"] / r0_base
    parms = set_beta_parameter(parms)

    parms["beta"][1,2] *= parms["connectivity_scenario"]
    parms["beta"][2,1] *= parms["connectivity_scenario"]
    ############################

    steps = parms["tf"]
    t = np.linspace(1, steps, steps)
    groups = parms["n_groups"]

    S, V, E1, E2, I1, I2, R, Y, u = initialize_population(steps, groups, parms)

    model = SEIRModel(parms)

    S, V, E1, E2, I1, I2, R, Y, u = run_model(model, u, t, steps, groups, S, V, E1, E2, I1, I2, R, Y)

    df = pl.DataFrame({
        't': np.repeat(t, groups),
        'group': np.tile(np.arange(groups), steps),
        'S': S.flatten(),
        'V': V.flatten(),
        'E1': E1.flatten(),
        'E2': E2.flatten(),
        'I1': I1.flatten(),
        'I2': I2.flatten(),
        'R': R.flatten(),
        'Y': Y.flatten()
    })

    return df

if __name__ == "__main__":
    parameter_sets = griddler.griddle.read("scripts/config.yaml")
    results_all = griddler.run_squash(griddler.replicated(simulate), parameter_sets)
    results = results_all.select(cs.by_name(['initial_coverage_scenario', 'connectivity_scenario', 't', 'group', 'S', 'V', 'E1', 'E2', 'I1', 'I2', 'R', 'Y', 'replicate']))
    results.write_csv("output/results_test.csv")
