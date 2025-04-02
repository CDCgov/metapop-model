import numpy as np
import polars as pl
import polars.selectors as cs
import griddler
import griddler.griddle
from metapop import SEIRModel
from metapop.helper import *

def simulate(parms):
    #### Set up rate params
    parms["sigma"] = time_to_rate(parms["latent_duration"])
    parms["gamma"] = time_to_rate(parms["infectious_duration"])
    parms["sigma_scaled"] = parms["sigma"] * parms["n_e_compartments"]
    parms["gamma_scaled"] = parms["gamma"] * parms["n_i_compartments"]

    #### Set beta matrix based on desired R0 and connectivity scenario ###
    parms["beta"] = construct_beta(parms)

    #### set up the model time steps
    steps = parms["tf"]
    t = np.linspace(1, steps, steps)

    #### Initialize population
    groups = parms["n_groups"]
    S, V, E1, E2, I1, I2, R, Y, X, u = initialize_population(steps, groups, parms)

    #### Run the model
    model = SEIRModel(parms)
    S, V, E1, E2, I1, I2, R, Y, X, u = run_model(model, u, t, steps, groups, S, V, E1, E2, I1, I2, R, Y, X)

    #### Flatten into a dataframe
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
        'Y': Y.flatten(),
        'X': X.flatten(),
    })

    return df

if __name__ == "__main__":
    parameter_sets = griddler.griddle.read("scripts/config.yaml")
    results_all = griddler.run_squash(griddler.replicated(simulate), parameter_sets)
    results = results_all.select(cs.by_name(['initial_coverage_scenario', 'k_21', 't', 'group', 'S', 'V', 'E1', 'E2', 'I1', 'I2', 'R', 'Y', 'X', 'replicate']))
    results.write_csv("output/results.csv")
