# This file is part of the metapop package. It contains the simulation methods
import numpy as np
import polars as pl
from .model import SEIRModel
# import what's needed from other metapop modules
from .helper import (
    construct_beta,
    initialize_population,
    build_vax_schedule,
    time_to_rate,
)
# if you want to use methods from metapop in this file under
# if __name__ == "__main__": you'll need to import them as:
# from metapop.helper import (
#     construct_beta,
#     initialize_population,
#     build_vax_schedule,
#     time_to_rate,
# )
### note: this is not recommended use within a file that is imported as a package module, but it can be useful for testing purposes

__all__ = [
    "run_model",
    "simulate",
]


# run a single simulation of the SEIR model given a model instance
def run_model(model, u, t, steps, groups, S, V, E1, E2, I1, I2, R, Y, X):
    """
    Update the population arrays based on the SEIR model.

    Args:
        model: The SEIR model instance.
        u: The initial state.
        t: The time array.
        steps: The number of time steps.
        groups: The number of groups.
        S, V, E1, E2, I1, I2, R, Y, X: The population arrays to be updated. Y is a infection counter (counted when they become infectious I1). X is vaccine uptake counter.

    Returns:
        S, V, E1, E2, I1, I2, R, Y, X, u
    """
    for j in range(1, steps):
        u = model.seirmodel(u, t[j])
        for group in range(groups):
            S[j, group], V[j, group], E1[j, group], E2[j, group], I1[j, group], I2[j, group], R[j, group], Y[j, group], X[j, group] = u[group]

    return S, V, E1, E2, I1, I2, R, Y, X, u

def simulate(parms):
    #### Set up rate params
    parms["sigma"] = time_to_rate(parms["latent_duration"])
    parms["gamma"] = time_to_rate(parms["infectious_duration"])
    parms["sigma_scaled"] = parms["sigma"] * parms["n_e_compartments"]
    parms["gamma_scaled"] = parms["gamma"] * parms["n_i_compartments"]

    #### Set beta matrix based on desired R0 and connectivity scenario ###
    parms["beta"] = construct_beta(parms)

    #### Set up vaccine schedule for group 2
    parms["vaccination_uptake_schedule"] = build_vax_schedule(parms)

    #### Set up the model time steps
    steps = parms["tf"]
    t = np.arange(1, steps + 1)

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
