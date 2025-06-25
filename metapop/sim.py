# This file is part of the metapop package. It contains the simulation methods
import griddler
import numpy as np
import polars as pl

# import what's needed from other metapop modules
from .helper import (
    Ind,
    build_vax_schedule,
    construct_beta,
    initialize_population,
    seed_from_string,
    time_to_rate,
)
from .model import SEIRModel

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
    "simulate_replicates",
    "get_time_array",
]


# Run a single simulation of the SEIR model given a model instance.
def run_model(
    model, u, t_array, steps, groups, S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X
):
    """
    Update the population arrays based on the SEIR model.

    Args:
        model: The SEIR model instance.
        u       (array): The initial state.
        t_array (array): The time array.
        steps     (int): The number of time steps.
        groups    (int): The number of groups.
        S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X: The population arrays to be updated. Y is a infection counter (counted when they become infectious I1). X is vaccination uptake counter.

    Returns:
        S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u
    """
    for j in range(1, steps):
        u = model.seirmodel(u, t_array[j])
        for group in range(groups):
            (
                S[j, group],
                V[j, group],
                SV[j, group],
                E1[j, group],
                E2[j, group],
                E1_V[j, group],
                E2_V[j, group],
                I1[j, group],
                I2[j, group],
                R[j, group],
                Y[j, group],
                X[j, group],
            ) = u[group]

    return S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u


def get_time_array(parms):
    """
    Get the time array for the simulation based on the total time and time step.

    Args:
        parms (dict): The parameters dictionary containing 'tf' (total time)

    Returns:
        np.ndarray: An array of time steps.
    """
    steps = parms["tf"]
    t_array = np.arange(1, steps + 1)
    return t_array


def vaccinate_on_day(
    model,
    u,
    t,
    groups,
    S,
    V,
    SV,
    E1,
    E2,
    E1_V,
    E2_V,
    I1,
    I2,
    R,
    Y,
    X,
    vaccination_uptake_schedule,
):
    """
    Check if vaccination should occur on the current day and update
    states if so. No other transitions occur in this function.

    Args:
        model: The SEIR model instance.
        u: The current state of the population.
        t: The current time step.
        groups: The number of groups in the population.
        S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X: The population arrays to be updated. Y is a infection counter (counted when they become infectious I1). X is vaccine uptake counter.
        vaccination_uptake_schedule: A schedule indicating when vaccinations occur.

    Returns:
        S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u

    """
    if t in vaccination_uptake_schedule:
        new_vaccinated, new_vaccine_failures, new_exposed_vaccinated = model.vaccinate(
            u, t
        )
        updated_susceptibles, updated_failures = model.get_updated_susceptibles(
            u, new_vaccinated, new_vaccine_failures
        )
        for group in range(groups):
            u[group][Ind.S.value] = updated_susceptibles[group]
            u[group][Ind.SV.value] = updated_failures[group]
            u[group][Ind.V.value] += new_vaccinated[group]
            u[group][Ind.X.value] += (
                new_vaccinated[group]
                + new_vaccine_failures[group]
                + new_exposed_vaccinated[group]
            )
            S[0, group] = updated_susceptibles[group]
            V[0, group] = u[group][Ind.V.value]
            SV[0, group] = updated_failures[group]
            X[0, group] = u[group][Ind.X.value]

    return S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u


def simulate(parms, seed):
    #### Set up rate params
    parms["sigma"] = time_to_rate(parms["latent_duration"])
    parms["gamma"] = time_to_rate(parms["infectious_duration"])
    parms["sigma_scaled"] = parms["sigma"] * parms["n_e_compartments"]
    parms["gamma_scaled"] = parms["gamma"] * parms["n_i_compartments"]

    #### Set beta matrix based on desired R0 and connectivity scenario ###
    parms["beta"] = construct_beta(parms)

    #### Set up the model time steps
    steps = parms["tf"]

    parms["t_array"] = get_time_array(parms)
    t_array = parms["t_array"]

    #### Set up vaccine schedule for group specified in parms
    parms["vaccination_uptake_schedule"] = build_vax_schedule(parms)

    #### Initialize population
    groups = parms["n_groups"]
    S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u = initialize_population(
        steps, groups, parms
    )

    #### Initialize the model
    model = SEIRModel(parms, seed)

    #### Vaccinate the population on the first day if applicable
    first_day = t_array[0]
    S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u = vaccinate_on_day(
        model,
        u,
        first_day,
        groups,
        S,
        V,
        SV,
        E1,
        E2,
        E1_V,
        E2_V,
        I1,
        I2,
        R,
        Y,
        X,
        parms["vaccination_uptake_schedule"],
    )

    #### Run the model
    S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u = run_model(
        model, u, t_array, steps, groups, S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X
    )

    #### Flatten into a dataframe
    df = pl.DataFrame(
        {
            "t": np.repeat(t_array, groups),
            "group": np.tile(np.arange(groups), steps),
            "S": S.flatten(),
            "V": V.flatten(),
            "SV": SV.flatten(),
            "E1": E1.flatten(),
            "E2": E2.flatten(),
            "E1_V": E1_V.flatten(),
            "E2_V": E2_V.flatten(),
            "I1": I1.flatten(),
            "I2": I2.flatten(),
            "R": R.flatten(),
            "Y": Y.flatten(),
            "X": X.flatten(),
        }
    )

    return df


def simulate_replicates(parameter_sets, simulate_fn=simulate):
    """
    Runs multiple replicates of the SEIR model for a grid of parameter sets.
    A unique seed is generated for each replicated based on a base seed,
    a stable hash of the parameter set, and the replicate index.
    Args:
        parameter_sets (Sequence[dict]): A sequence containing the parameters for the simulation.
            Parameter sets must include 'n_replicates' and 'seed'.
        simulate_fn (function): The function to run the simulation. Defaults to `simulate`.
    Returns:
        inner_simulate_replicates (function): A function that takes a parameter set and returns the simulation results.
    """

    def inner_simulate_replicates(parameter_set):
        assert isinstance(parameter_set, dict)

        n_replicates = parameter_set["n_replicates"]
        base_seed = parameter_set["seed"]
        rng_name = seed_from_string("model")

        outputs = []
        for i in range(n_replicates):
            seed = [base_seed, rng_name, i]
            current_params = parameter_set.copy()
            outputs.append(simulate_fn(current_params, seed))

        return pl.concat(
            [
                output.with_columns(pl.lit(i, dtype=pl.Int64).alias("replicate"))
                for i, output in enumerate(outputs)
            ]
        )

    return griddler.run_squash(inner_simulate_replicates, parameter_sets)
