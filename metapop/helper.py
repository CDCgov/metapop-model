import numpy as np

def set_beta_parameter(parms):
    """
    If beta ranges are supplied in the parameters, reset the beta parameter.

    Args:
        parms (dict): Dictionary containing the parameters, optionally including "beta_2_low" and "beta_2_high".

    Returns:
        dict: Parameters with optionally randomly generated beta_2_value.
    """
    if "beta_2_low" in parms and "beta_2_high" in parms:
        beta_2_value = np.random.uniform(parms["beta_2_low"], parms["beta_2_high"])
        parms["beta"][1][1] = beta_2_value
    return parms


# initialize populations
def initialize_population(steps, groups, parms):
    """
    Initialize the population arrays and the initial state based on the provided parameters.

    Args:
        steps (int): Number of time steps.
        groups (int): Number of groups.
        parms (dict): Dictionary containing the parameters, including "N", "initial_vaccine_coverage", and "I0".

    Returns:
        tuple: A tuple containing the initialized arrays (S, V, E1, E2, I1, I2, R, Y) and the initial state (u).
    """
    # arrays for each state
    S  = np.zeros((steps, groups))
    V  = np.zeros((steps, groups))
    E1 = np.zeros((steps, groups))
    E2 = np.zeros((steps, groups))
    I1 = np.zeros((steps, groups))
    I2 = np.zeros((steps, groups))
    R  = np.zeros((steps, groups))
    Y  = np.zeros((steps, groups))

    # initial states
    u = [[int(parms["N"][group] * (1 - parms["initial_vaccine_coverage"][group])) - parms["I0"][group],
          int(parms["N"][group] * (parms["initial_vaccine_coverage"][group])), # at some point we will need to ensure that these are integer values
          0,
          0,
          parms["I0"][group],
          0,
          0,
          0
         ] for group in range(groups)]

    # first time step is initial state
    for group in range(groups):
        S[0, group], V[0, group], E1[0, group], E2[0, group], I1[0, group], I2[0, group], R[0, group], Y[0, group] = u[group]

    return S, V, E1, E2, I1, I2, R, Y, u


# run model
def run_model(model, u, t, steps, groups, S, V, E1, E2, I1, I2, R, Y):
    """
    Update the population arrays based on the SEIR model.

    Args:
        model: The SEIR model instance.
        u: The initial state.
        t: The time array.
        steps: The number of time steps.
        groups: The number of groups.
        S, V, E1, E2, I1, I2, R, Y: The population arrays to be updated.

    Returns:
        S, V, E1, E2, I1, I2, R, Y, u
    """
    for j in range(1, steps):
        u = model.seirmodel(u, t[j])
        for group in range(groups):
            S[j, group], V[j, group], E1[j, group], E2[j, group], I1[j, group], I2[j, group], R[j, group], Y[j, group] = u[group]

    return S, V, E1, E2, I1, I2, R, Y, u
