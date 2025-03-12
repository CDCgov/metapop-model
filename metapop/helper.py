import numpy as np

def set_beta_parameter(parms):
    """
    If beta ranges are supplied in the parameters, reset the beta parameter.

    Args:
        parms (dict): Dictionary containing the parameters, optionally including "beta_2_low" and "beta_2_high".

    Returns:
        dict: Parameters with optionally randomly generated beta_2_value.
    """
    n_groups = parms["n_groups"]
    index = n_groups - 1 # python indexing
    if "beta_2_low" in parms and "beta_2_high" in parms:
        beta_2_value = np.random.uniform(parms["beta_2_low"], parms["beta_2_high"])
        parms["beta"][index][index] = beta_2_value
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
    u = [[int(parms["pop_sizes"][group] * (1 - parms["initial_vaccine_coverage"][group])) - parms["I0"][group],
          int(parms["pop_sizes"][group] * (parms["initial_vaccine_coverage"][group])), # at some point we will need to ensure that these are integer values
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

def get_infected(u, I_indices, groups):
    """
    Calculate the number of infected individuals for each group. If there is only one infected compartment, n_i_compartments=1, then return I for each group

    Args:
        u (list): The state of the system.
        I_indices (list): The indices of the I compartments.
        groups (int): The number of groups.

    Returns:
        np.array: An array of the number of infected individuals for each group.
    """
    return np.array([sum(u[group][i] for i in I_indices) for group in range(groups)])

def calculate_foi(beta, I_g, pop_sizes, target_group):
    """
    Calculate the force of infection (FOI) for a target group.

    Args:
        beta (np.array): The transmission rate matrix.
        I_g (np.array): The number of infected individuals in each group.
        pop_sizes (np.array): The population sizes of each group.
        target_group (int): The target group index.

    Returns:
        float: The force of infection for the target group.
    """
    return np.dot(beta[target_group], I_g / np.array(pop_sizes))

def rate_to_frac(rate):
    """
    Calculate the fraction of transitions based on the rate

    Args:
        rate (float): The rate

    Returns:
        float: The fraction that will transition.
    """
    return 1.0 - np.exp(-rate)

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
