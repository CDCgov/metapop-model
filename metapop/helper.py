import numpy as np
import numpy.linalg as la

def get_percapita_contact_matrix(parms):
    """
    Calculate the per capita contact matrix based on the total contacts, average per capita degrees per population, and the population sizes for a 3-group population.

    Args:
        parms (dict): Dictionary containing the parameters, including:
        k (int): Contacts total
        k_g1 (int): contacts general and sub pop 1
        k_21 (int): contacts between sub pop 1 and 2
        k_g2 (int): contacts general and sub pop 2
        pop_sizes (array): population sizes of each group

    Returns:
        np.array: The per capita contact matrix.
    """
    assert parms['n_groups'] == 3, "The number of groups (n_groups) must be 3 to use this function."

    assert parms["pop_sizes"][0] == np.max(parms["pop_sizes"]), "The first population must be the largest to represent the population."

    k_i = parms["k_i"]
    k_g1 = parms["k_g1"]
    k_21 = parms["k_21"]
    k_g2 = parms["k_g2"]
    pop_sizes = np.array(parms["pop_sizes"])

    edges_per_group = pop_sizes * k_i

    contacts = np.array([[0,                  k_g1 * pop_sizes[1], k_g2 * pop_sizes[2]],
                         [k_g1 * pop_sizes[1],       0,            k_21 * pop_sizes[1]],
                         [k_g2 * pop_sizes[2],k_21 * pop_sizes[1],                  0]])
    colsums = np.sum(contacts, axis=0)

    edges_to_assign = edges_per_group - colsums
    np.fill_diagonal(contacts, edges_to_assign)

    percapita_contacts = contacts / pop_sizes

    # this should go into a python test
    assert np.allclose(np.sum(percapita_contacts, axis=0), k_i), f"The columns of the per capita contact matrix must sum to the per capita degrees k_i. The percapita contact matrix is \n{percapita_contacts} and the sum of the columns is {np.sum(percapita_contacts, axis=0)}."

    return percapita_contacts


def make_beta_matrix(parms):
    """
    For a 3-group population where we assume general, subpop1, and subpop2:
    Define matrix based on total contacts per group (k) and specific
    interactions with general, and between subgroups 1 and 2

    Args:
        parms (dict): Dictionary containing beta parameters, including:
        k: Contacts total
        k_g1: contacts general and sub pop 1
        k_21: contacts between sub pop 1 and 2
        k_g2: contacts general and sub pop 2
        pop_sizes: array with population sizes of each group

    Returns:
        np.array: The matrix representing contact rates between groups.

    """
    assert parms['n_groups'] == 3, "The number of groups (n_groups) must be 3 to use this function."

    k = parms["k"]
    k_g1 = parms["k_g1"]
    k_21 = parms["k_21"]
    k_g2 = parms["k_g2"]
    pop_sizes = np.array(parms["pop_sizes"])

    edges_per_group = pop_sizes * k


    contacts = np.array([[0,                   k_g1 * pop_sizes[1], k_g2 * pop_sizes[2]],
                         [k_g1 * pop_sizes[1], 0,                   k_21 * pop_sizes[1]],
                         [k_g2 * pop_sizes[2], k_21 * pop_sizes[1],         0]])

    colsums = np.sum(contacts, axis=0)

    edges_to_assign = edges_per_group - colsums

    np.fill_diagonal(contacts, edges_to_assign)

    b = contacts / pop_sizes

    return b

def get_r0(beta_matrix, gamma_unscaled, pop_sizes, n_i_compartments):
    """
    Calculate the basic reproduction number (R0) matrix and return its spectral radius.

    Args:
        beta_matrix (np.array): The transmission rate matrix.
        gamma_unscaled (float): The unscaled recovery rate.
        pop_sizes (list or np.array): The population sizes of each group.
        n_i_compartments (int): The number of infectious compartments.

    Returns:
        float: The spectral radius of the R0 matrix, representing the basic reproduction number.
    """
    gamma_scaled = gamma_unscaled / n_i_compartments

    # Calculate the R0 matrix with row-wise multi
    X = (beta_matrix / gamma_scaled) * pop_sizes / sum(pop_sizes)

    # Calculate the eigenvalues of the R0 matrix
    eigen_all = la.eig(X)
    spectral_radius = np.max(np.abs(eigen_all[0]))

    return spectral_radius

def rescale_beta_matrix(unscaled_beta, factor):
    """
    Rescale the beta matrix by a given factor.

    Args:
        unscaled_beta (np.array): The original transmission rate matrix.
        factor (float): The factor by which to rescale the beta matrix.

    Returns:
        np.array: The rescaled transmission rate matrix.
    """

    beta_scaled = unscaled_beta * factor
    return beta_scaled

def calculate_beta_factor(r0_desired, current_r0):
    """
    Calculate the scaling factor for the beta matrix to achieve a desired R0.

    Args:
        r0_desired (float): The desired basic reproduction number (R0) from parms file.
        current_r0 (float): The current basic reproduction number (R0).

    Returns:
        float: The scaling factor for the beta matrix to get desired R0.
    """

    factor = r0_desired / current_r0
    return factor

def modify_beta_connectivity(base_beta, connectivity_factor):
    """
    Modify the beta matrix to adjust the connectivity between sub-groups in a 3 pop or more model.

    Args:
        base_beta (np.array): Transmission rate matrix which we want to adjust.
        connectivity_factor (float): The factor by which to adjust the connectivity between specific groups.

    Returns:
        np.array: The modified transmission rate matrix with adjusted connectivity.
    """
    # Assert statement to check that beta is at least 3x3 so this doesn't fail
    assert base_beta.shape[0] >= 3 and base_beta.shape[1] >= 3, "The base_beta matrix must be at least 3x3."

    final_beta = base_beta
    final_beta[1,2] *= connectivity_factor
    final_beta[2,1] *= connectivity_factor

    return final_beta

def construct_beta(parms):
    """
    Construct the scaled beta matrix based on the desired R0.

    Args:
        parms (dict): Dictionary containing the parameters, including:
            - k, kg1, kg2, k12
            - gamma (float): The recovery rate.
            - pop_sizes (list or np.array): The population sizes of each group.
            - n_i_compartments (int): The number of infectious compartments.
            - desired_r0 (float): The desired basic reproduction number (R0).

    Returns:
        np.array: The scaled beta matrix.
    """
    beta_unscaled = get_percapita_contact_matrix(parms)
    beta_modified_connectivity = modify_beta_connectivity(beta_unscaled, parms["connectivity_scenario"])
    r0_base = get_r0(beta_modified_connectivity, parms["gamma"], parms["pop_sizes"], parms["n_i_compartments"])
    beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
    beta_scaled = rescale_beta_matrix(beta_modified_connectivity, beta_factor)
    return beta_scaled

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
        beta (np.array): The contact matrix.
        I_g (np.array): The number of infected individuals in each group.
        pop_sizes (np.array): The population sizes of each group.
        target_group (int): The target group index.

    Returns:
        float: The force of infection for the target group.
    """

    foi = 0
    for j in range(len(pop_sizes)):
        foi += I_g[j] * beta[target_group, j] / pop_sizes[target_group]

    return foi

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
