import numpy as np
import numpy.linalg as la

def get_percapita_contact_matrix(parms):
    """
    Calculate the per capita contact matrix based on the total contacts, average per capita degrees per population, and the population sizes for a 3-group population.

    In this model we assume the 3-group population is a general population, subpop1, and subpop2, where subpop1 and subpop2 are smaller than the general population. The matrix is defined by the total per capita degree per group (k_i), the out degree from subpop1 to the general population, the out degree from subpop2 to the general population, and the out degree from subpop1 to subpop2.

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

    assert np.all(percapita_contacts >= 0), "The per capita contact matrix must have non-negative values."

    return percapita_contacts

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
    state_arrays = [np.zeros((steps, groups)) for _ in range(9)]
    S, V, E1, E2, I1, I2, R, Y, X = state_arrays

    # set up u vector
    u = []

    for group in range(groups):
        u_i = [0, int(parms["pop_sizes"][group] * parms["initial_vaccine_coverage"][group]), 0, 0, parms["I0"][group], 0, 0, 0, 0]
        u_i[0] = int(parms["pop_sizes"][group] - np.sum(u_i))
        u.append(u_i)

    # first time step is initial state
    for group in range(groups):
        S[0, group], V[0, group], E1[0, group], E2[0, group], I1[0, group], I2[0, group], R[0, group], Y[0, group], X[0, group]= u[group]

    return S, V, E1, E2, I1, I2, R, Y, X, u

def get_infected(u, I_indices, groups, parms, t):
    """
    Calculate the number of infected individuals for each group. If there is only one infected compartment, n_i_compartments=1, then return I for each group

    Args:
        u (list): The state of the system.
        I_indices (list): The indices of the I compartments.
        groups (int): The number of groups.

    Returns:
        np.array: An array of the number of infected individuals for each group.
    """
    if (parms["symptomatic_isolation"] & (t >= parms["symptomatic_isolation_day"])):
        # last I compartment
        i_max = max(I_indices)

        # Prerash infected = I1 infecteds
        pre_rash_infected = np.array([sum(u[group][i] for i in I_indices if i != i_max) for group in range(groups)])

        # Postrash infected that are not isolating
        post_rash = np.array([(u[group][i_max] * (1 - parms["isolation_success"])) for group in range(groups)])

        # Total infected = prerash (I1) + postrash infected (I2)
        infected = pre_rash_infected + post_rash

        return infected
    else:
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

def time_to_rate(duration):
    """
    Calculate the rate parameters of transitions based on the length of time in compartment

    Args:
        duration (float): The duration of time in compartment

    Returns:
        float: The rate parameter
    """
    return 1.0/duration

def build_vax_schedule(parms):
    """
    Build dictionary desribing vaccination schedule for group 2

    Args:
        parms(dict): contains columns vaccine_uptake_days and vaccine_uptake_doses
    Returns:
        dict: dictionary with days and doses
    """
    assert "vaccine_uptake_range" in parms, "vaccine_uptake_range must be provided in parms"
    assert "total_vaccine_uptake_doses" in parms, "total_vaccine_uptake_doses must be provided in parms"

    # Generate a sequence of days between the start and end of the vaccine_uptake_range
    vaccine_uptake_days = list(range(parms["vaccine_uptake_range"][0], parms["vaccine_uptake_range"][1] + 1))
    doses_per_day = round(parms["total_vaccine_uptake_doses"] / len(vaccine_uptake_days))

    # Create the schedule dictionary
    schedule = {day: doses_per_day for day in vaccine_uptake_days}
    return(schedule)

def vaccinate_groups(groups, u, t, vaccination_uptake_schedule, parms):
    """
    Incorporate active vaccination given an uptake schedule, for group 2 only

    Args:
        groups (int): The number of groups.
        u (list): The state of the system.
        t: timestep
        vaccination_uptake_schedule (dict): dictionary with keys of days and values of doses on that day for group 2
        vaccinated_group: group to vaccinate
    Returns:
        np.array: Number of susceptibles vaccinated on that day for each group (should only be group 2)
    """
    new_vaccinated = np.zeros(groups, dtype=int)
    vaccinated_group = parms["vaccinated_group"]

    if t in vaccination_uptake_schedule:
        S, V, E1, E2, I1, I2, R, Y, X = u[vaccinated_group] # get group 2
        vaccine_eligible = S + E1 + E2
        doses = vaccination_uptake_schedule[t]
        # Calculate the proportion of doses to assign to S, E1, and E2
        if vaccine_eligible > 0:
            S_doses = int(doses * S / vaccine_eligible)  # proportion of doses going to S
            S_doses = min(S_doses, S)  # Ensure S_doses does not exceed S
        else:
            S_doses = 0

        new_vaccinated[vaccinated_group] = S_doses

    return new_vaccinated

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
