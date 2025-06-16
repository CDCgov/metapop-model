import hashlib
import os

import numpy as np
import numpy.linalg as la

from .version import __version__

__all__ = [
    "get_percapita_contact_matrix",
    "get_r0",
    "rescale_beta_matrix",
    "calculate_beta_factor",
    "get_r0_one_group",
    "construct_beta",
    "initialize_population",
    "get_infected",
    "calculate_foi",
    "rate_to_frac",
    "time_to_rate",
    "build_vax_schedule",
    "vaccinate_groups",
    "seed_from_string",
    "get_metapop_info",
]


def get_percapita_contact_matrix(parms):
    """
    Calculate the per capita contact matrix based on the total contacts, average per capita degrees per population, and the population sizes for a 3-group population.

    In this model we assume the 3-group population is a general population, subpop1, and subpop2, where subpop1 and subpop2 are smaller than the general population. The matrix is defined by the total per capita degree per group (k_i), the out degree from subpop1 to the general population, the out degree from subpop2 to the general population, and the out degree from subpop1 to subpop2.

    Args:
        parms (dict): Dictionary containing the parameters, including:
        k_i (float): Contacts total
        k_g1 (float): contacts general and sub pop 1
        k_21 (float): contacts between sub pop 1 and 2
        k_g2 (float): contacts general and sub pop 2
        pop_sizes (array): population sizes of each group

    Returns:
        np.array: The per capita contact matrix.
    """
    assert (
        parms["n_groups"] == 3
    ), "The number of groups (n_groups) must be 3 to use this function."
    assert parms["pop_sizes"][0] == np.max(
        parms["pop_sizes"]
    ), "The first population must be the largest to represent the population."

    k_i = parms["k_i"]
    k_g1 = parms["k_g1"]
    k_21 = parms["k_21"]
    k_g2 = parms["k_g2"]
    pop_sizes = np.array(parms["pop_sizes"])

    edges_per_group = pop_sizes * k_i

    contacts = np.array(
        [
            [0, k_g1 * pop_sizes[1], k_g2 * pop_sizes[2]],
            [k_g1 * pop_sizes[1], 0, k_21 * pop_sizes[1]],
            [k_g2 * pop_sizes[2], k_21 * pop_sizes[1], 0],
        ]
    )
    colsums = np.sum(contacts, axis=0)

    edges_to_assign = edges_per_group - colsums
    np.fill_diagonal(contacts, edges_to_assign)

    percapita_contacts = contacts / pop_sizes

    # this should go into a python test
    assert np.allclose(
        np.sum(percapita_contacts, axis=0), k_i
    ), f"The columns of the per capita contact matrix must sum to the per capita degrees k_i. The percapita contact matrix is \n{percapita_contacts} and the sum of the columns is {np.sum(percapita_contacts, axis=0)}."

    assert np.all(
        percapita_contacts >= 0
    ), "The per capita contact matrix must have non-negative values."

    return percapita_contacts


def get_r0(beta_matrix, gamma, pop_sizes):
    """
    Calculate the basic reproduction number (R0) matrix and return its spectral radius when the beta matrix is at least a 2x2 array.

    Args:
        beta_matrix (np.array): The transmission rate matrix.
        gamma (float): The recovery rate.
        pop_sizes (list or np.array): The population sizes of each group.

    Returns:
        float: The spectral radius of the R0 matrix, representing the basic reproduction number.
    """

    # Calculate the R matrix with row-wise multiplication
    X = (beta_matrix / gamma) * pop_sizes / sum(pop_sizes)

    # More than one population, calculate R0 based on spectral radius of R matrix
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


def get_r0_one_group(k, gamma):
    """
    Calculate the basic reproduction number (R0) number when there's only one group.

    Args:
        k (list): Contacts per day (unscaled)
        gamma (float): The recovery rate.

    Returns:
        float: R0, contacts per day / recovery rate
    """
    X = k[0] / gamma

    return X


def construct_beta(parms):
    """
    Construct the scaled beta matrix based on the desired R0.

    Args:
        parms (dict): Dictionary containing the parameters, including:
            - k_i (float): Total per capita contacts per group.
            - k_g1 (float): Contacts between the general population and subpopulation 1.
            - k_g2 (float): Contacts between the general population and subpopulation 2.
            - k_21 (float): Contacts between subpopulation 1 and subpopulation 2.
            - gamma (float): The recovery rate.
            - pop_sizes (list or np.array): The population sizes of each group.
            - n_i_compartments (int): The number of infectious compartments.
            - desired_r0 (float): The desired basic reproduction number (R0).

    Returns:
        np.array: The scaled beta matrix.
    """
    if parms["n_groups"] == 3:
        beta_unscaled = get_percapita_contact_matrix(parms)
        r0_base = get_r0(beta_unscaled, parms["gamma"], parms["pop_sizes"])
        beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
        beta_scaled = rescale_beta_matrix(beta_unscaled, beta_factor)
    else:
        assert (
            parms["n_groups"] == 1
        ), "Setups only designed for one or three groups currently."
        # skip per capita contact matrix building, get R0 directly

        r0_base = get_r0_one_group(parms["k_i"], parms["gamma"])
        beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
        beta_scaled = rescale_beta_matrix(parms["k_i"][0], beta_factor)
        # beta_scaled = beta_scaled.reshape(1, 1)  # Reshape to 1x1 matrix for consistency
    return beta_scaled


def initialize_population(steps, groups, parms):
    """
        Initialize the population arrays and the initial state based on the provided parameters.

        Args:
            steps (int): Number of time steps.
            groups (int): Number of groups.
            parms (dict): Dictionary containing the parameters, including "N", "initial_vaccine_coverage", "vaccine_efficacy_2_dose" and "I0".

    Returns:
        tuple: A tuple containing the initialized arrays:
            - S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X (np.array): Arrays representing the state of the population.
            - u (list): The initial state vector for each group.
    """
    # arrays for each state
    state_arrays = [np.zeros((steps, groups)) for _ in range(12)]
    S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X = state_arrays

    # set up u vector
    u = []

    for group in range(groups):
        u_i = [
            0,  # S
            int(
                parms["pop_sizes"][group]
                * parms["initial_vaccine_coverage"][group]
                * parms["vaccine_efficacy_2_dose"]
            ),  # V
            0,  # SV
            0,  # E1
            0,  # E2
            0,  # E1_V
            0,  # E2_V
            parms["I0"][group],  # I1
            0,  # I2
            0,  # R
            0,  # Y
            0,  # X
        ]
        u_i[2] = int(
            parms["pop_sizes"][group] * parms["initial_vaccine_coverage"][group]
            - u_i[1]
        )  # SV

        u_i[0] = int(parms["pop_sizes"][group] - np.sum(u_i))  # S
        u.append(u_i)

    # first time step is initial state
    for group in range(groups):
        (
            S[0, group],
            V[0, group],
            SV[0, group],
            E1[0, group],
            E2[0, group],
            E1_V[0, group],
            E2_V[0, group],
            I1[0, group],
            I2[0, group],
            R[0, group],
            Y[0, group],
            X[0, group],
        ) = u[group]

    return S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u


def get_infected(u, I_indices, groups, parms, t):
    """
    Calculate the number of infected individuals for each group. If there is only one infected compartment, n_i_compartments=1, then return I for each group

    Args:
        u (list): The state of the system.
        I_indices (list): The indices of the I compartments.
        groups (int): The number of groups.
        parms (dict): Dictionary containing model parameters
        t (int): The current time step.

    Returns:
        np.array: An array of the number of infected individuals for each group.
    """
    assert (
        "symptomatic_isolation_start_day" in parms
    ), "Key 'symptomatic_isolation_start_day' is missing in parms."
    assert (
        "symptomatic_isolation_duration_days" in parms
    ), "Key 'symptomatic_isolation_duration_days' is missing in parms."
    assert (
        "pre_rash_isolation_start_day" in parms
    ), "Key 'pre_rash_isolation_start_day' is missing in parms."
    assert (
        "pre_rash_isolation_duration_days" in parms
    ), "Key 'pre_rash_isolation_duration_days' is missing in parms."

    isolation_start_day = parms["symptomatic_isolation_start_day"]
    isolation_duration_days = parms["symptomatic_isolation_duration_days"]
    pre_rash_isolation_start_day = parms["pre_rash_isolation_start_day"]
    pre_rash_isolation_duration_days = parms["pre_rash_isolation_duration_days"]

    isolation_range_days = range(
        isolation_start_day, isolation_start_day + isolation_duration_days
    )
    pre_rash_isolation_range_days = range(
        pre_rash_isolation_start_day,
        pre_rash_isolation_start_day + pre_rash_isolation_duration_days,
    )

    # last I compartment
    i_max = max(I_indices)

    # Handling pre_rash_isolation
    if t in pre_rash_isolation_range_days:
        # Prerash infected = I1 infecteds
        pre_rash_infected = np.array(
            [
                sum(
                    u[group][i] * (1 - parms["pre_rash_isolation_success"])
                    for i in I_indices
                    if i != i_max
                )
                for group in range(groups)
            ]
        )

    else:
        pre_rash_infected = np.array(
            [
                sum(u[group][i] for i in I_indices if i != i_max)
                for group in range(groups)
            ]
        )

    # Handling isolation
    if t in isolation_range_days:
        # Postrash infected that are not isolating
        post_rash = np.array(
            [
                (u[group][i_max] * (1 - parms["isolation_success"]))
                for group in range(groups)
            ]
        )

    else:
        post_rash = np.array([(u[group][i_max]) for group in range(groups)])

    infected = pre_rash_infected + post_rash
    return infected


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
        beta_value = beta if len(pop_sizes) == 1 else beta[target_group, j]
        foi += I_g[j] * beta_value / pop_sizes[target_group]
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
    return 1.0 / duration


def build_vax_schedule(parms):
    """
    Build dictionary describing vaccination schedule for group 2
    Vaccines are distributed evenly during the vaccine campiagn duration (see example)

    Args:
        parms (dict): Dictionary containing the parameters, including:
            - vaccine_uptake_start_day (int): The day the vaccination campaign starts.
            - vaccine_uptake_duration_days (int): The duration of the vaccination campaign.
            - total_vaccine_uptake_doses (int): The total number of vaccine doses available.
            - t_array (list): The array of time steps in the model.

    Returns:
        dict: A dictionary with days as keys and doses as values.

    Examples:
        The expected cumulative doses delivered by day t are calculated from the total doses delivered and campaign duration
        These expected doses are then rounded to integers and the differences across days are the doses delivered.

        >>> parms = {"vaccine_uptake_start_day": 0, "vaccine_uptake_duration_days": 10, "total_vaccine_uptake_doses": 25, "t_array": np.arange(1,30)}
        >>> build_vax_schedule(parms)
        {1: 2, 2: 3, 3: 3, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2, 9: 2, 10: 3}

        In the case of clipped vaccine campaigns, when the end of the simulation occurs before all doses have been delivered,
        the doses are scheduled as above but cut short and total doses scheduled does not match uptake doses

        >>> parms = {"vaccine_uptake_start_day": 0, "vaccine_uptake_duration_days": 10, "total_vaccine_uptake_doses": 25, "t_array": np.arange(1,6)}
        >>> build_vax_schedule(parms)
        {1: 2, 2: 3, 3: 3, 4: 2, 5: 2}

    """
    assert (
        "vaccine_uptake_start_day" in parms
    ), "vaccine_uptake_start_day must be provided in parms"
    assert (
        "vaccine_uptake_duration_days" in parms
    ), "vaccine_uptake_duration_days must be provided in parms"
    assert (
        "total_vaccine_uptake_doses" in parms
    ), "total_vaccine_uptake_doses must be provided in parms"

    # Generate a sequence of days between the start and end of the vaccine_uptake_range
    t_array = parms["t_array"]

    # if the start day is beyond the end of the time series, return a schedule with zero doses on that day
    if parms["vaccine_uptake_start_day"] >= len(t_array):
        start_day = parms["vaccine_uptake_start_day"] + 1
    else:
        start_day = t_array[parms["vaccine_uptake_start_day"]]

    # try to set the end day to the last day of the campaign if it's within the time series for simulation
    # if not, set it to the last day of the time series
    if parms["vaccine_uptake_start_day"] + parms["vaccine_uptake_duration_days"] < len(
        t_array
    ):
        end_day = t_array[
            parms["vaccine_uptake_start_day"] + parms["vaccine_uptake_duration_days"]
        ]
    else:
        end_day = t_array[-1] + 1

    vaccine_uptake_days = list(range(start_day, end_day))

    if len(vaccine_uptake_days) > 0:
        # Distribute doses evenly across time by rounding cumulative expected doses and taking difference
        avg_doses_per_day = (
            parms["total_vaccine_uptake_doses"] / parms["vaccine_uptake_duration_days"]
        )
        expected_cumu_doses = [
            (d + 1) * avg_doses_per_day for d in range(len(vaccine_uptake_days))
        ]
        cumu_doses = [round(dose) for dose in expected_cumu_doses]

        doses_per_day = [cumu_doses[0]]
        if len(cumu_doses) > 1:
            doses_per_day += [
                cumu_doses[i + 1] - cumu_doses[i] for i in range(len(cumu_doses) - 1)
            ]
        # Create the schedule dictionary
        schedule = {
            vaccine_uptake_days[day]: doses_per_day[day]
            for day in range(len(vaccine_uptake_days))
        }

    # If no days are specified, set the schedule to 0 doses for day the first day of the vaccine schedule as a
    # placeholder rather than an empty dictionary
    else:
        doses_per_day = 0
        schedule = {end_day: 0}

    # if vaccine campaign runs for the whole duration, assert that the total number of doses
    # administered matches the total number of doses specified as available
    # if the vaccine campaign is clipped by the end of the simulation, doses delivered will not match
    if len(vaccine_uptake_days) == parms["vaccine_uptake_duration_days"]:
        assert sum(schedule.values()) == parms["total_vaccine_uptake_doses"]

    return schedule


def vaccinate_groups(groups, u, t, vaccination_uptake_schedule, parms):
    """
    Incorporate active vaccination given an uptake schedule, for group 2 only

    Args:
        groups (int): The number of groups.
        u (list): The state of the system.
        t: timestep
        vaccination_uptake_schedule (dict): dictionary with keys of days and values of doses on that day for group 2
        parms (dict): Dictionary containing the parameters, including:
            - vaccinated_group (int): The group to be vaccinated.
            - vaccine_efficacy_1_dose (float): The efficacy of the vaccine after one dose.
    Returns:
        np.array: Number of susceptibles successfully vaccinated on that day for each group (should only be group 2)
        np.array: Number of susceptibles unsuccessfully vaccinated on that day for each group (should only be group 2), individuals remain susceptible but cannot be vaccinated again
    """
    new_vaccinated = np.zeros(groups, dtype=int)
    new_failures = np.zeros(groups, dtype=int)
    new_exposed_doses = np.zeros(groups, dtype=int)
    vaccinated_group = parms["vaccinated_group"]
    vaccine_efficacy = parms["vaccine_efficacy_1_dose"]

    if t in vaccination_uptake_schedule:
        S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X = u[
            vaccinated_group
        ]  # get group 2
        vaccine_eligible = S + E1 + E2
        doses = vaccination_uptake_schedule[t]
        # Calculate the proportion of doses to assign to S, E1, and E2
        if vaccine_eligible > 0:
            S_prop = S / vaccine_eligible
            S_doses = round(doses * S_prop)  # proportion of doses going to S
            S_doses = min(S_doses, S)  # Ensure S_doses does not exceed S

            E_doses = round(doses * (1 - S_prop))
            E_doses = min(E_doses, E1 + E2)  # Ensure E_doses does not exceed E1 + E2
        else:
            S_doses = 0
            E_doses = 0

        new_exposed_doses[vaccinated_group] = E_doses
        vaccine_failures = int(S_doses * (1 - vaccine_efficacy))
        new_vaccinated[vaccinated_group] = S_doses - vaccine_failures
        new_failures[vaccinated_group] = vaccine_failures

    return new_vaccinated, new_failures, new_exposed_doses


def seed_from_string(string):
    """
    Generate a stable seed from a string
    Args:
        string (str): The input string to generate the seed from.
    Returns:
        int: A stable seed derived from the input string.
    """
    hash_object = hashlib.blake2b(string.encode(), digest_size=10)
    seed = int(hash_object.hexdigest(), 16)
    return seed


def get_metapop_info():
    """
    Get metadata on the metapopulation model.

    Returns:
        dict: A dictionary containing metadata on the metapopulation model.
    """
    # get version info from version.py
    version = "unknown"
    try:
        version = __version__
    except Exception:
        pass

    # commit info from git
    commit = os.popen("git rev-parse HEAD").read().strip().split("\n")[-1][0:7]

    info = {
        "name": "Measles Metapopulation Model",
        "description": "A model simulating measles transmission in a metapopulation with public health interventions.",
        "version": version,  # version from version.py
        "commit": commit,
        "url": "https://github.com/cdcent/metapop-model",
        "email": "eocevent410@cdc.gov",
    }

    return info
