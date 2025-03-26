import os
from metapop import SEIRModel
from metapop.helper import* # noqa: F405
import yaml
import numpy as np
from numpy.testing import assert_allclose


def test_get_percapita_contact_matrix():
    # Define the parameters
    parms = {
        "k_i": np.array([10, 10, 10]),
        "k_g1": 1,
        "k_g2": 2,
        "k_21": 2,
        "n_groups": 3,
        "pop_sizes": [1000, 100, 100]
    }

    # Call the get_percapita_contact_matrix function
    percapita_contacts = get_percapita_contact_matrix(parms)

    # Check the result
    expected_percapita_contacts = np.array([
        [9.7, 1., 2.],
        [0.1, 7., 2.],
        [0.2, 2., 6.]
    ])
    assert np.array_equal(percapita_contacts, expected_percapita_contacts), f"Expected {expected_percapita_contacts}, but got {percapita_contacts} when using equal degree for all subgroups"

    # A scenario where the degree is different for each subgroup
    parms["k_i"] = np.array([10, 20, 15])

    percapita_contacts = get_percapita_contact_matrix(parms)

    expected_percapita_contacts = np.array([
        [9.7, 1., 2.],
        [0.1, 17., 2.],
        [0.2, 2., 11.]
    ])
    assert np.array_equal(percapita_contacts, expected_percapita_contacts), f"Expected {expected_percapita_contacts}, but got {percapita_contacts} when using different degree for each subgroup"

def test_get_r0():
    beta_matrix = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.3, 0.4, 0.1]
    ])
    gamma_unscaled = 0.1
    pop_sizes = np.array([1000, 200, 200])
    n_i_compartments = 2
    expected_r0 = 3.709795
    r0 = get_r0(beta_matrix, gamma_unscaled, pop_sizes, n_i_compartments)
    assert np.isclose(r0, expected_r0), f"Expected {expected_r0}, but got {r0}"

def test_construct_beta():
    parms = {
        "beta_within": 0.1,
        "beta_general": 0.05,
        "beta_small": 0.02,
        "k_i": [10, 10, 10],
        "k_g1": 1,
        "k_g2": 2,
        "k_21": 2,
        "gamma": 0.1,
        "pop_sizes": np.array([1000, 100, 100]),
        "n_i_compartments": 2,
        "desired_r0": 2.0,
        "n_groups": 3,
        "connectivity_scenario": 1.0
    }
    beta_unscaled = get_percapita_contact_matrix(parms)
    r0_base = get_r0(beta_unscaled, parms["gamma"], parms["pop_sizes"], parms["n_i_compartments"])
    beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
    beta_scaled = rescale_beta_matrix(beta_unscaled, beta_factor)
    expected_beta = modify_beta_connectivity(beta_scaled, parms["connectivity_scenario"])
    beta_scaled = construct_beta(parms)
    assert beta_scaled.shape == (3, 3), f"Expected shape (3, 3), but got {beta_scaled.shape}"
    assert np.allclose(beta_scaled, expected_beta), f"Expected {expected_beta}, but got {beta_scaled}"

def test_pop_initialization():
    parms = {
        "pop_sizes": [1000, 2000],
        "initial_vaccine_coverage": [0.0, 0.95],
        "I0": [10, 2]
    }
    steps = 3
    groups = 2

    S, V, E1, E2, I1, I2, R, Y, X, u = initialize_population(steps, groups, parms)

    # correct dimensions
    assert len(S) == steps
    assert len(S[0]) == groups

    # initial vaccination
    assert V[0][0] == 0
    assert_allclose(V[0][1], parms["pop_sizes"][1] * parms["initial_vaccine_coverage"][1], rtol=1e-2, atol=1e-8)

    # initial infections
    assert I1[0][0] == parms["I0"][0]
    assert I1[0][1] == parms["I0"][1]

    # initial state vector is correct
    assert len(u) == groups
    assert len(u[0]) == 9 # S V E1 E2 I1 I2 R Y X
    assert u[0][0] == S[0][0]

    assert sum(u[0]) == parms["pop_sizes"][0], f"Sum of u[0] ({sum(u[0])}) does not equal pop_sizes[0] ({parms['pop_sizes'][0]})"
    assert sum(u[1]) == parms["pop_sizes"][1], f"Sum of u[1] ({sum(u[1])}) does not equal pop_sizes[1] ({parms['pop_sizes'][1]})"


def test_calculate_foi_0():
    # Define the parameters
    beta = np.array([[0.1, 0.2], [0.3, 0.4]])
    I_g = np.array([0, 0])
    pop_sizes = np.array([100, 200])
    target_group = 0

    # Call the calculate_foi function
    foi = calculate_foi(beta, I_g, pop_sizes, target_group)

    # Check the result
    expected_foi = 0
    assert np.isclose(foi, expected_foi), f"Expected {expected_foi}, but got {foi}"

def test_calculate_foi():
    # Define the parameters
    beta = np.array([[0.1, 0.2], [0.3, 0.4]])
    I_g = np.array([1, 0])
    pop_sizes = np.array([100, 200])
    target_group = 0

    # Call the calculate_foi function
    foi = calculate_foi(beta, I_g, pop_sizes, target_group)

    # Check the result
    expected_foi = np.dot(beta[target_group], I_g / pop_sizes)
    assert np.isclose(foi, expected_foi), f"Expected {expected_foi}, but got {foi}"

def test_active_vaccination():
    parms = {
        "n_groups": 3,
        "vaccine_uptake_range": [10, 11],
        "total_vaccine_uptake_doses": 100,
        "vaccinated_group": 2
    }

    # Initial state for each group: Group 2 has no E or I individuals yet
    u = [
        [1000, 0, 100, 50, 0, 0, 0, 0, 0],  # Group 0: S, V, E1, E2, I1, I2, R, Y, X
        [700, 0, 60, 40, 0, 0, 0, 0, 0],    # Group 1: S, V, E1, E2, I1, I2, R, Y, X
        [600, 0, 0, 0, 0, 0, 0, 0, 0]     # Group 2: S, V, E1, E2, I1, I2, R, Y, X
    ]

    vaccination_uptake_schedule = build_vax_schedule(parms)

    # Assertions to check the expected results: in group 2, 100% of eligible are susceptible
    assert np.array_equal(vaccinate_groups(parms["n_groups"], u, 5, vaccination_uptake_schedule, parms), [0, 0, 0])
    assert np.array_equal(vaccinate_groups(parms["n_groups"], u, 10, vaccination_uptake_schedule, parms), [0, 0, 50])
    assert np.array_equal(vaccinate_groups(parms["n_groups"], u, 11, vaccination_uptake_schedule, parms), [0, 0, 50])
    assert np.array_equal(vaccinate_groups(parms["n_groups"], u, 12, vaccination_uptake_schedule, parms), [0, 0, 0])

    ### Check: If vaccine_doses >> than S population
    parms["total_vaccine_uptake_doses"] = 10000 # more doses than number of S in group 2
    vaccination_uptake_schedule = build_vax_schedule(parms)
    uptake = vaccinate_groups(parms["n_groups"], u, 10, vaccination_uptake_schedule, parms)

    assert uptake[parms["vaccinated_group"]] == u[parms['vaccinated_group']][0], "Group 2 uptake is size of Susceptible population"

def test_get_infected():
    # Define the initial state
    u = [
        [99,  0, 0, 0, 1, 2, 0, 0],  # S V E1 E2 I1 I2 R Y
        [100, 0, 0, 0, 3, 4, 0, 0]   # S V E1 E2 I1 I2 R Y
    ]

    # Define the indices of the I compartments
    I_indices = [4, 5]

    # Define the number of groups
    groups = 2

    # Define time
    t = 1

    # Define parms
    parms = {
        "symptomatic_isolation": False,
        "symptomatic_isolation_day": 400
    }

    # Call the get_infected function
    infected = get_infected(u, I_indices, groups, parms, t)

    # Check the results
    expected_infected = np.array([3, 7])  # Sum of I1 and I2 for each group
    assert np.array_equal(infected, expected_infected), f"Expected {expected_infected}, but got {infected}"

def test_rate_to_frac():
    # Define the rate
    rate = 0.0 # never happens

    # Call the rate_to_frac function
    frac = rate_to_frac(rate)

    # Check the result
    expected_frac = 0
    assert np.isclose(frac, expected_frac), f"Expected {expected_frac}, but got {frac}"

def test_run_model_once_with_config():
    # Define the parameters
    testdir = os.path.dirname(__file__)
    with open(os.path.join(testdir, "test_config.yaml"), "r") as file:
        config = yaml.safe_load(file)

    parms = config["baseline_parameters"]
    parms["initial_vaccine_coverage"] = [0.9, 0.5, 0.5] # add here, griddler has it as nested params
    parms["vaccine_uptake"] = False # setting here in case default config changes
    parms["connectivity_scenario"] = 1.0
    parms["symptomatic_isolation"] = False
    parms["symptomatic_isolation_day"] = 400
    # pulling k_21 from grid parameters
    parms['k_21'] = config['grid_parameters']['k_21'][0]

    parms["beta"] = construct_beta(parms)

    # Define the time array and steps
    groups = parms["n_groups"]
    steps = parms["tf"]
    t = np.linspace(1, steps, steps)

    S, V, E1, E2, I1, I2, R, Y, X, u = initialize_population(steps, groups, parms)

    # Create an instance of SEIRModel
    model = SEIRModel(parms)

    # Call the run_model function
    S, V, E1, E2, I1, I2, R, Y, X, u = run_model(model, u, t, steps, groups, S, V, E1, E2, I1, I2, R, Y, X)

    # Check the results
    assert S.shape ==  (steps, groups)
    assert V.shape ==  (steps, groups)
    assert E1.shape == (steps, groups)
    assert E2.shape == (steps, groups)
    assert I1.shape == (steps, groups)
    assert I2.shape == (steps, groups)
    assert R.shape ==  (steps, groups)
    assert Y.shape ==  (steps, groups)
    assert X.shape ==  (steps, groups)

    # Check initial vax = end vax
    if parms["vaccine_uptake"] is False:
        assert np.array_equal(V[0], V[-1]), "The starting V[0] should be the same as the ending V[0] when uptake is zero"
