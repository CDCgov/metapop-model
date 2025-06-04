import os

import numpy as np
import yaml
from numpy.testing import assert_allclose

from metapop.helper import *
from metapop.model import *
from metapop.sim import *  # noqa: F405


def test_get_percapita_contact_matrix():
    # Define the parameters
    parms = {
        "k_i": np.array([10, 10, 10]),
        "k_g1": 1,
        "k_g2": 2,
        "k_21": 2,
        "n_groups": 3,
        "pop_sizes": [1000, 100, 100],
    }

    # Call the get_percapita_contact_matrix function
    percapita_contacts = get_percapita_contact_matrix(parms)

    # Check the result
    expected_percapita_contacts = np.array(
        [
            [9.7, 1.0, 2.0],
            [0.1, 7.0, 2.0],
            [0.2, 2.0, 6.0],
        ]  # noqa: E501
    )
    assert np.array_equal(
        percapita_contacts, expected_percapita_contacts
    ), f"Expected {expected_percapita_contacts}, but got {percapita_contacts} when using equal degree for all subgroups"

    # A scenario where the degree is different for each subgroup
    parms["k_i"] = np.array([10, 20, 15])

    percapita_contacts = get_percapita_contact_matrix(parms)

    expected_percapita_contacts = np.array(
        [
            [9.7, 1.0, 2.0],
            [0.1, 17.0, 2.0],
            [0.2, 2.0, 11.0],
        ]  # noqa: E501
    )
    assert np.array_equal(
        percapita_contacts, expected_percapita_contacts
    ), f"Expected {expected_percapita_contacts}, but got {percapita_contacts} when using different degree for each subgroup"


def test_get_r0():
    # keeling and rohani example
    beta_matrix = np.array(
        [  # noqa: E231
            [10.0, 0.1],
            [0.1, 1.0],
        ]  # noqa: E501
    )
    gamma = 1.0
    pop_sizes = np.array([20, 80])
    expected_r0 = 2.001331855134916
    r0 = get_r0(beta_matrix, gamma, pop_sizes)
    assert np.isclose(r0, expected_r0), f"Expected {expected_r0}, but got {r0}"


def test_get_r0_one_group():
    parms = dict(
        k_i=np.array([10.0]),
        infectious_duration=9.0,
        desired_r0=2.0,
        n_groups=1,
    )
    parms["gamma"] = time_to_rate(parms["infectious_duration"])
    r0_base = get_r0_one_group(parms["k_i"], parms["gamma"])
    beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
    beta_scaled = rescale_beta_matrix(parms["k_i"][0], beta_factor)
    expected_r0 = parms["desired_r0"]

    r0 = get_r0_one_group([beta_scaled], parms["gamma"])
    assert np.isclose(r0, expected_r0), f"Expected {expected_r0}, but got {r0}"


def test_construct_beta():
    parms = {
        "k_i": [10, 10, 10],
        "k_g1": 1,
        "k_g2": 2,
        "k_21": 2,
        "gamma": 0.1,
        "pop_sizes": np.array([1000, 100, 100]),
        "n_i_compartments": 2,
        "desired_r0": 2.0,
        "n_groups": 3,
        "connectivity_scenario": 1.0,
    }
    beta_unscaled = get_percapita_contact_matrix(parms)
    r0_base = get_r0(beta_unscaled, parms["gamma"], parms["pop_sizes"])
    beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
    beta_scaled = rescale_beta_matrix(beta_unscaled, beta_factor)
    expected_beta = construct_beta(parms)
    assert beta_scaled.shape == (
        3,
        3,
    ), f"Expected shape (3, 3), but got {beta_scaled.shape}"
    assert np.allclose(
        beta_scaled, expected_beta
    ), f"Expected {expected_beta}, but got {beta_scaled}"

    # one_group setup
    parms = {
        "k_i": [10.0],
        "k_g1": 0,
        "k_g2": 0,
        "k_21": 0,
        "gamma": 0.1,
        "pop_sizes": np.array([1000]),
        "n_i_compartments": 2,
        "desired_r0": 2.0,
        "n_groups": 1,
    }
    parms["k_i"] = np.array(parms["k_i"])
    r0_base = get_r0_one_group(parms["k_i"], parms["gamma"])
    beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
    beta_scaled = rescale_beta_matrix(parms["k_i"], beta_factor)
    expected_beta = construct_beta(parms)
    assert np.allclose(
        r0_base, 100.0
    ), f"Expected beta / gamma = 100, but got {r0_base}"
    assert isinstance(
        beta_scaled, np.ndarray
    ), f"Expected np.ndarray, but got {type(beta_scaled)}"
    assert np.allclose(
        beta_scaled, expected_beta
    ), f"Expected {expected_beta}, but got {beta_scaled}"


def test_pop_initialization():
    parms = {
        "pop_sizes": [1000, 2000],
        "initial_vaccine_coverage": [0.0, 0.95],
        "I0": [10, 2],
        "vaccine_efficacy_2_dose": 0.95,
    }
    steps = 3
    groups = 2

    S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u = initialize_population(
        steps, groups, parms
    )

    # correct dimensions
    assert len(S) == steps
    assert len(S[0]) == groups

    # initial vaccination
    assert V[0][0] == 0
    assert_allclose(
        V[0][1],
        parms["pop_sizes"][1]
        * parms["initial_vaccine_coverage"][1]
        * parms["vaccine_efficacy_2_dose"],
        rtol=1e-2,
        atol=1e-8,
    )
    assert_allclose(
        V[0][1] + SV[0][1],
        parms["pop_sizes"][1] * parms["initial_vaccine_coverage"][1],
        rtol=1e-2,
        atol=1e-8,
    )

    # initial infections
    assert I1[0][0] == parms["I0"][0]
    assert I1[0][1] == parms["I0"][1]

    # initial state vector is correct
    assert len(u) == groups
    assert len(u[0]) == 12  # S V SV E1 E2 E1_V, E2_V I1 I2 R Y X
    assert u[0][0] == S[0][0]

    assert (
        sum(u[0]) == parms["pop_sizes"][0]
    ), f"Sum of u[0] ({sum(u[0])}) does not equal pop_sizes[0] ({parms['pop_sizes'][0]})"
    assert (
        sum(u[1]) == parms["pop_sizes"][1]
    ), f"Sum of u[1] ({sum(u[1])}) does not equal pop_sizes[1] ({parms['pop_sizes'][1]})"


def test_calculate_foi_0():
    # Define the parameters
    beta = np.array(
        [
            [0.1, 0.2],  # noqa: E231
            [0.3, 0.4],  # noqa: E231
        ]
    )
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
    beta = np.array(
        [
            [0.1, 0.2],  # noqa: E231
            [0.3, 0.4],  # noqa: E231
        ]
    )
    I_g = np.array([1, 0])
    pop_sizes = np.array([100, 200])
    target_group = 0

    # Call the calculate_foi function
    foi = calculate_foi(beta, I_g, pop_sizes, target_group)

    # Check the result
    expected_foi = np.dot(beta[target_group], I_g / pop_sizes)
    assert np.isclose(foi, expected_foi), f"Expected {expected_foi}, but got {foi}"


def test_vaccination_schedule_not_empty():
    """
    Test that if vaccine_uptake_duration_days is 0, then the schedule is not
    empty but contains day 0 and with 0 doses.
    """
    parms = dict(
        n_groups=3,
        vaccine_uptake_start_day=10,
        vaccine_uptake_duration_days=0,
        total_vaccine_uptake_doses=0,
        vaccinated_group=2,
        tf=20,
    )
    parms["t_array"] = get_time_array(parms)
    vaccination_uptake_schedule = build_vax_schedule(parms)
    assert vaccination_uptake_schedule == {parms["vaccine_uptake_start_day"] + 1: 0}


def test_vaccines_administered_on_single_day():
    """
    Test that if vaccine_uptake_duration_days is 1, then all vaccine doses i.e.,
    vaccinations happen on that single day.
    """
    parms = dict(
        n_groups=3,
        vaccine_uptake_start_day=10,
        vaccine_uptake_duration_days=1,
        total_vaccine_uptake_doses=100,
        vaccinated_group=2,
        vaccine_efficacy_1_dose=0.93,
        tf=20,
    )

    # Initial state for each group: Group 2 has no E or I individuals yet
    u = [  #  S, V, SV,  E1, E2, E1_V, E2_V, I1, I2, R, Y, X
        [1000, 0, 0, 100, 50, 0, 0, 0, 0, 0, 0, 0],
        [700, 0, 60, 40, 0, 0, 0, 0, 0, 0, 0, 0],
        [600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    parms["t_array"] = get_time_array(parms)
    vaccination_uptake_schedule = build_vax_schedule(parms)

    # assert that all vaccine doses are administered on the same day
    fails = int(
        parms["total_vaccine_uptake_doses"] * (1 - parms["vaccine_efficacy_1_dose"])
    )
    assert np.array_equal(
        vaccinate_groups(parms["n_groups"], u, 11, vaccination_uptake_schedule, parms),
        ([0, 0, 100 - fails], [0, 0, fails], [0, 0, 0]),
    )


def test_active_vaccination():
    """
    Test that vaccination happens on the right days when vaccination campaign
    starts after the first day in the model.
    """
    # Check that vaccine uptake happens on 2 days
    parms = {
        "n_groups": 3,
        "vaccine_uptake_start_day": 10,
        "vaccine_uptake_duration_days": 4,
        "total_vaccine_uptake_doses": 100,
        "vaccinated_group": 2,
        "vaccine_efficacy_1_dose": 0.93,
        "tf": 20,
    }

    # Initial state for each group: Group 2 has no E or I individuals yet
    u = [  #  S, V, SV,  E1,  E2, E1_V, E2_V, I1, I2, R, Y, X
        [1000, 0, 0, 100, 50, 0, 0, 0, 0, 0, 0, 0],
        [700, 0, 0, 60, 40, 0, 0, 0, 0, 0, 0, 0],
        [600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    parms["t_array"] = get_time_array(parms)
    vaccination_uptake_schedule = build_vax_schedule(parms)

    # Assertions to check the expected results: in group 2, 100% of eligible are susceptible
    # no vaccines on day 5 - the campaign has not started yet
    assert np.array_equal(
        vaccinate_groups(parms["n_groups"], u, 5, vaccination_uptake_schedule, parms),
        ([0, 0, 0], [0, 0, 0], [0, 0, 0]),
    )
    # Check that on every day of the vaccination campaign, the uptake is 25
    doses_per_day = int(
        parms["total_vaccine_uptake_doses"] / parms["vaccine_uptake_duration_days"]
    )
    for day in vaccination_uptake_schedule:
        assert np.array_equal(
            sum(
                vaccinate_groups(
                    parms["n_groups"], u, day, vaccination_uptake_schedule, parms
                )
            ),
            [0, 0, doses_per_day],
        )

    day_after_vaccination = max(vaccination_uptake_schedule.keys()) + 1

    assert np.array_equal(
        vaccinate_groups(
            parms["n_groups"],
            u,
            day_after_vaccination,
            vaccination_uptake_schedule,
            parms,
        ),
        ([0, 0, 0], [0, 0, 0], [0, 0, 0]),
    )


def test_vaccine_doses_greater_than_population():
    """
    Test that if the number of vaccine doses is greater than the susceptible population
    in the vaccinated group, the uptake is capped at that size.
    """
    # Check if vaccine_doses >> than S population
    parms = {
        "n_groups": 3,
        "vaccine_uptake_start_day": 10,
        "vaccine_uptake_duration_days": 1,
        "total_vaccine_uptake_doses": 10000,
        "vaccinated_group": 2,
        "vaccine_efficacy_1_dose": 0.93,
        "tf": 20,
    }

    # Initial state for each group: Group 2 has no E or I individuals yet
    u = [  #  S, V, SV,  E1,  E2, E1_V, E2_V, I1, I2, R, Y, X
        [1000, 0, 0, 100, 50, 0, 0, 0, 0, 0, 0, 0],
        [700, 0, 0, 60, 40, 0, 0, 0, 0, 0, 0, 0],
        [600, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    initial_S_group_2 = u[parms["vaccinated_group"]][0]

    parms["t_array"] = get_time_array(parms)
    vaccination_uptake_schedule = build_vax_schedule(parms)

    success, fails, vaccinated_exposed = vaccinate_groups(
        parms["n_groups"], u, 11, vaccination_uptake_schedule, parms
    )

    failed_vax = int(initial_S_group_2 * (1 - parms["vaccine_efficacy_1_dose"]))
    assert (
        success[parms["vaccinated_group"]] == initial_S_group_2 - failed_vax
    ), "Group 2 success is size of Susceptible population (accounting for vaccine efficacy)"

    assert (
        fails[parms["vaccinated_group"]] == failed_vax
    ), "Group 2 fails is size of Susceptible population (accounting for vaccine efficacy)"


def test_get_infected():
    # Define the initial state
    u = [
        [99, 0, 0, 0, 1, 2, 0, 0],  # S V E1 E2 I1 I2 R Y
        [100, 0, 0, 0, 3, 4, 0, 0],  # S V E1 E2 I1 I2 R Y
    ]

    # Define the indices of the I compartments
    I_indices = [4, 5]

    # Define the number of groups
    groups = 2

    # Define time
    t = 1

    # Define parms
    parms = {
        "symptomatic_isolation_start_day": 400,
        "symptomatic_isolation_duration_days": 100,
        "pre_rash_isolation_start_day": 400,
        "pre_rash_isolation_duration_days": 100,
    }

    # Call the get_infected function
    infected = get_infected(u, I_indices, groups, parms, t)

    # Check the results
    expected_infected = np.array([3, 7])  # Sum of I1 and I2 for each group
    assert np.array_equal(
        infected, expected_infected
    ), f"Expected {expected_infected}, but got {infected}"


def test_symptomatic_isolation():
    parms = {
        "symptomatic_isolation_start_day": 10,
        "symptomatic_isolation_duration_days": 1,
        "isolation_success": 1.0,
        "pre_rash_isolation_start_day": 400,
        "pre_rash_isolation_duration_days": 100,
    }

    # Initial state for each group: Group 2 has no E or I individuals yet
    u = [
        [1000, 0, 0, 0, 100, 100, 0, 0, 0],  # Group 0: S, V, E1, E2, I1, I2, R, Y, X
        [600, 0, 0, 0, 0, 0, 0, 0, 0],  # Group 2: S, V, E1, E2, I1, I2, R, Y, X
    ]

    # Define the indices of the I compartments
    I_indices = [4, 5]

    # Define the number of groups
    groups = 2

    # First test that no isolation happens before isolation day
    t = 1

    # Call the get_infected function
    infected = get_infected(u, I_indices, groups, parms, t)

    # Check the results
    expected_infected = np.array([200, 0])  # Sum of I1 and I2 for each group
    assert np.array_equal(
        infected, expected_infected
    ), f"Expected {expected_infected}, but got {infected}"

    # Next test that isolation happens on isolation day
    t2 = 10

    # Call the get_infected function
    infected2 = get_infected(u, I_indices, groups, parms, t2)

    # Check the results
    expected_infected2 = np.array([100, 0])  # Should just be I1
    assert np.array_equal(
        infected2, expected_infected2
    ), f"Expected {expected_infected2}, but got {infected2}"

    # Next test that no longer happening after duration
    t3 = 100

    # Call the get_infected function
    infected3 = get_infected(u, I_indices, groups, parms, t3)

    # Check the results, we should now be back to pre-isolation values
    assert np.array_equal(
        infected3, expected_infected
    ), f"Expected {expected_infected}, but got {infected3}"


def test_pre_rash_isolation():
    parms = {
        "symptomatic_isolation_start_day": 400,
        "symptomatic_isolation_duration_days": 100,
        "pre_rash_isolation_start_day": 10,
        "pre_rash_isolation_duration_days": 1,
        "pre_rash_isolation_success": 1.0,
    }

    # Initial state for each group: Group 2 has no E or I individuals yet
    u = [
        [1000, 0, 0, 0, 100, 100, 0, 0, 0],  # Group 0: S, V, E1, E2, I1, I2, R, Y, X
        [600, 0, 0, 0, 0, 0, 0, 0, 0],  # Group 2: S, V, E1, E2, I1, I2, R, Y, X
    ]

    # Define the indices of the I compartments
    I_indices = [4, 5]

    # Define the number of groups
    groups = 2

    # First test that no isolation happens before isolation day
    t = 1

    # Call the get_infected function
    infected = get_infected(u, I_indices, groups, parms, t)

    # Check the results
    expected_infected = np.array([200, 0])  # Sum of I1 and I2 for each group
    assert np.array_equal(
        infected, expected_infected
    ), f"Expected {expected_infected}, but got {infected}"

    # Next test that isolation happens on isolation day
    t2 = 10

    # Call the get_infected function
    infected2 = get_infected(u, I_indices, groups, parms, t2)

    # Check the results
    expected_infected2 = np.array([100, 0])  # Should just be I2
    assert np.array_equal(
        infected2, expected_infected2
    ), f"Expected {expected_infected2}, but got {infected2}"

    # Next test that no longer happening after duration
    t3 = 100

    # Call the get_infected function
    infected3 = get_infected(u, I_indices, groups, parms, t3)

    # Check the results, we should now be back to pre-isolation values
    assert np.array_equal(
        infected3, expected_infected
    ), f"Expected {expected_infected}, but got {infected3}"


def test_pre_post_isolation():
    parms = {
        "symptomatic_isolation_start_day": 10,
        "symptomatic_isolation_duration_days": 1,
        "isolation_success": 1.0,
        "pre_rash_isolation_start_day": 10,
        "pre_rash_isolation_duration_days": 1,
        "pre_rash_isolation_success": 1.0,
    }

    # Initial state for each group: Group 2 has no E or I individuals yet
    u = [
        [1000, 0, 0, 0, 100, 100, 0, 0, 0],  # Group 0: S, V, E1, E2, I1, I2, R, Y, X
        [600, 0, 0, 0, 0, 0, 0, 0, 0],  # Group 2: S, V, E1, E2, I1, I2, R, Y, X
    ]

    # Define the indices of the I compartments
    I_indices = [4, 5]

    # Define the number of groups
    groups = 2

    # First test that no isolation happens before isolation day
    t = 1

    # Call the get_infected function
    infected = get_infected(u, I_indices, groups, parms, t)

    # Check the results
    expected_infected = np.array([200, 0])  # Sum of I1 and I2 for each group
    assert np.array_equal(
        infected, expected_infected
    ), f"Expected {expected_infected}, but got {infected}"

    # Next test that isolation happens on isolation day
    t2 = 10

    # Call the get_infected function
    infected2 = get_infected(u, I_indices, groups, parms, t2)

    # Check the results
    expected_infected2 = np.array([0, 0])  # Should just be no infections
    assert np.array_equal(
        infected2, expected_infected2
    ), f"Expected {expected_infected2}, but got {infected2}"

    # Next test that no longer happening after duration
    t3 = 100

    # Call the get_infected function
    infected3 = get_infected(u, I_indices, groups, parms, t3)

    # Check the results, we should now be back to pre-isolation values
    assert np.array_equal(
        infected3, expected_infected
    ), f"Expected {expected_infected}, but got {infected3}"


def test_rate_to_frac():
    # Define the rate
    rate = 0.0  # never happens

    # Call the rate_to_frac function
    frac = rate_to_frac(rate)

    # Check the result
    expected_frac = 0
    assert np.isclose(frac, expected_frac), f"Expected {expected_frac}, but got {frac}"


def test_run_model_once_with_config():
    # Define the parameters
    testdir = os.path.dirname(__file__)
    with open(os.path.join(testdir, "test_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    parms = config["baseline_parameters"]

    # cast rate params
    parms["sigma"] = time_to_rate(parms["latent_duration"])
    parms["gamma"] = time_to_rate(parms["infectious_duration"])
    parms["sigma_scaled"] = parms["sigma"] * parms["n_e_compartments"]
    parms["gamma_scaled"] = parms["gamma"] * parms["n_i_compartments"]

    # matrix construction
    parms["beta"] = construct_beta(parms)

    # Define the time array and steps
    groups = parms["n_groups"]
    steps = parms["tf"]
    t = np.linspace(1, steps, steps)

    S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u = initialize_population(
        steps, groups, parms
    )

    # Individuals with vaccine
    SV_initial = (u[0][2], u[1][2], u[2][2])
    E1_V_initial = (u[0][5], u[1][5], u[2][5])

    # assert the sum of E1_V_initial is 0
    assert sum(E1_V_initial) == 0, "E1_V_initial should be 0"

    # Create an instance of SEIRModel
    model = SEIRModel(parms, seed=123)

    # Call the run_model function
    S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X, u = run_model(
        model, u, t, steps, groups, S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X
    )

    # Check the results
    assert S.shape == (steps, groups)
    assert V.shape == (steps, groups)
    assert SV.shape == (steps, groups)
    assert E1.shape == (steps, groups)
    assert E2.shape == (steps, groups)
    assert E1_V.shape == (steps, groups)
    assert E2_V.shape == (steps, groups)
    assert I1.shape == (steps, groups)
    assert I2.shape == (steps, groups)
    assert R.shape == (steps, groups)
    assert Y.shape == (steps, groups)
    assert X.shape == (steps, groups)

    # Check initial vax = end vax
    if parms["total_vaccine_uptake_doses"] == 0:
        assert np.array_equal(
            V[0], V[-1]
        ), "The starting V[0] should be the same as the ending V[0] when uptake is zero"

    # Check that SV_initial is smaller than it is at the end (group 0)
    # if people are leaving SV, then they should be populating E1_V
    if SV_initial[0] > SV[-1][0]:
        assert (
            sum(sum(E1_V)) > 0.0
        ), "Individuals should be entering E1_V after SV, i.e., this vector should be getting populated through the simulation"


def test_seed_from_string():
    # Check that seeds are 10-digit integers
    result = seed_from_string("test")
    assert isinstance(result, int), "Seed should be an integer"

    # Check that seeds are different
    assert seed_from_string("test") == seed_from_string("test")
    assert seed_from_string("test") != seed_from_string("test2")


def test_get_metapop_info():
    # Call the get_metadata_info function
    metadata = get_metapop_info()

    # Check the result
    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert "version" in metadata, "Metadata should contain 'version' key"
    assert "commit" in metadata, "Metadata should contain 'commit' key"
    assert metadata["url"] == "https://github.com/cdcent/metapop-model"
