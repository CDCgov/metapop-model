import os
import yaml
from metapop.app_helper import *


# read in the test config file
testdir = os.path.dirname(__file__)
with open(os.path.join(testdir, "test_config.yaml"), "r") as file:
    config = yaml.safe_load(file)


def test_get_scenario_results():
    """Test the get_scenario_results function."""
    # get_scenario_results expects a list of parameter sets
    parms = [config["baseline_parameters"]]
    results = get_scenario_results(parms)
    assert results is not None, "Expected results to be not None"


def test_read_parameters():
    """Test the read_parameters function."""
    # read_parameters expects a file name
    parms = read_parameters(os.path.join(testdir, "test_config.yaml"))
    assert isinstance(parms, dict), "Expected parameters to be a dictionary"
    assert len(parms) > 0, "Expected parameters to be not empty"


def test_get_outcome_options():
    """Test the get_outcome_options function."""
    # outcome_options expects a list of parameter sets
    options = get_outcome_options()
    assert isinstance(options, tuple), "Expected options to be a tuple"
    expected_values = (
        "Daily Infections",
        "Daily Incidence",
        "Daily Cumulative Incidence",
        "Weekly Incidence",
        "Weekly Cumulative Incidence",
    )
    assert all(value in options for value in expected_values), \
        f"Expected values {expected_values} to be in options, but got {options}"


def test_get_outcome_mapping():
    """Test the get_outcome_mapping function."""
    # get_outcome_mapping expects a list of parameter sets

    outcome_mapping = get_outcome_mapping()
    assert isinstance(outcome_mapping, dict), "Expected mapping to be a dictionary"
    expected_keys = [
        "Daily Infections",
        "Daily Incidence",
        "Daily Cumulative Incidence",
        "Weekly Incidence",
        "Weekly Cumulative Incidence",
    ]
    assert all(key in outcome_mapping for key in expected_keys), \
        f"Expected keys {expected_keys} to be in mapping, but got {list(outcome_mapping.keys())}"


def test_get_list_keys():
    parms = dict(
        pop_sizes=[100, 200, 300],
        n_groups=3,
    )

    list_keys = get_list_keys(parms)

    assert list_keys == ['pop_sizes'], f"Expected ['pop_sizes'], but got {list_keys}"


def test_get_min_values():
    min_values = get_min_values()
    assert isinstance(min_values, dict), "Expected parameters to be a dictionary"
    expected_I0 = [0, 0, 0]
    assert min_values["I0"] == expected_I0, f"Expected min I0 values to be {expected_I0}, but got {min_values['I0']}"

    # assert that when you provide a dictionary, the returned minimum values are updated
    parms = dict(
        pop_sizes = [10000, 50, 80],
        latent_duration = 5.,
    )
    min_values = get_min_values(parms)
    assert isinstance(min_values, dict), "Expected parameters to be a dictionary"
    assert min_values["pop_sizes"] == parms["pop_sizes"], \
        f"Expected min pop_sizes to be {parms['pop_sizes']}, but got {min_values['pop_sizes']}"
    assert min_values["latent_duration"] == parms["latent_duration"], \
        f"Expected min latent_duration to be {parms['latent_duration']}, but got {min_values['latent_duration']}"


def test_get_max_values():
    max_values = get_max_values()
    assert isinstance(max_values, dict), "Expected parameters to be a dictionary"
    expected_I0 = [100, 100, 100]
    assert max_values["I0"] == expected_I0, f"Expected max I0 values to be {expected_I0}, but got {max_values['I0']}"

    # assert that when you provide a dictionary, the returned maximum values are updated
    parms = dict(
        pop_sizes = [50_000, 5_000, 8_000],
        isolation_success = 0.85,
    )
    max_values = get_max_values(parms)
    assert isinstance(max_values, dict), "Expected parameters to be a dictionary"
    assert max_values["pop_sizes"] == parms["pop_sizes"], \
        f"Expected max pop_sizes to be {parms['pop_sizes']}, but got {max_values['pop_sizes']}"
    assert max_values["isolation_success"] == parms["isolation_success"], \
        f"Expected max isolation_success to be {parms['isolation_success']}, but got {max_values['isolation_success']}"
