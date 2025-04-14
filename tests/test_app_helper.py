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
