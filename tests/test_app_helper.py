import os
from math import isclose

import numpy as np
import polars as pl
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
        "Daily Incidence",
        "Daily Cumulative Incidence",
        "Weekly Incidence",
        "Weekly Cumulative Incidence",
    )
    assert all(
        value in options for value in expected_values
    ), f"Expected values {expected_values} to be in options, but got {options}"


def test_get_outcome_mapping():
    """Test the get_outcome_mapping function."""
    # get_outcome_mapping expects a list of parameter sets

    outcome_mapping = get_outcome_mapping()
    assert isinstance(outcome_mapping, dict), "Expected mapping to be a dictionary"
    expected_keys = [
        "Daily Incidence",
        "Daily Cumulative Incidence",
        "Weekly Incidence",
        "Weekly Cumulative Incidence",
    ]
    assert all(
        key in outcome_mapping for key in expected_keys
    ), f"Expected keys {expected_keys} to be in mapping, but got {list(outcome_mapping.keys())}"


def test_get_list_keys():
    parms = dict(
        pop_sizes=[100, 200, 300],
        n_groups=3,
    )

    list_keys = get_list_keys(parms)

    assert list_keys == ["pop_sizes"], f"Expected ['pop_sizes'], but got {list_keys}"


def test_get_min_values():
    """Test the get_min_values function."""
    min_values = get_min_values()
    assert isinstance(min_values, dict), "Expected parameters to be a dictionary"
    expected_I0 = [0, 0, 0]
    assert (
        min_values["I0"] == expected_I0
    ), f"Expected min I0 values to be {expected_I0}, but got {min_values['I0']}"

    # assert that when you provide a dictionary, the returned minimum values are updated
    parms = dict(
        pop_sizes=[10000, 50, 80],
        latent_duration=5.0,
    )
    min_values = get_min_values(parms)
    assert isinstance(min_values, dict), "Expected parameters to be a dictionary"
    assert (
        min_values["pop_sizes"] == parms["pop_sizes"]
    ), f"Expected min pop_sizes to be {parms['pop_sizes']}, but got {min_values['pop_sizes']}"
    assert (
        min_values["latent_duration"] == parms["latent_duration"]
    ), f"Expected min latent_duration to be {parms['latent_duration']}, but got {min_values['latent_duration']}"


def test_get_max_values():
    """Test the get_max_values function."""
    max_values = get_max_values()
    assert isinstance(max_values, dict), "Expected parameters to be a dictionary"
    expected_I0 = [10, 10, 10]
    assert (
        max_values["I0"] == expected_I0
    ), f"Expected max I0 values to be {expected_I0}, but got {max_values['I0']}"

    # assert that when you provide a dictionary, the returned maximum values are updated
    parms = dict(
        pop_sizes=[50_000, 5_000, 8_000],
        isolation_success=0.85,
    )
    max_values = get_max_values(parms)
    assert isinstance(max_values, dict), "Expected parameters to be a dictionary"
    assert (
        max_values["pop_sizes"] == parms["pop_sizes"]
    ), f"Expected max pop_sizes to be {parms['pop_sizes']}, but got {max_values['pop_sizes']}"
    assert (
        max_values["isolation_success"] == parms["isolation_success"]
    ), f"Expected max isolation_success to be {parms['isolation_success']}, but got {max_values['isolation_success']}"


def test_get_step_values():
    """Test the get_step_values function."""
    step_values = get_step_values()
    assert isinstance(step_values, dict), "Expected parameters to be a dictionary"
    expected_I0 = 1
    assert (
        step_values["I0"] == expected_I0
    ), f"Expected step I0 values to be {expected_I0}, but got {step_values['I0']}"

    # assert that when you provide a dictionary, the returned step values are updated
    parms = dict(
        pop_sizes=50,
    )
    step_values = get_step_values(parms)
    assert isinstance(step_values, dict), "Expected parameters to be a dictionary"
    assert (
        step_values["pop_sizes"] == parms["pop_sizes"]
    ), f"Expected step pop_sizes to be {parms['pop_sizes']}, but got {step_values['pop_sizes']}"


def test_get_helpers():
    """Test the get_helpers function."""
    helpers = get_helpers()
    assert isinstance(helpers, dict), "Expected helpers to be a dictionary"
    expected_r0 = "Basic reproduction number R0. R0 cannot be negative"
    assert (
        helpers["desired_r0"] == expected_r0
    ), f"Expected R0 helper to be {expected_r0}, but got {helpers['desired_r0']}"

    # assert that when you provide a dictionary, the returned helpers are updated
    parms = dict(latent_duration="Latent duration in days.")
    helpers = get_helpers(parms)
    assert isinstance(helpers, dict), "Expected helpers to be a dictionary"
    assert (
        helpers["latent_duration"] == parms["latent_duration"]
    ), f"Expected latent_duration helper to be {parms['latent_duration']}, but got {helpers['latent_duration']}"


def test_get_formats():
    """Test the get_formats function."""
    formats = get_formats()
    assert isinstance(formats, dict), "Expected formats to be a dictionary"
    expected_R0 = "%.1f"
    assert (
        formats["desired_r0"] == expected_R0
    ), f"Expected R0 string format to be {expected_R0}, but got {formats['desired_r0']}"

    # assert that when you provide a dictionary, the returned formats are updated
    parms = dict(
        latent_duration="%.2f",
    )
    formats = get_formats(parms)
    assert isinstance(formats, dict), "Expected formats to be a dictionary"
    assert (
        formats["latent_duration"] == parms["latent_duration"]
    ), f"Expected latent_duration string format to be {parms['latent_duration']}, but got {formats['latent_duration']}"


def test_get_base_widget_idkeys():
    """Test the get_base_widget_idkeys function."""
    base_keys = get_base_widget_idkeys()
    assert isinstance(base_keys, dict), "Expected base keys to be a dictionary"
    expected_k_i = ["k_i_0", "k_i_1", "k_i_2"]
    assert (
        base_keys["k_i"] == expected_k_i
    ), f"Expected k_i keys to be {expected_k_i}, but got {base_keys['k_i']}"

    # assert that when you provide a dictionary, the returned base keys are updated
    parms = dict(
        k_i=["k_i_0", "k_i_1", "k_i_2", "k_i_3", "k_i_4", "k_i_5"],
    )
    base_keys = get_base_widget_idkeys(parms)
    assert isinstance(base_keys, dict), "Expected base keys to be a dictionary"
    assert (
        base_keys["k_i"] == parms["k_i"]
    ), f"Expected k_i keys to be {parms['k_i']}, but got {base_keys['k_i']}"


def test_get_widget_idkeys():
    """Test the get_widget_idkeys function."""

    widget_idkeys = get_widget_idkeys(5)
    assert isinstance(widget_idkeys, dict), "Expected widget idkeys to be a dictionary"
    expected_r0 = "desired_r0_5"
    assert (
        widget_idkeys["desired_r0"] == expected_r0
    ), f"Expected R0 idkey to be {expected_r0}, but got {widget_idkeys['desired_r0']}"


def test_rescale_prop_vax():
    """Test the rescale_prop_vax function."""
    # Input parameters
    parms = {
        "pop_sizes": 100,  # Population sizes for 3 groups
        "initial_vaccine_coverage": 0.89,  # Initial vaccine coverage for each group
        "I0": 1,  # Initial infections for each group
        "total_vaccine_uptake_doses": 0.5,  # Proportional vaccine uptake in percentage
    }

    # Expected result
    expected_total_doses = int(
        np.sum(
            (
                np.array(parms["pop_sizes"])
                - np.array(parms["pop_sizes"])
                * np.array(parms["initial_vaccine_coverage"])
                - np.array(parms["I0"])
            )
            * (parms["total_vaccine_uptake_doses"] / 100.0)
        )
    )

    # Call the function
    rescaled_parms = rescale_prop_vax(parms)

    assert (
        rescaled_parms["total_vaccine_uptake_doses"] == expected_total_doses
    ), f"Expected total vaccine uptake doses to be {expected_total_doses}, but got {rescaled_parms['total_vaccine_uptake_doses']}"


def test_get_median_from_episize_one_group():
    # Odd number of replicates to return a single median
    n_reps = 101

    episize_df = pl.DataFrame(
        {
            "replicate": range(n_reps),
            "R": np.random.normal(0, 0.5, n_reps),
            "S": 0,
            "group": 0,
            "t": 1,
        }
    )

    dummy_traj = get_median_trajectory_from_episize(episize_df)
    r = dummy_traj["R"].item()

    # check that r is near 0
    assert isclose(r, 0, abs_tol=0.1), f"Expected R to be close to 0, but got {r}"

    # and equal to the median
    assert r == np.median(
        episize_df["R"]
    ), f"Expected R to be {np.median(episize_df['R'])}, but got {r}"


def test_get_median_from_episize_multiple_groups():
    n_reps = 101

    # Create two groups, one with pop size of about 0 (S=0, R~Norm(0, 0.5)), the other of popsize of about 0 (s=10, R~Norm(0, 0.5))
    episize_df = pl.DataFrame(
        {
            "replicate": list(range(n_reps)) * 2,
            "R": np.concatenate(
                [np.random.normal(0, 0.5, n_reps), np.random.normal(0, 0.5, n_reps)]
            ),
            "S": [0] * n_reps + [10] * n_reps,
            "group": [0] * n_reps + [1] * n_reps,
            "t": 1,
        }
    )

    dummy_traj = get_median_trajectory_from_episize(episize_df)

    # Check that Group 0, the base group, has a median r equal to function output
    r_group = dummy_traj.filter(pl.col("group") == 0)["R"].item()
    median_group = np.median(episize_df.filter(pl.col("group") == 0)["R"])

    assert (
        r_group == median_group
    ), f"Expected R for group 1 to be {median_group}, but got {r_group}"


def test_get_median_from_peak_time():
    # Odd number of replicates so that median is exactly equal to observed value
    n_reps = 101
    base_replicate_trajectory = pl.DataFrame(
        {
            "I1": [0, 1, 2, 1, 0],
            "I2": [0, 0, 0, 0, 0],
            "t_id": [0, 1, 2, 3, 4],
            "group": 0,
        }
    )
    # Create multiple trajectories from base and bind together
    for replicate in range(n_reps):
        current = base_replicate_trajectory.with_columns(
            (pl.col("t_id") + np.random.normal(0, 0.5, 1)).alias("t"),
            pl.lit(replicate).alias("replicate"),
        )
        if replicate == 0:
            all_trajectories = current
        else:
            all_trajectories.vstack(current)

    # Get the median trajectory
    dummy_traj = get_median_trajectory_from_peak_time(all_trajectories)
    # Check that the median trajectory is correct
    true_median = (
        all_trajectories.filter(pl.col("t_id") == 2).select("t").median().item()
    )

    assert (
        dummy_traj["t"][2] == true_median
    ), f"Expected median t to be {true_median}, but got {dummy_traj['t'][2]}"
