import os

import numpy as np
import polars as pl
import pytest
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
        "Daily incidence",
        "Daily cumulative incidence",
        "Weekly incidence",
        "Weekly cumulative incidence",
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
        "Daily incidence",
        "Daily cumulative incidence",
        "Weekly incidence",
        "Weekly cumulative incidence",
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
    expected_r0 = "The basic reproductive number captures contact rates and the probability of infection given contact with an infectious individual. Some communities may have different contact patterns — for example, in communities with larger households or higher population density. R0 values for measles are typically estimated to be between 12 and 18 (see Detailed Methods)"
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


def test_get_base_session_state_idkeys():
    """Test the get_base_session_state_idkeys function."""
    base_keys = get_base_session_state_idkeys()
    assert isinstance(base_keys, dict), "Expected base keys to be a dictionary"
    expected_k_i = ["k_i_0", "k_i_1", "k_i_2"]
    assert (
        base_keys["k_i"] == expected_k_i
    ), f"Expected k_i keys to be {expected_k_i}, but got {base_keys['k_i']}"

    # assert that when you provide a dictionary, the returned base keys are updated
    parms = dict(
        k_i=["k_i_0", "k_i_1", "k_i_2", "k_i_3", "k_i_4", "k_i_5"],
    )
    base_keys = get_base_session_state_idkeys(parms)
    assert isinstance(base_keys, dict), "Expected base keys to be a dictionary"
    assert (
        base_keys["k_i"] == parms["k_i"]
    ), f"Expected k_i keys to be {parms['k_i']}, but got {base_keys['k_i']}"


def test_get_session_state_idkeys():
    """Test the get_session_state_idkeys function."""

    session_state_idkeys = get_session_state_idkeys(5)
    assert isinstance(
        session_state_idkeys, dict
    ), "Expected session state idkeys to be a dictionary"
    expected_r0 = "desired_r0_5"
    assert (
        session_state_idkeys["desired_r0"] == expected_r0
    ), f"Expected R0 idkey to be {expected_r0}, but got {session_state_idkeys['desired_r0']}"


def test_rescale_prop_vax():
    """Test the rescale_prop_vax function."""
    # Input parameters
    parms = {
        "pop_sizes": 100,  # Population sizes for 3 groups
        "initial_vaccine_coverage": 0.89,  # Initial vaccine coverage for each group
        "I0": 1,  # Introduced infections for each group
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
            "R": [x / 100.0 for x in range(n_reps)],
            "S": 0,
            "group": 0,
            "t": 1,
        }
    )

    dummy_traj_id = get_median_trajectory_from_episize(episize_df)
    r = episize_df.filter(pl.col("replicate") == dummy_traj_id)["R"].item()

    # and equal to the median
    assert r == np.median(
        episize_df["R"]
    ), f"Expected R to be {np.median(episize_df['R'])}, but got {r}"


def test_get_median_from_episize_multiple_groups():
    n_reps = 101

    # Create two groups with different final epidemic sizes
    episize_df = pl.DataFrame(
        {
            "replicate": list(range(n_reps)) * 2,
            "R": [x / 100.0 for x in range(n_reps)] + [x / 50.0 for x in range(n_reps)],
            "S": [0] * n_reps + [10] * n_reps,
            "group": [0] * n_reps + [1] * n_reps,
            "t": 1,
        }
    )

    for group in range(2):
        dummy_traj_id = get_median_trajectory_from_episize(episize_df, base_group=group)

        # Check that Group 0, the base group, has a median r equal to function output
        r_group = episize_df.filter(
            (pl.col("group") == group) & (pl.col("replicate") == dummy_traj_id)
        )["R"].item()

        median_group = np.median(episize_df.filter((pl.col("group") == group))["R"])

        assert (
            r_group == median_group
        ), f"Expected R for group {group} to be {median_group}, but got {r_group}"


# Test selection of median trajectory from peak time with multiple peaks in each replicate
def test_get_median_from_peak_time_multiple_maxes():
    # Odd number of replicates so that median is exactly equal to observed value
    n_reps = 101

    # Peak infection times are at 1 and 3, meaning time id 2 should be selected
    base_replicate_trajectory = pl.DataFrame(
        {
            "I": [0, 2, 1, 2, 0],
            "t_id": [0, 1, 2, 3, 4],
            "group": 0,
        }
    )
    # Create multiple trajectories from base and bind together
    for replicate in range(n_reps):
        current = base_replicate_trajectory.with_columns(
            (pl.col("t_id") + pl.lit(replicate / 100)).alias("t"),
            pl.lit(replicate).alias("replicate"),
        )
        if replicate == 0:
            all_trajectories = current
        else:
            all_trajectories = all_trajectories.vstack(current)

    # Get the median trajectory
    dummy_traj_id = get_median_trajectory_from_peak_time(all_trajectories)
    # Check that the median trajectory is correct
    true_median = (
        all_trajectories.filter(pl.col("t_id") == 2).select("t").median().item()
    )

    observed_median = all_trajectories.filter(
        (pl.col("t_id") == 2) & (pl.col("replicate") == dummy_traj_id)
    )["t"].item()

    assert (
        observed_median == true_median
    ), f"Expected median t to be {true_median}, but got {observed_median}"


# Use a single peak infection time but have multiple groups
def test_get_median_from_peak_time_multiple_groups():
    # Odd number of replicates so that median is exactly equal to observed value
    n_reps = 101
    base_replicate_trajectory = pl.DataFrame(
        {
            "I": [0, 1, 2, 1, 0],
            "t_id": [0, 1, 2, 3, 4],
        }
    )
    # Create multiple trajectories from base and bind together, distinguishing the median times for each group
    for replicate in range(n_reps):
        for group in range(2):
            # Create a unique time for each replicate
            unique_time = (replicate / 100) + group
            current = base_replicate_trajectory.with_columns(
                (pl.col("t_id") + pl.lit(unique_time)).alias("t"),
                pl.lit(replicate).alias("replicate"),
                pl.lit(group).alias("group"),
            )
            if replicate == 0 and group == 0:
                all_trajectories = current
            else:
                all_trajectories = all_trajectories.vstack(current)

    for group in range(2):
        # Get the median trajectory
        dummy_traj_id = get_median_trajectory_from_peak_time(
            all_trajectories, base_group=group
        )
        # Check that the median trajectory is correct for each base group
        true_median = (
            all_trajectories.filter((pl.col("t_id") == 2) & (pl.col("group") == group))
            .select("t")
            .median()
            .item()
        )

        observed_median = all_trajectories.filter(
            (pl.col("group") == group)
            & (pl.col("t_id") == 2)
            & (pl.col("replicate") == dummy_traj_id)
        )["t"].item()

        assert (
            observed_median == true_median
        ), f"Expected median t in group {group} to be {true_median}, but got {observed_median}"


def ks_compare(base: list, compared_array: list, expect: bool, threshold: float):
    base_scenario = pl.DataFrame({"Total": base, "Scenario": ["base"] * len(base)})
    compared_scenario = pl.DataFrame(
        {"Total": compared_array, "Scenario": ["compared"] * len(compared_array)}
    )
    assert (
        totals_same_by_ks(
            base_scenario.vstack(compared_scenario), ["base", "compared"], threshold
        )
        == expect
    )


def test_totals_same_by_ks():
    base = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]
    similar = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
    with_outlier = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 15]
    moderate_different = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4]
    different = [6, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10]

    ks_compare(base, base, True, 0.05)
    ks_compare(base, similar, True, 0.05)
    ks_compare(base, with_outlier, True, 0.05)
    ks_compare(base, moderate_different, False, 0.05)
    ks_compare(base, different, False, 0.05)


def test_indistinguishable_scenarios():
    # modify to be a single population
    parms = config["baseline_parameters"]
    parms["n_replicates"] = 2000
    parms["n_groups"] = 1
    parms["initial_vaccine_coverage"] = [0.99]
    parms["pop_sizes"] = [2000]
    parms["I0"] = [5]

    parms["vaccinated_group"] = 0
    parms["vaccine_uptake_start_day"] = 0
    parms["vaccine_uptake_duration_days"] = 0
    parms["total_vaccine_uptake_doses"] = 0

    # Write out scenario names
    scenario_names = ["no_intervention", "intervention"]

    # Make the no intervention simulation outputs
    reps_no_intervention = get_scenario_results([parms])
    reps_no_intervention = reps_no_intervention.with_columns(
        pl.lit(scenario_names[0]).alias("Scenario")
    )

    # Make the vax intervention scenario and change base seed
    # The vax intervention should vaccinate all individuals in the population who are not infected
    # or vaccinated, whether protected or who had a vaccine failure.
    parms["vaccine_uptake_start_day"] = 75
    parms["vaccine_uptake_duration_days"] = 20
    parms["seed"] = 1234

    # Set vaccine doses to be equivalent to 100% of unvaccinated susceptibles as in the app slider
    parms["total_vaccine_uptake_doses"] = int(
        parms["pop_sizes"][0] * (1 - parms["initial_vaccine_coverage"][0])
        - parms["I0"][0]
    )
    reps_with_vax = get_scenario_results([parms])
    reps_with_vax = reps_with_vax.with_columns(
        pl.lit(scenario_names[1]).alias("Scenario")
    )

    # Combine results as in app.py
    combined_results = (
        reps_with_vax.vstack(reps_no_intervention)
        .with_columns(
            pl.col("t").cast(pl.Int64),
            pl.col("group").cast(pl.Int64),
            pl.col("Y").cast(pl.Int64),
        )
        .filter(pl.col("t") == pl.col("t").max())
        .group_by("Scenario", "replicate")
        .agg(pl.col("Y").sum().alias("Total"))
        .sort(["Scenario", "replicate"])
    )

    # Scenarios for these sets should be indistinguishable
    assert totals_same_by_ks(combined_results, scenario_names, 0.05)

    # Average difference between replicates across scenarios should be zero
    diffs = combined_results.filter(pl.col("Scenario") == scenario_names[0]).select(
        "Total"
    ) - combined_results.filter(pl.col("Scenario") == scenario_names[1]).select("Total")

    assert (
        abs(diffs["Total"].mean()) < 0.1
    ), f"Expected the mean difference to be close to 0, but got {diffs['Total'].mean()}"


def test_relative_difference():
    # Create a sample DataFrame
    scenarios = ["A", "B"]
    a_base_vals = np.arange(100, 191, 10)
    b_compare_vals = np.arange(105, 196, 10)

    data = pl.DataFrame(
        {
            "Scenario": ["A"] * 10 + ["B"] * 10,
            "Total": a_base_vals.tolist() + b_compare_vals.tolist(),
        }
    )

    rel_diffs = []
    for base in a_base_vals:
        for compare in b_compare_vals:
            diff = 100 * (base - compare) / np.mean(a_base_vals)
            rel_diffs.append(diff)

    expected_reldiff = np.mean(rel_diffs)
    expected_reldiff_low = np.quantile(rel_diffs, 0.025)
    expected_reldiff_high = np.quantile(rel_diffs, 0.975)
    expected_reldiff = [expected_reldiff_low, expected_reldiff, expected_reldiff_high]

    # Test relative_difference for Total column
    total_reldiff = relative_difference(data, col_name="Total", group_values=scenarios)
    assert (
        len(total_reldiff) == 3
    ), f"Expected total_reldiff to have 3 elements, but got {len(total_reldiff)}"
    print(total_reldiff)
    for i in range(3):
        assert (
            total_reldiff[i] == expected_reldiff[i]
        ), f"Expected {i}th element of relative difference to be {expected_reldiff[i]}, but got {total_reldiff[i]}"


def test_relative_difference_against_self():
    # Create a sample DataFrame
    scenarios = ["A", "B"]
    a_base_vals = np.arange(100, 191, 10)

    data = pl.DataFrame(
        {
            "Scenario": ["A"] * 10 + ["B"] * 10,
            "Total": a_base_vals.tolist() * 2,
        }
    )

    # Test relative_difference for Total column
    total_reldiff = relative_difference(data, col_name="Total", group_values=scenarios)
    # lwr should be approximately equal to upr and mean should be approximately zero
    assert total_reldiff[1] == pytest.approx(0.0, rel=1e-6)
    assert -total_reldiff[0] == pytest.approx(total_reldiff[2], rel=1e-6)


def test_relative_difference_assertion():
    with pytest.raises(AssertionError):
        relative_difference(
            pl.DataFrame({"Scenario": ["A", "B", "C"], "Total": [1, 2, 3]}),
            col_name="Total",
            group_values=["A", "B", "C"],
        )


def test_relative_difference_identifier():
    # Create a sample DataFrame
    scenarios = ["A", "B"]
    a_base_vals = [100] * 10
    b_compare_vals = [105] * 10

    data = pl.DataFrame(
        {
            "Scenario": ["A"] * 10 + ["B"] * 10,
            "Total": a_base_vals + b_compare_vals,
            "replicate": list(range(10)) * 2,
        }
    )
    expected_reldiff = [-5, -5, -5]

    # Test relative_difference for Total column
    total_reldiff = relative_difference(
        data, col_name="Total", group_values=scenarios, identifier="replicate"
    )
    assert (
        len(total_reldiff) == 3
    ), f"Expected total_reldiff to have 3 elements, but got {len(total_reldiff)}"

    for i in range(3):
        assert (
            total_reldiff[i] == expected_reldiff[i]
        ), f"Expected {i}th element of relative difference to be {expected_reldiff[i]}, but got {total_reldiff[i]}"


def test_relative_difference_identifier_unequal_length():
    """Should trim the values that don't have identifier matches in A for B compare scenarios"""
    # Create a sample DataFrame
    scenarios = ["A", "B"]
    a_base_vals = [100] * 10
    # Additional five values that should alter calculation of reldiff unless properly removed by the helper
    b_compare_vals = [105] * 10 + [10] * 5

    data = pl.DataFrame(
        {
            "Scenario": ["A"] * 10 + ["B"] * 15,
            "Total": a_base_vals + b_compare_vals,
            "replicate": np.arange(0, 10).tolist() + np.arange(0, 15).tolist(),
        }
    )
    expected_reldiff = [-5, -5, -5]

    # Test relative_difference for Total column
    total_reldiff = relative_difference(
        data, col_name="Total", group_values=scenarios, identifier="replicate"
    )
    assert (
        len(total_reldiff) == 3
    ), f"Expected total_reldiff to have 3 elements, but got {len(total_reldiff)}"

    for i in range(3):
        assert (
            total_reldiff[i] == expected_reldiff[i]
        ), f"Expected {i}th element of relative difference to be {expected_reldiff[i]}, but got {total_reldiff[i]}"


def test_relative_difference_identifier_no_matches():
    with pytest.raises(
        ValueError,
        match="No matching comparisons based on column replicate. Check input data frame or try running with all pairwise comparisons instead.",
    ):
        # Create a sample DataFrame
        scenarios = ["A", "B"]
        a_base_vals = [100] * 10
        b_compare_vals = [105] * 10

        data = pl.DataFrame(
            {
                "Scenario": ["A"] * 10 + ["B"] * 10,
                "Total": a_base_vals + b_compare_vals,
                "replicate": np.arange(0, 10).tolist() + np.arange(10, 20).tolist(),
            }
        )

        # Test relative_difference for Total column
        relative_difference(
            data, col_name="Total", group_values=scenarios, identifier="replicate"
        )
