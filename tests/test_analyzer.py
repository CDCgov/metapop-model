import os
import warnings

import numpy as np
import polars as pl
import pytest

from metapop.analyzer import (
    add_daily_incidence_scenario,
    add_week_column,
    create_filename,
    create_intervention_summary_table,
    get_relative_difference,
    trim_string_column,
)

testdir = os.path.dirname(__file__)


###### Utility function for testing add_daily_incidence_scenario #####
@pytest.fixture
def sample_simulation_data():
    """Create sample simulation data for testing."""
    return pl.DataFrame(
        {
            "replicate": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "intervention_scenario": [
                "none",
                "none",
                "none",
                "isolation",
                "isolation",
                "isolation",
                "none",
                "none",
                "none",
                "isolation",
                "isolation",
                "isolation",
            ],
            "group": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "t": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "Y": [5, 12, 20, 3, 8, 15, 7, 15, 25, 4, 10, 18],
        }
    )


##### Utility function for testing create_intervention_summary_table #####
@pytest.fixture
def sample_intervention_data():
    """Create sample intervention data for testing."""
    # Create data with multiple replicates, scenarios, and immunity levels
    data = []
    scenarios = ["none", "isolation", "quarantine"]
    immunity_levels = [0.8, 0.9]
    replicates = [1, 2, 3]

    for replicate in replicates:
        for scenario in scenarios:
            for immunity in immunity_levels:
                # Simulate different outbreak sizes based on scenario
                base_size = (
                    100 if scenario == "none" else 80 if scenario == "isolation" else 60
                )
                # Add some random variation
                final_y = base_size + replicate

                data.append(
                    {
                        "replicate": replicate,
                        "intervention_scenario": scenario,
                        "initial_vaccine_coverage": immunity,
                        "group": 0,
                        "t": 365,  # Final day
                        "Y": final_y,
                    }
                )

    return pl.DataFrame(data)


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
    total_reldiff = get_relative_difference(
        data, col_name="Total", group_values=scenarios
    )
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

    # Test get_relative_difference for Total column
    total_reldiff = get_relative_difference(
        data, col_name="Total", group_values=scenarios
    )
    # lwr should be approximately equal to upr and mean should be approximately zero
    assert total_reldiff[1] == pytest.approx(0.0, rel=1e-6)
    assert -total_reldiff[0] == pytest.approx(total_reldiff[2], rel=1e-6)


def test_relative_difference_assertion():
    with pytest.raises(AssertionError):
        get_relative_difference(
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

    # Test get_relative_difference for Total column
    total_reldiff = get_relative_difference(
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
    total_reldiff = get_relative_difference(
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

        # Test get_relative_difference for Total column
        get_relative_difference(
            data, col_name="Total", group_values=scenarios, identifier="replicate"
        )


def test_add_daily_incidence_scenario_basic(sample_simulation_data):
    """Test basic functionality of add_daily_incidence_scenario."""
    result = add_daily_incidence_scenario(sample_simulation_data, groups=[0])

    # Check that Incidence column is added
    assert "Incidence" in result.columns

    # Check that number of rows is preserved
    assert len(result) == len(sample_simulation_data)

    # Check that original columns are preserved
    original_cols = set(sample_simulation_data.columns)
    assert original_cols.issubset(set(result.columns))


def test_add_daily_incidence_scenario_calculation(sample_simulation_data):
    """Test that incidence calculations are correct."""
    result = add_daily_incidence_scenario(sample_simulation_data, groups=[0])

    # Sort to match expected order
    result = result.sort(["replicate", "intervention_scenario", "t"])

    # For replicate 1, scenario "isolation", group 0: Y = [3, 8, 15]
    # Expected incidence: [3, 5, 7]
    isolation_r1 = result.filter(
        (pl.col("replicate") == 1) & (pl.col("intervention_scenario") == "isolation")
    ).sort("t")

    expected_incidence = [3, 5, 7]  # 3 (first), 8-3=5, 15-8=7
    assert isolation_r1["Incidence"].to_list() == expected_incidence


def test_add_daily_incidence_scenario_unsorted_data():
    """Test that function handles unsorted time data correctly."""
    df = pl.DataFrame(
        {
            "replicate": [1, 1, 1],
            "intervention_scenario": ["none", "none", "none"],
            "group": [0, 0, 0],
            "t": [3, 1, 2],  # Unsorted times
            "Y": [20, 5, 12],  # Corresponding Y values
        }
    )

    result = add_daily_incidence_scenario(df, groups=[0])

    # Should sort by t and calculate correctly
    # After sorting: t=[1,2,3], Y=[5,12,20]
    # Expected incidence: [5, 7, 8] (5, 12-5=7, 20-12=8)
    sorted_result = result.sort("t")
    assert sorted_result["Incidence"].to_list() == [5, 7, 8]  # Fixed: first value is Y


def test_add_daily_incidence_scenario_decreasing_y():
    """Test with decreasing Y values (edge case)."""
    df = pl.DataFrame(
        {
            "replicate": [1, 1, 1],
            "intervention_scenario": ["none", "none", "none"],
            "group": [0, 0, 0],
            "t": [1, 2, 3],
            "Y": [10, 8, 5],  # Decreasing cumulative values
        }
    )

    result = add_daily_incidence_scenario(df, groups=[0])

    # Expected incidence: [10, -2, -3] (10, 8-10=-2, 5-8=-3)
    sorted_result = result.sort("t")
    assert sorted_result["Incidence"].to_list() == [10, -2, -3]


def test_create_intervention_summary_table_basic(sample_intervention_data):
    """Test basic functionality of create_intervention_summary_table."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = create_intervention_summary_table(
            sample_intervention_data,
            rng=np.random.default_rng(123),
            IHR=0.15,
            scenario_order=["isolation", "quarantine"],
        )

    # Check that result is a DataFrame
    assert isinstance(result, pl.DataFrame)

    # Check expected columns exist
    expected_columns = [
        "scenario",
        "baseline_immunity",
        "infections_no_interventions",
        "infections_interventions",
        "infections_relative_difference_pct",
        "hospitalizations_no_interventions",
        "hospitalizations_interventions",
        "hospitalizations_relative_difference_pct",
    ]
    assert all(col in result.columns for col in expected_columns)

    # Should have 2 scenarios × 2 immunity levels = 4 rows (not 8 anymore)
    assert len(result) == 4

    # Should include isolation and quarantine, but not "none"
    scenarios = result["scenario"].unique().to_list()
    assert "isolation" in scenarios
    assert "quarantine" in scenarios
    assert "none" not in scenarios
    assert len(scenarios) == 2

    immunity_levels = result["baseline_immunity"].unique().sort().to_list()
    assert immunity_levels == [0.8, 0.9]


##### Test functions for create_filename ########
def test_create_filename_base_only():
    """Test filename creation with base name only."""
    result = create_filename("results")
    assert result == "results.csv"


def test_create_filename_all_parameters():
    """Test filename creation with all parameters."""
    result = create_filename("results", date="2025_08_07", suffix="final", fmt=".json")
    assert result == "results_2025_08_07_final.json"


def test_create_filename_empty_date_and_suffix():
    """Test filename creation with empty date and suffix."""
    result = create_filename("results", date="", suffix="")
    assert result == "results.csv"


##### Test functions for trim_string_column ####
def test_trim_string_column_default():
    """Test trimming with default char_length (should be 0, so no change)."""
    sample_df = pl.DataFrame(
        {
            "intervention_scenario": [
                "none80",
                "isolation_only80",
                "isolation_and_quarantine80",
            ],
            "other_column": [1, 2, 3],
        }
    )
    result = trim_string_column(sample_df, "intervention_scenario")
    expected = sample_df  # No change since default char_length is 0
    assert result.equals(expected)


def test_trim_string_column_remove_2_chars():
    """Test removing 2 characters from end."""
    sample_df = pl.DataFrame(
        {
            "intervention_scenario": [
                "none80",
                "isolation_only80",
                "isolation_and_quarantine80",
            ],
            "other_column": [1, 2, 3],
        }
    )
    result = trim_string_column(sample_df, "intervention_scenario", char_length=2)
    expected_values = ["none", "isolation_only", "isolation_and_quarantine"]
    assert result["intervention_scenario"].to_list() == expected_values
    assert result["other_column"].to_list() == [1, 2, 3]  # Other columns unchanged


####### Test functions for add_week_column ######
def test_add_week_column():
    """Test adding week column with basic day values."""
    df = pl.DataFrame({"t": [0, 1, 7, 8, 14, 15, 21, 22]})
    result = add_week_column(df)

    expected_weeks = [0, 1, 1, 2, 2, 3, 3, 4]
    assert result["week"].to_list() == expected_weeks
    assert result["t"].to_list() == [
        0,
        1,
        7,
        8,
        14,
        15,
        21,
        22,
    ]  # Original column preserved


def test_add_week_column_preserves_other_columns():
    """Test that add_week_column preserves other columns in the DataFrame."""
    df = pl.DataFrame(
        {
            "t": [1, 8, 15],
            "replicate": [1, 1, 1],
            "Y": [10, 20, 30],
            "intervention_scenario": ["none", "isolation", "quarantine"],
        }
    )
    result = add_week_column(df)

    # Check week column is added correctly
    assert result["week"].to_list() == [1, 2, 3]

    # Check other columns are preserved
    assert result["replicate"].to_list() == [1, 1, 1]
    assert result["Y"].to_list() == [10, 20, 30]
    assert result["intervention_scenario"].to_list() == [
        "none",
        "isolation",
        "quarantine",
    ]
    assert result["t"].to_list() == [1, 8, 15]

    # Check total number of columns
    assert len(result.columns) == 5  # Original 4 + 1 new week column
