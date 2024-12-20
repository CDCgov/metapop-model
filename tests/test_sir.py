import numpy as np
from sir import SEIRModel
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose
import pytest

def test_exposed():
    # Define the parameters
    parms = {
        "beta": np.array([[1, 2], [3, 4]]),
        "N": [100, 100],
        "dt": 1.0,
        "n_groups": 2
    }

    # Initial state for each group
    u = [
        [99, 0, 1, 0, 0],  # Group 0: 99 susceptible, 0 exposed, 1 infected, 0 recovered, 0 cumulative infections
        [100, 0, 0, 0, 0]  # Group 1: 100 susceptible, 0 exposed, 0 infected, 0 recovered, 0 cumulative infections
    ]

    # Create an instance of SEIRModel
    model = SEIRModel(parms)

    # Call the exposed method
    new_exposed = model.exposed(u)

    # Check the results
    expected_exposed = [
        np.random.binomial(99, 1.0 - np.exp(-0.01)),
        np.random.binomial(100, 1.0 - np.exp(-0.03))
    ]

    assert len(new_exposed) == 2
    assert_allclose(new_exposed, expected_exposed)

def test_exposed_group1_zero_population():
    # Define the parameters
    parms = {
        "beta": np.array([[1, 2], [3, 4]]),
        "N": [100, 0],
        "dt": 1,
        "n_groups": 2
    }

    # Initial state for each group
    u = [
        [99, 0, 1, 0, 0],  # Group 0: 99 susceptible, 0 exposed, 1 infected, 0 recovered, 0 cumulative infections
        [0, 0, 0, 0, 0]    # Group 1: 0 susceptible, 0 exposed, 0 infected, 0 recovered, 0 cumulative infections
    ]

    # Create an instance of SEIRModel
    model = SEIRModel(parms)

    # Call the exposed method
    new_exposed = model.exposed(u)

    # Check the results
    expected_exposed = [
        np.random.binomial(99, 1.0 - np.exp(-0.01)),
        0  # No new exposures in group 1 because the population size is 0
    ]

    assert len(new_exposed) == 2
    assert_allclose(new_exposed, expected_exposed)
