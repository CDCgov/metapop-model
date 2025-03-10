from metapop.helper import set_beta_parameter
import numpy as np

def test_set_beta_parameter():
    # Set a seed for reproducibility
    np.random.seed(42)

    # Define the parameters
    parms = {
        "beta_2_low": 0.1,
        "beta_2_high": 0.5,
        "beta": [[0, 0], [0, 0]]
    }

    # Call the function
    parms = set_beta_parameter(parms)

    # Check if the beta_2_value is within the expected range
    assert 0.1 <= parms["beta"][1][1] <= 0.5


# should write a test for initial vaccination coverage

# should write a test that no one changes in intiail vaccination coverage
