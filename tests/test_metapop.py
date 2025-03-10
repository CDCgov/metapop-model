from metapop import SEIRModel  # Ensure this import path is correct
import numpy as np
import yaml

def test_only_expose_susceptible():
    # Define the parameters
    parms = {
        "beta": np.array([[2, 2], [2, 2]]),
        "sigma1": 0.25,
        "N": [100, 100],
        "dt": 1,
        "n_groups": 2
    }

    # Initial state for each group
    u = [
        [99, 0, 0, 0, 1, 0, 0, 0],     # SV E1 E2 I1 I2 R Y
        [ 0, 0, 0, 0, 0, 0, 100, 0]    # group has no susceptibles
    ]

    # Create an instance of SEIRModel
    model = SEIRModel(parms)

    # Call the exposed method
    new_exposed = model.exposed(u)
    print(new_exposed)
    assert new_exposed[1] == [0, 0] # No new exposures in this group bc no susceptibles
    assert len(new_exposed) == parms['n_groups']

def check_config():
    with open("scripts/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    parms = config["baseline_parameters"]

    n_groups = parms["n_groups"]

    # Check that N and I0 are of length equal to n_groups
    assert len(parms["N"]) == n_groups, f"N should be of length {n_groups}"
    assert len(parms["I0"]) == n_groups, f"I0 should be of length {n_groups}"

    # Check that beta is a square n_groups x n_groups array
    beta = np.array(parms["beta"])
    assert beta.shape == (n_groups, n_groups), f"beta should be a {n_groups}x{n_groups} array"
