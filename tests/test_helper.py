from metapop import SEIRModel
from metapop.helper import set_beta_parameter
from metapop.helper import initialize_population
from metapop.helper import run_model
import yaml
import numpy as np
from numpy.testing import assert_allclose

def test_set_beta_parameter():
    # Set a seed for reproducibility
    np.random.seed(42)

    # Define the parameters
    parms = {
        "beta_2_low": 0.1,
        "beta_2_high": 0.5,
        "beta": [[0, 0], [0, 0]]
    }

    beta_original_shape = np.shape(parms["beta"])

    # Call the function
    parms = set_beta_parameter(parms)

    # Check to make sure same as old dimensions
    assert beta_original_shape == np.shape(parms["beta"])

    # Check if the beta_2_value is within the expected range
    assert 0.1 <= parms["beta"][1][1] <= 0.5

def test_pop_initialization():
    parms = {
        "N": [1000, 2000],
        "initial_vaccine_coverage": [0.0, 0.99],
        "I0": [10, 0]
    }
    steps = 3
    groups = 2

    S, V, E1, E2, I1, I2, R, Y, u = initialize_population(steps, groups, parms)

    # correct dimensions
    assert len(S) == steps
    assert len(S[0]) == groups

    # initial vaccination
    assert V[0][0] == 0
    assert_allclose(V[0][1], parms["N"][1] * parms["initial_vaccine_coverage"][1], rtol=1e-2, atol=1e-8)

    # initial infections
    assert I1[0][0] == parms["I0"][0]
    assert I1[0][1] == parms["I0"][1]

    # initial state vector is correct
    assert len(u) == groups
    assert len(u[0]) == 8 # S V E1 E2 I1 I2 R Y
    assert u[0][0] == S[0][0]

def test_run_model():
    # Define the parameters
    with open("scripts/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    parms = config["baseline_parameters"]
    parms["initial_vaccine_coverage"] = [0.9, 0.5] # add here, griddler handles in nested params
    parms["vaccine_uptake"] = False # setting here in case default config changes

    # Define the time array and steps
    groups = parms["n_groups"]
    steps = parms["tf"]
    t = np.linspace(1, steps, steps)

    S, V, E1, E2, I1, I2, R, Y, u = initialize_population(steps, groups, parms)

    # Create an instance of SEIRModel
    model = SEIRModel(parms)

    # Call the run_model function
    S, V, E1, E2, I1, I2, R, Y, u = run_model(model, u, t, steps, groups, S, V, E1, E2, I1, I2, R, Y)

    # Check the results
    assert S.shape ==  (steps, groups)
    assert V.shape ==  (steps, groups)
    assert E1.shape == (steps, groups)
    assert E2.shape == (steps, groups)
    assert I1.shape == (steps, groups)
    assert I2.shape == (steps, groups)
    assert R.shape ==  (steps, groups)
    assert Y.shape ==  (steps, groups)

    # Check initial vax = end vax
    if parms["vaccine_uptake"] is False:
        assert np.array_equal(V[0], V[-1]), "The starting V[0] should be the same as the ending V[0] when uptake is zero"
