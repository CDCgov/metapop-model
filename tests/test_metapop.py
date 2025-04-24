import numpy as np
from metapop.model import *


def test_only_expose_susceptible():
    # Define the parameters
    parms = {
        "beta": np.array([[2, 2], [2, 2]]),
        "k_i": [10, 10, 10],
        "sigma": 0.75,
        "sigma_scaled": 0.75 * 2,
        "n_e_compartments": 2,
        "n_i_compartments": 2,
        "pop_sizes": [100, 100],
        "n_groups": 2,
        "symptomatic_isolation_start_day": 400,
        "symptomatic_isolation_duration_days": 100,
        "pre_rash_isolation_start_day": 400,
        "pre_rash_isolation_duration_days": 100,
        "isolation_success": 0.0,
        "pre_rash_isolation_success": 0.0,
    }

    # Initial state for each group
    u = [
        [99, 0, 0, 0, 1, 0, 0, 0, 0],  # S V E1 E2 I1 I2 R Y X
        [0, 0, 0, 0, 0, 0, 100, 0, 0],  # group has no susceptibles
    ]

    # Set time
    t = 1

    # set susceptibles
    current_susceptibles = [99, 0]

    # Create an instance of SEIRModel
    model = SEIRModel(parms)

    # Call the exposed method
    new_exposed, old_exposed = model.exposed(u, current_susceptibles, t)
    assert new_exposed[1] == 0  # No new exposures in this group bc no susceptibles
    assert len(new_exposed) == parms["n_groups"]
    assert len(old_exposed) == parms["n_groups"]
