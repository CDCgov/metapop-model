import numpy as np

from metapop.helper import build_vax_schedule
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
        [99, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # S V SV E1 E2 E1_V E2_V I1 I2 R Y X
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0],  # group has no susceptibles
    ]

    # Set time
    t = 1

    # set susceptibles
    current_susceptibles = [99, 0]
    current_vax_fails = [0, 0]

    # Create an instance of SEIRModel
    model = SEIRModel(parms)

    # Call the exposed method
    new_exposed, new_exposed_vax, old_exposed, old_exposed_vax = model.exposed(
        u, current_susceptibles, current_vax_fails, t
    )
    assert new_exposed[1] == 0  # No new exposures in this group bc no susceptibles
    assert len(new_exposed) == parms["n_groups"]
    assert len(old_exposed) == parms["n_groups"]


def test_aon_vaccine():
    """Test all or nothing vaccine"""
    parms = dict(
        vaccine_efficacy_1_dose=1.0,
        vaccine_efficacy_2_dose=1.0,
        total_vaccine_uptake_doses=50,
        vaccine_uptake_start_day=1,
        vaccine_uptake_duration_days=1,
        vaccinated_group=0,
        k_i=10,
        n_groups=1,
        symptomatic_isolation_start_day=400,
        symptomatic_isolation_duration_days=100,
        pre_rash_isolation_start_day=400,
        pre_rash_isolation_duration_days=100,
        beta=np.array([[20]]),
        pop_sizes=[1000],
        sigma_scaled=1 / 10.0,
    )
    # Build vax schedule
    parms["vaccination_uptake_schedule"] = build_vax_schedule(parms)

    # Set time
    t = 1

    # Initial state for each group
    u = [[690, 300, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0]]
    initial_S = u[0][Ind.S.value]
    initial_SV = u[0][Ind.SV.value]

    ### test: 100% effective vaccine
    model = SEIRModel(parms)

    # vaccinate working properly
    new_v, new_sv = model.vaccinate(u, t)
    assert new_sv == [0]
    assert new_v == [50]

    # get_update_susceptibles working properly
    current_susceptibles, current_failures = model.get_updated_susceptibles(
        u, new_v, new_sv
    )
    assert current_failures == [0]
    assert current_susceptibles == [690 - 50]
    assert sum(current_susceptibles) < initial_S
    assert sum(current_failures) == initial_SV

    # exposed working properly, only moving people from S to E1
    new_e1, new_e1_V, new_e2, new_e2_V = model.exposed(
        u, current_susceptibles, current_failures, t
    )
    assert sum(new_e1_V) == 0

    ### test: 0% effective vaccine
    parms["vaccine_efficacy_1_dose"] = 0.0
    # u = [[690, 500, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0]] # increase SV to 500
    model = SEIRModel(parms)

    # vaccinate working properly
    new_v, new_sv = model.vaccinate(u, t)
    assert new_sv == [50]
    assert new_v == [0]

    # get_update_susceptibles working properly
    current_susceptibles, current_failures = model.get_updated_susceptibles(
        u, new_v, new_sv
    )
    assert current_failures == [50]
    assert current_susceptibles == [690 - 50]

    # exposed working properly, both e1 and e1_v growing
    new_e1, new_e1_V, new_e2, new_e2_V = model.exposed(
        u, current_susceptibles, current_failures, t
    )
    assert sum(new_e1[0]) > 0
    # not guaranteed to have infections in SV -- maybe make it run abunch of times and check
    assert sum(new_e1_V[0]) > 0
