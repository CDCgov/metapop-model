from unittest.mock import Mock

import polars as pl

from metapop.sim import simulate_replicates


def test_simulate_replicates_seed():
    BASE_SEED = 42
    N_REPLICATES = 10

    sim_mock = Mock()
    sim_mock.return_value = pl.DataFrame()

    param_sets = [
        {"seed": BASE_SEED, "n_replicates": N_REPLICATES, "x": 1},
        {"seed": BASE_SEED, "n_replicates": N_REPLICATES, "x": 2},
    ]

    simulate_replicates(param_sets, simulate_fn=sim_mock)

    # Check that the correct number of replicates were called
    assert sim_mock.call_count == N_REPLICATES * 2

    # The second argument to the simulate function should be a seed
    p1_r1_seed = sim_mock.call_args_list[0][0][1]
    p1_r2_seed = sim_mock.call_args_list[1][0][1]
    p2_r1_seed = sim_mock.call_args_list[N_REPLICATES][0][1]

    # Seeds should have three elements:
    # The base seed, the stable hash of the parameter set, and the replicate index
    assert isinstance(p1_r1_seed, list)
    assert len(p1_r1_seed) == 3

    assert p1_r1_seed[0] == BASE_SEED
    seed1_hash = p1_r1_seed[1]
    assert p1_r1_seed[2] == 0  # First replicate

    assert p1_r2_seed[0] == BASE_SEED
    assert p1_r2_seed[1] == seed1_hash
    assert p1_r2_seed[2] == 1  # Second replicate

    assert p2_r1_seed[0] == BASE_SEED
    assert p2_r1_seed[1] == seed1_hash
    assert p2_r1_seed[2] == 0  # First replicate
