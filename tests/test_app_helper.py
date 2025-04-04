from metapop.app_helper import *

def test_get_list_keys():
    parms = dict(
        pop_sizes=[100, 200, 300],
        n_groups=3,
    )

    list_keys = get_list_keys(parms)

    assert list_keys == ['pop_sizes'], f"Expected ['pop_sizes'], but got {list_keys}"
