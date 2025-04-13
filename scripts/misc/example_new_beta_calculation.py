# This script shows examples of how to calculate the beta matrix for a single
# population to achieve a desired R0, and how to calculate the beta matrix for
# a multi-population model to achieve a desired R0 for the whole population.

from metapop.helper import *
import numpy as np
import griddler
import griddler.griddle

if __name__ == "__main__":

    # one population example
    print("one population example")
    parameter_sets = griddler.griddle.read("scripts/onepop/onepop_config.yaml")
    parms = parameter_sets[0]
    parms['k_i'] = np.array(parms['k_i']).reshape((1, ))
    parms['gamma'] = 1. / parms["infectious_duration"]
    r0_base = get_r0_one_group(parms['k_i'], parms['gamma'])
    beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
    beta_scaled = rescale_beta_matrix(parms['k_i'][0], beta_factor)
    r0 = get_r0_one_group([beta_scaled], parms['gamma'])
    print("r0 base", r0_base)
    print("beta scaled", beta_scaled)
    print("r0 scaled", r0)
    assert np.allclose(r0_base * beta_factor, parms["desired_r0"]), f"r0 {r0_base * beta_factor} not equal to desired r0 {parms['desired_r0']}"

    # multipop example
    print("\nmetapop example")
    parameter_sets = griddler.griddle.read("scripts/connectivity/config.yaml")
    parms = parameter_sets[0]
    print("k_i", parms["k_i"])
    parms['gamma'] = 1. / parms["infectious_duration"]
    beta_unscaled = get_percapita_contact_matrix(parms)
    r0_base = get_r0(beta_unscaled, parms['gamma'], parms['pop_sizes'])
    print("r0 base", r0_base)
    beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
    beta_scaled = rescale_beta_matrix(beta_unscaled, beta_factor)
    r0 = get_r0(beta_scaled, parms['gamma'], parms['pop_sizes'])
    print("r0 scaled", r0)
    assert np.allclose(r0, parms["desired_r0"]), f"r0 {r0} not equal to desired r0 {parms['desired_r0']}"
