from metapop.helper import *
import griddler
import griddler.griddle

if __name__ == "__main__":
    parameter_sets = griddler.griddle.read("scripts_one_pop/one_pop_config.yaml")
    for parms in parameter_sets[0:1]:
        parms['k_i'] = 10
        parms['gamma'] = 1.0 / parms["infectious_duration"]
        r0_base = get_r0_one_group(parms['k_i'], parms['gamma'])
        beta_factor = calculate_beta_factor(parms["desired_r0"], r0_base)
        beta_scaled = rescale_beta_matrix(parms['k_i'], beta_factor)
        print(r0_base)
        print(beta_scaled)
