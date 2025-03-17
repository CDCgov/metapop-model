import metapop as mp
import griddler
import griddler.griddle

if __name__ == "__main__":
    parameter_sets = griddler.griddle.read("scripts/config.yaml")
    for parms in parameter_sets[0:2]:
        parms['k_i'] = [10, 20, 15]
        percapita_contacts = mp.get_percapita_contact_matrix(parms)
        print(parms)
        print(percapita_contacts)
