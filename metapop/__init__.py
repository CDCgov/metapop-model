import numpy as np
from metapop.helper import* # noqa: F405

class SEIRModel:
    def __init__(self, parms):
        self.parms = parms

        # convert some lists to arrays
        self.parms['k_i'] = np.array(parms['k_i'])

        # define internal model variables
        self.groups = parms["n_groups"]
        self.E_indices = np.arange(2, # S V are first and E starts at 2
                                   2 + parms["n_e_compartments"])
        self.I_indices = np.arange(max(self.E_indices), # I starts after E indices
                                   max(self.E_indices) + parms["n_i_compartments"])
    def exposed(self, u):
        new_exposed = []
        old_exposed = []

        # Extract the number of infected individuals for each group
        I_g = get_infected(u, self.I_indices, self.groups)

        for target_group in range(self.groups):
            S, V, E1, E2, I1, I2, R, Y = u[target_group]

            # Get new Infections, for beta: rows are to, columns are from
            foi =  calculate_foi(self.parms["beta"], I_g, self.parms["pop_sizes"], target_group)
            new_e_frac = rate_to_frac(foi)
            new_exposed.append(np.random.binomial(S, new_e_frac))

            # Get within E chain movement (E1 -> E2)
            e1_to_e2_frac = rate_to_frac(self.parms["sigma"])
            old_exposed.append(np.random.binomial(E1, e1_to_e2_frac))

        return [new_exposed, old_exposed]

    def vaccinate(self, u):
        new_vaccinated = []

        if (self.parms["vaccine_uptake"]):
            #do something
            for group in range(self.groups):
                new_vaccinated.append(10000) #filler
        else:
            for group in range(self.groups):
                new_vaccinated.append(0)
        return new_vaccinated

    def infectious(self, u):
        new_infectious = []
        old_infectious = []
        for group in range(self.groups):
            S, V, E1, E2, I1, I2, R, Y = u[group]
            new_i_frac = rate_to_frac(self.parms["sigma"])
            new_infectious.append(np.random.binomial(E2, new_i_frac))
            i1_to_i2_frac = rate_to_frac(self.parms["gamma"])
            old_infectious.append(np.random.binomial(I1, i1_to_i2_frac))
        return [new_infectious, old_infectious]

    def recovery(self, u):
        new_recoveries = []
        for group in range(self.groups):
            S, V,  E1, E2, I1, I2, R, Y = u[group]
            new_r_frac = rate_to_frac(self.parms["gamma"])
            new_recoveries.append(np.random.binomial(I2, new_r_frac))
        return new_recoveries

    def seirmodel(self, u, t): # add v group, possibly vaccinated group with vaccine failure going back eventually, might need more complexity for ongoing vaccine campaign
        new_u = []
        new_vaccinated = self.vaccinate(u)
        new_exposed, old_exposed = self.exposed(u)
        new_infectious, old_infectious = self.infectious(u)
        new_recoveries = self.recovery(u)
        for group in range(self.groups):
            S, V, E1, E2, I1, I2, R, Y = u[group]
            new_S = S - new_exposed[group] # - new_vaccinated[group]
            new_V = V + new_vaccinated[group]
            new_E1 = E1 + new_exposed[group] - old_exposed[group] # - new_vaccinated[group]? #handle these by percentage?
            new_E2 = E2 + old_exposed[group] - new_infectious[group] # - new_vaccinated[group]?
            new_I1 = I1 + new_infectious[group] - old_infectious[group]
            new_I2 = I2 + old_infectious[group] - new_recoveries[group]
            new_R = R + new_recoveries[group]
            new_Y = Y + new_exposed[group]
            new_u.append([new_S, new_V, new_E1, new_E2, new_I1, new_I2, new_R, new_Y])
        return new_u
