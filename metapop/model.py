# This file is part of the metapop package. It contains the SEIRModel class
# implementation
import numpy as np
from enum import Enum
# import what's needed from other metapop modules
from .helper import (
    get_infected,
    calculate_foi,
    rate_to_frac,
    vaccinate_groups,
)
# if you want to use methods from metapop in this file under
# if __name__ == "__main__": you'll need to import them as:
# from metapop.helper import (
#     get_infected,
#     calculate_foi,
#     rate_to_frac,
#     vaccinate_groups,
# )
### note: this is not recommended use within a file that is imported as a package module, but it can be useful for testing purposes

__all__ = ["Ind", "SEIRModel"]


class Ind(Enum):
    S = 0
    V = 1
    E1 = 2
    E2 = 3
    I1 = 4
    I2 = 5
    R = 6
    Y = 7
    X = 8

class SEIRModel:
    def __init__(self, parms):
        self.parms = parms

        # convert some lists to arrays
        self.parms['k_i'] = np.array(parms['k_i'])

        # define internal model variables
        self.groups = parms["n_groups"]

    def exposed(self, u, current_susceptibles, t):
        new_exposed = []
        old_exposed = []

        # Extract the number of infected individuals for each group
        I_g = get_infected(u, [Ind.I1.value, Ind.I2.value], self.groups, self.parms, t)

        for target_group in range(self.groups):
            S = current_susceptibles[target_group]

            # Get new Infections, for beta: rows are to, columns are from
            foi =  calculate_foi(self.parms["beta"], I_g, self.parms["pop_sizes"], target_group)
            new_e_frac = rate_to_frac(foi)
            new_exposed.append(np.random.binomial(S, new_e_frac))  # Ensure S is non-negative

            # Get within E chain movement (E1 -> E2)
            e1_to_e2_frac = rate_to_frac(self.parms["sigma_scaled"])
            old_exposed.append(np.random.binomial(u[target_group][Ind.E1.value], e1_to_e2_frac))

        return [new_exposed, old_exposed]

    def vaccinate(self, u, t):
        new_vaccinated = []

        if self.parms["total_vaccine_uptake_doses"] > 0:
            vaccination_uptake_schedule = self.parms["vaccination_uptake_schedule"]
            new_vaccinated = vaccinate_groups(self.groups, u, t, vaccination_uptake_schedule, self.parms)
        else:
            for group in range(self.groups):
                new_vaccinated.append(0)

        return new_vaccinated

    def infectious(self, u):
        new_infectious = []
        old_infectious = []
        for group in range(self.groups):
            new_i_frac = rate_to_frac(self.parms["sigma_scaled"])
            new_infectious.append(np.random.binomial(u[group][Ind.E2.value], new_i_frac))
            i1_to_i2_frac = rate_to_frac(self.parms["gamma_scaled"])
            old_infectious.append(np.random.binomial(u[group][Ind.I1.value], i1_to_i2_frac))
        return [new_infectious, old_infectious]

    def recovery(self, u, t):
        new_recoveries = []
        for group in range(self.groups):
            new_r_frac = rate_to_frac(self.parms["gamma_scaled"])
            new_recoveries.append(np.random.binomial(u[group][Ind.I2.value], new_r_frac))
        return new_recoveries

    def get_updated_susceptibles(self, u, new_vaccinated):
        """
        Get the number of susceptibles in each target group based on the new vaccinated individuals.
        Args:
            u (list): The state vector of the system.
            new_vaccinated (list): The number of new vaccinated individuals for each group on this day.
        Returns:
            list: The updated number of susceptibles for each target group to pass on.
        """
        updated_susceptibles = []
        for target_group in range(len(u)):
            S = u[target_group][Ind.S.value] - new_vaccinated[target_group]
            updated_susceptibles.append(S)
        return updated_susceptibles


    def seirmodel(self, u, t):
        new_u = []
        new_vaccinated = self.vaccinate(u, t)
        current_susceptibles = self.get_updated_susceptibles(u, new_vaccinated)
        new_exposed, old_exposed = self.exposed(u, current_susceptibles, t)
        new_infectious, old_infectious = self.infectious(u)
        new_recoveries = self.recovery(u, t)
        for group in range(self.groups):
            S, V, E1, E2, I1, I2, R, Y, X = u[group]
            new_S = S - new_exposed[group] - new_vaccinated[group]
            new_V = V + new_vaccinated[group]
            new_E1 = E1 + new_exposed[group] - old_exposed[group]
            new_E2 = E2 + old_exposed[group] - new_infectious[group]
            new_I1 = I1 + new_infectious[group] - old_infectious[group]
            new_I2 = I2 + old_infectious[group] - new_recoveries[group]
            new_R = R + new_recoveries[group]
            new_Y = Y + new_infectious[group]
            new_X = X + new_vaccinated[group]
            new_u.append([new_S, new_V, new_E1, new_E2, new_I1, new_I2, new_R, new_Y, new_X])

        return new_u
