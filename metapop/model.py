# This file is part of the metapop package. It contains the SEIRModel class
# implementation
from enum import Enum

import numpy as np

# import what's needed from other metapop modules
from .helper import (
    calculate_foi,
    get_infected,
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
    SV = 2
    E1 = 3
    E2 = 4
    E1_V = 5
    E2_V = 6
    I1 = 7
    I2 = 8
    R = 9
    Y = 10
    X = 11


class SEIRModel:
    def __init__(self, parms, seed):
        self.parms = parms

        # convert some lists to arrays
        self.parms["k_i"] = np.array(parms["k_i"])

        # define internal model variables
        self.groups = parms["n_groups"]

        self.rng = np.random.default_rng(seed)

    def exposed(self, u, current_susceptibles, current_vacc_fails, t):
        new_E1 = []
        new_E1_V = []
        new_E2 = []
        new_E2_V = []

        # Extract the number of infected individuals for each group
        I_g = get_infected(u, [Ind.I1.value, Ind.I2.value], self.groups, self.parms, t)

        for target_group in range(self.groups):
            S_val = current_susceptibles[target_group]
            VF_val = current_vacc_fails[target_group]

            # Get new Infections, for beta: rows are to, columns are from
            foi = calculate_foi(
                self.parms["beta"], I_g, self.parms["pop_sizes"], target_group
            )
            new_e_frac = rate_to_frac(foi)
            new_E1.append(self.rng.binomial(S_val, new_e_frac))

            new_E1_V.append(self.rng.binomial(VF_val, new_e_frac))

            # Get within E chain movement (E1 -> E2)
            e1_to_e2_frac = rate_to_frac(self.parms["sigma_scaled"])
            new_E2.append(
                self.rng.binomial(u[target_group][Ind.E1.value], e1_to_e2_frac)
            )
            new_E2_V.append(
                self.rng.binomial(u[target_group][Ind.E1_V.value], e1_to_e2_frac)
            )

        return new_E1, new_E1_V, new_E2, new_E2_V

    def vaccinate(self, u, t):
        new_V = []
        new_SV = []
        if self.parms["total_vaccine_uptake_doses"] > 0:
            vaccination_uptake_schedule = self.parms["vaccination_uptake_schedule"]
            new_V, new_SV, new_EV = vaccinate_groups(
                self.groups, u, t, vaccination_uptake_schedule, self.parms
            )
        else:
            for group in range(self.groups):
                new_V.append(0)
                new_SV.append(0)
                new_EV.append(0)

        return new_V, new_SV, new_EV

    def infectious(self, u):
        new_I1 = []
        new_I1_V = []
        new_I2 = []
        for group in range(self.groups):
            new_i_frac = rate_to_frac(self.parms["sigma_scaled"])
            new_I1.append(self.rng.binomial(u[group][Ind.E2.value], new_i_frac))
            new_I1_V.append(self.rng.binomial(u[group][Ind.E2_V.value], new_i_frac))

            i1_to_i2_frac = rate_to_frac(self.parms["gamma_scaled"])
            new_I2.append(self.rng.binomial(u[group][Ind.I1.value], i1_to_i2_frac))
        return new_I1, new_I1_V, new_I2

    def recovery(self, u, t):
        new_R = []
        for group in range(self.groups):
            new_r_frac = rate_to_frac(self.parms["gamma_scaled"])
            new_R.append(self.rng.binomial(u[group][Ind.I2.value], new_r_frac))
        return new_R

    def get_updated_susceptibles(self, u, new_vaccinated, new_failures):
        """
        Get the number of susceptibles in each target group based on the new vaccinated individuals.
        Args:
            u (list): The state vector of the system.
            new_vaccinated (list): The number of new vaccinated individuals for each group on this day.
        Returns:
            list: The updated number of susceptibles for each target group to pass on.
        """
        updated_susceptibles = []
        updated_failures = []
        for target_group in range(self.groups):
            S_val = (
                u[target_group][Ind.S.value]
                - new_vaccinated[target_group]
                - new_failures[target_group]
            )
            updated_susceptibles.append(S_val)

        for target_group in range(self.groups):
            SV_val = u[target_group][Ind.SV.value] + new_failures[target_group]
            updated_failures.append(SV_val)

        return updated_susceptibles, updated_failures

    def seirmodel(self, u, t):
        new_u = []
        s_v, s_sv, e_v = self.vaccinate(u, t)
        current_susceptibles, current_failures = self.get_updated_susceptibles(
            u, s_v, s_sv
        )
        s_e1, sv_e1v, e1_e2, e1v_e2v = self.exposed(
            u, current_susceptibles, current_failures, t
        )
        e2_i1, e2v_i1, i1_i2 = self.infectious(u)
        i2_r = self.recovery(u, t)
        for group in range(self.groups):
            S, V, SV, E1, E2, E1_V, E2_V, I1, I2, R, Y, X = u[group]
            new_S = S - s_e1[group] - s_v[group] - s_sv[group]
            new_V = V + s_v[group]
            new_SV = SV + s_sv[group] - sv_e1v[group]
            new_E1 = E1 + s_e1[group] - e1_e2[group]
            new_E2 = E2 + e1_e2[group] - e2_i1[group]
            new_E1_V = E1_V + sv_e1v[group] - e1v_e2v[group]
            new_E2_V = E2_V + e1v_e2v[group] - e2v_i1[group]
            new_I1 = I1 + e2v_i1[group] + e2_i1[group] - i1_i2[group]
            new_I2 = I2 + i1_i2[group] - i2_r[group]
            new_R = R + i2_r[group]
            new_Y = Y + sv_e1v[group] + s_e1[group]
            new_X = X + s_sv[group] + s_v[group] + e_v[group]
            new_u.append(
                [
                    new_S,
                    new_V,
                    new_SV,
                    new_E1,
                    new_E2,
                    new_E1_V,
                    new_E2_V,
                    new_I1,
                    new_I2,
                    new_R,
                    new_Y,
                    new_X,
                ]
            )

        return new_u
