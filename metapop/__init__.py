import numpy as np
from metapop.helper import* # noqa: F405

class SEIRModel:
    def __init__(self, parms):
        self.parms = parms

        # convert some lists to arrays
        self.parms['k_i'] = np.array(parms['k_i'])

        # define internal model variables
        self.groups = parms["n_groups"]
        self.S = 0
        self.V = 1
        self.E1 = 2
        self.E2 = 3
        self.I1 = 4
        self.I2 = 5
        self.R = 6
        self.Y = 7
        self.X = 8

    def exposed(self, u, current_susceptibles):
        new_exposed = []
        old_exposed = []

        # Extract the number of infected individuals for each group
        I_g = get_infected(u, [self.I1, self.I2], self.groups)

        for target_group in range(self.groups):
            S = current_susceptibles[target_group]

            # Get new Infections, for beta: rows are to, columns are from
            foi =  calculate_foi(self.parms["beta"], I_g, self.parms["pop_sizes"], target_group)
            new_e_frac = rate_to_frac(foi)
            new_exposed.append(np.random.binomial(S, new_e_frac))  # Ensure S is non-negative

            # Get within E chain movement (E1 -> E2)
            e1_to_e2_frac = rate_to_frac(self.parms["sigma"])
            old_exposed.append(np.random.binomial(u[target_group][self.E1], e1_to_e2_frac))

        return [new_exposed, old_exposed]

    def vaccinate(self, u, t):
        new_vaccinated = []

        if self.parms["vaccine_uptake"]:
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
            new_i_frac = rate_to_frac(self.parms["sigma"])
            new_infectious.append(np.random.binomial(u[group][self.E2], new_i_frac))
            i1_to_i2_frac = rate_to_frac(self.parms["gamma"])
            old_infectious.append(np.random.binomial(u[group][self.I1], i1_to_i2_frac))
        return [new_infectious, old_infectious]

    def recovery(self, u):
        new_recoveries = []
        for group in range(self.groups):
            new_r_frac = rate_to_frac(self.parms["gamma"])
            new_recoveries.append(np.random.binomial(u[group][self.I2], new_r_frac))
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
            S = u[target_group][self.S] - new_vaccinated[target_group]
            updated_susceptibles.append(S)
        return updated_susceptibles


    def seirmodel(self, u, t):
        new_u = []
        new_vaccinated = self.vaccinate(u, t)
        current_susceptibles = self.get_updated_susceptibles(u, new_vaccinated)
        new_exposed, old_exposed = self.exposed(u, current_susceptibles)
        new_infectious, old_infectious = self.infectious(u)
        new_recoveries = self.recovery(u)
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
