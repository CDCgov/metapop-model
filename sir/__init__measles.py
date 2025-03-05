import numpy as np
import math

class SEIRModel:
    def __init__(self, parms):
        self.parms = parms
        self.groups = parms["n_groups"]

    def exposed(self, u):
        new_exposed = []
        old_exposed = []
        # Extract the number of infected individuals for each group,
        # eventually want to split contact and infectiousness, 
        # right now they are specified by beta, just not by I1/2
        I1_g = np.array([u[group][4] for group in range(self.groups)])
        I2_g = np.array([u[group][5] for group in range(self.groups)])
        I_g = I1_g + I2_g
        for target_group in range(self.groups):
            S, V, E1,  E2, I1, I2, R, Y = u[target_group]
            # rows are to, columns are from
            foi = np.dot(self.parms["beta"][target_group] , I_g / np.array(self.parms["N"]))
            e1frac = 1.0 - np.exp(-foi * self.parms["dt"])
            new_exposed.append(np.random.binomial(S, e1frac))
            e2frac = 1.0 - math.exp(-self.parms["sigma1"] * self.parms["dt"])
            old_exposed.append(np.random.binomial(E1, e2frac))
        return [new_exposed, old_exposed]
    
    #def vaccinated(self, u):
        #new_vaccinated = []
        #if intervention vaccinate at rate
    #    return new_vaccinated

    def infectious(self, u):
        new_infectious = []
        old_infectious = []
        for group in range(self.groups):
            S, V, E1, E2, I1, I2, R, Y = u[group]
            i1frac = 1.0 - math.exp(-self.parms["sigma2"] * self.parms["dt"])
            new_infectious.append(np.random.binomial(E2, i1frac))
            i2frac = 1.0 - math.exp(-self.parms["gamma1"] * self.parms["dt"])
            old_infectious.append(np.random.binomial(I1, i2frac))
        return [new_infectious, old_infectious]

    def recovery(self, u):
        new_recoveries = []
        for group in range(self.groups):
            S, V,  E1, E2, I1, I2, R, Y = u[group]
            rfrac = 1.0 - math.exp(-self.parms["gamma2"] * self.parms["dt"])
            new_recoveries.append(np.random.binomial(I2, rfrac))
        return new_recoveries

    def seirmodel(self, u, t): # add v group, possibly vaccinated group with vaccine failure going back eventually, might need more complexity for ongoing vaccine campaign
        new_u = []
        #new_vaccinated = self.vaccinated(u)
        new_exposed = self.exposed(u)[0]
        old_exposed = self.exposed(u)[1] #
        new_infectious = self.infectious(u)[0]# 
        old_infectious = self.infectious(u)[1]
        new_recoveries = self.recovery(u)#
        for group in range(self.groups):
            S, V, E1, E2, I1, I2, R, Y = u[group]
            new_S = S - new_exposed[group] # - new_vaccinated[group]
            new_V = 0 #V + new_vaccinated[group]
            new_E1 = E1 + new_exposed[group] - old_exposed[group] # - new_vaccinated[group]? #handle these by percentage?
            new_E2 = E2 + old_exposed[group] - new_infectious[group] # - new_vaccinated[group]?
            new_I1 = I1 + new_infectious[group] - old_infectious[group]
            new_I2 = I2 + old_infectious[group] - new_recoveries[group]
            new_R = R + new_recoveries[group]
            new_Y = Y + new_exposed[group]
            new_u.append([new_S, new_V, new_E1, new_E2, new_I1, new_I2, new_R, new_Y])
        return new_u
