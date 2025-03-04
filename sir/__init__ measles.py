import numpy as np
import math

class SEIRModel:
    def __init__(self, parms):
        self.parms = parms
        self.groups = parms["n_groups"]

    def exposed(self, u): #need old exposed eventally, old should be more infectious
        # could we just handle both here?
        new_exposed = []
        old_exposed = []
        # Extract the number of infected individuals for each group
        I_g = np.array([u[group][2] for group in range(self.groups)])
        for target_group in range(self.groups):
            S, V, E1,  E2, I1, I2, R, Y = u[target_group]
            # rows are to, columns are from
            foi = np.dot(self.parms["beta"][target_group] , I_g / np.array(self.parms["N"]))
            ifrac = 1.0 - np.exp(-foi * self.parms["dt"])
            new_exposed.append(np.random.binomial(S, ifrac))
            e1frac = 1.0 - math.exp(-self.parms["sigma1"] * self.parms["dt"])
            old_exposed.append(np.random.binomial(E1, e1frac))
        return {new_exposed, old_exposed}

    def infectious(self, u): #need infection secondary
        new_infectious = []
        old_infectious = []
        for group in range(self.groups):
            S, V, E1, E2, I1, I2, R, Y = u[group]
            e2frac = 1.0 - math.exp(-self.parms["sigma2"] * self.parms["dt"]) #this should probably be a combo of E1+E2
            new_infectious.append(np.random.binomial(E2, e2frac))
            i2frac = 1.0 - math.exp(-self.parms["treatment"] * self.parms["dt"])
            old_infectious.append(np.random.binomial(I2, i2frac))
        return {new_infectious, old_infectious}

    def recovery(self, u):
        new_recoveries = []
        for group in range(self.groups):
            S, V,  E1, E2, I1, I2, R, Y = u[group]
            rfrac = 1.0 - math.exp(-self.parms["gamma"] * self.parms["dt"])
            new_recoveries.append(np.random.binomial(I2, rfrac))
        return new_recoveries

    def seirmodel(self, u, t): # add v group, possibly vaccinated group with vaccine failure going back eventually, might need more complexity for ongoing vaccine campaign
        new_u = []
        new_exposed = self.exposed(u)[0]
        old_exposed = self.exposed(u)[1] #
        new_infectious = self.infectious(u)[0]# 
        old_infectious = self.infectious(u)[1]
        new_recoveries = self.recovery(u)#
        for group in range(self.groups):
            S, V, E1, E2, I1, I2, R, Y = u[group]
            new_S = S - new_exposed[group]
            new_V = 0
            new_E1 = E1 + new_exposed[group] - old_exposed[group]
            new_E2 = E2 + old_exposed[group] - new_infectious[group]
            new_I1 = I1 + new_infectious[group] - old_infectious[group]
            new_I2 = I2 + old_infectious[group] - new_recoveries[group]
            new_R = R + new_recoveries[group]
            new_Y = Y + new_exposed[group]
            new_u.append([new_S, new_V, new_E1, new_E2, new_I1, new_I2, new_R, new_Y])
        return new_u
