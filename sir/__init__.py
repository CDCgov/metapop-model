import numpy as np
import math

class SEIRModel:
    def __init__(self, parms):
        self.parms = parms
        self.groups = parms["n_groups"]

    def exposed(self, u):
        new_exposed = []
        # Extract the number of infected individuals for each group
        I_g = np.array([u[group][2] for group in range(self.groups)])
        for target_group in range(self.groups):
            S, E, I, R, Y = u[target_group]
            # rows are to, columns are from
            foi = np.dot(self.parms["beta"][target_group] , I_g / np.array(self.parms["N"]))
            ifrac = 1.0 - np.exp(-foi * self.parms["dt"])
            new_exposed.append(np.random.binomial(S, ifrac))
        return new_exposed

    def infectious(self, u):
        new_infectious = []
        for group in range(self.groups):
            S, E, I, R, Y = u[group]
            efrac = 1.0 - math.exp(-self.parms["sigma"] * self.parms["dt"])
            new_infectious.append(np.random.binomial(E, efrac))
        return new_infectious

    def recovery(self, u):
        new_recoveries = []
        for group in range(self.groups):
            S, E, I, R, Y = u[group]
            rfrac = 1.0 - math.exp(-self.parms["gamma"] * self.parms["dt"])
            new_recoveries.append(np.random.binomial(I, rfrac))
        return new_recoveries

    def seirmodel(self, u, t):
        new_u = []
        new_exposed = self.exposed(u)
        new_infectious = self.infectious(u)
        new_recoveries = self.recovery(u)
        for group in range(self.groups):
            S, E, I, R, Y = u[group]
            new_S = S - new_exposed[group]
            new_E = E + new_exposed[group] - new_infectious[group]
            new_I = I + new_infectious[group] - new_recoveries[group]
            new_R = R + new_recoveries[group]
            new_Y = Y + new_exposed[group]
            new_u.append([new_S, new_E, new_I, new_R, new_Y])
        return new_u
