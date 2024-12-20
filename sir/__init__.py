import numpy as np
import math

def sirmodel(u,parms,t):
    S,I,R,Y=u
    foi = parms["beta"]*(I+parms["iota"])/parms["N"]
    ifrac = 1.0 - math.exp(-foi*parms["dt"])
    rfrac = 1.0 - math.exp(-parms["gamma"]*parms["dt"])
    infection = np.random.binomial(S,ifrac)
    recovery = np.random.binomial(I,rfrac)
    return [S-infection,I+infection-recovery,R+recovery,Y+infection ]

