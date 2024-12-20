import numpy as np
import sirenv
import pandas as pd
import griddler
import griddle.griddle

def simulate(parms):

    t = np.linspace(0,parms["tf"], parms["tl"])
    S = np.zeros(parms["tl"])
    I = np.zeros(parms["tl"])
    R = np.zeros(parms["tl"])
    Y = np.zeros(parms["tl"])
    u = [parms["N"] - parms["I0"], parms["I0"],0,0]
    S[0],I[0],R[0],Y[0] = u
    for j in range(1,parms["tl"]):
        u = sirenv.sir(u,parms,t[j])
        S[j],I[j],R[j],Y[j] = u
    return {'t':t,'S':S,'I':I,'R':R,'Y':Y}

if __name__=="__main__":
    parameter_sets = griddler.griddle.read("scripts/config.yaml")

    results_all = griddler.run_squash(simulate, parameter_sets)

    results_all.write_csv(f"scripts/results.csv")
