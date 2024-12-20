import numpy as np
import polars as pl
import polars.selectors as cs
import griddler
import griddler.griddle
from sir import sirmodel

def simulate(parms):

    t = np.linspace(0,parms["tf"], parms["tl"])
    S = np.zeros(parms["tl"])
    I = np.zeros(parms["tl"])
    R = np.zeros(parms["tl"])
    Y = np.zeros(parms["tl"])
    u = [parms["N"] - parms["I0"], parms["I0"],0,0]
    S[0],I[0],R[0],Y[0] = u
    for j in range(1,parms["tl"]):
        u = sirmodel(u,parms,t[j])
        S[j],I[j],R[j],Y[j] = u

    df = pl.DataFrame({
        't': t,
        'S': S,
        'I': I,
        'R': R,
        'Y': Y
    })

    return df

if __name__=="__main__":
    parameter_sets = griddler.griddle.read("scripts/config.yaml")
    results_all = griddler.run_squash(griddler.replicated(simulate), parameter_sets)
    print(results_all.columns)
    cols_to_select = [
        "t", "S", "I", "R", "replicate"
    ]
    results = (
        results_all
        .select(cs.by_name(cols_to_select))
    )

    with pl.Config(tbl_rows=-1):
        print(results)

    # save results
    results.write_csv(f"scripts/results.csv")
