import numpy as np
import polars as pl
import polars.selectors as cs
import griddler
import griddler.griddle
from sir import SEIRModel

def simulate(parms):

    beta_2_value = None
    if "beta_2_low" in parms and "beta_2_high" in parms:
        beta_2_value = np.random.uniform(parms["beta_2_low"], parms["beta_2_high"])
        parms["beta"][1][1] = beta_2_value

    t = np.linspace(0, parms["tf"], parms["tl"])
    groups = parms["n_groups"]
    S = np.zeros((parms["tl"], groups))
    E = np.zeros((parms["tl"], groups))
    I = np.zeros((parms["tl"], groups))
    R = np.zeros((parms["tl"], groups))
    Y = np.zeros((parms["tl"], groups))
    u = [[parms["N"][group] - parms["I0"][group],
          0,
          parms["I0"][group],
          0,
          0
         ] for group in range(groups)]
    for group in range(groups):
        S[0, group], E[0, group], I[0, group], R[0, group], Y[0, group] = u[group]

    model = SEIRModel(parms)

    for j in range(1, parms["tl"]):
        u = model.seirmodel(u, t[j])
        for group in range(groups):
            S[j, group], E[j, group], I[j, group], R[j, group], Y[j, group] = u[group]

    df = pl.DataFrame({
        't': np.repeat(t, groups),
        'group': np.tile(np.arange(groups), parms["tl"]),
        'S': S.flatten(),
        'E': E.flatten(),
        'I': I.flatten(),
        'R': R.flatten(),
        'Y': Y.flatten(),
        'beta_2_value': [beta_2_value] * (parms["tl"] * groups)
    })

    return df

if __name__ == "__main__":
    parameter_sets = griddler.griddle.read("scripts/config.yaml")
    results_all = griddler.run_squash(griddler.replicated(simulate), parameter_sets)
    print(results_all.columns)
    results = results_all.select(cs.by_name(["t", "group", "Y", "replicate", "beta_2_value"]))
    # with pl.Config(tbl_rows = -1):
    #     print(results)
    results.write_csv(f"output/results_100_beta.csv")