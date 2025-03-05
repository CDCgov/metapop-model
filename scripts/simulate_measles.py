import numpy as np
import polars as pl
import polars.selectors as cs
import griddler
import griddler.griddle
from sir.__init__measles import SEIRModel

def simulate(parms):

    beta_2_value = None
    if "beta_2_low" in parms and "beta_2_high" in parms:
        beta_2_value = np.random.uniform(parms["beta_2_low"], parms["beta_2_high"])
        parms["beta"][1][1] = beta_2_value

    t = np.linspace(0, parms["tf"], parms["tl"])
    groups = parms["n_groups"]
    S = np.zeros((parms["tl"], groups))
    V = np.zeros((parms["tl"], groups))
    E1 = np.zeros((parms["tl"], groups))
    E2 = np.zeros((parms["tl"], groups))
    I1 = np.zeros((parms["tl"], groups))
    I2 = np.zeros((parms["tl"], groups))
    R = np.zeros((parms["tl"], groups))
    Y = np.zeros((parms["tl"], groups))
    u = [[parms["N"][group] - parms["I0"][group],
          0,
          0,
          0,
          parms["I0"][group],
          0,
          0,
          0
         ] for group in range(groups)]
    for group in range(groups):
        S[0, group], V[0, group], E1[0, group], E2[0, group], I1[0, group], I2[0, group], R[0, group], Y[0, group] = u[group]

    model = SEIRModel(parms)

    for j in range(1, parms["tl"]):
        u = model.seirmodel(u, t[j])
        for group in range(groups):
            S[j, group], V[j, group], E1[j, group], E2[j, group], I1[j, group], I2[j, group], R[j, group], Y[j, group] = u[group]

    df = pl.DataFrame({
        't': np.repeat(t, groups),
        'group': np.tile(np.arange(groups), parms["tl"]),
        'S': S.flatten(),
        'V': V.flatten(),
        'E1': E1.flatten(),
        'E2': E2.flatten(),
        'I1': I1.flatten(),
        'I2': I2.flatten(),
        'R': R.flatten(),
        'Y': Y.flatten(),
        'beta_2_value': [beta_2_value] * (parms["tl"] * groups)
    })

    return df

if __name__ == "__main__":
    parameter_sets = griddler.griddle.read("scripts/config_measles.yaml")
    results_all = griddler.run_squash(griddler.replicated(simulate), parameter_sets)
    print(results_all.columns)
    results = results_all.select(cs.by_name(["t", "group", "Y", "replicate", "beta_2_value"]))
    # with pl.Config(tbl_rows = -1):
    #     print(results)
    results_tot = results_all.select(cs.by_name(['t', 'group', 'S', 'V', 'E1', 'E2', 'I1', 'I2', 'R', 'Y', 'beta_2_value']))
    results_tot.write_csv("output/results_all_100_beta.csv")
    
