# Visualize incidence and cumulative incidence time curves
import polars as pl
import metapop as mt

# read the results - make sure you run scripts.simulate.py first
results = pl.read_csv("output/results.csv")

# update the defaults
plkwargs = dict()
plkwargs['vax_levs'] = ["low", "medium", "optimistic"]
plkwargs['k_21_vals'] = [0.01, 0.1]
plkwargs['groups'] = results['group'].unique().to_numpy()
plkwargs['colors'] = ["#20419a", "#cf4828", "#f78f47"]
assert len(plkwargs['colors']) >= len(plkwargs['groups']), "Not enough colors for all groups"

# plot all scenarios
mt.plot_daily_infectious(results, **plkwargs)
mt.plot_interval_incidence(results, **plkwargs)
mt.plot_interval_incidence(results,interval=7, **plkwargs)
mt.plot_interval_cumulative_incidence(results, **plkwargs)

# try to plot just one scenario
plkwargs_1 = plkwargs.copy()
plkwargs_1['figname'] = "daily_infectious_curves_one_scenario.png"
plkwargs_1['vax_levs'] = ["low"]
plkwargs_1['k_21_vals'] = [0.01]
plkwargs_1['margins'] = dict(left=0.18, right=0.9, top=0.9, bottom=0.15)

mt.plot_daily_infectious(results, **plkwargs_1)

plkwargs_1['figname'] = "daily_incidence_curves_one_scenario.png"
mt.plot_interval_incidence(results, **plkwargs_1)

plkwargs_1['figname'] = "weekly_incidence_curves_one_scenario.png"
mt.plot_interval_incidence(results, interval=7, **plkwargs_1)

plkwargs_1['figname'] = "daily_cumulative_incidence_curves_one_scenario.png"
mt.plot_interval_cumulative_incidence(results, **plkwargs_1)
