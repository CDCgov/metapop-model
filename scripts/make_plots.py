import matplotlib.pyplot as plt
import polars as pl
import numpy as np


results = pl.read_csv("output/results_all_100_beta.csv")
data = pl.read_csv("data/2025-02-28_TX_Epi_Report.csv")

unique_replicates = results['replicate'].unique()
unique_Y = results["Y"].unique()

def plot_replicate(results, replicate):
    plt.style.use("ggplot")
    replicate_res = results.filter(pl.col("replicate") == replicate)
    replicate_res = replicate_res.filter(pl.col("group") == 1)
    fig = plt.figure(facecolor='w')
    infection_curve = fig.add_subplot(111, facecolor = '#dddddd', axisbelow = True)
    infection_curve.plot("t", "S", 'b', data = replicate_res, alpha = 0.5, lw = 2, label = 'Susceptible')
    infection_curve.plot("t", "E1", 'r', data = replicate_res, alpha = 0.5, lw = 2, label = 'Exposed')
    infection_curve.plot("t", "I1", 'r', data = replicate_res, alpha = 0.5, lw = 2, label = 'Infected')
    infection_curve.plot("t", "R", 'g', data = replicate_res, alpha = 0.5, lw = 2, label = 'Recovered')
    infection_curve.set_xlabel('Time /days')
    infection_curve.set_ylabel('Number')
    legend = plt.legend(title = "Population", loc = 5, bbox_to_anchor = (1.25, 0.5))
    legend.get_frame().set_linewidth(0)
    legend.get_frame().set_facecolor("white")
    plt.savefig('output/infection_curve.png')


final_res = results.filter(pl.col("t") == 200)
#plt.figure()
plt.hist("Y", data = final_res.filter(pl.col("group") == 1), stacked = True, color = 'blue')
plt.hist("Y", data = final_res.filter(pl.col("group") == 0), stacked = True, color = 'green')# this is not stacked as far as I can tell, depending on the order they change
plt.savefig('output/final_size.png')

plot_replicate(results, 0)

