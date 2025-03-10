import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np

results = pl.read_csv("output/results.csv")
data = pl.read_csv("data/2025-02-28_TX_Epi_Report.csv")

#### Visualization of one simulation
# Filter the data for replicate == 0 and initial_coverage_scenario == "low"
replicate_0_results = results.filter((pl.col("replicate") == 0) & (pl.col("initial_coverage_scenario") == "low"))

# Plot the line plot for each group
plt.figure()

for group in [0, 1]:
    group_data = replicate_0_results.filter(pl.col("group") == group)
    sum_I1_I2 = group_data["I1"] + group_data["I2"]
    plt.plot(group_data["t"], sum_I1_I2, label=f'Group {group}')

plt.xlabel('Time (Days)')
plt.ylabel('Infected Population')
plt.legend()
plt.savefig('output/infection_curve.png')
plt.close()

#### Faceted Histogram
# Filter the data for t=200
filtered_results = results.filter(pl.col("t") == 200)

# Convert to pandas DataFrame for seaborn compatibility
filtered_results_df = filtered_results.to_pandas()

# Calculate the bin edges for consistent bin width
min_value = filtered_results_df["Y"].min()
max_value = filtered_results_df["Y"].max()
bins = np.linspace(min_value, max_value, 21)  # 20 bins

# Create the faceted histogram with the same bin width for all histograms
g = sns.FacetGrid(filtered_results_df, row="initial_coverage_scenario", sharex=True, sharey=True, height=4, aspect=2)
g.map_dataframe(sns.histplot, x="Y", hue="group", multiple="stack", bins=bins, palette=['blue', 'orange'])

# Set the x-axis to log scale
g.set(xscale="log")

g.set_axis_labels('Final size (log scale)', 'Number of simulations')
g.add_legend(title='Group')

plt.savefig('output/final_size_log_scale.png')
plt.close()
