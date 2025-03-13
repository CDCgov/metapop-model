import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
from numpy import unique

results = pl.read_csv("output/results_test.csv")
groups = unique(results['group'])
plot_cols = ["#156082", "#3b7d23", "#78206e"]
sns.set_context("notebook", font_scale=1.5)

#### Visualization of one simulation ####
# Filter the data for replicate == 0 and initial_coverage_scenario == "low"
replicate_0_results = results.filter((pl.col("replicate") == 0) & (pl.col("initial_coverage_scenario") == "optimistic") & (pl.col("connectivity_scenario") == 1.0))

# Plot the line plot for each group
plt.figure()

for group, color in zip(groups, plot_cols):
    group_data = replicate_0_results.filter(pl.col("group") == group)
    sum_I1_I2 = group_data["I1"] + group_data["I2"]
    plt.plot(group_data["t"], sum_I1_I2, label=f'Group {group}', color=color)

plt.xlabel('Time (Days)')
plt.ylabel('Infected Population')
plt.legend()
plt.tight_layout()

plt.savefig('output/infection_curve.png')
plt.close()

#### Faceted Histogram Sub Groups ####
vax_scenario_order = ["low", "medium","optimistic"]

filtered_results = results.filter(pl.col("t") == 365)

# Convert to pandas DataFrame for seaborn compatibility
filtered_results_df = filtered_results.to_pandas()

# Calculate the bin edges for consistent bin width
min_value = filtered_results_df["Y"].min()
max_value = filtered_results_df["Y"].max()
bins = np.linspace(min_value, max_value, 100)  # 40 bins

# Create the faceted histogram with the same bin width for all histograms
g = sns.FacetGrid(filtered_results_df,
                  row="initial_coverage_scenario",
                  col = "connectivity_scenario",
                  row_order=vax_scenario_order,
                  sharex=True, sharey=True, height=4, aspect=3)
g.map_dataframe(sns.histplot, x="Y", hue="group", multiple="stack", bins=bins, palette=plot_cols)
# To set the x-axis to log scale: g.set(xscale="log")

g.set_axis_labels('Final size', 'Number of simulations')
g.add_legend(title='Group')
g.tight_layout()

g.savefig('output/final_size_groups.png')

#### Faceted Histogram Total outbreak size ####
# Filter the data for t=365

filtered_results = results.filter(pl.col("t") == 365)

# Convert to pandas DataFrame for seaborn compatibility
filtered_results_df = filtered_results.to_pandas()

# Group by simulation, vaccination scenario, and connectivity scenario, and sum the Y column
grouped_results = filtered_results_df.groupby(['replicate', 'initial_coverage_scenario', 'connectivity_scenario']).agg({'Y': 'sum'}).reset_index()
grouped_results.rename(columns={'Y': 'total_outbreak_size'}, inplace=True)

# Calculate the bin edges for consistent bin width
min_value = grouped_results["total_outbreak_size"].min()
max_value = grouped_results["total_outbreak_size"].max()
bins = np.linspace(min_value, max_value, 100)  # 20 bins

# Create the faceted histogram with the same bin width for all histograms
g = sns.FacetGrid(grouped_results,
                  row="initial_coverage_scenario",
                  col="connectivity_scenario",
                  row_order=vax_scenario_order,
                  sharex=True, sharey=True, height=4, aspect=3)
g.map_dataframe(sns.histplot, x="total_outbreak_size", bins=bins)
g.set_axis_labels('Total Outbreak Size', 'Number of Simulations')

plt.tight_layout()
plt.savefig('output/final_size_total.png')
plt.close()
