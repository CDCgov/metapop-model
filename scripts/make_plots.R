library(tidyverse)
library(ggplot2)

results <- read_csv("output/results_test.csv")

plot_cols = c("#156082",  "#78206e", "#3b7d23")

#### Cumulative plot
filtered_results <- results |>
    filter(replicate == 15,
           initial_coverage_scenario == "optimistic",
           connectivity_scenario == 1,
           beta_small == 0.001)

total_results <- filtered_results |>
    group_by(t) |>
    summarize(total_Y = sum(Y))

filtered_results |>
    ggplot(aes(t, Y, col=factor(group))) +
    geom_line(size = 1) +
    geom_line(data = total_results, aes(t, total_Y),
              color = "black", size = 1) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    #facet_wrap(~replicate) +
    labs(x="Days", y = "Cumulative Infections", col = "Group")


#### Incidence plot
filtered_results |>
    ggplot(aes(t, I1+I2, col=factor(group))) +
    geom_line(size = 1) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    labs(x="Days", y = "Incident Infections", col = "Group")

#### Overeall final size plot
results |>
    filter(t == 365) |>
    group_by(replicate, initial_coverage_scenario, connectivity_scenario) |>
    summarise(final_size = sum(Y)) |> # total sum across groups
    ggplot(aes(final_size + 1)) +
    geom_histogram() +
    theme_minimal(base_size = 18) +
    labs(x="Final Outbreak Size", y = "Number of Simulations") +
    facet_grid(connectivity_scenario ~ initial_coverage_scenario)


#### Grouped outbreak size plots
results |>
    filter(t == 365) |>
    ggplot(aes(Y, fill = factor(group))) +
    geom_histogram() +
    #scale_x_log10() +
    facet_grid(connectivity_scenario ~ initial_coverage_scenario) +
    scale_fill_manual(values = plot_cols) +
    theme_minimal(base_size = 18) +
    labs(x="Final Group Outbreak Size", y="Number of simulations",fill ="Group")

#### Summary table
outbreak_size <- 300
res_table <- results |>
    filter(t == 365, Y >= 30) |>
    group_by(group, initial_coverage_scenario,
             connectivity_scenario, beta_small) |>
    count() |>
    mutate(n = round(n/1000, 2)) |>
    pivot_wider(names_from = group, values_from = n) |>
    select(initial_coverage_scenario, connectivity_scenario, Sub1 = `1`, Sub2 =`2`, General = `0`, beta_small)
write_csv(res_table, "output/res_table.csv")
