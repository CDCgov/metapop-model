library(tidyverse)
library(ggplot2)

date <- ""
suffix <- ""
R0 <- 12

# Function to create filename for experiment results
create_filename <- function(base_name = "output/interventions/results", date = "", suffix = "", format=".csv") {
  filename <- base_name
  if (date != '') {
    filename <- paste0(filename, "_", date)
  }
  if (suffix != '') {
    filename <- paste0(filename, "_", suffix)
  }
  filename <- paste0(filename, format)
  return(filename)
}

# Use the function to create the filename for results
filename <- create_filename(date = date, suffix = suffix)
# read in the results
results <- read_csv(filename)

# get the number of replicates
reps <- max(results$replicate)
plot_reps <- 20 # sims to plot in incidence curves
plot_cols <- c("#20419a", "#cf4828", "#f78f47")

# Get 20 simulations for plots
isolation_days <- c(400, 0)
filtered_results <- results |>
    filter(
        replicate %in% 1:plot_reps
    )|>
    mutate(symptomatic_isolation_day = factor(symptomatic_isolation_day, levels = isolation_days))

#### Cumulative and incidence plots ####
p <- filtered_results |>
    ggplot() +
    geom_line(
        aes(t, Y,
            col = factor(group),
            group = interaction(replicate, group)
        ),
        alpha = 0.5
    ) +
    facet_grid(symptomatic_isolation_day ~ total_vaccine_uptake_doses,
        labeller = label_both
    ) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    labs(x = "Days", y = "Cumulative Infections", col = "Group")

ggsave(filename = paste0(
    "output/interventions/cumulative_curves",
    12, ".png"
), plot = p, width = 10, height = 8)

#### Incidence plot ####
incidence_results <- filtered_results |>
    arrange(t) |>
    group_by(replicate, group, total_vaccine_uptake_doses, symptomatic_isolation_day) |>
    mutate(Y_diff = Y - lag(Y, default = 0)) |>
    select(
        t, replicate, group, total_vaccine_uptake_doses,
        symptomatic_isolation_day, Y, Y_diff
    ) |>
    mutate(week = t %/% 7) |>
    group_by(week, replicate, group, total_vaccine_uptake_doses, symptomatic_isolation_day) |>
    summarise(weekly_Y_diff = sum(Y_diff, na.rm = TRUE))


p <- incidence_results |>
    ggplot(aes(week, weekly_Y_diff,
        col = factor(group),
        group = interaction(replicate, group)
    )) +
    geom_line(alpha = 0.25) +
    facet_grid(symptomatic_isolation_day ~ total_vaccine_uptake_doses,
        labeller = label_both
    ) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    labs(x = "Week", y = "Weekly Incident Infections", col = "Group")

ggsave(filename = paste0(
    "output/interventions/incidence_curves",
    12, ".png"
), plot = p, width = 10, height = 8)


#### Overall final size plot
for (i in c(12)) {
    p <- results |>
        filter(
            t == 365
        ) |>
        group_by(replicate, total_vaccine_uptake_doses, symptomatic_isolation_day) |>
        summarise(final_size = sum(Y)) |> # total sum across groups
        ggplot(aes(final_size)) +
        # scale_x_log10() +
        geom_histogram(bins = 50) +
        theme_minimal(base_size = 18) +
        labs(x = "Final Outbreak Size", y = "Number of Simulations") +
        facet_grid(symptomatic_isolation_day ~ total_vaccine_uptake_doses,
            labeller = label_both
        )

    ggsave(filename = paste0(
        "output/interventions/overall_final_size",
        i, ".png"
    ), plot = p, width = 10, height = 8)
}


#### Grouped outbreak size plots ####
for (i in c(12)) {
    p <- results |>
        filter(
            t == 365
        ) |>
        ggplot(aes(Y + 1, fill = factor(group))) +
        geom_histogram(position = "identity", alpha = 0.5) +
        scale_x_log10() +
        facet_grid(symptomatic_isolation_day ~ total_vaccine_uptake_doses,
            labeller = label_both
        ) +
        scale_fill_manual(values = plot_cols) +
        theme_minimal(base_size = 18) +
        labs(
            title = "R0=12", x = "Final Group Outbreak Size",
            y = "Number of simulations", fill = "Group"
        )
    ggsave(filename = paste0(
        "output/interventions/group_final_size_r0",
        i, ".png"
    ), plot = p, width = 10, height = 8)
}

#### Percent of susceptible infected cumulative
# should read in from config
coverages <- c(0.95, 0.8, 0.8)
pop_sizes <- c(40000, 5000, 5000)

filtered_categories <- filtered_results |>
    mutate(sus_population = case_when(
        group == 0 ~ (1 - coverages[1]) * pop_sizes[1], # nolint
        group == 1 ~ (1 - coverages[2]) * pop_sizes[2], # nolint
        group == 2 ~ (1 - coverages[3]) * pop_sizes[3]
    )) |> # nolint
    mutate(Y_prop_sus = Y / sus_population)


p <- filtered_categories |>
    ggplot() +
    geom_line(
        aes(t, Y_prop_sus,
            col = factor(group),
            group = interaction(replicate, group)
        ),
        alpha = 0.5
    ) +
    facet_grid(symptomatic_isolation_day ~ total_vaccine_uptake_doses,
        labeller = label_both
    ) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    # facet_wrap(~replicate) +
    labs(x = "Days", y = "Cumulative Infections", col = "Group")

ggsave(filename = paste0(
    "output/interventions/cumulative_sus_infected.png",
    12, ".png"
), plot = p, width = 10, height = 8)


#### Summary table
outbreak_sizes <- c(50, 100)
for (i in outbreak_sizes) {
    res_table <- results |>
        filter(t == 365, Y >= i) |>
        group_by(
            group,
            total_vaccine_uptake_doses,
            symptomatic_isolation_day
        ) |>
        count() |>
        mutate(n = round(n / reps, 2) * 100) |>
        pivot_wider(names_from = group, values_from = n) |>
        select(
            TotalUptake = total_vaccine_uptake_doses,
            IsolationDay = symptomatic_isolation_day,
            Sub1 = `1`, Sub2 = `2`,
            General = `0`
        )

    write_csv(
        res_table,
        paste0("output/interventions/res_table_outbreak_size", i, ".csv")
    )
}
