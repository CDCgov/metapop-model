library(tidyverse)
library(ggplot2)

date <- ""
suffix <- ""
R0 <- 12 # not varied

# Function to create filename for experiment results
create_filename <- function(base_name = "output/one_pop/results", date = "", suffix = "", format = ".csv") {
    filename <- base_name
    if (date != "") {
        filename <- paste0(filename, "_", date)
    }
    if (suffix != "") {
        filename <- paste0(filename, "_", suffix)
    }
    filename <- paste0(filename, format)
    return(filename)
}

# Use the function to create the filename for results
filename <- create_filename( date = date, suffix = suffix)
# read in the results
results <- read_csv(filename)

# get the number of replicates
reps <- max(results$replicate)
plot_reps <- 20 # sims to plot in incidence curves
plot_cols <- c("#20419a", "#cf4828", "#f78f47")

# Get 20 simulations for plots
pop_sizes <- c(5000) # read from config
vax_levs <- c("low") # also: "medium", "optimistic"
uptake_levs <- c(0, 250)
filtered_results <- results |>
    filter(
        replicate %in% 1:plot_reps, initial_coverage_scenario %in% vax_levs
    )

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
    facet_grid(total_vaccine_uptake_doses ~ initial_coverage_scenario,
        labeller = label_both
    ) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    labs(x = "Days", y = "Cumulative Infections", col = "Group")

ggsave(filename = paste0(
    "output/one_pop/cumulative_curves",
    12, ".png"
), plot = p, width = 10, height = 8)

#### Incidence plot ####
incidence_results <- filtered_results |>
    arrange(t) |>
    group_by(replicate, group, initial_coverage_scenario, total_vaccine_uptake_doses) |>
    mutate(Y_diff = Y - lag(Y, default = 0)) |>
    select(
        t, replicate, group, initial_coverage_scenario,
        total_vaccine_uptake_doses, Y, Y_diff
    ) |>
    mutate(week = t %/% 7) |>
    group_by(week, replicate, group, initial_coverage_scenario, total_vaccine_uptake_doses) |>
    summarise(weekly_Y_diff = sum(Y_diff, na.rm = TRUE))


p <- incidence_results |>
    ggplot(aes(week, weekly_Y_diff,
        col = factor(group),
        group = interaction(replicate, group)
    )) +
    geom_line(alpha = 0.25) +
    facet_grid(total_vaccine_uptake_doses ~ initial_coverage_scenario,
        labeller = label_both
    ) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    labs(x = "Week", y = "Weekly Incident Infections", col = "Group")

ggsave(filename = paste0(
    "output/one_pop/incidence_curves",
    12, ".png"
), plot = p, width = 10, height = 8)


#### Overeall final size plot
for (i in c(R0)) {
    p <- results |>
        filter(
            t == 365,
            initial_coverage_scenario %in% vax_levs
        ) |>
        group_by(replicate, initial_coverage_scenario, total_vaccine_uptake_doses) |>
        summarise(final_size = sum(Y)) |> # total sum across groups
        ggplot(aes(final_size)) +
        # scale_x_log10() +
        geom_histogram(bins = 50) +
        theme_minimal(base_size = 18) +
        labs(x = "Final Outbreak Size", y = "Number of Simulations") +
        facet_grid(total_vaccine_uptake_doses ~ initial_coverage_scenario,
            labeller = label_both
        )

    ggsave(filename = paste0(
        "output/one_pop/overall_final_size",
        i, ".png"
    ), plot = p, width = 10, height = 8)
}

#### Percent of susceptible infected cumulative
coverage_scenarios <- data.frame(
    initial_coverage_scenario = c("low", "medium", "optimistic"), # nolint
    coverage_0 = c(0.8, 0.9, 0.95)
)

filtered_categories <- filtered_results |>
    left_join(coverage_scenarios, by = "initial_coverage_scenario") |>
    mutate(sus_population = case_when(
        group == 0 ~ (1 - coverage_0) * pop_sizes[1] # nolint
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
    facet_grid(total_vaccine_uptake_doses ~ initial_coverage_scenario,
        labeller = label_both
    ) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    # facet_wrap(~replicate) +
    labs(x = "Days", y = "Cumulative Infections", col = "Group")

ggsave(filename = paste0(
    "output/one_pop/cumulative_sus_infected.png",
    12, ".png"
), plot = p, width = 10, height = 8)


#### Summary table
outbreak_sizes <- c(50, 100)
for (i in outbreak_sizes) {
    res_table <- results |>
        filter(t == 365) |>
        mutate(outbreak = ifelse(Y >= i, 1, 0)) |>
        group_by(
            initial_coverage_scenario,
            total_vaccine_uptake_doses
        ) |>
        summarise(n = sum(outbreak)) |>
        mutate(n = round(n / reps, 2)) |>
        select(
            InitialCoverage = initial_coverage_scenario,
            DailyDoses = total_vaccine_uptake_doses,
            `OutbreakProbability` = n
        )

    write_csv(
        res_table,
        paste0("output/one_pop/res_table_outbreak_size", i, ".csv")
    )
}
