library(tidyverse)
library(ggplot2)
library(ggridges)

results <- read_csv("output/results_test.csv")

plot_cols <- c("#156082", "#78206e", "#3b7d23")

#### Cumulative and incidence plots ####
vax_levs <- c("low", "optimistic")
sub_conns <- c(0.001, 0.5)
desired_r0s <- c(12)
filtered_results <- results |>
    filter(
        replicate %in% 1:20,
        initial_coverage_scenario %in% vax_levs,
        beta_small %in% sub_conns,
        desired_r0 %in% desired_r0s
    )

total_results <- filtered_results |>
    group_by(
        t, replicate,
        desired_r0,
        initial_coverage_scenario,
        beta_small
    ) |>
    summarize(total_Y = sum(Y))

p <- filtered_results |>
    ggplot() +
    geom_line(
        aes(t, Y,
            col = factor(group),
            group = interaction(replicate, group)
        ),
        alpha = 0.5
    ) +
    geom_line(
        data = total_results,
        aes(t, total_Y, group = replicate),
        color = "darkgrey",
        alpha = 0.5
    ) +
    facet_grid(beta_small ~ initial_coverage_scenario,
        labeller = label_both
    ) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    # facet_wrap(~replicate) +
    labs(x = "Days", y = "Cumulative Infections", col = "Group")

ggsave(filename = paste0(
        "output/cumulative_curves_r0",
        12, ".png"
    ), plot = p, width = 10, height = 8)

#### Incidence plot ####
# filtered_results |>
#     ggplot(aes(t, I1 + I2, col = factor(group))) +
#     geom_line(size = 1) +
#     theme_minimal(base_size = 18) +
#     scale_color_manual(values = plot_cols) +
#     labs(x = "Days", y = "Incident Infections", col = "Group")


#### Overeall final size plot
for (i in c(8, 12)) {
    p <- results |>
        filter(
            t == 365,
            desired_r0 == i,
            initial_coverage_scenario %in% vax_levs,
            beta_small %in% sub_conns
        ) |>
        group_by(replicate, initial_coverage_scenario, beta_small) |>
        summarise(final_size = sum(Y)) |> # total sum across groups
        ggplot(aes(final_size)) +
        # scale_x_log10() +
        geom_histogram(bins = 50) +
        theme_minimal(base_size = 18) +
        labs(x = "Final Outbreak Size", y = "Number of Simulations") +
        facet_grid(beta_small ~ initial_coverage_scenario, labeller = label_both)

    ggsave(filename = paste0(
        "output/overall_final_size_r0",
        i, ".png"
    ), plot = p, width = 10, height = 8)
}


#### Grouped outbreak size plots ####
for (i in c(8, 12)) {
    p <- results |>
        filter(
            t == 365, beta_small %in% sub_conns,
            initial_coverage_scenario %in% vax_levs,
            desired_r0 == i
        ) |>
        ggplot(aes(Y + 1, fill = factor(group))) +
        geom_histogram(position = "identity", alpha = 0.5) +
        scale_x_log10() +
        facet_grid(beta_small ~ initial_coverage_scenario,
            labeller = label_both
        ) +
        scale_fill_manual(values = plot_cols) +
        theme_minimal(base_size = 18) +
        labs(
            title = "R0=8", x = "Final Group Outbreak Size",
            y = "Number of simulations", fill = "Group"
        )
    ggsave(filename = paste0(
        "output/group_final_size_r0",
        i, ".png"
    ), plot = p, width = 10, height = 8)
}

#### Summary table
outbreak_sizes <- c(300, 1000)
for (i in outbreak_sizes) {
    res_table <- results |>
        filter(t == 365, Y >= 30, desired_r0 == 12) |>
        group_by(
            group, initial_coverage_scenario,
            beta_small
        ) |>
        count() |>
        mutate(n = round(n / 1000, 2)) |>
        pivot_wider(names_from = group, values_from = n) |>
        select(
            InitialCoverage = initial_coverage_scenario,
            SubPopConnectivity = beta_small,
            Sub1 = `1`, Sub2 = `2`,
            General = `0`
        )

    write_csv(res_table,
              paste0("output/res_table_outbreak_size", i, ".csv"))
}
