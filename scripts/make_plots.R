library(tidyverse)
library(ggplot2)

results <- read_csv("output/results_test.csv")
reps <- max(results$replicate)

plot_cols <- c("#20419a", "#cf4828", "#f78f47")

#### Cumulative and incidence plots ####
vax_levs <- c("low", "medium", "optimistic")
sub_conns <- c(0.01, 0.1)
filtered_results <- results |>
    filter(
        replicate %in% 1:20,
        initial_coverage_scenario %in% vax_levs,
        k_21 %in% sub_conns
    )

p <- filtered_results |>
    ggplot() +
    geom_line(
        aes(t, Y,
            col = factor(group),
            group = interaction(replicate, group)
        ),
        alpha = 0.5
    ) +
    facet_grid(k_21 ~ initial_coverage_scenario,
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
p <- filtered_results |>
    ggplot(aes(t, I1 + I2,
    col = factor(group),
    group=interaction(replicate, group))) +
    geom_line(alpha = 0.5) +
    facet_grid(k_21 ~ initial_coverage_scenario,
        labeller = label_both
    ) +
    theme_minimal(base_size = 18) +
    scale_color_manual(values = plot_cols) +
    labs(x = "Days", y = "Incident Infections", col = "Group")
ggsave(filename = paste0(
        "output/incidence_curves_r0",
        12, ".png"
    ), plot = p, width = 10, height = 8)


#### Overeall final size plot
for (i in c(12)) {
    p <- results |>
        filter(
            t == 365,
            initial_coverage_scenario %in% vax_levs,
            k_21 %in% sub_conns
        ) |>
        group_by(replicate, initial_coverage_scenario, k_21) |>
        summarise(final_size = sum(Y)) |> # total sum across groups
        ggplot(aes(final_size)) +
        # scale_x_log10() +
        geom_histogram(bins = 50) +
        theme_minimal(base_size = 18) +
        labs(x = "Final Outbreak Size", y = "Number of Simulations") +
        facet_grid(k_21 ~ initial_coverage_scenario,
            labeller = label_both)

    ggsave(filename = paste0(
        "output/overall_final_size_r0",
        i, ".png"
    ), plot = p, width = 10, height = 8)
}


#### Grouped outbreak size plots ####
for (i in c(12)) {
    p <- results |>
        filter(
            t == 365, k_21 %in% sub_conns,
            initial_coverage_scenario %in% vax_levs
        ) |>
        ggplot(aes(Y + 1, fill = factor(group))) +
        geom_histogram(position = "identity", alpha = 0.5) +
        scale_x_log10() +
        facet_grid(k_21 ~ initial_coverage_scenario,
            labeller = label_both
        ) +
        scale_fill_manual(values = plot_cols) +
        theme_minimal(base_size = 18) +
        labs(
            title = "R0=12", x = "Final Group Outbreak Size",
            y = "Number of simulations", fill = "Group"
        )
    ggsave(filename = paste0(
        "output/group_final_size_r0",
        i, ".png"
    ), plot = p, width = 10, height = 8)
}

#### Summary table
outbreak_sizes <- c(300, 500)
for (i in outbreak_sizes) {
    res_table <- results |>
        filter(t == 365, Y >= i) |>
        group_by(
            group,
            initial_coverage_scenario,
            k_21
        ) |>
        count() |>
        mutate(n = round(n / reps, 2) * 100) |>
        pivot_wider(names_from = group, values_from = n) |>
        select(
            InitialCoverage = initial_coverage_scenario,
            SubPopConnectivity = k_21,
            Sub1 = `1`, Sub2 = `2`,
            General = `0`
        )

    write_csv(res_table,
              paste0("output/res_table_outbreak_size", i, ".csv"))
}
