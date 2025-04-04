library(tidyverse)
library(dplyr)

# functions for plots/tables:
#   extracting cumulative, incident, weekly, attack rates
#   to be reused in plots/analyses

create_filename <- function(base_name, date = "",
                            suffix = "", format = ".csv") {
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

get_weekly_inc_from_cum <- function(simulation_results,
                                    grouping_params) {
    simulation_results |>
        arrange(t) |>
        group_by(replicate, group, !!!syms(grouping_params)) |>
        mutate(Y_diff = Y - lag(Y, default = 0)) |>
        select(
            t, replicate, group, all_of(grouping_params), Y, Y_diff
        ) |>
        mutate(week = t %/% 7) |>
        group_by(week, replicate, group, !!!syms(grouping_params)) |>
        summarise(weekly_Y_diff = sum(Y_diff, na.rm = TRUE), .groups = "drop")
}
