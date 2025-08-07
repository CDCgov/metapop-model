library(tidyverse)
library(ggplot2)
source("scripts/analyzer.R")


# Use the function to create the filename for results
filename <- create_filename("output/comparison/vax/results")
# read in the results
results <- read_csv(filename)

# hospitalizations
IHR <- 0.15


final_results <- results |>
  filter(t == 365) |>
  mutate(Scenario = str_sub(intervention_scenario, end = -3)) |>
  group_by(Scenario, initial_vaccine_coverage) |>
  mutate(hospitalizations = rbinom(n(), Y, IHR)) |>
  summarise(
    upper_incidence = quantile(Y, 0.975),
    lower_incidence = quantile(Y, 0.025),
    incidence = median(Y),
    upper_hospitalizations = quantile(hospitalizations, 0.975),
    lower_hospitalizations = quantile(hospitalizations, 0.025),
    hospitalizations = median(hospitalizations)
  ) |>
  ungroup()

baseline_incidence <- final_results |>
  filter(Scenario == "none") |>
  select(incidence, initial_vaccine_coverage) |>
  rename(baseline_incidence = incidence) |>
  ungroup()

baseline_hospitalizations <- final_results |>
  filter(Scenario == "none") |>
  select(hospitalizations, initial_vaccine_coverage) |>
  rename(baseline_hospitalizations = hospitalizations) |>
  ungroup()

final_results <- final_results |>
  left_join(baseline_incidence, by = "initial_vaccine_coverage") |>
  left_join(baseline_hospitalizations, by = "initial_vaccine_coverage") |>
  mutate(
    relative_difference_incidence =
      (baseline_incidence - incidence) / baseline_incidence,
    relative_difference_hospitalizations =
      (baseline_hospitalizations - hospitalizations) / baseline_hospitalizations
  ) |>
  select(
    Scenario, initial_vaccine_coverage,
    incidence, upper_incidence, lower_incidence,
    hospitalizations, upper_hospitalizations, lower_hospitalizations,
    relative_difference_incidence, relative_difference_hospitalizations
  )


write_csv(
  final_results,
  paste0("output/comparison/vax/final_results_table.csv")
)
