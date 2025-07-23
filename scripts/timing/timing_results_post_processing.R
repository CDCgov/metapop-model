library(tidyverse)
library(ggplot2)
source("scripts/analyzer.R")


# Use the function to create the filename for results
filename <- create_filename("output/timing/results")
# read in the results
results <- read_csv(filename)

# hospitalizations
IHR <- 0.15


final_results <- results |>
  filter(t == 365) |>
  mutate(Scenario = intervention_delay) |>
  group_by(Scenario, initial_vaccine_coverage) |>
  mutate(hospitalizations = rbinom(n(), Y, IHR)) |>
  summarise(
    upper_incidence = quantile(Y, 0.95),
    lower_incidence = quantile(Y, 0.05),
    incidence = median(Y),
    upper_hospitalizations = quantile(hospitalizations, 0.95),
    lower_hospitalizations = quantile(Y, 0.05),
    hospitalizations = median(hospitalizations)
  ) |>
  ungroup()

write.csv(final_results, create_filename("output/timing/final_results",
                                         date = as.character(Sys.Date())))

final_results |>
  ggplot(aes(Scenario, incidence, color = factor(initial_vaccine_coverage))) +
  geom_point() +
  geom_errorbar(aes(ymin = lower_incidence, ymax = upper_incidence),
                width = 0.2) +
  labs(
    color = "Initial Vaccine Coverage",
    x = "Intervention Delay (days)",
    y = "Outbreak size"
  ) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
