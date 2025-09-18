import matplotlib.pyplot as plt
import pandas as pd

pop = [0.056, 0.165, 0.779]
cov = [0.9, 0.93, 0.95, 0.9]
# [0.0, 0.0, 0.93,  0.95]
print(pop[0] * cov[0] + pop[1] * cov[2] + pop[2] * cov[3])


def make_dataframes(pop, cov):
    df_pop = pd.DataFrame(
        [
            {
                "population": "<5",
                "percentage": 100 * pop[0],
            },
            {
                "population": "5-17",
                "percentage": 100 * pop[1],
            },
            {
                "population": "18+",
                "percentage": 100 * pop[2],
            },
        ]
    )
    df_coverage = pd.DataFrame(
        [
            {
                "threshold": "24 months",
                "coverage": 100 * cov[0],
            },
            {
                "threshold": "5 years (kindergarten)",
                "coverage": 100 * cov[1],
            },
            {
                "threshold": "18+ years",
                "coverage": 100 * cov[2],
            },
            {
                "threshold": "19+ years",
                "coverage": 100 * cov[3],
            },
        ]
    )
    return df_pop, df_coverage


def get_baseline_immunity(population, coverage):
    """
    Calculate the baseline immunity given the values in the calculator

    Args:
        population (list): The percent of the population in each goup.
        coverage (list): The percent of individuals in each group with prior immunity.

    Returns:
        immunity_value (float): The percent of the population with immunity.
    """

    immunity_value = (
        # 0-1years
        population[0] * 0 * 1 / 5
        +
        # 1-2years
        population[0] * coverage[0] / (2 * 5)
        +
        # 2-5years
        +population[0] * (coverage[1] + coverage[0]) * 3 / (2 * 5)
        # 5-18years
        + population[1] * (coverage[2] + coverage[1]) / 2
        # over 18 years
        + population[2] * (coverage[3])
    ) / (100 * 100)

    return round(immunity_value, 2)


def get_step_immunity(population, coverage):
    immunity_value = (
        0 * population[0] * 2 / 5
        + coverage[0] * population[0] * 3 / 5
        + coverage[1] * population[1]
        + coverage[2] * population[2]
    ) / (100 * 100)
    return round(immunity_value, 2)


df_pop, df_coverage = make_dataframes(pop, cov)

baseline_immun = get_baseline_immunity(df_pop["percentage"], df_coverage["coverage"])
step_baseline_immune = get_step_immunity(df_pop["percentage"], df_coverage["coverage"])


x = pd.concat(
    [pd.DataFrame([df_pop["percentage"][0]]), df_pop["percentage"]], ignore_index=True
)

x.loc[0, "0"] = x[0][0] * 2 / 5
x.loc[1, "0"] = x[0][1] * 3 / 5
x.loc[2, "0"] = x[0][2] + x[0][1]
x.loc[3, "0"] = x[0][3] + x[0][2]


x = [1, 2, 5, 18, 19, 30]

y = pd.concat(
    [
        pd.DataFrame([0]),
        df_coverage["coverage"],
        pd.DataFrame([df_coverage.loc[3, "coverage"]]),
    ],
    ignore_index=True,
)


plt.step(x, y, where="post", color="#3F5D7D", label="Step Function")
plt.plot(x, y, color="#fb7e38", label="Linear Interpolation")

# Set axis ranges
plt.xlim(0, 30)
plt.ylim(0, 100)
plt.xlabel("Population Age", fontsize=14)
plt.ylabel("Coverage", fontsize=14)
plt.title("Differences in interpolation method", fontsize=18)
for y in range(0, 100, 10):
    plt.plot(
        range(0, 31), [y] * len(range(0, 31)), "--", lw=0.5, color="black", alpha=0.25
    )

for x in range(0, 30, 5):
    plt.axvline([x], 0, 100, linestyle="--", lw=0.5, color="black", alpha=0.25)

ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.legend(bbox_to_anchor=(0.7, 0.6))


plt.savefig("output/immunity_interpolation.png")

print(baseline_immun)
print(step_baseline_immune)


# def make_table(df_pop, df_cov):


# I think we should force 0-12 months to be vaccine free
