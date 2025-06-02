import os

import yaml

from metapop import simulate

testdir = os.path.dirname(__file__)


def test_vaccine_schedule_day_1():
    """
    Test that vaccines are given on day 1 if the schedule specified that.
    """
    with open(os.path.join(testdir, "test_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    parms = config["baseline_parameters"]

    # reset scenario so that there are no infections but vaccines should be given and all on day 1
    parms["I0"] = [0, 0, 0]
    parms["tf"] = 10
    parms["vaccine_uptake_start_day"] = 0
    parms["vaccine_uptake_duration_days"] = 1
    parms["total_vaccine_uptake_doses"] = 100
    parms["vaccine_efficacy_1_dose"] = 1
    parms["vaccine_efficacy_2_dose"] = 1
    parms["vaccinated_group"] = 2

    df = simulate(parms, seed=parms["seed"])
    # print(df.head(n=10))
    # for key, value in parms.items():
    #     print(f"{key}: {value}")

    # Check that the number of vaccinated individuals is equal to the total vaccine uptake doses
    vdf = df.filter(df["group"] == parms["vaccinated_group"])
    vaccinations_on_day_1 = vdf.filter(vdf["t"] == 1)["X"][0]
    print("\nVaccines starting on day 1 and continuing for 1 day total")
    print(vdf)
    print(f"Day 1: {vaccinations_on_day_1}\n")
    assert (
        vaccinations_on_day_1 == parms["total_vaccine_uptake_doses"]
    ), f"Expected {parms['total_vaccine_uptake_doses']} but got {vaccinations_on_day_1}"


def test_vaccine_schedule_including_day_1():
    with open(os.path.join(testdir, "test_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    parms = config["baseline_parameters"]

    # reset scenario so that there are no infections but vaccines should be
    # given starting on the first day and continuing for 5 days total
    parms["I0"] = [0, 0, 0]
    parms["tf"] = 10
    parms["vaccine_uptake_start_day"] = 0
    parms["vaccine_uptake_duration_days"] = 5
    parms["total_vaccine_uptake_doses"] = 100
    parms["vaccine_efficacy_1_dose"] = 1
    parms["vaccine_efficacy_2_dose"] = 1
    parms["vaccinated_group"] = 2

    df = simulate(parms, seed=parms["seed"])
    vdf = df.filter(df["group"] == parms["vaccinated_group"])

    vaccine_uptake_start_day_in_model = 1 + parms["vaccine_uptake_start_day"]
    vaccine_uptake_end_day_in_model = (
        1 + parms["vaccine_uptake_start_day"] + parms["vaccine_uptake_duration_days"]
    )

    print("Vaccines starting on day 1 and continuing for 5 days total")
    for day in range(
        vaccine_uptake_start_day_in_model, vaccine_uptake_end_day_in_model
    ):
        vaccinations_on_day = vdf.filter(vdf["t"] == day)["X"][0]
        print(f"Day {day}: {vaccinations_on_day}")

    # Check all vaccines administered by last day of campaign
    vaccines_administered = vdf.filter(vdf["t"] == vaccine_uptake_end_day_in_model)[
        "X"
    ][0]
    print(f"Total vaccines from day 1 to 5: {vaccines_administered}\n")

    assert (
        vaccines_administered == parms["total_vaccine_uptake_doses"]
    ), f"Expected {parms['total_vaccine_uptake_doses']} but got {vaccines_administered}"


def test_vaccine_schedule_starting_day_2():
    with open(os.path.join(testdir, "test_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    parms = config["baseline_parameters"]

    # reset scenario so that there are no infections but vaccines should be
    # given starting on the first day and continuing for 5 days total
    parms["I0"] = [0, 0, 0]
    parms["tf"] = 10
    parms["vaccine_uptake_start_day"] = 1
    parms["vaccine_uptake_duration_days"] = 5
    parms["total_vaccine_uptake_doses"] = 100
    parms["vaccine_efficacy_1_dose"] = 1
    parms["vaccine_efficacy_2_dose"] = 1
    parms["vaccinated_group"] = 2

    df = simulate(parms, seed=parms["seed"])
    vdf = df.filter(df["group"] == parms["vaccinated_group"])

    vaccine_uptake_start_day_in_model = 1 + parms["vaccine_uptake_start_day"]
    vaccine_uptake_end_day_in_model = (
        1 + parms["vaccine_uptake_start_day"] + parms["vaccine_uptake_duration_days"]
    )

    print("Vaccines starting on day 2 and continuing for 5 days total")

    for day in range(
        vaccine_uptake_start_day_in_model, vaccine_uptake_end_day_in_model
    ):
        vaccinations_on_day = vdf.filter(vdf["t"] == day)["X"][0]
        print(f"Day {day}: {vaccinations_on_day}")

    # Check all vaccines administered by last day of campaign
    vaccines_administered = vdf.filter(vdf["t"] == vaccine_uptake_end_day_in_model)[
        "X"
    ][0]
    print(f"Total vaccines from day 2 to 6: {vaccines_administered}\n")
    assert (
        vaccines_administered == parms["total_vaccine_uptake_doses"]
    ), f"Expected {parms['total_vaccine_uptake_doses']} but got {vaccines_administered}"


def test_vaccine_schedule_starting_after_day_2():
    with open(os.path.join(testdir, "test_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    parms = config["baseline_parameters"]

    # reset scenario so that there are no infections but vaccines should be
    # given starting on the first day and continuing for 5 days total
    parms["I0"] = [0, 0, 0]
    parms["tf"] = 10
    parms["vaccine_uptake_start_day"] = 2
    parms["vaccine_uptake_duration_days"] = 5
    parms["total_vaccine_uptake_doses"] = 100
    parms["vaccine_efficacy_1_dose"] = 1
    parms["vaccine_efficacy_2_dose"] = 1
    parms["vaccinated_group"] = 2

    df = simulate(parms, seed=parms["seed"])
    vdf = df.filter(df["group"] == parms["vaccinated_group"])

    vaccine_uptake_start_day_in_model = 1 + parms["vaccine_uptake_start_day"]
    vaccine_uptake_end_day_in_model = (
        1 + parms["vaccine_uptake_start_day"] + parms["vaccine_uptake_duration_days"]
    )

    print("Vaccines starting on day 3 and continuing for 5 days total")

    for day in range(
        vaccine_uptake_start_day_in_model, vaccine_uptake_end_day_in_model
    ):
        vaccinations_on_day = vdf.filter(vdf["t"] == day)["X"][0]
        print(f"Day {day}: {vaccinations_on_day}")

    # Check all vaccines administered by last day of campaign
    vaccines_administered = vdf.filter(vdf["t"] == vaccine_uptake_end_day_in_model)[
        "X"
    ][0]
    print(f"Total vaccines from day 3 to 7: {vaccines_administered}\n")
    assert (
        vaccines_administered == parms["total_vaccine_uptake_doses"]
    ), f"Expected {parms['total_vaccine_uptake_doses']} but got {vaccines_administered}"
