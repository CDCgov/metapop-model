import math
import os

import yaml

from metapop import simulate
from metapop.helper import build_vax_schedule
from metapop.sim import get_time_array

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
    parms["tf"] = 3
    parms["vaccine_uptake_start_day"] = 0
    parms["vaccine_uptake_duration_days"] = 1
    parms["total_vaccine_uptake_doses"] = 100
    parms["vaccine_efficacy_1_dose"] = 1
    parms["vaccine_efficacy_2_dose"] = 1
    parms["vaccinated_group"] = 2

    df = simulate(parms, seed=parms["seed"])

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
    # given starting on the second day and continuing for 5 days total
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


def test_vaccine_schedule_clipped_past_end_of_simulation():
    with open(os.path.join(testdir, "test_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    parms = config["baseline_parameters"]
    # reset scenario so that there are no infections but vaccines should be
    # given starting on the eighth day after introduction (day 9), lasting for 5 days total, 3 days past the end of the simulation
    # so the vaccination schedule built is only for 2 days
    parms["I0"] = [0, 0, 0]
    parms["tf"] = 10
    parms["vaccine_uptake_start_day"] = 8
    parms["vaccine_uptake_duration_days"] = 5
    parms["total_vaccine_uptake_doses"] = 100
    parms["vaccine_efficacy_1_dose"] = 1
    parms["vaccine_efficacy_2_dose"] = 1
    parms["vaccinated_group"] = 2

    parms["t_array"] = get_time_array(parms)

    schedule = build_vax_schedule(parms)
    print(
        "Vaccines starting on day 9 and continuing for 5 days total, clipped to 2 days"
    )
    for day in schedule:
        vaccinations_on_day = schedule[day]
        print(f"Day {day}: {vaccinations_on_day}")
    assert (
        len(schedule) == parms["tf"] - parms["vaccine_uptake_start_day"]
    ), "Expected vaccination schedule to be clipped to 2 days"


def test_vaccine_schedule_starting_on_last_day_of_simulation():
    with open(os.path.join(testdir, "test_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    parms = config["baseline_parameters"]

    # reset scenario so that there are no infections but vaccines should be
    # given starting on the eighth day after introduction (day 9), lasting for 5 days total, 3 days past the end of the simulation
    # so the vaccination schedule built is only for 2 days
    parms["I0"] = [0, 0, 0]
    parms["tf"] = 365
    parms["vaccine_uptake_start_day"] = 365
    parms["vaccine_uptake_duration_days"] = 5
    parms["total_vaccine_uptake_doses"] = 100
    parms["vaccine_efficacy_1_dose"] = 1
    parms["vaccine_efficacy_2_dose"] = 1
    parms["vaccinated_group"] = 2

    parms["t_array"] = get_time_array(parms)

    schedule = build_vax_schedule(parms)
    print(schedule)
    print(f"The last day of simulation is {parms['tf']}")
    print(
        f"Vaccines are starting {parms['vaccine_uptake_start_day']} days after introduction, which is day {parms['vaccine_uptake_start_day'] + 1} in the simulation"
    )
    assert schedule == {
        parms["vaccine_uptake_start_day"] + 1: 0
    }, "No vaccines should be administered past the end of simulation"


def uneven_doses_per_day(duration: int, doses: int):
    """
    Distribute `doses` vaccines across a campagin of `duration` days with assert checks

    Example:
        If expected doses per day is (25 doses / 10 duration) = 2.5, cumulative doses are
        2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0
        With integers using round(), where odd integers round up on 0.5
        2, 5, 8, 10, 12, 15, 18, 20, 22, 25
        And delivered doses by day are the floor() or ceil() of the average
        2, 3, 3, 2,  2,  3,  3,  2,  2,  3
        With the final schedule being
        {1: 2, 2: 3, 3: 3, 4: 2, 5: 2, 6: 3, 7: 3, 8: 2, 9: 2, 10: 3}
    """
    with open(os.path.join(testdir, "test_config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    parms = config["baseline_parameters"]
    # reset scenario so that there are no infections but vaccines should be
    # distributed unevenly over ten days because the expected doses per day is  not equal to integers.

    parms["I0"] = [0, 0, 0]
    parms["tf"] = duration
    parms["vaccine_uptake_start_day"] = 0
    parms["vaccine_uptake_duration_days"] = duration
    parms["total_vaccine_uptake_doses"] = doses
    parms["vaccine_efficacy_1_dose"] = 1
    parms["vaccine_efficacy_2_dose"] = 1
    parms["vaccinated_group"] = 2

    parms["t_array"] = get_time_array(parms)

    avg = parms["total_vaccine_uptake_doses"] / parms["vaccine_uptake_duration_days"]

    schedule = build_vax_schedule(parms)

    for i in range(
        parms["vaccine_uptake_start_day"] + 1,
        parms["vaccine_uptake_start_day"] + parms["vaccine_uptake_duration_days"],
    ):
        assert schedule[i] in [math.floor(avg), math.ceil(avg)]
    assert sum(schedule.values()) == parms["total_vaccine_uptake_doses"]


def test_hi_lo_uneven_doses():
    # Test greater than one dose per day on average
    uneven_doses_per_day(duration=10, doses=25)
    # Test less than one dose per day on average
    uneven_doses_per_day(duration=10, doses=9)
    # Test zero doses
    uneven_doses_per_day(duration=10, doses=0)
