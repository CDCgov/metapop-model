# flake8: noqa
"""
Streamlit app for the metapopulation measles outbreak model.

This script provides an interactive web interface for simulating measles outbreaks
under different intervention scenarios (reactive vaccination, isolation, quarantine).
Users can adjust model parameters, run stochastic simulations, and visualize results.

Key Features:
- Sidebar for parameter input and scenario configuration
- Dynamic charts comparing intervention and no-intervention scenarios
- Outbreak summary statistics and downloadable results
- Detailed documentation and assumptions
"""

import os
import importlib
import pandas as pd
import streamlit as st
import numpy as np
import polars as pl
import altair as alt
from st_flexible_callout_elements import flexible_callout
import io
import sys
import base64
import datetime

# import what's needed from other metapop modules
from .app_helper import (
    get_scenario_results,
    read_parameters,
    get_show_parameter_mapping,
    get_advanced_parameter_mapping,
    get_outcome_options,
    get_outcome_mapping,
    app_editors,
    set_parms_to_zero,
    rescale_prop_vax,
    get_widget_types,
    get_min_values,
    get_max_values,
    get_step_values,
    get_helpers,
    get_formats,
    get_widget_idkeys,
    update_intervention_parameters_from_widget,
    reset,
    add_daily_incidence,
    get_interval_results,
    get_table,
    get_median_trajectory_from_episize,
    get_median_trajectory_from_peak_time,
    totals_same_by_ks,
)
from .helper import (
    Ind,
    build_vax_schedule,
    initialize_population,
    seed_from_string,
    get_metapop_info,
)
from .sim import get_time_array
# if you want to use the methods from metapop in this file under
# if __name__ == "__main__": you'll need to import them as:
# from metapop.app_helper import (
#     get_scenario_results,
#     read_parameters,
#     get_default_full_parameters,
#     get_show_parameter_mapping,
#     get_advanced_parameter_mapping,
#     get_outcome_options,
#     get_outcome_mapping,
#     get_parms_from_table,
#     correct_parameter_types,
#     add_daily_incidence,
#     get_interval_results,
#     create_chart,
#     calculate_outbreak_summary,
#     get_table,
# )
### note: this is not recommended use within a file that is imported as a package module, but it can be useful for testing purposes

__all__ = [
    "app",
]


def app(replicates=20):
    """
    Main Streamlit app function for the measles outbreak simulator.

    Args:
        replicates (int): Number of simulation replicates to plot for each scenario.
    """
    st.set_page_config(layout="wide")
    st.title("Measles Outbreak Simulator")

    # Show development warning if not in production
    if os.environ.get("MODE", "PRODUCTION") == "DEVELOPMENT":
        st.warning(f"Building from dev branch")

    st.text(
        "This interactive tool illustrates the impact of "
        "vaccination, isolation, and quarantine measures on the "
        "size of measles outbreaks following introduction of measles into "
        "a community by comparing scenarios with and without interventions."
    )

    # Load default parameters from YAML config
    filepath = os.path.join(
        os.path.dirname(__file__), "app_assets", "onepop_config.yaml"
    )
    parms = read_parameters(filepath)

    # Set up random number generators for plotting and hospitalizations
    plot_rng = np.random.default_rng([parms["seed"], seed_from_string("plot")])
    hosp_rng = np.random.default_rng(
        [parms["seed"], seed_from_string("hospitalizations")]
    )

    scenario_names = ["No Interventions", "Interventions"]
    show_parameter_mapping = get_show_parameter_mapping(parms)
    advanced_parameter_mapping = get_advanced_parameter_mapping()

    # --- Sidebar: Model Inputs and Parameter Editing ---
    with st.sidebar:
        st.text(
            "This tool is meant for use at the beginning of an outbreak at the county level or finer geographic scale. "
            "It is not intended to provide an exact forecast of measles infections in any community. "
            "Hover over the (?) for more information about each parameter."
        )
        st.header("Model Inputs")

        # Get widget types, min/max values, steps, helpers, formats, and keys for widgets
        widget_types = get_widget_types()
        min_values = dict(pop_sizes=[1000, 100, 100], I0=[1, 0, 0])
        min_values = get_min_values(min_values)
        max_values = dict(
            initial_vaccine_coverage=[0.99, 0.99, 0.99],
            vaccine_uptake_start_day=364,
        )
        max_values = get_max_values(max_values)
        steps = get_step_values()
        helpers = get_helpers()
        formats = get_formats()
        keys0 = get_widget_idkeys(0)  # shared parameters
        keys1 = get_widget_idkeys(1)  # scenario 1
        keys2 = get_widget_idkeys(2)  # scenario 2

        # Customize helper texts for clarity
        helpers["I0"][0] = "The model currently has a maximum of 10 initial infections."
        helpers["pop_sizes"][0] = (
            "The model currently has a minimum of 1,000 people and a maximum of 100,000 people. As population sizes get larger, the assumption of a well-mixed population becomes less valid."
        )
        helpers["initial_vaccine_coverage"][0] = (
            "The percent of the population with any immunity against measles, including both through MMR vaccination and through past infection."
        )

        # Add a section for a Reset button for all parameters
        col_reset = st.columns(1)[0]

        # --- Shared Parameters in order of appearance in the app ---
        shared_keys = [
            "I0",
            "pop_sizes",
            "initial_vaccine_coverage",
        ]
        # shared parameters that are lists
        shared_list_keys = [
            "I0",
            "pop_sizes",
            "initial_vaccine_coverage",
        ]

        # add a section for the shared parameters in the sidebar panel
        col0 = st.columns(1)[0]

        col0.subheader(
            "Shared parameters",
        )

        col0.text(
            "Type in a population size and baseline immunity, as "
            "well as the number of people initially infected with measles in the population. "
        )
        subheader = ""
        edited_parms = app_editors(
            col0,
            subheader,
            parms,
            shared_keys,
            shared_list_keys,
            show_parameter_mapping,
            widget_types,
            min_values,
            max_values,
            steps,
            helpers,
            formats,
            keys0,
        )

        # --- Scenario Parameters ---
        ordered_keys_vax = [
            "total_vaccine_uptake_doses",
            "vaccine_uptake_start_day",
            "vaccine_uptake_duration_days",
        ]
        ordered_keys_npi = [
            "isolation_on",
            "isolation_adherence",
            "pre_rash_isolation_on",
            "pre_rash_isolation_adherence",
        ]
        # parameters that are lists or arrays
        list_parameter_keys = []

        # Scenario parameters
        # parameters for scenario 1 are not shown
        st.header(
            "Interventions scenario",
            help="The adherence to both isolation and quarantine, "
            "as well as the vaccine uptake and start time and duration of the vaccination campaign, "
            "can be specified.",
        )
        st.text(
            "Choose interventions to simulate and compare with a "
            "scenario with no active interventions. Interventions can be applied "
            "independently or in combination with each other. "
            "The results are compared to a baseline scenario that does not "
            "have a vaccination campaign, nor isolation or quarantine interventions incorporated."
        )
        st.text(
            "When quarantine and isolation are turned on, they are applied to the entire duration of the simulation."
        )

        # Set scenario 1 (no intervention) parameters to zero for interventions
        edited_parms1 = set_parms_to_zero(
            edited_parms,
            [
                "pre_rash_isolation_adherence",
                "isolation_adherence",
                "total_vaccine_uptake_doses",
            ],
        )
        assert (
            edited_parms1["total_vaccine_uptake_doses"] == 0
        ), "Total vaccine uptake doses should be 0 for the no intervention scenario"
        assert (
            edited_parms1["pre_rash_isolation_adherence"] == 0
        ), "Pre-rash isolation adherence should be 0 for the no intervention scenario"
        assert (
            edited_parms1["isolation_adherence"] == 0
        ), "Isolation adherence should be 0 for the no intervention scenario"

        # Intervention scenario parameter editors
        col_intervention_vax = st.columns(1)[0]

        # placeholder for text about the vaccine campaign doses and other intervention effects in the siderbar
        # defining this here allows us to place it above the advanced options section
        col_intervention_text = st.columns(1)[0]
        col_intervention_npi = st.columns(1)[0]

        # For the intervention scenario, user defines values
        edited_parms2 = app_editors(
            col_intervention_vax,
            "",
            edited_parms,
            ordered_keys_vax,
            list_parameter_keys,
            show_parameter_mapping,
            widget_types,
            min_values,
            max_values,
            steps,
            helpers,
            formats,
            keys2,
        )
        edited_parms2 = app_editors(
            col_intervention_npi,
            "",
            edited_parms2,
            ordered_keys_npi,
            list_parameter_keys,
            show_parameter_mapping,
            widget_types,
            min_values,
            max_values,
            steps,
            helpers,
            formats,
            keys2,
        )

        # This app gives users the ability to specify the proportion of people without prior immunity who will get vaccinated during an active vaccination campaign
        # The metapop model under the hood takes in a parameter for the number of doses to be administered during the vaccination campaign
        # Rescale vaccine uptake from a proportion to the number of doses administered during vaccination campaign
        if parms["use_prop_vaccine_uptake"]:
            edited_parms1 = rescale_prop_vax(edited_parms1)
            edited_parms2 = rescale_prop_vax(edited_parms2)

        # --- Disease Parameters ---
        with st.expander("Disease parameters"):
            st.text(
                "These options allow changes to parameter assumptions about "
                "measles natural history parameters."
            )
            advanced_ordered_keys = [
                "desired_r0",
                "latent_duration",
                "infectious_duration",
                "IHR",
            ]

            # show additional advanced parameters if there are multiple population groups
            if parms["n_groups"] > 1:
                advanced_ordered_keys = advanced_ordered_keys + [
                    "k_i_0",
                    "k_g1",
                    "k_g2",
                    "k_21",
                ]

            # advanced parameters that are lists or arrays
            advanced_list_keys = ["k_i"]

            # set the parameters for scenario 2
            edited_advanced_parms2 = app_editors(
                st.container(),
                "",
                edited_parms2,
                advanced_ordered_keys,
                advanced_list_keys,
                advanced_parameter_mapping,
                widget_types,
                min_values,
                max_values,
                steps,
                helpers,
                formats,
                keys1,
                disabled=False,
            )

            # shared advanced parameters to scenario 1
            edited_advanced_parms1 = (
                edited_parms1  # copy the no intervention scenario parameters so far
            )
            # copy the non intervention advanced parameters from scenario 2 to scenario 1
            for key in advanced_ordered_keys:
                if key not in ("pre_rash_isolation_adherence", "isolation_adherence"):
                    edited_advanced_parms1[key] = edited_advanced_parms2[key]

        # Final assertions for scenario 1
        assert (
            edited_advanced_parms1["total_vaccine_uptake_doses"] == 0
        ), "Total vaccine uptake doses should be 0 for the no intervention scenario"
        assert (
            edited_advanced_parms1["pre_rash_isolation_adherence"] == 0
        ), "Pre-rash isolation adherence should be 0 for the no intervention scenario"
        assert (
            edited_advanced_parms1["isolation_adherence"] == 0
        ), "Isolation adherence should be 0 for the no intervention scenario"

        # --- About Section ---
        info = get_metapop_info()
        st.header("About this app")
        st.caption(f"metapop version: {info['version']}")
        st.caption(f"commit hash: {info['commit']}")

        # show the url of the repo
        url = info["url"]
        image_url = "https://hdpng.com/images/ok-calling-myself-a-developer-would-be-a-stretch-but-still-two-years-ago-i-had-no-clue-what-git-and-github-was-and-how-to-do-a-pr-256.png"
        # image_url = os.path.join(os.path.dirname(__file__), "app_assets", "github-mark.png")
        st.markdown(
            f'<a href="{url}" target="_blank">'
            f'<img src="{image_url}" width="30" style="vertical-align:middle; margin-right:8px;">'
            "Source code"
            "</a>",
            unsafe_allow_html=True,
        )
        st.caption(f"Questions? Contact us: {info['email']}")

    # --- End of Sidebar ---

    # --- Scenario Logic ---
    # Determine if interventions are on or off
    if (
        (
            edited_parms2["total_vaccine_uptake_doses"] == 0
            or edited_parms2["vaccine_uptake_duration_days"] == 0
        )
        and edited_parms2["pre_rash_isolation_on"] == False
        and edited_parms2["isolation_on"] == False
    ):
        interventions = "Off"
    else:
        interventions = "On"

    # Reset parameters if reset button is clicked
    with col_reset:
        reset_button = st.button(
            "Reset parameters",
            on_click=reset,
            args=(
                parms,
                widget_types,
            ),
        )

    # Update intervention parameters from widgets
    edited_intervention_parms1 = update_intervention_parameters_from_widget(
        edited_advanced_parms1
    )
    edited_intervention_parms2 = update_intervention_parameters_from_widget(
        edited_advanced_parms2
    )

    # Enforce logical constraints on interventions: isolation > quarantine
    if (
        edited_parms2["pre_rash_isolation_on"] == True
        and edited_parms2["isolation_on"] == False
    ):
        st.error(
            "Isolation must be activated for quarantine to be activated. "
            "Please adjust the intervention toggles"
        )
        return

    if (
        edited_intervention_parms2["isolation_adherence"]
        < edited_intervention_parms2["pre_rash_isolation_adherence"]
        and edited_parms2["pre_rash_isolation_on"] == True
        and edited_parms2["isolation_on"] == True
    ):
        st.error(
            "Isolation adherence should be greater than or equal to quarantine adherence. "
            "Please adjust the parameters."
        )
        return

    # Prepare scenario parameter sets
    updated_parms1 = edited_intervention_parms1.copy()
    updated_parms2 = edited_intervention_parms2.copy()
    scenario1 = [updated_parms1]
    scenario2 = [updated_parms2]

    # --- Initial Population State and Vaccination Schedule ---
    initial_states = initialize_population(1, 1, updated_parms2)

    # last element returned by initialize_population is the initial state vector
    u_ind = Ind.max_value() + 1

    warning_message = ""

    # Warn if no one to vaccinate and no other interventions. This effectively means that there are
    #  no other  interventions being modeled in the second scenario.
    if (
        (initial_states[u_ind][0][Ind.S.value] == 0)
        and (not scenario2[0]["isolation_on"])
        and (not scenario2[0]["pre_rash_isolation_on"])
    ):
        warning_message += (
            "With these initial conditions, there are no people to vaccinate in this population and other interventions are turned off. "
            "Please turn on isolation or quarantine or adjust the population size or baseline immunity."
            "\n\n"
        )

    # Build vaccine schedule and warn if no doses will be administered
    edited_intervention_parms2["t_array"] = get_time_array(edited_intervention_parms2)
    schedule = build_vax_schedule(edited_intervention_parms2)

    # if doses are zero, warn the user
    if sum(schedule.values()) == 0:
        # if no other warning message defined yet, create this one instead
        if warning_message == "":
            warning_message += (
                "With the selected vaccination campaign parameters, no vaccine doses will be administered during the campaign."
                " This may happen if the vaccination uptake percentage for the susceptible population is too low or if the campaign duration is zero days."
                " Please review vaccination campaign parameters to administer at least one dose during the campaign."
            )

    if warning_message != "":
        st.warning(
            warning_message,
            icon="⚠️",
        )

    # --- Plot Options ---
    outcome_option = st.selectbox(
        "Metric",
        get_outcome_options(),
        index=0,  # by default display weekly incidence
        placeholder="Select an outcome to plot",
    )

    # --- Run Simulations and Display Results ---
    chart_placeholder = st.empty()

    # Map the selected option to the outcome variable
    outcome_mapping = get_outcome_mapping()
    outcome = outcome_mapping[outcome_option]

    # run the model with the updated parameters
    chart_placeholder.text("Running scenarios...")
    results1 = get_scenario_results(scenario1)
    results2 = get_scenario_results(scenario2)

    # Display number of doses administered
    with col_intervention_text:
        st.text(
            f"Total vaccines scheduled to be administered during campaign: {sum(schedule.values())} doses",
            help="This number is calculated based on user input for the percentage of the non-immune population that gets vaccinated during the campaign. If the campaign starts late, the actual number of doses administered may be lower due to there being not enough eligible individuals left to vaccinate.",
        )

    # Save full results for summary tables
    fullresults1 = results1
    fullresults2 = results2

    # Extract unique groups
    groups = results1["group"].unique().to_list()

    # Add daily incidence columns
    chart_placeholder.text("Adding daily incidence...")
    results1 = add_daily_incidence(results1, groups)
    results2 = add_daily_incidence(results2, groups)

    # Get weekly interval results
    interval = 7
    chart_placeholder.text("Getting interval results...")
    interval_results1 = get_interval_results(results1, groups, interval)
    interval_results2 = get_interval_results(results2, groups, interval)

    # Rename columns for plotting
    app_column_mapping = {f"inc_{interval}": "Winc", "Y": "WCI"}
    interval_results1 = interval_results1.rename(app_column_mapping)
    interval_results2 = interval_results2.rename(app_column_mapping)

    # Default to cumulative daily incidence if outcome not available
    if outcome not in ["Y", "inc", "Winc", "WCI"]:
        print("outcome not available yet, defaulting to Cumulative Daily Incidence")
        outcome = "Y"

    # Prepare data for Altair plots
    if outcome_option in [
        "Daily Incidence",
        "Daily Cumulative Incidence",
    ]:
        alt_results1 = results1
        alt_results2 = results2
        x = "t:Q"
        time_label = "Time (days)"
        vax_start = min(schedule.keys())
        vax_end = max(schedule.keys())
    elif outcome_option in ["Weekly Incidence", "Weekly Cumulative Incidence"]:
        alt_results1 = interval_results1
        alt_results2 = interval_results2
        x = "interval_t:Q"
        time_label = "Time (weeks)"
        vax_start = min(schedule.keys()) / interval
        vax_end = max(schedule.keys()) / interval

    # get median trajectory for each scenario (based weekly results for ALL sims, not just smaller sample)
    ave_traj1 = get_median_trajectory_from_peak_time(interval_results1)
    ave_traj2 = get_median_trajectory_from_peak_time(interval_results2)

    # Filter for median trajectory results
    ave_results1 = alt_results1.filter(pl.col("replicate") == ave_traj1).with_columns(
        pl.lit(scenario_names[0]).alias("scenario")
    )
    ave_results2 = alt_results2.filter(pl.col("replicate") == ave_traj2).with_columns(
        pl.lit(scenario_names[1]).alias("scenario")
    )
    combined_ave_results = ave_results1.vstack(ave_results2)

    # Get subset of results for plotting
    alt_results1 = alt_results1.with_columns(
        pl.lit(scenario_names[0]).alias("scenario")
    )
    alt_results2 = alt_results2.with_columns(
        pl.lit(scenario_names[1]).alias("scenario")
    )
    replicate_inds = plot_rng.choice(
        results1["replicate"].unique().to_numpy(), replicates, replace=False
    )
    combined_alt_results = alt_results1.vstack(alt_results2).filter(
        pl.col("replicate").is_in(replicate_inds)
    )

    # --- Build Altair Chart ---
    # Chart title and subtitle
    if interventions == "On":
        pre_rash_isolation_adherence = 0
        isolation_adherence = 0
        if edited_intervention_parms2["pre_rash_isolation_on"]:
            pre_rash_isolation_adherence = edited_intervention_parms2[
                "pre_rash_isolation_adherence"
            ]
        if edited_intervention_parms2["isolation_on"]:
            isolation_adherence = edited_intervention_parms2["isolation_adherence"]
        pre_rash_isolation_adherence_pct = int(pre_rash_isolation_adherence * 100)
        isolation_adherence_pct = int(isolation_adherence * 100)
        mean_doses_administered = round(
            results2.filter(pl.col("t") == edited_intervention_parms2["t_array"][-1])
            .select("X")
            .mean()
            .item()
        )
        title = alt.TitleParams(
            "Outcome Comparison by Scenario",
            subtitle=[
                (f"Vaccine campaign: {mean_doses_administered} doses administered"),
                f"Quarantine adherence: {pre_rash_isolation_adherence_pct}%",
                f"Isolation adherence: {isolation_adherence_pct}%",
            ],
            subtitleColor="#808080",
        )
    else:
        combined_alt_results = alt_results1.filter(
            pl.col("replicate").is_in(replicate_inds)
        )
        combined_ave_results = ave_results1
        title = "No Intervention Scenario"

    chart_placeholder.text("Building charts...")
    chart = (
        alt.Chart(combined_alt_results)
        .mark_line(opacity=0.3, strokeWidth=0.75, clip=True)
        .encode(
            x=alt.X(x, title=time_label),
            y=alt.Y(outcome, title=outcome_option),
            color=alt.Color(
                "scenario",
                title="Scenario",
                scale=alt.Scale(
                    domain=[scenario_names[0], scenario_names[1]],
                    range=["#FB7E38", "#0057b7"],
                ),
            ),
            detail="replicate",
            tooltip=["scenario", "t", outcome],
        )
        .properties(title=title, width=800, height=400)
    )

    # Add vaccine campaign period as a shaded box if applicable
    if edited_intervention_parms2["total_vaccine_uptake_doses"] > 0:
        vax = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x_start": [vax_start],
                        "x_end": [vax_end],
                    }
                )
            )
            .mark_rect(opacity=0.1, color="grey")
            .encode(
                x=alt.X("x_start:Q", title=time_label),
                x2="x_end:Q",
            )
        )
        chart = chart + vax

    # Add bold line for median trajectory
    ave_line = (
        alt.Chart(combined_ave_results.to_pandas())
        .mark_line(opacity=1.0, strokeWidth=3.0, clip=True)
        .encode(
            x=alt.X(x, title=time_label),
            y=alt.Y(outcome, title=outcome_option),
            color=alt.Color(
                "scenario",
                title="Scenario",
                scale=alt.Scale(
                    domain=[scenario_names[0], scenario_names[1]],
                    range=["#cf4828", "#20419a"],
                ),
            ),
            tooltip=["scenario", "t", outcome],
        )
    )
    chart = chart + ave_line

    # Add annotation if no interventions are selected
    if interventions == "Off":
        annotation = (
            alt.Chart(
                pd.DataFrame(
                    {"text": ["Use at least one intervention to compare scenarios"]}
                )
            )
            .mark_text(align="center", baseline="top", color="grey", fontSize=18)
            .encode(text="text:N", y=alt.value(10))
        )
        chart = chart + annotation

    chart = chart.properties(padding={"top": 10, "bottom": 30, "left": 30, "right": 40})
    chart_placeholder.altair_chart(chart, use_container_width=True)

    # --- Chart Description ---
    st.markdown(
        '<p style="font-size:14px;">'
        "Each thin line represents an individual simulation of the stochastic "
        "model. All simulations within a given scenario (i.e., shown with "
        "the same color) are run under the same set of parameters, and "
        "differences between each individual simulation are due to random "
        "variation in contact rates. Bolded lines show the simulation that possessed "
        "the median time of peak prevalence across all epidemic trajectories for "
        "each scenario. If a vaccination campaign is activated, the time period over "
        "which vaccines are distributed is shown by gray box."
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Outbreak Summary Stats ---
    st.subheader("Simulation summary")
    with st.expander("Show intervention strategies", expanded=False):
        columns = st.columns(2)
        flexible_callout(
            (
                "No Interventions:<br><ul>"
                f"<li> Vaccines administered during campaign: {int(updated_parms1['total_vaccine_uptake_doses'])}</li>"
                f"<li> Adherence to quarantine among pre-symptomatic infectious individuals: {int(updated_parms1['pre_rash_isolation_adherence'] * 100)}%</li>"
                f"<li> Adherence to isolation among symptomatic infectious individuals:  {int(updated_parms1['isolation_adherence'] * 100)}%</li></ul>"
            ),
            background_color="#feeadf",
            font_color="#8f3604",
            container=columns[0],
        )
        if interventions == "On":
            pre_rash_isolation_adherence = 0
            isolation_adherence = 0
            if edited_intervention_parms2["pre_rash_isolation_on"]:
                pre_rash_isolation_adherence = edited_intervention_parms2[
                    "pre_rash_isolation_adherence"
                ]
            if edited_intervention_parms2["isolation_on"]:
                isolation_adherence = edited_intervention_parms2["isolation_adherence"]

            callout_text = "Interventions:<br><ul>"
            if (
                edited_intervention_parms2["total_vaccine_uptake_doses"] == 0
                or edited_intervention_parms2["vaccine_uptake_duration_days"] == 0
            ):
                callout_text += "<li> Vaccines administered during campaign: 0</li>"
            else:
                callout_text += f"<li> Vaccines administered during campaign: {mean_doses_administered} between day {min(schedule.keys())} and day {max(schedule.keys())}</li>"
            callout_text += f"<li> Adherence to quarantine among pre-symptomatic infectious individuals: {int(pre_rash_isolation_adherence * 100)}%</li>"
            callout_text += f"<li> Adherence to isolation among symptomatic infectious individuals: {int(isolation_adherence * 100)}%</li></ul>"

            flexible_callout(
                callout_text,
                background_color="#cbe4ff",
                font_color="#001833",
                container=columns[1],
            )

    # Prepare outbreak summary table
    fullresults1 = fullresults1.with_columns(
        pl.lit(scenario_names[0]).alias("Scenario")
    )
    fullresults2 = fullresults2.with_columns(
        pl.lit(scenario_names[1]).alias("Scenario")
    )
    combined_results = (
        fullresults2.vstack(fullresults1)
        .with_columns(
            pl.col("t").cast(pl.Int64),
            pl.col("group").cast(pl.Int64),
            pl.col("Y").cast(pl.Int64),
        )
        .filter(pl.col("t") == pl.col("t").max())
        .group_by("Scenario", "replicate")
        .agg(pl.col("Y").sum().alias("Total"))
        .sort(["Scenario", "replicate"])
    )

    outbreak_summary = get_table(
        combined_results, edited_intervention_parms2["IHR"], hosp_rng
    )

    # This function will always return false until scipy is included and ks_2samp can be run
    no_scenario_difference = totals_same_by_ks(combined_results, scenario_names)

    if interventions == "Off":
        outbreak_summary = outbreak_summary.select("", scenario_names[0])

    # Highlight outbreak summary
    if interventions == "On":
        relative_difference = (
            outbreak_summary.filter(pl.col("") == "Infections, mean (95% CI)")
            .select("Relative Difference (%)")
            .to_numpy()[0][0]
        )

        intervention_text = f"Adding "
        interventions = []
        if edited_intervention_parms2["total_vaccine_uptake_doses"] > 0:
            interventions.append("vaccination")
        if edited_intervention_parms2["pre_rash_isolation_success"] > 0:
            interventions.append("quarantine")
        if edited_intervention_parms2["isolation_success"] > 0:
            interventions.append("isolation")

        if len(interventions) > 1:
            intervention_text += (
                ", ".join(interventions[:-1]) + " and " + interventions[-1]
            )
        elif interventions:
            intervention_text += interventions[0]

        st.text(
            f"{intervention_text} decreases total measles infections by {float(relative_difference):.0f}% "
            f"in a population of size {edited_parms2['pop_sizes'][0]} "
            f"with baseline immunity of {round(edited_parms2['initial_vaccine_coverage'][0] * 100)}%."
        )

        # Always false until scipy is included in the project.
        if no_scenario_difference:
            st.warning(
                "The two scenarios are statistically indistinguishable based on a 2 sample K-S test. "
                "In this case, the relative difference is not a reliable metric.",
                icon="⚠️",
            )

        # if the Relative Difference is NaN, set it to ""
        # outbreak_summary = outbreak_summary.with_columns(
        #     pl.col("Relative Difference (%)").fill_nan("")
        # )

    if "pyodide" in sys.modules:
        # Workaround for stlite not displying st.dataframe correctly in this case.
        # Unclear if this is a stlite pyarrow problem (see notes at
        # https://github.com/whitphx/stlite?tab=readme-ov-file#limitations
        # about `st.dataframe()`.)
        st.markdown(outbreak_summary.to_pandas().to_markdown(index=False))
        csv_buffer = io.StringIO()
        outbreak_summary.to_pandas().to_csv(csv_buffer, index=False)
        base64_encoded = base64.b64encode(csv_buffer.getvalue().encode("utf-8")).decode(
            "utf-8"
        )
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        data_uri = f'<p style="font-size:10px;text-align:right;"><a download="measles-sim-{time}.csv" href="data:text/csv;base64,{base64_encoded}">Download as CSV</a></p>'
        st.markdown(data_uri, unsafe_allow_html=True)
    else:
        st.dataframe(outbreak_summary)

    # --- Detailed Methods Section ---
    with st.expander("Detailed methods", expanded=False):
        st.markdown(
            """
            <p style="font-size:14px;">
            This model examines measles transmission in a population after an
            introduction of measles. This is a stochastic compartmental SVEIR
            model with all-or-nothing vaccination. Because the model is
            stochastic, simulations using the same parameters can yield
            different epidemic outcomes. We therefore run 100 individual
            simulations to produce a range of possible outcomes and use these
            simulations to estimate outcome uncertainty for each parameter set.
            <br><br>

            <p style="font-size:14px;">
            People who are immune at the beginning of the simulation, either
            through past vaccination or previous infection, begin in the
            "Vaccinated" compartment. Individuals who have received the vaccine
            but are not immune (due to vaccine failure) are tracked in a separate
            "Vaccinated but Susceptible" compartment, and have the same
            susceptibility as individuals in the "Susceptible" compartment.
            <br><br>

            <p style="font-size:14px;">
            Users can explore the impact of interventions, including vaccination,
            isolation, and quarantine measures ("Interventions" scenario)
            compared to a baseline scenario without active interventions ("No
            Interventions". When they are implemented, isolation and quarantine
            measures begin on the same day as the introduced measles infections are
            identified via rash onset and run for the duration of simulation.
            The start and end time of the vaccination campaign can be specified.
            <br><br>

            <p style="font-size:14px;">
            We conduct a two-sample K-S test to determine if the
            total measles infections from the "Interventions" scenario differ
            from the total measles infections of the "No Interventions" baseline
            scenario and present information if scenario results are
            indistinguishable.<br><br>

            <b style="font-size:14px;">Assumptions</b>
            <p style="font-size:14px;">We note that this modeling approach
            makes several simplifying assumptions, including the following:</p>
            <ul>

            <li style="font-size:14px;">This is a homogenous mixing model without age or spatial structure,
            meaning all individuals have the same probability of contact with
            each other (also known as a well-mixed population model). In larger populations, this may overestimate the size
            and duration of an outbreak.</li>

            <li style="font-size:14px;">The MMR vaccine is modeled as an
            "all-or-nothing" vaccine, meaning that it is perfectly effective
            for some people (corresponding to the efficacy for one or two doses of MMR) and has no efficacy for others.
            </li>

            <li style="font-size:14px;">To initialize the population with existing immunity, users input a "baseline
            immunity" value. Because there is no age structure in the model, this value is assumed to
            account for existing vaccination coverage or prior infection over the entire population. It's assumed that
            existing vaccination coverage and prior infection result in 97% protection against future infection
            (the estimated efficacy of two doses of MMR) and this is multiplied by baseline immunity to initialize
            the size of the Vaccinated population. The remaining 3% of individuals are assumed to lack protection
            and are initialized into a separate state (Vaccinated but Susceptible).
            </li>

            <li style="font-size:14px;"> Vaccines administered during the vaccination campaign are
            assumed to be first doses of MMR. The user-defined vaccination campaign timing is defined by the
            vaccination start day (days after the initial infection is introduced) and the duration of the campaign (in days).
            It's assumed that the number of doses administered per day is equal to the total number of doses
            divided by the duration of the campaign (but may be rounded to the nearest integer value).
            If the campaign ends after the simulation ends, the vaccination campaign will be shortened to run until the last simulation day while keeping the rate of doses administered per day the same . In this case, the total number of doses scheduled to be administered will not match the vaccination uptake proportion input specified in the sidebar.
            </li>

            <li style="font-size:14px;"> We assume that both susceptible and
            exposed individuals who are not yet infectious are eligible to get
            vaccinated during the vaccination campaign. We also assume that exposed
            individuals are not yet aware of their exposure status and so they
            are equally likely to seek vaccination. After vaccination, only
            susceptible individuals become immune, while exposed individuals
            remain in the exposed state and continue with infection progression
            as normal. The number of doses administered may be lower than the
            number of doses scheduled if by the time of the campaign, the daily
            dose rate scheduled exceeds the number of individuals eligible for vaccination.
            </li>

            <b style="font-size:14px;">Model Parameters</b>
            <ul>
            <li style="font-size:14px;"> The basic reproductive number (R<sub>0</sub>),
            captures contact rates and the probability of infection given
            contact with an infectious individual. R<sub>0</sub> for measles is
            generally estimated to be between 12 and 18
            <a href='https://www.ecdc.europa.eu/en/measles/facts' target='_blank'>[Factsheet about measles]</a>.
            Communities with higher contact rates — for example populations with
            higher population density or larger households
            <a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC8765757/' target='_blank'>[Social contact patterns and implications for infectious disease transmission – a systematic review and meta-analysis of contact surveys | eLife]</a>
            — may have higher R<sub>0</sub>. The probability of infection given
            contact with an infectious individuals is very high for measles;
            the household attack rate is estimated to be 90% among unvaccinated contacts
            <a href='https://www.cdc.gov/yellow-book/hcp/travel-associated-infections-diseases/measles-rubeola.html#:~:text=Measles%20is%20among%20the%20most,global%20eradication%20of%20measles%20feasible' target='_blank'>[CDC Yellow Book: Measles (Rubeola)]</a>.
            </li>

            <li style="font-size:14px;">The latent period is generally
            estimated to be around 11 days
            <a href='https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html' target='_blank'>[Measles Clinical Diagnosis Fact Sheet | Measles (Rubeola) | CDC]</a>.
            </li>

            <li style="font-size:14px;">The infectious period is generally
            estimated to be around 9 days, with an upper bound of 11 days
            <a href='https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html' target='_blank'>[Measles Clinical Diagnosis Fact Sheet | Measles (Rubeola) | CDC]</a>.
            </li>

            <li style="font-size:14px;">Measles rash onset is generally
            estimated to be on day 5 of this infectious period
            <a href='https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html' target='_blank'>[Measles Clinical Diagnosis Fact Sheet | Measles (Rubeola) | CDC]</a>.
            In this model, isolation when sick is assumed to start halfway through the infectious period.
            </li>

            <li style="font-size:14px;"> We assume vaccine efficacy for individuals
            vaccinated during the campaign is 93%, the estimate for one dose of MMR
            <a href='https://www.cdc.gov/measles/vaccines/index.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fvaccines%2Fvpd%2Fmmr%2Fpublic%2Findex.html' target='_blank'>[MMR Vaccine Information]</a>.
            </li>

            <li style="font-size:14px;"> We assume protection against infection for individuals
            with prior infection or vaccination at the start of the simulation is on average 97% for the population, the estimate for two doses of MMR
            <a href='https://www.cdc.gov/measles/vaccines/index.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fvaccines%2Fvpd%2Fmmr%2Fpublic%2Findex.html' target='_blank'>[MMR Vaccine Information]</a>.
            </li>

            <li style="font-size:14px;">Isolation when sick is estimated to be
            approximately 75% effective at reducing transmission when comparing
            people who do isolate when sick to people who do not
            <a href='https://academic.oup.com/cid/article/75/1/152/6424734' target='_blank'>[Impact of Isolation and Exclusion as a Public Health Strategy to Contain Measles Virus Transmission During a Measles Outbreak | Clinical Infectious Diseases | Oxford Academic]</a>.
            In this model, since isolation starts only at rash onset, isolation
            reduces transmission by 100% during the second half of the
            infectious period, leading to a reduction of 50% overall.
            </li>

            <li style="font-size:14px;">
            Quarantine for people who are unvaccinated but have been
            exposed is estimated to be 44-76% effective at reducing transmission
            when comparing those who do quarantine to those who do not.
            We assume a 60% reduction in transmission, which is the mean of this range.
            <a href='https://academic.oup.com/cid/article/75/1/152/6424734' target='_blank'>[Impact of Isolation and Exclusion as a Public Health Strategy to Contain Measles Virus Transmission During a Measles Outbreak | Clinical Infectious Diseases | Oxford Academic]</a>
            </li>

            <li style="font-size:14px;">
            The infection hospitalization ratio (IHR) has been estimated at 20%
            in past outbreaks, but we allow users to vary this value between 5% and 25%
            <a href='https://www.cdc.gov/measles/signs-symptoms/index.html' target='blank'>[Measles Symptoms and Complications | Measles (Rubeola) | CDC]</a>.
            </li>
            </ul>
            </p>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    app()
