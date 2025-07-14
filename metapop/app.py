# flake8: noqa
"""
Streamlit app for the metapopulation measles outbreak model.

This script provides an interactive web interface for simulating measles
outbreaks in a population under different intervention scenarios (reactive
vaccination, isolation, quarantine). Users can adjust model parameters,
run stochastic simulations, and visualize results.

Key Features:
- Sidebar for parameter input and scenario configuration
- Dynamic charts comparing intervention and no-intervention scenarios
- Outbreak summary statistics and downloadable results
- Detailed documentation and assumptions
"""

import os
import copy
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
    get_session_state_idkeys,
    update_intervention_parameters_from_widget,
    reset,
    add_daily_incidence,
    get_interval_results,
    get_table,
    get_median_trajectory_from_episize,
    get_median_trajectory_from_peak_time,
    totals_same_by_ks,
    img_to_bytes,
    img_to_html,
    is_light_color,
    get_github_logo_path,
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

    This is a Streamlit app that illustrates the impact of layering multiple
    intervention strategies to mitigate a measles outbreak in a single
    community. This interactive tool allows users to see the impact of
    layering 3 active intervention during the outbreak: a vaccination campaign,
    an isolation intervention, and a quarantine intervention, as well as
    a baseline immunity level in the population acquired prior to the outbreak.
    The app simulates the spread of measles following introduction into a
    single community and visualizes outbreak size and incidence under
    different scenarios.

    Args:
        replicates (int): Number of simulation replicates to plot for each scenario. Defaults to 20.

    Returns:
        None: The function runs a Streamlit app and does not return anything.
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

    # Info about the this app (version, date, commit)
    info = get_metapop_info()

    parms = read_parameters(filepath)

    # Ensures that the cache gets invalidated when code changes
    parms["cache_key"] = info["commit"]

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
        min_values = dict(
            pop_sizes=[1000, 100, 100],
            I0=[1, 0, 0],
            vaccine_uptake_start_day=4,
            symptomatic_isolation_start_day=4,
            pre_rash_isolation_start_day=4,
        )
        min_values = get_min_values(min_values)
        max_values = dict(
            initial_vaccine_coverage=[0.99, 0.99, 0.99],
            vaccine_uptake_start_day=364,
            pre_rash_isolation_start_day=364,
            symptomatic_isolation_start_day=364,
        )
        max_values = get_max_values(max_values)
        steps = get_step_values()
        helpers = get_helpers()
        formats = get_formats()

        # --- Note on the use of session state keys ---

        # Session state id keys are defined for each of the shared parameters
        # and the scenarios parameters. These are used to uniquely identify
        # each interactive element which allows users to edit the model
        # parameters used for simulations.

        # Each interactive element must have its own unique key which gets
        # passed on to the Streamlit session state for tracking.

        # By providing these widgets with unique keys, we ensure that the values
        # are correctly associated with the parameters they represent and the
        # values can be reset when the user clicks the reset button. Pressing
        # the reset button results in the parameters being reset to their default values.

        # Each session state key is mapped to a model parameter and the default
        # value for that parameter in stored in the `parms` dictionary. When the
        # reset button is clicked, the app finds the original values for all input
        # elements users can modify and resets the session state value associated
        # with the session state key.

        # If a set of session state keys are not used, this means that the
        # parameters will not be shown and modifiable by users in the sidebar panel.
        # The No Interventions scenario parameters are not shown in the sidebar panel,
        # so it is possible that the keys for this scenario are not used.

        session_state_keys0 = get_session_state_idkeys(0)  # shared parameters
        session_state_keys1 = get_session_state_idkeys(1)  # scenario 1
        session_state_keys2 = get_session_state_idkeys(2)  # scenario 2

        # Customize helper texts for clarity
        helpers["I0"][0] = (
            "The model allows for a maximum of 10 infections introduced in the population."
        )
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
        # this function will return a dictionary of edited scenario parameters
        # based on user input for parameters shared between all scenarios
        # this way we can avoid modifying the original default parameters in parms
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
            session_state_keys0,
        )

        # Intervention scenario and parameters
        st.header(
            "Interventions scenario",
            help="The adherence, duration, and start time of both isolation and quarantine interventions, "
            "as well as the vaccine uptake and start time and duration of the vaccination campaign, "
            "can be specified.",
        )
        st.text(
            "Choose interventions to simulate and compare with a "
            "scenario with no active interventions. Interventions can be applied "
            "independently or in combination with each other. "
            "The results are compared to a baseline scenario that does not "
            "have a vaccination campaign, nor isolation or quarantine interventions incorporated. "
            "In this model, day 1 corresponds to when infected people are introduced into the community and day 5 is the average day of rash onset "
            "for introduced infections (see Detailed Methods). "
        )

        # --- Intervention scenario Parameters ---

        # order of vaccination parameters in the sidebar
        ordered_keys_vax = [
            "vaccine_uptake",
            "total_vaccine_uptake_doses",
            "vaccine_uptake_start_day",
            "vaccine_uptake_duration_days",
        ]
        list_vax_parameter_keys = []  # parameters that are lists or arrays

        # order of non-pharmaceutical intervention (NPI) parameters in the sidebar
        ordered_keys_npi = [
            "isolation_on",
            "isolation_adherence",
            "symptomatic_isolation_start_day",
            "symptomatic_isolation_duration_days",
            "pre_rash_isolation_on",
            "pre_rash_isolation_adherence",
            "pre_rash_isolation_start_day",
            "pre_rash_isolation_duration_days",
        ]
        # parameters that are lists or arrays
        list_npi_parameter_keys = []

        # Set up separate parameter dictionaries for each scenario using the
        # edited_parms as the base dictionary

        # For the no intervention scenario, create edited_parms1 as the parameter dictionary
        # parameters for the no intervention scenario are not shown, but the values are
        # copied from edited_parms and intervention parameters are set to zero
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
        # defining this here allows us to place it above the isolation and quarantine section
        col_intervention_text = st.columns(1)[0]
        col_intervention_npi = st.columns(1)[0]

        # For the intervention scenario, user defines values for the vaccination campaign
        # use edited_parms as the base dictionary and create edited_parms2 with
        # user defined inputs from the sidebar - this is what app_editors will return
        edited_parms2 = app_editors(
            col_intervention_vax,
            "",
            edited_parms,
            ordered_keys_vax,
            list_vax_parameter_keys,
            show_parameter_mapping,
            widget_types,
            min_values,
            max_values,
            steps,
            helpers,
            formats,
            session_state_keys2,
        )

        # create a new section for isolation and quarantine inputs for the intervention scenario
        edited_parms2 = app_editors(
            col_intervention_npi,
            "",
            edited_parms2,
            ordered_keys_npi,
            list_npi_parameter_keys,
            show_parameter_mapping,
            widget_types,
            min_values,
            max_values,
            steps,
            helpers,
            formats,
            session_state_keys2,
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

            # advanced parameters that are lists or arrays - empty for this app
            advanced_list_keys = []

            # set the advanced parameters for scenario 2
            edited_parms2 = app_editors(
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
                session_state_keys2,
                disabled=False,
            )

            # shared advanced parameters to scenario 1
            # copy the non intervention advanced parameters from scenario 2 to scenario 1
            for key in advanced_ordered_keys:
                # do not copy adherence parameters
                if key not in ("pre_rash_isolation_adherence", "isolation_adherence"):
                    edited_parms1[key] = edited_parms2[key]
            for key in advanced_list_keys:
                # do not copy adherence parameters
                if key not in ("pre_rash_isolation_adherence", "isolation_adherence"):
                    edited_parms1[key] = edited_parms2[key]
        # Final assertions for scenario 1
        # check that the intervention parameters for scenario 1 are still set to zero for interventions
        assert (
            edited_parms1["total_vaccine_uptake_doses"] == 0
        ), "Total vaccine uptake doses should be 0 for the no intervention scenario"
        assert (
            edited_parms1["pre_rash_isolation_adherence"] == 0
        ), "Pre-rash isolation adherence should be 0 for the no intervention scenario"
        assert (
            edited_parms1["isolation_adherence"] == 0
        ), "Isolation adherence should be 0 for the no intervention scenario"

        # --- About this app section ---
        st.header("About this app")
        st.caption(f"metapop version: {info['version']}")
        st.caption(f"commit hash: {info['commit']}")

        # show the url of the repo
        url = info["url"]

        # Choose GitHub logo based on Streamlit browser theme at runtime
        is_light = is_light_color()
        image_path = get_github_logo_path(is_light)

        markdown_content = ""
        markdown_content += f'<a href="{url}" target="_blank">'
        markdown_content += f"{img_to_html(image_path)}"
        markdown_content += "Source code"
        markdown_content += "</a>"

        st.markdown(
            markdown_content,
            help="GitHub Logo from: https://github.com/logos",
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

    # set model parameters based on app inputs - this will update internal parameters that are combinations of user inputs
    # these dictionaries will be used to run the model
    edited_parms1 = update_intervention_parameters_from_widget(edited_parms1)
    edited_parms2 = update_intervention_parameters_from_widget(edited_parms2)

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

    ### Dictate that isolation > quarantine
    if (
        edited_parms2["isolation_adherence"]
        < edited_parms2["pre_rash_isolation_adherence"]
        and edited_parms2["pre_rash_isolation_on"] == True
        and edited_parms2["isolation_on"] == True
    ):
        st.error(
            "Isolation adherence should be greater than or equal to quarantine adherence. "
            "Please adjust the parameters."
        )
        return

    # create two scenarios to run with pygriddler
    scenario1 = [edited_parms1]
    scenario2 = [edited_parms2]

    # --- Initial Population State and Vaccination Schedule ---

    # check the initial population state and see if conditions allow for vaccination to run
    # build population state vectors, first elements are state vectors over time

    # the last element of the first axis is the complete population state vector, "u"
    # the first element of the 2nd axis is for group 0
    # the first element of the 3rd axis is for the susceptible population
    initial_states = initialize_population(
        1,
        1,
        edited_parms2,
    )

    # last element returned by initialize_population is the initial state vector
    u_ind = Ind.max_value() + 1

    warning_message = ""

    # Warn if no one to vaccinate and no other interventions. This effectively means that there are
    #  no other  interventions being modeled in the second scenario.
    if (
        (initial_states[u_ind][0][Ind.S.value] == 0)
        and (not scenario2[0]["vaccine_uptake"])
        and (not scenario2[0]["isolation_on"])
        and (not scenario2[0]["pre_rash_isolation_on"])
    ):
        warning_message += (
            "With these initial conditions, there are no people to vaccinate in this population and other interventions are turned off. "
            "Please turn on isolation or quarantine or adjust the population size or baseline immunity."
            "\n\n"
        )

    # Build vaccine schedule and warn if no doses will be administered
    # create a parameter dictionary of scenario 2 to calculate and expose the vaccine schedule
    intervention_parms2 = edited_parms2.copy()
    intervention_parms2["t_array"] = get_time_array(intervention_parms2)
    schedule = build_vax_schedule(intervention_parms2)

    # if doses are zero, warn the user
    if sum(schedule.values()) == 0 and scenario2[0]["vaccine_uptake"]:
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

    use_cache = os.environ.get("USE_CACHE", "true").strip().lower() == "true"
    results1 = get_scenario_results(scenario1, use_cache)
    results2 = get_scenario_results(scenario2, use_cache)

    # Display number of doses administered
    if scenario2[0]["vaccine_uptake"]:
        with col_intervention_text:
            st.text(
                f"Total vaccines scheduled to be administered during campaign: {sum(schedule.values())} doses",
                help="This number is calculated based on user input for the percentage of the non-immune population that gets vaccinated during the campaign. If the campaign starts late, the actual number of doses administered may be lower due to there being not enough eligible individuals left to vaccinate.",
            )

    # Save a copy of the full results for summary tables so that results1 and results2 can be modified for visualization
    fullresults1 = copy.deepcopy(results1)
    fullresults2 = copy.deepcopy(results2)

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
    app_column_mapping = {
        f"inc_{interval}": "Weekly Incidence",
        "Y": "Weekly Cumulative Incidence",
    }
    interval_results1 = interval_results1.rename(app_column_mapping)
    interval_results2 = interval_results2.rename(app_column_mapping)

    # Rename columns in daily results for app display after results have been calculated
    app_column_mapping = {"Y": "Cumulative Incidence"}
    results1 = results1.rename(app_column_mapping)
    results2 = results2.rename(app_column_mapping)

    # Default to cumulative daily incidence if outcome not available
    if outcome not in [
        "Cumulative Incidence",
        "Incidence",
        "Weekly Incidence",
        "Weekly Cumulative Incidence",
    ]:
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
        if edited_parms2["pre_rash_isolation_on"]:
            pre_rash_isolation_adherence = edited_parms2["pre_rash_isolation_adherence"]
        if edited_parms2["isolation_on"]:
            isolation_adherence = edited_parms2["isolation_adherence"]
        pre_rash_isolation_adherence_pct = int(pre_rash_isolation_adherence * 100)
        isolation_adherence_pct = int(isolation_adherence * 100)

        pre_rash_isolation_end_day = (
            edited_parms2["pre_rash_isolation_start_day"]
            + edited_parms2["pre_rash_isolation_duration_days"]
        )
        pre_rash_isolation_end_day = min(
            edited_parms2["tf"], pre_rash_isolation_end_day
        )

        symptomatic_isolation_end_day = (
            edited_parms2["symptomatic_isolation_start_day"]
            + edited_parms2["symptomatic_isolation_duration_days"]
        )
        symptomatic_isolation_end_day = min(
            edited_parms2["tf"], symptomatic_isolation_end_day
        )

        mean_doses_administered = round(
            results2.filter(pl.col("t") == results2["t"].max())
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

    # base Altair chart
    chart = alt.Chart(combined_alt_results).encode(x=alt.X(x, title=time_label))

    no_intervention_color = "#fb7e38"  # orange
    # intervention_color = "#20419a"  # blue
    intervention_color = "#0057b7"  # blue
    # vaccine_campaign_color = "#ffe100"  # yellow
    vaccine_campaign_color = "#209a79"

    trajectories = (
        chart.mark_line(opacity=0.35, strokeWidth=0.75, clip=True)
        .encode(
            y=alt.Y(outcome, title=outcome_option),
            color=alt.Color(
                "scenario",
                title="Scenario",
                scale=alt.Scale(
                    domain=[scenario_names[0], scenario_names[1]],
                    range=[no_intervention_color, intervention_color],
                ),
                legend=None,
            ),
            detail="replicate",
        )
        .properties(title=title, width=800, height=400)
    )

    #  Add bold line for median trajectory
    ave_chart = alt.Chart(combined_ave_results).encode(x=alt.X(x, title=time_label))
    ave_line = ave_chart.mark_line(opacity=1.0, strokeWidth=3.0, clip=True).encode(
        y=alt.Y(outcome, title=outcome_option),
        color=alt.Color(
            "scenario",
            title="Scenario",
            scale=alt.Scale(
                domain=[scenario_names[0], scenario_names[1]],
                range=[
                    no_intervention_color,
                    intervention_color,
                ],
            ),
        ),
    )

    # Add vaccine campaign period as a shaded box if applicable
    if edited_parms2["total_vaccine_uptake_doses"] > 0:
        # Draw two vertical dashed lines for the start and end of the vaccine campaign
        vax_df = pd.DataFrame(
            {
                "x_start": [vax_start],
                "x_end": [vax_end],
                "Intervention": ["Vaccine campaign period"],
            }
        )
        # Transparent window for campaign period, with legend label
        vax_window = (
            alt.Chart(vax_df)
            .mark_rect(
                opacity=0.33,
                color=vaccine_campaign_color,
                stroke=vaccine_campaign_color,
                strokeWidth=1.5,
            )
            .encode(
                x=alt.X("x_start:Q", title=time_label),
                x2="x_end:Q",
                color=alt.Color(
                    "Intervention:N",
                    legend=alt.Legend(title="Vaccine Campaign"),
                    scale=alt.Scale(
                        domain=["Vaccine campaign period"],
                        range=[vaccine_campaign_color],
                    ),
                ),
            )
        )
        # Vertical lines for campaign start/end
        vax_lines_df = pd.DataFrame(
            {
                "x": [vax_start, vax_end],
                "Intervention": ["Vaccine campaign start", "Vaccine campaign end"],
            }
        )
        vax_lines = (
            alt.Chart(vax_lines_df)
            .mark_rule(
                color=vaccine_campaign_color,
                strokeDash=[6, 4],
                strokeWidth=2,
            )
            .encode(
                x=alt.X("x:Q", title=time_label),
                # Remove color encoding to avoid legend entry
            )
        )
        vax = vax_window + vax_lines
    else:
        # If no vaccine campaign, set vax to an empty chart
        vax = (
            alt.Chart(pd.DataFrame({"x": []}))
            .mark_line()
            .encode(x=alt.X("x:Q", title=time_label))
        )

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
    else:
        # Add annotation for the vaccine campaign period
        annotation = (
            alt.Chart(pd.DataFrame({"text": [""]}))
            .mark_text(align="center", baseline="top", color="grey", fontSize=18)
            .encode(text="text:N", y=alt.value(10))
        )

    layer = alt.layer(vax, trajectories, ave_line, annotation).resolve_scale(
        color="independent"
    )
    chart_placeholder.altair_chart(layer, use_container_width=True)

    # --- Chart Description ---
    st.markdown(
        '<p style="font-size:14px;">'
        "Each thin line represents counts of new daily or weekly rash onsets "
        "from an individual simulation of the stochastic model. "
        "Introduced infections arrive in the community on day 1 and have an average rash onset "
        "time on day 5, the first day at which interventions can begin. "
        "All simulations within a given scenario (i.e., shown with "
        "the same color) are run under the same set of parameters, and "
        "differences between each individual simulation are due to random "
        "variation in contact rates. Bolded lines show the simulation that possessed "
        "the median time of peak prevalence across all epidemic trajectories for "
        "each scenario. If a vaccination campaign is activated, the time period over "
        "which vaccines are distributed is shown by a shaded window between two dashed lines. The model does not account "
        "for case ascertainment, so the number of new rash onsets represents the true number of infections "
        "in the population. "
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Outbreak Summary Stats ---
    st.subheader("Simulation summary")
    with st.expander("Show intervention strategies", expanded=True):
        columns = st.columns(2)
        # Create a callout box and text describing the No Interventions scenario
        flexible_callout(
            (
                "No Interventions:<br><ul>"
                f"<li> Vaccines administered during campaign: {int(edited_parms1['total_vaccine_uptake_doses'])}</li>"
                f"<li> Adherence to isolation among symptomatic infectious individuals:  {int(edited_parms1['isolation_adherence'] * 100)}%</li>"
                f"<li> Adherence to quarantine among pre-symptomatic infectious individuals: {int(edited_parms1['pre_rash_isolation_adherence'] * 100)}%</li></ul>"
            ),
            background_color="#feeadf",
            font_color="#8f3604",
            container=columns[0],
        )
        # Create a callout box and text describing the interventions
        # implemented in the Interventions scenario when it's being displayed (i.e. "On")
        if interventions == "On":
            callout_text = "Interventions:<br><ul>"
            if (
                edited_parms2["total_vaccine_uptake_doses"] == 0
                or edited_parms2["vaccine_uptake_duration_days"] == 0
            ):
                callout_text += "<li> Vaccines administered during campaign: 0</li>"
            else:
                callout_text += (
                    f"<li> Vaccines administered during campaign: {mean_doses_administered} "
                    f"between day {min(schedule.keys())} and day {max(schedule.keys())}</li>"
                )
            callout_text += f"<li> Adherence to isolation among symptomatic infectious individuals: {isolation_adherence_pct}%"
            if isolation_adherence_pct > 0:
                callout_text += (
                    f" from day {edited_parms2['symptomatic_isolation_start_day']+1} "
                    f"through day {symptomatic_isolation_end_day}</li>"
                )
            else:
                callout_text += "</li>"
            callout_text += f"<li> Adherence to quarantine among pre-symptomatic infectious individuals: {pre_rash_isolation_adherence_pct}%"
            if pre_rash_isolation_adherence_pct > 0:
                callout_text += (
                    f" from day {edited_parms2['pre_rash_isolation_start_day']+1} "
                    f"through day {pre_rash_isolation_end_day}</li>"
                )
            else:
                callout_text += "</li>"
            callout_text += f"<i>All intervention start times are relative to when infections are introduced into the community (day 1). The minimum intervention start time is day 5 (when introduced infections have rash onset, on average).</i>"

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
        combined_results,
        edited_parms2["IHR"],
        hosp_rng,
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
            .item()
            .split("%")[0]
        )

        intervention_text = f"Adding "
        interventions = []
        if edited_parms2["total_vaccine_uptake_doses"] > 0:
            interventions.append("vaccination")
        if edited_parms2["pre_rash_isolation_success"] > 0:
            interventions.append("quarantine")
        if edited_parms2["isolation_success"] > 0:
            interventions.append("isolation")
        if len(interventions) > 1:
            intervention_text += (
                ", ".join(interventions[:-1]) + " and " + interventions[-1]
            )
        elif interventions:
            intervention_text += interventions[0]

        st.text(
            f"{intervention_text} decreases total measles infections by {relative_difference}% "
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
            </p>

            <p style="font-size:14px;">
            People who are immune at the beginning of the simulation, either
            through past vaccination or previous infection, begin in the
            "Vaccinated" compartment. Individuals who have received the vaccine
            but are not immune (due to vaccine failure) are tracked in a separate
            "Vaccinated but Susceptible" compartment, and have the same
            susceptibility as individuals in the "Susceptible" compartment.
            </p>

            <p style="font-size:14px;">
            Users can explore the impact of interventions, including vaccination,
            isolation, and quarantine measures ("Interventions" scenario)
            compared to a baseline scenario without active interventions ("No
            Interventions". The start day and duration of all three intervention
            measures (isolation, quarantine, and vaccination) can be specified by
            the user.
            </p>

            <p style="font-size:14px;">
            In this model, day 1 corresponds to when introduced infections arrive in the community.
            Introduced infections are assumed to arrive in their pre-rash infectious stage and are
            modeled to become symptomatic, on average, 4 days later on day 5.
            In this model day 5 is the earliest day most communities would be aware of measles cases
            and begin public health interventions.
            </p>

            <p style="font-size:14px;">
            We show the estimated difference between total infection in both
            scenarios relative to the mean values from the no intervention
            scenario and round to the nearest integer percentage, doing the
            same for total hospitalizations.
            We then conduct a two-sample K-S test to determine if the
            total measles infections from the "Interventions" scenario differ
            from the total measles infections of the "No Interventions" baseline
            scenario and present information if scenario results are
            indistinguishable.
            </p>

            <p style="font-size:14px;">
            This model does not account for case ascertainment of infections,
            which may vary over time during an outbreak.
            </p>

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

            <li style="font-size:14px;"> Following the incubation period,
            we assume that infected individuals first enter a pre-rash infectious state
            before developing a rash and entering a separate infectious state defined by rash onset.
            We assume that the duration of time spent in each of these two states is the same,
            with a mean of 4.5 days each.
            </li>

            <li style="font-size:14px;"> To incorporate quarantine measures based on
            potential exposure to infectious individuals,
            we model quarantine as a reduction in transmission from individuals in the pre-rash
            onset, infectious compartment. Specifically, we assume that a proportion of these
            individuals have their transmission potential fully reduced to zero.
            We assume isolation acts in a similar fashion to reduce transmission of infectious individuals
            that have a rash.
            The proportion of individuals whose transmission is reduced to zero by
            quarantine or isolation is defined by the adherence parameters
            in the sidebar and multiplied by the efficacy of each of these interventions, given
            in the Parameters section below.
            </li>

            <li style="font-size:14px;"> If individuals quarantine prior to rash onset, it's assumed
            that they will also isolate on rash onset. Thus, the model is constrained such that
            the proportion of individuals adhering to quarantine is less than or equal
            to the proportion of individuals adhering (and that isolation must be used for quarantine
            to be used).
            </li>

            <b style="font-size:14px;">Model Parameters</b>
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
