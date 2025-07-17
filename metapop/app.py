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
import toml

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
    st.set_page_config(page_title="CDC Measles Outbreak Simulator", layout="wide")
    st.title("CDC Measles Outbreak Simulator")

    # Show development warning if not in production
    if os.environ.get("MODE", "PRODUCTION") == "DEVELOPMENT":
        st.warning(f"Building from dev branch")

    st.text(
        "This interactive tool helps public health decision-makers to "
        "explore the potential impact of three key public health "
        "interventions - isolation, quarantine, and vaccination - on "
        "measles outbreaks. A summary of the model and how public health "
        "interventions can prevent or slow the spread of measles in "
        "communities is available for download."
    )

    # Load default parameters from YAML config
    filepath = os.path.join(
        os.path.dirname(__file__), "app_assets", "onepop_config.yaml"
    )

    # Load primary color
    slpath = os.path.join(os.getcwd(), ".streamlit", "config.toml")

    slconfig = toml.load(slpath)
    primary_color = slconfig["theme"]["primaryColor"]

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

    scenario_names = ["No interventions", "Interventions"]
    show_parameter_mapping = get_show_parameter_mapping(parms)
    advanced_parameter_mapping = get_advanced_parameter_mapping()

    # --- Sidebar: Model Inputs and Parameter Editing ---
    with st.sidebar:
        st.text(
            "This simulator is designed for use in the early stages of an outbreak to "
            "understand the potential impact of isolation, quarantine, and vaccination "
            "on the size of a measles outbreak and is most applicable at the county "
            "level or smaller geographic scale. "
            "It is not an exact forecast of measles infections in any community. "
            "Hover over the ? icon for more information about each parameter, and "
            "read our Behind the Model to learn more about the modeling methods "
            "(link to Behind the Model to be added after clearance approval). "
        )

        # Get widget types, min/max values, steps, helpers, formats, and keys for widgets
        widget_types = get_widget_types()
        min_values = dict(
            pop_sizes=[1000, 100, 100],
            I0=[1, 0, 0],
            vaccine_uptake_start_day=0,
            symptomatic_isolation_start_day=0,
            pre_rash_isolation_start_day=0,
        )
        min_values = get_min_values(min_values)
        max_values = dict(
            initial_vaccine_coverage=[0.99, 0.99, 0.99],
            vaccine_uptake_start_day=180,
            pre_rash_isolation_start_day=180,
            symptomatic_isolation_start_day=180,
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
        # The No interventions scenario parameters are not shown in the sidebar panel,
        # so it is possible that the keys for this scenario are not used.

        session_state_keys0 = get_session_state_idkeys(0)  # shared parameters
        session_state_keys1 = get_session_state_idkeys(1)  # scenario 1
        session_state_keys2 = get_session_state_idkeys(2)  # scenario 2

        # Customize helper texts for clarity
        helpers["I0"][0] = (
            "The model currently allows for a maximum of 10 initial introductions in the population. "
            "This value represents recent importations of people who are infectious transmitting to others at the beginning of the outbreak. "
            "It is not meant to represent the total number of infections reported in a population to date. "
        )
        helpers["pop_sizes"][0] = (
            "The model currently has a minimum of 1,000 people and a maximum of 100,000 people and assumes a well-mixed (homogeneous) population. "
            "As population sizes get larger, the assumption that everyone in a community is equally likely to interact with "
            "everyone else in the community becomes less valid and the model might be less appropriate."
        )
        helpers["initial_vaccine_coverage"][0] = (
            "The percent of the population with prior immunity to measles, including through either MMR vaccination or through past infection."
        )

        # Add a section for a Reset button for all parameters
        col_reset = st.columns(1)[0]

        # --- Shared Parameters in order of appearance in the app ---
        shared_keys = [
            "pop_sizes",
            "initial_vaccine_coverage",
            "I0",
        ]
        # shared parameters that are lists
        shared_list_keys = [
            "pop_sizes",
            "initial_vaccine_coverage",
            "I0",
        ]

        # add a section for the shared parameters in the sidebar panel
        col0 = st.columns(1)[0]

        col0.subheader(
            "Population characteristics",
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
            "Interventions",
            help="The adherence to isolation and quarantine, "
            "vaccine uptake, and the start time, and duration of each intervention "
            "can be specified.",
        )
        # placeholder for the introduction text about interventions
        col_intervention_intro = st.columns(1)[0]

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

        ave_first_rash_onset = np.floor(edited_parms2["infectious_duration"] / 2 + 1)

        intervention_intro = (
            "Choose interventions to compare with a "
            "scenario with no active interventions. Interventions can be applied "
            "alone or in combination. "
            "The results are compared to a no intervention scenario with no "
            "isolation, quarantine, or vaccination campaign. "
            "In this model, day 1 corresponds to when infected people are introduced "
            f"into the community. "
        )
        col_intervention_intro.text(intervention_intro)

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
        markdown_content += f'{img_to_html(image_path, "GitHub Logo")}'
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
            or edited_parms2["vaccine_uptake"] == False
        )
        and (
            edited_parms2["pre_rash_isolation_on"] == False
            or edited_parms2["pre_rash_isolation_adherence"] == 0
        )
        and (
            edited_parms2["isolation_on"] == False
            or edited_parms2["isolation_adherence"] == 0
        )
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
                "Total vaccines scheduled to be administered during campaign: "
                f"{sum(schedule.values())} doses",
                help=(
                    "This number is calculated based on user input for the "
                    "percentage of the population without prior immunity that is "
                    "vaccinated during the campaign. If the campaign starts "
                    "after a substantial number of new infections have occurred in "
                    "the simulation, the actual number of doses administered may be "
                    "lower due to a limited number of non-immune individuals "
                    "remaining."
                ),
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
        f"inc_{interval}": "Weekly incidence",
        "Y": "Weekly cumulative incidence",
    }
    interval_results1 = interval_results1.rename(app_column_mapping)
    interval_results2 = interval_results2.rename(app_column_mapping)

    # Rename columns in daily results for app display after results have been calculated
    app_column_mapping = {"Y": "Cumulative incidence"}
    results1 = results1.rename(app_column_mapping)
    results2 = results2.rename(app_column_mapping)

    # Default to cumulative daily incidence if outcome not available
    if outcome not in [
        "Cumulative incidence",
        "Incidence",
        "Weekly incidence",
        "Weekly cumulative incidence",
    ]:
        print("outcome not available yet, defaulting to 'Daily cumulative incidence'")
        outcome = "Y"

    # Prepare data for Altair plots
    if outcome_option in [
        "Daily incidence",
        "Daily cumulative incidence",
    ]:
        alt_results1 = results1
        alt_results2 = results2
        x = "t:Q"
        time_label = "Time (days)"
        vax_start = min(schedule.keys())
        vax_end = max(schedule.keys())
    elif outcome_option in ["Weekly incidence", "Weekly cumulative incidence"]:
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
        if len(schedule) > 0:
            dose_per_day = np.sum(list(schedule.values())) / len(schedule)
        else:
            dose_per_day = 0

        dose_per_day_text = ""
        if np.round(dose_per_day) > 1:
            dose_per_day_text = (
                f"(equivalent to around {np.round(dose_per_day):.0f} doses per day)"
            )
        elif np.round(dose_per_day) == 1:
            dose_per_day_text = "(equivalent to around 1 dose per day)"
        elif np.round(dose_per_day) < 1:
            dose_per_day_text = "(equivalent to less than 1 dose per day)"

        title = alt.TitleParams(
            "Simulated measles epidemic curve with and without public health interventions",
            subtitle=[
                f"Population size: {edited_parms2['pop_sizes'][0]} people, {edited_parms2['I0'][0]} initial introductions",
                f"Vaccine campaign: {mean_doses_administered} doses administered",
                f"Isolation adherence: {isolation_adherence_pct}%",
                f"Quarantine adherence: {pre_rash_isolation_adherence_pct}%",
            ],
            subtitleColor="#808080",
        )
    else:
        combined_alt_results = alt_results1.filter(
            pl.col("replicate").is_in(replicate_inds)
        )
        combined_ave_results = ave_results1
        title = "No intervention scenario"

    chart_placeholder.text("Building charts...")

    # base Altair chart
    chart = alt.Chart(combined_alt_results).encode(x=alt.X(x, title=time_label))

    no_intervention_color = "#fb7e38"  # orange
    intervention_color = "#0057b7"  # blue
    vaccine_campaign_color = "#909090"  # grey

    trajectories = (
        chart.mark_line(opacity=0.35, strokeWidth=1.0, clip=True)
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
            tooltip=[],  # Empty tooltip for the trajectories
        )
        .properties(title=title, width=800, height=400)
    )

    #  Add bold line for median trajectory
    ave_chart = alt.Chart(combined_ave_results).encode(x=alt.X(x, title=time_label))
    ave_line = ave_chart.mark_line(opacity=1.0, strokeWidth=4.0, clip=True).encode(
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
        tooltip=[],  # Empty tooltip for the median line
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
                opacity=0.18,
                color=vaccine_campaign_color,
                stroke=vaccine_campaign_color,
                strokeWidth=1.5,
            )
            .encode(
                x=alt.X("x_start:Q", title=time_label),
                x2="x_end:Q",
                color=alt.Color(
                    "Intervention:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=["Vaccine campaign period"],
                        range=[vaccine_campaign_color],
                    ),
                ),
                tooltip=[],  # Empty tooltip for the shaded window
            )
        )
        # Create a dummy DataFrame to be used so that the vaccine window is
        # shown in the legend but the color associated does not have the same
        # opacity as the shaded window and instead is a bit darker
        # streamlit and altair do not allow for a legend entry to have a
        # different opacity than the actual mark, so we create a dummy DataFrame
        # with the same color as the vaccine campaign window but with no data
        # to be used in the legend
        dummy_vax_df = pd.DataFrame(
            {
                "x_start": [np.nan],
                "x_end": [np.nan],
                "Intervention": ["Vaccine campaign period"],
            }
        )

        dummy_vax_window = (
            alt.Chart(dummy_vax_df)
            .mark_rect(
                opacity=0.9,
                color=vaccine_campaign_color,
                strokeWidth=1.5,
            )
            .encode(
                x=alt.X("x_start:Q", title=time_label),
                x2="x_end:Q",
                color=alt.Color(
                    "Intervention:N",
                    legend=alt.Legend(title="Vaccine campaign"),
                    scale=alt.Scale(
                        domain=["Vaccine campaign period"],
                        range=[vaccine_campaign_color],
                    ),
                ),
                tooltip=[],  # Empty tooltip for the dummy window
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
                tooltip=[],  # Empty tooltip for the vertical lines
            )
        )
        vax = vax_window + vax_lines
    else:
        # If no vaccine campaign, set vax to an empty chart
        vax = (
            alt.Chart(pd.DataFrame({"x": []}))
            .mark_line()
            .encode(
                x=alt.X("x:Q", title=time_label),
                tooltip=[],  # Empty tooltip for the empty chart
            )
        )

        dummy_vax_window = (
            alt.Chart(pd.DataFrame({"x": []}))
            .mark_line()
            .encode(
                x=alt.X("x:Q", title=time_label),
                tooltip=[],  # Empty tooltip for the dummy window
            )
        )

    # Add annotation if no interventions are selected
    if interventions == "Off":
        annotation = (
            alt.Chart(
                pd.DataFrame(
                    {"text": ["Use at least one intervention to compare scenarios"]}
                )
            )
            .mark_text(align="center", baseline="top", color=primary_color, fontSize=18)
            .encode(text="text:N", y=alt.value(10))
        )
    else:
        # Add annotation for the vaccine campaign period
        annotation = (
            alt.Chart(pd.DataFrame({"text": [""]}))
            .mark_text(align="center", baseline="top", color="grey", fontSize=18)
            .encode(text="text:N", y=alt.value(10))
        )

    layer = alt.layer(
        vax, trajectories, ave_line, dummy_vax_window, annotation
    ).resolve_scale(color="independent")
    chart_placeholder.altair_chart(layer, use_container_width=True)

    # --- Chart Description ---
    st.markdown(
        '<p style="font-size:14px;">'
        "Each thin line represents an individual simulation. The model runs 100 simulations "
        "for each scenario to generate results, and 20 randomly selected simulations are "
        "plotted here. All simulations for a given scenario (i.e., shown with "
        "the same color) are run under the same set of parameters, and "
        "differences between each individual simulation are due to random "
        "variation in contact rates. Read more about our modeling methods (link to Behind the Model to be added after clearance approval). "
        "Bolded lines show the simulation closest to the median time of peak prevalence across all epidemic trajectories for "
        "each scenario. If a vaccination campaign is modeled, the time period over "
        "which vaccines are distributed is shown by the shaded box. "
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Outbreak Summary Stats ---
    st.subheader("Simulation summary")
    with st.expander("Show intervention strategies", expanded=True):
        columns = st.columns(2)
        # Create a callout box and text describing the No interventions scenario
        flexible_callout(
            (
                "No interventions:<br><ul>"
                "<li>In the no intervention scenario, there is no isolation for individuals "
                "showing measles-specific symptoms, no quarantine of people exposed to "
                "measles with no evidence of prior immunity, and no vaccination campaign "
                "for people with no prior immunity.</li></ul>"
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
                    f"between day {min(schedule.keys())} and day {max(schedule.keys())} "
                    f"{dose_per_day_text}</li>"
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
            callout_text += f"</ul><em>All intervention start times are relative to when infections are introduced into the community (day 1).</em>"

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
            outbreak_summary.filter(
                pl.col("") == "Infections, median (95% prediction interval)"
            )
            .select("Relative difference (%)")
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

        # if the Relative difference is NaN, set it to ""
        # outbreak_summary = outbreak_summary.with_columns(
        #     pl.col("Relative difference (%)").fill_nan("")
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
            f"""
            For a more detailed description of the methods and parameters used in this model,
            see our Behind The Model (link to Behind the Model to be added after clearance approval).
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    app()
