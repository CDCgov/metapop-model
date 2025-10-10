# This file contains helper functions for the metapop app.
import base64
import copy
import datetime
import io
import os
import uuid
from pathlib import Path

import altair as alt
import griddler
import griddler.griddle
import numpy as np
import polars as pl
import scipy.stats as stats
import streamlit as st

# import what's needed from other metapop modules
from .sim import simulate_replicates

# if you want to use methods from metapop in this file under
# if __name__ == "__main__": you'll need to import them as:
# from metapop.sim import simulate
### note: this is not recommended use within a file that is imported as a package module, but it can be useful for testing purposes

__all__ = [
    "get_scenario_results",
    "get_scenario_results_cached",
    "read_parameters",
    "get_default_full_parameters",
    "get_default_show_parameters_table",
    "get_advanced_parameters_table",
    "get_show_parameter_mapping",
    "get_advanced_parameter_mapping",
    "get_outcome_options",
    "get_outcome_mapping",
    "get_list_keys",
    "get_keys_in_list",
    "repack_list_parameters",
    "app_editors",
    "get_min_values",
    "get_max_values",
    "get_step_values",
    "get_helpers",
    "get_formats",
    "get_base_session_state_idkeys",
    "get_session_state_idkeys",
    "get_parameter_key_for_session_key",
    "generate_random_key",
    "reset",
    "update_coverage",
    "edit_baseline_immunity",
    "calc_immunity",
    "button_to_calculate_immunity",
    "apply_calculated_immunity",
    "get_parms_from_table",
    "update_parms_from_table",
    "correct_parameter_types",
    "update_intervention_parameters_from_widget",
    "add_daily_incidence",
    "get_interval_cumulative_incidence",
    "get_interval_results",
    "create_chart",
    "set_parms_to_zero",
    "rescale_prop_vax",
    "get_median_trajectory_from_episize",
    "get_median_trajectory_from_peak_time",
    "totals_same_by_ks",
    "img_to_bytes",
    "img_to_html",
    "is_light_color",
    "get_github_logo_path",
    "render_chart_title",
    "combine_incidence_results",
    "csv_download_button",
]

CACHE_TTL = 60 * 60 * 24 * 7  # 1 week in seconds


@st.cache_data(show_spinner=False, ttl=CACHE_TTL, max_entries=20)
def get_scenario_results_cached(parms):
    return get_scenario_results(parms)


### Methods to simulate the model for the app ###
def get_scenario_results(parms, use_cache=False):
    """
    Run simulations for a grid set of parameters and return the combined
    results Dataframe.

    Args:
        parms        (list): List of dictionaries containing model parameters.
        scenario_name (str): Name of the scenario to run.

    Returns:

    """
    if use_cache:
        return get_scenario_results_cached(parms)

    results = simulate_replicates(parms)
    # cast group to string
    results = results.with_columns(pl.col("group").cast(pl.Utf8))
    # select subset of values to return
    results = results.select(
        [
            "k_21",
            "t",
            "group",
            "S",
            "V",
            "SV",
            "E1",
            "E2",
            "E1_V",
            "E2_V",
            "I1",
            "I2",
            "R",
            "Y",
            "X",
            "replicate",
        ]
    )
    # add a column for total infections
    results = results.with_columns((pl.col("I1") + pl.col("I2")).alias("I"))
    return results


### Methods to read in default parameters ###
def read_parameters(filepath="scripts/app/app_config.yaml"):
    """
    Read parameters from a YAML file and return the first set of parameters.

    Args:
        filepath (str): Path to the YAML file containing the model parameters.

    Returns:
        dict: A dictionary of parameters for the metapopulation model.
    """
    parameter_sets = griddler.griddle.read(filepath)
    parms = parameter_sets[0]
    return parms


def get_default_full_parameters():
    """
    Read in the default parameters for the metapopulation model from a YAML
    file and return it as a Dataframe for two scenarios to be updated with user
    input.

    Returns:
        pl.DataFrame: DataFrame containing the default parameters and their
        values for two scenarios.
    """
    # read in parms, some of which are lists
    filepath = os.path.join(os.path.dirname(__file__), "app_assets", "app_config.yaml")
    parms = read_parameters(filepath)

    # get keys that are lists, unpack them and add to the dictionary
    list_keys = get_list_keys(parms)

    for key in list_keys:
        for i, value in enumerate(parms[key]):
            parms["{}_{}".format(key, i)] = value

        del parms[key]

    keys = [key for key in parms.keys()]
    values = [parms[key] for key in keys]

    defaults = pl.DataFrame(
        {
            "Parameter": keys,
            "No interventions": values,
            "Interventions": values,
        },
        strict=False,
    )
    return defaults


def get_default_show_parameters_table():
    """
    Get a Dataframe of the default simulation parameters that users always see
    in the app sidebar. This Dataframe contains default values for two
    scenarios that can be updated by the user through other methods.

    Returns:
        pl.DataFrame: DataFrame containing the default parameters and their
        values for two scenarios.
    """

    full_defaults = get_default_full_parameters()
    show_parameter_mapping = get_show_parameter_mapping()
    show_defaults = full_defaults.filter(
        pl.col("Parameter").is_in(show_parameter_mapping.keys())
    )

    # replace specific values with integers
    for key in ["No interventions", "Interventions"]:
        show_defaults = show_defaults.with_columns(
            pl.when(pl.col(key).str.to_lowercase() == "true")
            .then(1)
            .when(pl.col(key).str.to_lowercase() == "false")
            .then(0)
            .otherwise(pl.col(key))
            .alias(key)
        )

    # cast to float
    show_defaults = show_defaults.with_columns(
        pl.col("No interventions").cast(pl.Float64)
    )
    show_defaults = show_defaults.with_columns(pl.col("Interventions").cast(pl.Float64))

    # renaming keys with longer names
    show_defaults = show_defaults.with_columns(
        pl.Series(
            name="Parameter",
            values=[
                show_parameter_mapping.get(key)
                for key in show_defaults["Parameter"].to_list()
            ],
        )
    )

    return show_defaults


def get_advanced_parameters_table():
    full_defaults = get_default_full_parameters()
    show_parameter_mapping = get_show_parameter_mapping()
    advanced_parameter_mapping = get_advanced_parameter_mapping()
    advanced_defaults = full_defaults.filter(
        ~pl.col("Parameter").is_in(show_parameter_mapping.keys())
        & pl.col("Parameter").is_in(advanced_parameter_mapping.keys())
    )

    # replace specific values with integers
    for key in ["No interventions", "Interventions"]:
        advanced_defaults = advanced_defaults.with_columns(
            pl.when(pl.col(key).str.to_lowercase() == "true")
            .then(1)
            .when(pl.col(key).str.to_lowercase() == "false")
            .then(0)
            .otherwise(pl.col(key))
            .alias(key)
        )

    # cast to float
    advanced_defaults = advanced_defaults.with_columns(
        pl.col("No interventions").cast(pl.Float64)
    )
    advanced_defaults = advanced_defaults.with_columns(
        pl.col("Interventions").cast(pl.Float64)
    )

    # renaming keys with longer names
    advanced_defaults = advanced_defaults.with_columns(
        pl.Series(
            name="Parameter",
            values=[
                advanced_parameter_mapping.get(key)
                for key in advanced_defaults["Parameter"].to_list()
            ],
        )
    )
    return advanced_defaults


### Methods to handle how parameters are displayed in the app ###
def get_show_parameter_mapping(parms=None):
    """
    Get a mapping of parameter names to their display names.

    Args:
        parms (dict, optional): A dictionary of parameters. If provided, it will
            adjust the display names based on the number of groups specified.

    Returns:
        dict: A dictionary mapping parameter names to their display names.
    """
    # Define the mapping of parameter names to display names
    show_mapping = dict(
        # n_groups = "number of groups",
        # desired_r0="R0",
        # k = "Average degree",
        # k_i_0 = "Average degree per person in large population",
        # k_i_1 = "Average degree per person in small population 1",
        # k_i_2 = "Average degree per person in small population 2",
        # k_g1 = "Average degree of small population 1 connecting to large population",
        # k_g2 = "Average degree of small population 2 connecting to large population",
        # k_21 = "Connectivity between smaller populations",        # n_e_compartments = "Number of exposed compartments",
        # latent_duration = "Latent period (days)",
        # n_i_compartments = "Number of infectious compartments",
        # infectious_duration = "Infectious period (days)",
        pop_sizes_0="Size of large population",
        pop_sizes_1="Size of small population 1",
        pop_sizes_2="Size of small population 2",
        I0_0="Infections introduced in the large population",
        I0_1="Infections introduced in small population 1",
        I0_2="Infections introduced in small population 2",
        vaccine_uptake="Enable vaccination campaign",
        total_vaccine_uptake_doses="Percent of people without prior immunity that get vaccinated",
        vaccine_uptake_start_day="Start of vaccination campaign (days after introduction)",
        vaccine_uptake_duration_days="Duration of vaccination campaign (days)",
        vaccinated_group="Vaccinated group",
        calculator_on="Enable Baseline Immunity Calculator",
        isolation_on="Enable isolation",
        isolation_adherence="Isolation adherence",
        isolation_reduction="Reduction in transmission due to isolation",
        symptomatic_isolation_start_day="Start of isolation intervention (days after introduction)",
        symptomatic_isolation_duration_days="Duration of isolation intervention (days)",
        pre_rash_isolation_on="Enable quarantine ",
        pre_rash_isolation_adherence="Quarantine adherence",
        pre_rash_isolation_reduction="Reduction in transmission due to quarantine",
        pre_rash_isolation_start_day="Start of quarantine intervention (days after introduction)",
        pre_rash_isolation_duration_days="Duration of quarantine intervention (days)",
        tf="Time steps",
        # n_replicates = "Number of replicates",
        # seed = "Random seed",
        initial_vaccine_coverage_0="Baseline immunity in large population",
        initial_vaccine_coverage_1="Baseline immunity in small population 1",
        initial_vaccine_coverage_2="Baseline immunity in small population 2",
        population_percentages_0="Percent of population in first age group",
        population_percentages_1="Percent of population in second age group",
        population_percentages_2="Percent of population in third age group",
        vaccine_coverages_0="Vaccine coverage by first age point",
        vaccine_coverages_1="Vaccine coverage by second age point",
        vaccine_coverages_2="Vaccine coverage by third age point",
    )

    if parms is not None and isinstance(parms, dict):
        if parms["n_groups"] == 1:
            show_mapping["pop_sizes_0"] = "Population size"
            show_mapping["I0_0"] = "Initial introductions"
            show_mapping["initial_vaccine_coverage_0"] = "Baseline immunity"

    return show_mapping


def get_advanced_parameter_mapping():
    """
    Get a mapping of advanced parameter names to their display names.

    Returns:
        dict: A dictionary mapping advanced parameter names to their display names.
    """
    # Define the mapping of advanced parameter names to display names
    advanced_mapping = dict(
        desired_r0=r"$R_0$",
        n_groups="Number of groups",
        infectious_duration="Infectious period (days)",
        latent_duration="Latent period (days)",
        pre_rash_isolation_adherence="Quarantine adherence",
        pre_rash_isolation_reduction="Reduction in transmission due to quarantine",
        isolation_adherence="Isolation adherence",
        isolation_reduction="Reduction in transmission due to isolation",
        # n_e_compartments="Number of exposed compartments",
        # n_i_compartments="Number of infectious compartments",
        # tf="Number of time steps",
        # pop_sizes_0="Size of large population",
        # pop_sizes_1="Size of small population 1",
        # pop_sizes_2="Size of small population 2",
        # k_i = "Average degree",
        k_i_0="Average degree for large population",
        k_i_1="Average degree for small population 1",
        k_i_2="Average degree for small population 2",
        k_g1="Average degree of small population 1 connecting to large population",
        k_g2="Average degree of small population 2 connecting to large population",
        k_21="Connectivity between smaller populations",
        IHR="Infection hospitalization ratio",
    )
    return advanced_mapping


def get_outcome_options():
    """
    Get the available outcome options for the app.

    Returns:
        tuple: A tuple containing the available outcome options.
    """
    return (
        "Weekly incident infections",
        "Weekly cumulative incident infections",
        "Daily incident infections",
        "Daily cumulative incident infections",
    )


def get_outcome_mapping():
    """
    Get a mapping of outcome names to their corresponding codes.

    Returns:
        dict: A dictionary mapping outcome names to their corresponding output
        column name.
    """
    # Define the mapping of outcome names to their corresponding codes
    return {
        "Weekly incident infections": "Weekly incident infections",
        "Weekly cumulative incident infections": "Weekly cumulative incident infections",
        "Daily incident infections": "Daily incident infections",
        "Daily cumulative incident infections": "Daily cumulative incident infections",
    }


### Methods to handle parameter keys based on their value types ###
def get_list_keys(parms):
    """
    Get the keys of parameters that have list values.

    Args:
        parms (dict): A dictionary of model parameters.

    Returns:
        list: The keys of the parameters dictionary that have list values.
    """
    list_keys = [key for key, value in parms.items() if isinstance(value, list)]
    # Sort to ensure deterministic order
    list_keys = sorted(list_keys)
    return list_keys


def get_keys_in_list(parms, updated_parms):
    """
    Get the expanded keys of parameters from updated_parms that map to keys
    that have list values in the parms dictionary.

    Args:
        parms         (dict): The original parameters dictionary.
        updated_parms (dict): The updated parameters dictionary.

    Returns:
        list: The keys of the parameters that are in the list keys of the updated parameters.
    """
    list_keys = get_list_keys(parms)
    keys_in_list = [
        key
        for key in updated_parms.keys()
        if any(key.startswith(list_key) for list_key in list_keys)
    ]
    keys_in_list = [key for key in sorted(keys_in_list)]
    return keys_in_list


def repack_list_parameters(parms, updated_parms, keys_in_list):
    """
    Repack the list parameters in the updated parameters dictionary.

    Args:
        parms         (dict): The original parameters dictionary.
        updated_parms (dict): The updated parameters dictionary.
        keys_in_list  (list): The keys of the parameters that are in the list keys of the updated parameters.

    Returns:
        dict: The updated parameters dictionary with repacked list parameters.
    """
    for key in keys_in_list:
        key_split = key.split("_")
        list_key = "_".join(key_split[:-1])

        if list_key not in updated_parms:
            updated_parms[list_key] = []
        if isinstance(parms[list_key][0], int) and not isinstance(
            parms[list_key][0], bool
        ):
            updated_parms[list_key].append(int(updated_parms[key]))
        elif isinstance(parms[list_key][0], bool):
            updated_parms[list_key].append(
                True
                if updated_parms[key] in [True, "TRUE", "True", "true", "1"]
                else False
            )
        elif isinstance(parms[list_key][0], float):
            updated_parms[list_key].append(float(updated_parms[key]))
        elif isinstance(parms[list_key][0], str):
            updated_parms[list_key].append(str(updated_parms[key]))

    for key in keys_in_list:
        del updated_parms[key]

    return updated_parms


### Set given parameters to zero ###
def set_parms_to_zero(parms, keys_to_set):
    """
    Set specified parameters in the given dictionary to zero.

    Args:
        parms (dict): The original parameters dictionary.
        keys_to_set (list): A list of parameter keys to set to zero.

    Returns:
        dict: A new dictionary with the specified parameters set to zero.
    """
    edited_parms = copy.deepcopy(parms)

    for key in keys_to_set:
        edited_parms[key] = 0.0

    return edited_parms


def rescale_prop_vax(edited_parms):
    """
    This function rescales the total vaccine uptake doses from a percentage
    to an absolute number based on the based on the population sizes,
    initial vaccine coverage, total vaccine uptake doses, and the initial
    number of infections or introductions. This method is used to translate
    user inputs for the total vaccine uptake doses into a value type which the
    metapop model expects to use for scenario specification.

    Args:
        edited_parms (dict): A dictionary of model parameters containing
                             population sizes, initial vaccine coverage,
                             total vaccine uptake doses, and the initial
                             number of infections or introductions.
    Returns:
        dict: An updated model parameters dictionary with rescaled total
        vaccine uptake doses.
    """
    pop_sizes = np.array(edited_parms["pop_sizes"])
    if edited_parms.get("calculator_on", False):
        initial_vaccine_coverage = st.session_state.immunity
    else:
        initial_vaccine_coverage = np.array(edited_parms["initial_vaccine_coverage"])
    prop_vaccine_uptake_doses = edited_parms["total_vaccine_uptake_doses"] / 100.0
    edited_parms["total_vaccine_uptake_doses"] = int(
        (pop_sizes - pop_sizes * initial_vaccine_coverage - edited_parms["I0"])
        * prop_vaccine_uptake_doses
    )
    return edited_parms


### Methods to create user inputs interfaces ###
def app_editors(
    element,
    scenario_name,
    parms,
    ordered_keys,
    list_keys,
    show_parameter_mapping,
    widget_types,
    min_values,
    max_values,
    steps,
    helpers,
    formats,
    element_keys,
    disabled=False,
):
    """
    Create the a Streamlit app section allowing users to modify or edit model
    parameters for a scenario. The section is created within the provided
    Streamlit element. This method also generates a new copy of the parameters
    dictionary with user-modified values and returns it.

    Args:
        element (st container object): The Streamlit element to place the widgets in.
        scenario_name           (str): The name of the scenario.
        parms                  (dict): A dictionary of parameters to modify.
        ordered_keys           (list): An ordered list of the parameters to create widgets for within the Streamlit element.
        list_keys              (list): The keys of the parameters that have list values.
        widget_types           (dict): The types of widget or user interface for each parameter.
        show_parameter_mapping (dict): The mapping of parameter names to display names.
        min_values             (dict): The minimum values for the parameters.
        max_values             (dict): The maximum values for the parameters.
        steps                  (dict): The step sizes for the parameters.
        helpers                (dict): The help text for the parameters.
        formats                (dict): The formats for the parameters.
        element_keys           (dict): The keys for the Streamlit elements. These are also known as session state keys.
        disabled               (bool): Whether the widgets should be disabled. Defaults to False.

    Returns:
        edited_parms: A copy of the parms dictionary with user modified values.
        This method also creates a Streamlit section with widgets for users to
        modify values for each parameter specified in `ordered_keys`.
    """
    edited_parms = copy.deepcopy(parms)

    with element:
        st.subheader(scenario_name)

        for key in ordered_keys:
            if key not in list_keys:
                callback = None
                if key == "calculator_on":
                    callback = coerce_calculator(
                        element_keys,
                    )
                if key == "pre_rash_isolation_on":
                    callback = coerce_quarantine_to_isolation(
                        element_keys,
                    )
                if widget_types[key] == "slider":
                    if key in [
                        "total_vaccine_uptake_doses",
                        "vaccine_uptake_start_day",
                        "vaccine_uptake_duration_days",
                    ]:
                        disabled_slider = not edited_parms["vaccine_uptake"]
                    elif key in [
                        "isolation_adherence",
                        "symptomatic_isolation_start_day",
                        "symptomatic_isolation_duration_days",
                    ]:
                        disabled_slider = not edited_parms["isolation_on"]
                    elif key in [
                        "pre_rash_isolation_adherence",
                        "pre_rash_isolation_start_day",
                        "pre_rash_isolation_duration_days",
                    ]:
                        disabled_slider = not edited_parms["pre_rash_isolation_on"]
                    else:
                        disabled_slider = disabled
                if widget_types[key] == "slider":
                    value = st.slider(
                        show_parameter_mapping[key],
                        min_value=min_values[key],
                        max_value=max_values[key],
                        value=parms[key],
                        step=steps[key],
                        help=helpers[key],
                        format=formats[key],
                        key=element_keys[key],
                        disabled=disabled_slider,
                    )
                elif widget_types[key] == "number_input":
                    value = st.number_input(
                        show_parameter_mapping[key],
                        min_value=min_values[key],
                        max_value=max_values[key],
                        value=parms[key],
                        step=steps[key],
                        help=helpers[key],
                        format=formats[key],
                        key=element_keys[key],
                        disabled=disabled,
                    )
                elif widget_types[key] == "toggle":
                    if key == "calculator_on":
                        toggle_on = False
                    else:
                        toggle_on = True
                    if key == "isolation_on":
                        toggle_on = True
                    else:
                        toggle_on = False
                    if key == "pre_rash_isolation_on":
                        # if isolation is not on, pre-rash isolation cannot be turned on
                        disabled_toggle = not edited_parms["isolation_on"]
                    else:
                        disabled_toggle = disabled
                    value = st.toggle(
                        show_parameter_mapping[key],
                        value=toggle_on,
                        help=helpers[key],
                        key=element_keys[key],
                        disabled=disabled_toggle,
                        on_change=callback,
                    )
                else:
                    pass
                edited_parms[key] = value
            if key in list_keys:
                for index in range(len(parms[key])):
                    if widget_types[key] == "slider":
                        value = st.slider(
                            show_parameter_mapping[f"{key}_{index}"],
                            min_value=min_values[key][index],
                            max_value=max_values[key][index],
                            value=parms[key][index],
                            step=steps[key],
                            help=helpers[key][index],
                            format=formats[key],
                            key=element_keys[key][index],
                            disabled=disabled,
                        )
                    elif widget_types[key] == "number_input":
                        if (
                            key == "initial_vaccine_coverage"
                            and "calculator_on" in edited_parms
                            and edited_parms["calculator_on"]
                        ):
                            value = st.empty()
                        else:
                            value = st.number_input(
                                show_parameter_mapping[f"{key}_{index}"],
                                min_value=min_values[key][index],
                                max_value=max_values[key][index],
                                value=parms[key][index],
                                step=steps[key],
                                help=helpers[key][index],
                                format=formats[key],
                                key=element_keys[key][index],
                                disabled=disabled,
                            )
                    elif widget_types[key] == "toggle":
                        value = st.toggle(
                            show_parameter_mapping[f"{key}_{index}"],
                            value=[True if parms[key][index] > 0 else False],
                            help=helpers[key][index],
                            key=element_keys[key][index],
                            disabled=disabled,
                        )
                    else:
                        pass
                    edited_parms[key][index] = value
    return edited_parms


def get_widget_types(widget_types=None):
    """
    Get the types of widgets for each of the app parameters. This method returns
    a dictionary of widget types for the app parameters. If a widget_types
    dictionary is provided, it will update the defaults with the provided values.

    Args:
        widget_types (dict): Optional widget types dictionary.

    Returns:
        dict: A dictionary of widget types for the app parameters.
    """
    defaults = dict(
        desired_r0="slider",
        k_i="slider",
        k_g1="number_input",
        k_g2="number_input",
        k_21="number_input",
        pop_sizes="number_input",
        latent_duration="slider",
        infectious_duration="slider",
        I0="number_input",
        initial_vaccine_coverage="number_input",
        vaccine_coverages="number_input",
        calc_set_immunity_button="button",
        population_percentages="number_input",
        vaccine_uptake="toggle",
        vaccine_uptake_start_day="slider",
        vaccine_uptake_duration_days="slider",
        total_vaccine_uptake_doses="slider",
        vaccinated_group="number_input",
        calculator_on="toggle",
        isolation_on="toggle",
        isolation_adherence="slider",
        isolation_reduction="slider",
        symptomatic_isolation_start_day="slider",
        symptomatic_isolation_duration_days="slider",
        pre_rash_isolation_on="toggle",
        pre_rash_isolation_adherence="slider",
        pre_rash_isolation_reduction="slider",
        pre_rash_isolation_start_day="slider",
        pre_rash_isolation_duration_days="slider",
        tf="number_input",
        IHR="slider",
    )
    if widget_types is not None and isinstance(widget_types, dict):
        # update with parms if provided
        defaults.update(widget_types)
    return defaults


def get_min_values(parms=None):
    """
    Get the minimum values for the app parameters. This method returns a
    dictionary of minimum values for the app parameters. If a parms dictionary
    is provided, it will update the defaults with the provided values.

    Args:
        parms (dict): Optional parameters dictionary.

    Returns:
        dict: A dictionary of minimum values for the app parameters.
    """
    defaults = dict(
        desired_r0=10.0,
        k_i=[0.0, 0.0, 0.0],
        k_g1=0.0,
        k_g2=0.0,
        k_21=0.0,
        pop_sizes=[15000, 100, 100],
        latent_duration=6.0,
        infectious_duration=5.0,
        I0=[0, 0, 0],
        initial_vaccine_coverage=[0.0, 0.0, 0.0],
        vaccine_coverages=[0.0, 0.0, 0.0],
        population_percentages=[0.0, 0.0, 0.0],
        vaccine_uptake_start_day=0,
        vaccine_uptake_duration_days=0,
        total_vaccine_uptake_doses=0.0,
        vaccinated_group=0,
        isolation_adherence=0.0,
        isolation_reduction=0.25,
        symptomatic_isolation_start_day=0,
        symptomatic_isolation_duration_days=0,
        pre_rash_isolation_adherence=0.0,
        pre_rash_isolation_reduction=0.25,
        pre_rash_isolation_start_day=0,
        pre_rash_isolation_duration_days=0,
        tf=30,
        IHR=0.05,
    )
    # update with parms if provided
    if parms is not None and isinstance(parms, dict):
        defaults.update(parms)
    return defaults


def get_max_values(parms=None):
    """
    Get the maximum values for the app parameters. This method returns a
    dictionary of maximum values for the app parameters. If a parms dictionary
    is provided, it will update the defaults with the provided values.

    Args:
        parms (dict): Optional parameters dictionary.

    Returns:
        dict: A dictionary of maximum values for the app parameters.
    """
    defaults = dict(
        desired_r0=18.0,
        k_i=[50.0, 50.0, 50.0],
        k_g1=50.0,
        k_g2=50.0,
        k_21=50.0,
        pop_sizes=[100_000, 15_000, 15_000],
        latent_duration=18.0,
        infectious_duration=11.0,
        I0=[10, 10, 10],
        initial_vaccine_coverage=[1.0, 1.0, 1.0],
        vaccine_coverages=[1.0, 1.0, 1.0],
        population_percentages=[1.0, 1.0, 1.0],
        vaccine_uptake_start_day=365,
        vaccine_uptake_duration_days=365,
        total_vaccine_uptake_doses=100.0,
        vaccinated_group=2,
        isolation_adherence=1.0,
        isolation_reduction=1.0,
        symptomatic_isolation_start_day=365,
        symptomatic_isolation_duration_days=365,
        pre_rash_isolation_adherence=1.0,
        pre_rash_isolation_reduction=1.0,
        pre_rash_isolation_start_day=365,
        pre_rash_isolation_duration_days=365,
        tf=400,
        IHR=0.25,
    )
    # update with parms if provided
    if parms is not None and isinstance(parms, dict):
        defaults.update(parms)
    return defaults


def get_step_values(parms=None):
    """
    Get the step or increment values for the app parameters. This method returns a
    dictionary of step values for the app parameters. If a parms dictionary is
    provided, it will update the defaults with the provided values.

    Args:
        parms (dict): Optional parameters dictionary.

    Returns:
        dict: A dictionary of step or increment values for the app parameters.
    """
    defaults = dict(
        desired_r0=0.1,
        k_i=0.1,
        k_g1=0.01,
        k_g2=0.01,
        k_21=0.01,
        pop_sizes=100,
        latent_duration=0.1,
        infectious_duration=0.1,
        I0=1,
        initial_vaccine_coverage=0.01,
        vaccine_coverages=0.01,
        population_percentages=0.01,
        vaccine_uptake_start_day=1,
        vaccine_uptake_duration_days=1,
        total_vaccine_uptake_doses=5.0,
        vaccinated_group=1,
        isolation_adherence=0.01,
        isolation_reduction=0.01,
        symptomatic_isolation_start_day=1,
        symptomatic_isolation_duration_days=1,
        pre_rash_isolation_adherence=0.01,
        pre_rash_isolation_reduction=0.01,
        pre_rash_isolation_start_day=1,
        pre_rash_isolation_duration_days=1,
        tf=1,
        IHR=0.01,
    )
    # update with parms if provided
    if parms is not None and isinstance(parms, dict):
        defaults.update(parms)
    return defaults


def get_helpers(parms=None):
    """
    Get the help text for the app parameters. This method returns a dictionary of help
    text for the app parameters. If a parms dictionary is provided, it will update the
    defaults with the provided values.

    Args:
        parms (dict): Optional parameters dictionary.

    Returns:
        dict: A dictionary of help text for the app parameters.
    """
    defaults = dict(
        desired_r0=r"The basic reproductive number captures contact rates and the probability of infection given contact with an infectious person. Some communities may have different contact patterns—for example, in communities with larger households or higher population density. $R_0$ values for measles are typically estimated to be between 12 and 18 (see Detailed Methods).",
        k_i=[
            "Average daily contacts for large population",
            "Average daily contacts for small population 1",
            "Average daily contacts for small population 2",
        ],
        k_g1="Average daily contact per person in small population 1 with people in the large population",
        k_g2="Average daily contact per person in small population 2 with people in the large population",
        k_21="Average daily contact between people in the small populations",
        pop_sizes=[
            "Size of the large population",
            "Size of the small population 1",
            "Size of the small population 2",
        ],
        latent_duration="The number of days from when a person is infected to when they become infectious.",
        infectious_duration="The total number of days a person who is infected with measles is infectious. In this model, rash onset occurs halfway through the selected infectious period.",
        I0=[
            "Introductions in large population",
            "Introductions in small population 1",
            "Introductions in small population 2",
        ],
        initial_vaccine_coverage=[
            "Baseline immunity in large population",
            "Baseline immunity in small population 1",
            "Baseline immunity in small population 2",
        ],
        vaccine_coverages=[
            "Vaccine coverage by first age point",
            "Vaccine coverage by second age point",
            "Vaccine coverage by third age point",
        ],
        population_percentages=[
            "Percent of population in first age group",
            "Percent of population in second age group",
            "Percent of population in third age group",
        ],
        vaccine_uptake="If turned on, initiates a vaccination campaign. The percent of unvaccinated people to receive a dose and duration of the campaign to distribute those doses are specified in the sliders below.",
        vaccine_uptake_start_day="Number of days after introduction of infections in the community that the vaccination campaign will start. The default is “4 days” (after introduction), which corresponds to day 5 in the model. Day 5 is the average time of rash onset occurrence for the introduced infections given an infectious period of 9 days, and so is assumed to be the first day that measles infections would be identified in the population. Vaccination campaigns can start up to 180 days or approximately 6 months after introduction to the community.",
        vaccine_uptake_duration_days="The model assumes vaccine doses are distributed at a constant rate for the duration of the campaign. Vaccination campaigns can last up to 180 days or approximately 6 months.",
        total_vaccine_uptake_doses="In this model, we administer one dose of the MMR vaccine per person vaccinated during the campaign, with 93% effectiveness among those vaccinated and an all-or-nothing vaccine.",
        vaccinated_group="Population receiving the vaccine",
        calculator_on="If turned on, use the calculated immunity from the calculator below to simulate. Note that even though there are age values in the calculator, this toggle will not add age structure to the simulation.",
        isolation_on="If turned on, reduces transmission by symptomatic people who adhere to isolation measures (the percentage as selected under “Isolation adherence”) by 100% during the symptomatic period. For more information on how isolation is implemented in the model, please see the Behind the Model, linked in Detailed Methods.",
        isolation_adherence="Percent of symptomatic people who will follow isolation guidance when isolation is turned on. To modify this parameter, enable isolation.",
        isolation_reduction="Percent reduction in transmission due to isolation. Only used if isolation is turned on.",
        symptomatic_isolation_start_day="Number of days after introduction that the isolation intervention will start. The default is “4 days” (after introduction), which corresponds to day 5 in the model. Day 5 is the average time of rash onset occurrence for the introduced infections given an infectious period of 9 days, and so is assumed to be the first day that measles infections would be identified in the population. Isolation can start up to 180 days or approximately 6 months after introduction to the community.",
        symptomatic_isolation_duration_days="Duration of symptomatic isolation intervention. After the isolation intervention duration ends, all infectious people will resume normal contact with others during the symptomatic infectious period. Only used if isolation is turned on.",
        pre_rash_isolation_on="If turned on, reduces transmission by people who are exposed but pre-symptomatic and adhere to quarantine measures (the percentage as selected under “Quarantine adherence”) by 60% during the pre-symptomatic period. For more information on how quarantine is implemented in the model, please see the Behind the Model, linked in Detailed Methods.",
        pre_rash_isolation_adherence="Percent of pre-symptomatic people who will follow quarantine guidance when quarantine is turned on. To modify this parameter, enable quarantine and isolation.",
        pre_rash_isolation_reduction="Percent reduction in transmission due quarantine. Only used if quarantine is turned on.",
        pre_rash_isolation_start_day="Number of days after introduction that the quarantine intervention will start. The default is “4 days”, which corresponds to day 5 in the model. Day 5 is the average time of rash onset occurrence for the introduced infections given an infectious period of 9 days, and so is assumed to be the first day that measles infections would be identified in the population. Quarantine can start up to 180 days or approximately 6 months after introduction to the community.",
        pre_rash_isolation_duration_days="Duration of pre-symptomatic quarantine intervention. After the quarantine intervention duration ends, all infectious people will resume normal contact with others during the pre-symptomatic infectious period. Only used if quarantine is turned on.",
        tf="Number of time steps to simulate",
        IHR="Proportion of infected people who are hospitalized.",
    )
    if parms is not None and isinstance(parms, dict):
        # update with parms if provided
        defaults.update(parms)
    return defaults


def get_formats(parms=None):
    """
    Get the formats for the app parameters. This method returns a dictionary of
    formats for the app parameters. If a parms dictionary is provided, it will
    update the defaults with the provided values.

    Args:
        parms (dict): Optional parameters dictionary.

    Returns:
        dict: A dictionary of formats for the app parameters.
    """
    defaults = dict(
        desired_r0="%.1f",
        k_i="%.1f",
        k_g1="%.2f",
        k_g2="%.2f",
        k_21="%.2f",
        pop_sizes="%.0d",
        latent_duration="%.1f",
        infectious_duration="%.1f",
        I0="%.0d",
        initial_vaccine_coverage="%.2f",
        vaccine_coverages="%.2f",
        population_percentages="%.2f",
        vaccine_uptake_start_day="%.0d",
        vaccine_uptake_duration_days="%.0d",
        total_vaccine_uptake_doses="%.1f",
        vaccinated_group="%.0d",
        isolation_adherence="%.2f",
        isolation_reduction="%.2f",
        symptomatic_isolation_start_day="%.0d",
        symptomatic_isolation_duration_days="%.0d",
        pre_rash_isolation_adherence="%.2f",
        pre_rash_isolation_reduction="%.2f",
        pre_rash_isolation_start_day="%.0d",
        pre_rash_isolation_duration_days="%.0d",
        tf="%.0d",
        IHR="%.2f",
    )
    if parms is not None and isinstance(parms, dict):
        # update with parms if provided
        defaults.update(parms)
    return defaults


def get_base_session_state_idkeys(parms=None):
    """
    Get the base session state ID keys for the app parameters. This method
    returns a dictionary of session state ID keys for the app parameters.
    If a parms dictionary is provided, it will update the defaults with the
    provided values.

    Args:
        parms (dict): Optional parameters dictionary.

    Returns:
        dict: A dictionary of session state ID keys for the app parameters.
    """
    ss_idkeys = dict(
        desired_r0="desired_r0",
        k_i=["k_i_0", "k_i_1", "k_i_2"],
        k_g1="k_g1",
        k_g2="k_g2",
        k_21="k_21",
        pop_sizes=["pop_sizes_0", "pop_sizes_1", "pop_sizes_2"],
        latent_duration="latent_duration",
        infectious_duration="infectious_duration",
        I0=["I0_0", "I0_1", "I0_2"],
        initial_vaccine_coverage=[
            "initial_vaccine_coverage_0",
            "initial_vaccine_coverage_1",
            "initial_vaccine_coverage_2",
        ],
        vaccine_coverages=[
            "vaccine_coverages_0",
            "vaccine_coverages_1",
            "vaccine_coverages_2",
        ],
        population_percentages=[  # Should this just use the pop_sizes infrastructure
            "population_percentages_0",
            "population_percentages_1",
            "population_percentages_2",
        ],
        vaccine_uptake="vaccine_uptake",
        vaccine_uptake_start_day="vaccine_uptake_start_day",
        vaccine_uptake_duration_days="vaccine_uptake_duration_days",
        total_vaccine_uptake_doses="total_vaccine_uptake_doses",
        vaccinated_group="vaccinated_group",
        calculator_on="calculator_on",
        isolation_on="isolation_on",
        isolation_adherence="isolation_adherence",
        isolation_reduction="isolation_reduction",
        symptomatic_isolation_start_day="symptomatic_isolation_start_day",
        symptomatic_isolation_duration_days="symptomatic_isolation_duration_days",
        pre_rash_isolation_on="pre_rash_isolation_on",
        pre_rash_isolation_adherence="pre_rash_isolation_adherence",
        pre_rash_isolation_reduction="pre_rash_isolation_reduction",
        pre_rash_isolation_start_day="pre_rash_isolation_start_day",
        pre_rash_isolation_duration_days="pre_rash_isolation_duration_days",
        tf="tf",
        IHR="IHR",
    )
    if parms is not None and isinstance(parms, dict):
        # update with parms if provided
        ss_idkeys.update(parms)
    return ss_idkeys


def get_session_state_idkeys(num):
    """
    Get the session state ID keys for the app parameters, appending a number
    to each key to ensure uniqueness across multiple session state widgets
    with similar parameters.

    Args:
        num (int): The number to append to each key.

    Returns:
        dict: A dictionary of session state ID keys for the app parameters with
        a number appended for uniqueness.
    """
    ss_idkeys = get_base_session_state_idkeys()
    for key, value in ss_idkeys.items():
        if isinstance(value, list):
            ss_idkeys[key] = [
                f"{ss_idkeys[key][i]}_{num}" for i in range(len(ss_idkeys[key]))
            ]
        else:
            ss_idkeys[key] = f"{ss_idkeys[key]}_{num}"

    return ss_idkeys


def get_parameter_key_for_session_key(session_key):
    """
    Get the parameter key for a given session key.

    Args:
        session_key (str): The session key to look up.

    Returns:
        str, int: The corresponding model parameter key and index if applicable.
    """

    # Go from session key to model parameter key, possibly with an index if the
    # parameter is a list or array for the metapop model

    split_key = session_key.split("_")
    # default values for key and index
    key, index = "", ""

    # find all session keys that are model parameters, assuming that we stitch
    # them together with a model parameter and numbers to indicate the index if
    # the parameter value is a list
    # first check if the last element of the split key is a number
    if len(split_key) > 1 and split_key[-1].isdigit():
        # remove the value at the end of the key - this is used for naming
        # purposes to make each key unique
        split_key = split_key[:-1]

        # check if the last element of the split key is a number - this means
        # the value of this parameter was stored in a list in the model parameter
        if split_key[-1].isdigit():
            index = int(split_key[-1])
            split_key = split_key[:-1]
        key = "_".join(split_key)
    else:
        key = "_".join(split_key)

    return key, index


def coerce_quarantine_to_isolation(element_keys):
    """
    Callback function for isolation_on. If isolation is turned off,
    quarantine is coerced to be off as well.

    Args:
        element_keys (dict): A dictionary containing the keys for the Streamlit elements.

    Returns:
        None
    """
    if element_keys["isolation_on"] not in st.session_state:
        st.session_state[element_keys["isolation_on"]] = False

    if not st.session_state[element_keys["isolation_on"]]:
        st.session_state[element_keys["pre_rash_isolation_on"]] = False
        st.write("To enable quarantine, isolation must be enabled.")


def coerce_calculator(element_keys):
    """
    Callback function for calculator_on. If calculator is turned off,
    use baseline immunity values.

    Args:
        element_keys (dict): A dictionary containing the keys for the Streamlit elements.

    Returns:
        None
    """
    if element_keys["calculator_on"] not in st.session_state:
        st.session_state[element_keys["calculator_on"]] = False


def update_intervention_parameters_from_widget(parms):
    """
    Update the model parameters for interventions based on the values in the
    session state. Modifies the input dictionary `parms` in place.

    Args:
        parms (dict): The original parameters dictionary.

    Returns:
        dict: The `parms` dictionary with updated values in place.
    """
    # symptomatic isolation
    if parms["isolation_on"]:
        parms["isolation_success"] = (
            parms["isolation_adherence"] * parms["isolation_reduction"]
        )
    else:
        parms["isolation_success"] = 0.0

    # pre-symptomatic isolation
    if parms["pre_rash_isolation_on"]:
        parms["pre_rash_isolation_success"] = (
            parms["pre_rash_isolation_adherence"]
            * parms["pre_rash_isolation_reduction"]
        )
    else:
        parms["pre_rash_isolation_success"] = 0.0

    if not parms["vaccine_uptake"]:
        parms["total_vaccine_uptake_doses"] = 0

    if parms["vaccine_uptake_duration_days"] == 0:
        parms["total_vaccine_uptake_doses"] = 0

    return parms


def generate_random_key():
    """
    Generate a random key for the session state.

    Returns:
        str: A random key string.
    """
    return str(uuid.uuid4()) + "_random_key"


def reset(defaults, widget_types):
    """
    Reset the session state widget values to their default values.

    Args:
        defaults (dict): The default values for the parameters.
        widget_types (dict): The types of widgets for the parameters.

    Returns: None
    """
    for session_key in st.session_state.keys():
        key, index = get_parameter_key_for_session_key(session_key)

        # skip if key is empty
        if key == "":
            continue

        # special case for data editor keys related to the baseline immunity calculator
        # we will generate a new random key for each editor to force a reset rather
        # than resetting the table itself
        elif key in [
            "pop_editor_key",
            "coverage_editor_key",
        ]:
            value = generate_random_key()

        # special cases for data editors related to baseline immunity calculator
        elif key == "pop_table_base":
            value = initialize_pop_table(defaults)

        elif key == "cov_table_base":
            value = initialize_vacc_table(defaults)

        # delete certain keys from session state to force streamlit to recreate them when reset is called
        elif key in [
            "table_changed",
            "immunity",
            "invalid_population_percentage",
            "calc_set_immunity_button_clicked",
        ]:
            del st.session_state[session_key]
            continue

        # continue if the key is one of the data editor keys
        elif key in [
            "default_pop_table",
            "default_cov_table",
            "pop_table",
            "cov_table",
            "calc_set_immunity_button",
            "reset",
        ]:
            continue

        elif "random_key" in session_key:
            continue

        elif key != "" and index == "":
            value = defaults[key]

        elif key != "" and isinstance(index, int) and key in defaults:
            value = defaults[key][index]

        else:
            raise ValueError(f"Invalid index type: {type(index)} for key: {key}")

        # by default, turn toggles off
        if widget_types.get(key, None) == "toggle":
            value = False

        if key == "calculator_on":
            value = False

        if key == "isolation_on":
            value = True

        # set the session state value to the default value
        st.session_state[session_key] = value

    st.session_state["reset"] = True


def initialize_pop_table(parms):
    """
    Initialize the population table DataFrame from parameters.

    Args:
        parms (dict): Parameters dictionary containing population data

    Returns:
        pl.DataFrame: DataFrame with population distribution data
    """
    immunity_age_groups = parms["calculator_pop_labels"]
    population_percentages = parms["calculator_pop_sizes"]

    df_pop = pl.DataFrame(
        [
            {"population": age, "percentage": 100 * pct}
            for age, pct in zip(immunity_age_groups, population_percentages)
        ]
    )

    return df_pop


def initialize_vacc_table(parms):
    """
    Initialize the vaccination coverage table DataFrame from parameters.

    Args:
        parms (dict): Parameters dictionary containing vaccine coverage data

    Returns:
        pl.DataFrame: DataFrame with vaccination coverage data
    """
    coverage_age_groups = parms["calculator_coverage_labels"]
    vaccine_coverages = parms["calculator_coverage_values"]

    df_coverage = pl.DataFrame(
        [
            {"threshold": age, "coverage": 100 * pct}
            for age, pct in zip(coverage_age_groups, vaccine_coverages)
        ]
    )

    return df_coverage


def table_changed():
    """
    Callback function to set table_changed flag in session state when a table is edited.
    """
    st.session_state.table_changed = True


@st.fragment
def edit_baseline_immunity(parms):
    """
    Create data editors for population and coverage tables and store results in session state.
    Updates session_state variables: pop_table and cov_table. Creates its own fragment in streamlit.

    Args:
        parms (dict): Parameters dictionary

    Returns:
        None
    """
    if "table_changed" not in st.session_state:
        st.session_state.table_changed = False

    st.session_state.pop_table = st.data_editor(
        st.session_state.default_pop_table,
        column_config={
            "population": "Population Age",
            "percentage": st.column_config.NumberColumn(
                "Percent of population (%)",
                help="What percent of the population is in this age group?",
                min_value=0,
                max_value=100,
                step=0.1,
                format="%.1f",
            ),
        },
        on_change=table_changed,
        disabled=["population"],
        hide_index=True,
        key=st.session_state.pop_editor_key,
    )

    try:
        total_percentage = st.session_state.pop_table["percentage"].sum()
    except KeyError:
        total_percentage = 100

    if np.round(abs(total_percentage - 100), 1) >= 0.1:  # Allow small rounding errors
        st.session_state.invalid_population_percentage = True
        st.warning(
            f"⚠️ Population percentages sum to {total_percentage:.1f}%. "
            "Please adjust the values so they sum to 100%."
        )
    else:
        st.session_state.invalid_population_percentage = False

    st.session_state.cov_table = st.data_editor(
        st.session_state.default_cov_table,
        column_config={
            "threshold": "Age Threshold",
            "coverage": st.column_config.NumberColumn(
                "Immunity Coverage",
                help="What percent of the population is immune by this age?",
                min_value=0,
                max_value=100,
                step=0.1,
                format="%.1f",
            ),
        },
        disabled=["threshold"],
        on_change=table_changed,
        hide_index=True,
        key=st.session_state.coverage_editor_key,
    )

    calc_immunity()

    button_to_calculate_immunity()


def get_baseline_immunity(population, coverage):
    """
    Calculate the baseline immunity given the values in the calculator

    Args:
        population (list): The percent of the population in each group.
        coverage (list): The percent of individuals in each group with prior immunity.

    Returns:
        immunity_value (float): The percent of the population with immunity.
    """

    # convert to proportions
    population = [p / 100 for p in population]
    coverage = [c / 100 for c in coverage]

    coverage_cutoffs = [2, 5, 18]

    immunity_value = 0

    # Handle the 0-2 year group within the first population group
    # 1-2 years get partial coverage
    one_year_coverage = (
        (1 / coverage_cutoffs[1]) * population[0] * (coverage[0] + 0) / 2
    )
    immunity_value += one_year_coverage

    # first cutoff to second cutoff: partial coverage (2-5 years)
    age_range_years = coverage_cutoffs[1] - coverage_cutoffs[0]  # 5-2 = 3 years
    immunity_value += (
        (age_range_years / (coverage_cutoffs[1] - 0))
        * population[0]
        * (coverage[1] + coverage[0])
        / 2
    )

    # second cutoff to third cutoff: (5-18 years)
    immunity_value += population[1] * (coverage[2] + coverage[1]) / 2

    # over third cutoff: (18+ years)
    immunity_value += population[2] * coverage[3]

    return round(immunity_value, 2)


def update_coverage(edited_df_coverage, default_values):
    """
    Update coverage DataFrame by filling missing values with default values.

    Args:
        edited_df_coverage (pl.DataFrame): DataFrame with 'coverage' column that may contain missing values
        default_values (pl.DataFrame): DataFrame with default 'coverage' values to use for missing entries. Must not contain null values.

    Returns:
        pl.DataFrame: Updated DataFrame with missing values filled from defaults
    """
    # Make a copy to avoid modifying the original DataFrame
    updated_df = edited_df_coverage.clone()

    has_missing_coverage = False

    # Check that we have enough default values
    if len(default_values) < len(updated_df):
        raise ValueError(
            f"Not enough default values ({len(default_values)}) for DataFrame rows ({len(updated_df)})"
        )

    if default_values["coverage"].is_null().any():
        raise ValueError(
            "Default values contain null entries, cannot fill missing coverage."
        )

    # check if any coverage values are missing from updated_df before we fill in values with defaults
    if updated_df["coverage"].is_null().any():
        has_missing_coverage = True

    # rename the column with default values to avoid conflict during join
    default_values = default_values.rename({"coverage": "default_coverage"})

    # create a joined dataframe to merge coverage values
    joined_df = updated_df.join(default_values, on="threshold", how="left")

    # fill in missing coverage values with default values (already converted to percentages)
    filled_df = joined_df.with_columns(
        pl.col("coverage").fill_null(pl.col("default_coverage"))
    )

    # drop the default coverage column
    updated_df = filled_df.drop("default_coverage")

    return updated_df, has_missing_coverage


def calc_immunity():
    """
    Calculate and display baseline immunity based on the values in the
    calculator tables.

    Args:
        parms (dict): Parameters dictionary containing default calculator values

    Returns:
        None. Updates session state variable 'immunity' with calculated value,
        session state variable 'table_changed' to indicate if table was changed,
        and displays a message with the calculated immunity as well as a warning
        message indicating if the user needs to press the button to recalculate
        the immunity value because they have changed table inputs.
    """
    # create table with default values from parms dictionary
    default_values = st.session_state.cov_table.clone()

    # change values of "coverage" column to current default values
    # not suggesting this stays in the code base long-term, but for now this
    # prevents a change from the previous default values we have for the
    # calculator. We will revisit this shortly and decide what the defaults
    # will be and store them in the config file for the app
    default_values = default_values.with_columns(
        pl.Series("coverage", [70, 85, 90, 95])
    )

    # update coverage table with default values for missing entries
    st.session_state.cov_table, has_missing_coverage = update_coverage(
        st.session_state.cov_table,
        default_values,
    )
    st.session_state.immunity = get_baseline_immunity(
        st.session_state.pop_table["percentage"],
        st.session_state.cov_table["coverage"],
    )

    immunity_text = f"{st.session_state.immunity * 100:.0f}"

    if st.session_state.table_changed:
        st.warning(
            "Values changed. Please click the button to recalculate baseline immunity."
        )
    else:
        if has_missing_coverage:
            message = f"Assuming coverage points with no data available are national estimates, the estimate for baseline immunity is {immunity_text}%"
        else:
            message = f"Based on these values, the estimate for baseline immunity is {immunity_text}%"
        st.success(message)


def click_set_immunity_button():
    """
    Callback function for the calculate and set immunity button. Sets flag in
    session state.
    """
    st.session_state.calc_set_immunity_button_clicked = True
    st.session_state.table_changed = False


def button_to_calculate_immunity():
    """
    Create a button to calculate and set baseline immunity.
    """
    if "calc_set_immunity_button_clicked" not in st.session_state:
        st.session_state.calc_set_immunity_button_clicked = False

    if st.button(
        "Calculate and Set Baseline Immunity",
        disabled=st.session_state.invalid_population_percentage,
        on_click=click_set_immunity_button,
    ):
        st.rerun()


def apply_calculated_immunity(parms):
    """
    Apply calculated immunity if calculator is enabled and immunity has been calculated.
    """
    if parms.get("calculator_on", False) and "immunity" in st.session_state:
        # Override all baseline immunity values with the calculated value
        if isinstance(parms["initial_vaccine_coverage"], list):
            for i in range(len(parms["initial_vaccine_coverage"])):
                parms["initial_vaccine_coverage"][i] = st.session_state.immunity
        else:
            parms["initial_vaccine_coverage"] = st.session_state.immunity
    return parms


### Methods to handle extraction of user inputs and updating parameter dictionaries to send for simulation ##
def get_parms_from_table(table, value_col="Scenario 1"):
    """
    Extract a parameter dictionary from a table.

    Args:
        table (pl.DataFrame): The input table containing parameters.
        value_col      (str): The column name for the parameter values.

    Returns:
        dict: A dictionary containing the parameters.
    """
    # get parameter dictionary from a table
    parms = dict()
    # expect the table to have the following columns
    # Parameter, No interventions, Interventions
    for key, value in zip(table["Parameter"].to_list(), table[value_col].to_list()):
        parms[key] = value
    return parms


def update_parms_from_table(parms, table, parameters_mapping, value_col="Scenario 1"):
    """
    Update the parameter dictionary with new values from a user input table.

    Args:
        parms              (dict): The original parameters dictionary.
        table      (pl.DataFrame): The input table containing parameters.
        parameters_mapping (dict): A mapping of parameter names to their display names.
        value_col           (str): The column name for the parameter values.

    Returns:
        dict: The updated parameters dictionary.
    """
    # get updated values from user through the sidebar
    for key, value in zip(table["Parameter"].to_list(), table[value_col].to_list()):
        original_key = next((k for k, v in parameters_mapping.items() if v == key), key)
        parms[original_key] = value
    return parms


def correct_parameter_types(original_parms, parms_from_table):
    """
    Correct the parameter types in the updated parameters dictionary.

    Args:
        original_parms   (dict): The original parameters dictionary.
        parms_from_table (dict): The updated parameters dictionary from the table.

    Returns:
        dict: The updated parameters dictionary with corrected types.
    """
    for key, value in original_parms.items():
        if isinstance(value, int) and not isinstance(value, bool):
            parms_from_table[key] = int(parms_from_table[key])
        elif isinstance(value, bool):
            if parms_from_table[key] in [True, "TRUE", "True", "true", "1"]:
                parms_from_table[key] = True
            else:
                parms_from_table[key] = False
        elif isinstance(value, float):
            parms_from_table[key] = float(parms_from_table[key])
        elif isinstance(value, str):
            parms_from_table[key] = str(parms_from_table[key])
    return parms_from_table


### Methods to calculate different metrics from simulation results ###
def add_daily_incidence(results, groups):
    """
    Add daily incidence to the results DataFrame."

    Args:
        results (pl.DataFrame): The results DataFrame.
        groups          (list): List of group indices.

    Returns:
        pl.DataFrame: The updated results DataFrame with daily incidence added.
    """
    # add a column for daily incidence
    results = results.with_columns(pl.lit(None).alias("Incidence"))
    unique_replicates = results.select("replicate").unique().to_series().to_list()
    updated_rows = []

    for replicate in unique_replicates:
        tempdf = results.filter(pl.col("replicate") == replicate)
        for group in groups:
            group_data = tempdf.filter(pl.col("group") == group)
            group_data = group_data.sort("t")
            inc = group_data["Y"] - group_data["Y"].shift(1)
            group_data = group_data.with_columns(inc.alias("Incidence"))
            updated_rows.append(group_data)

    results = pl.concat(updated_rows, how="vertical")
    return results


def get_interval_cumulative_incidence(results, groups, interval=7):
    """
    Calculate cumulative incidence over specified intervals.

    Args:
        results (pl.DataFrame): The results DataFrame.
        groups          (list): List of group indices.
        interval         (int): The interval in days. Defaults to 7 for a week.

    Returns:
        pl.DataFrame: The updated results DataFrame with interval cumulative incidence added.
    """
    interval_results = results.clone()
    interval_results.sort(["replicate", "group", "t"])
    # extract time points for every interval days
    time_points = results["t"].unique().sort().gather_every(interval)
    # make a single time array
    interval_points = np.arange(len(time_points), dtype=float)
    interval_results = interval_results.filter(pl.col("t").is_in(time_points))

    # tile the interval time points for each group and replicate
    unique_replicates = results.select("replicate").unique().to_series().to_list()
    repeated_interval_points = np.tile(
        interval_points, len(groups) * len(unique_replicates)
    )
    # add the interval time points to the interval results table
    interval_results = interval_results.with_columns(
        pl.Series(name="interval_t", values=repeated_interval_points)
    )

    # now Y is the cumulative incidence at each time point and interval_t is the interval time point
    return interval_results


def get_interval_results(results, groups, interval=7):
    """
    Calculate interval results for cumulative incidence.

    Args:
        results (pl.DataFrame): The results DataFrame.
        groups          (list): List of group indices.
        interval         (int): The interval in days. Defaults to 7 for a week.

    Returns:
         pl.DataFrame: The updated results DataFrame with interval cumulative incidence added.""
    """
    # get a table with results, in particular cumulative incidence at each interval time point
    # in this table, Y is the cumulative incidence

    interval_results = get_interval_cumulative_incidence(results, groups, interval)
    # now process this table to get the interval incidence
    unique_replicates = results.select("replicate").unique().to_series().to_list()

    updated_rows = []
    for replicate in unique_replicates:
        tempdf = interval_results.filter(pl.col("replicate") == replicate)
        for group in groups:
            group_data = tempdf.filter(pl.col("group") == group)
            group_data = group_data.sort("t")
            inc = group_data["Y"] - group_data["Y"].shift(1)
            group_data = group_data.with_columns(inc.alias(f"inc_{interval}"))
            updated_rows.append(group_data)
    interval_results = pl.concat(updated_rows)
    # drop column inc
    interval_results = interval_results.drop("Incidence")
    return interval_results


### Methods to create charts for the app ###
def create_chart(
    alt_results,
    outcome_option,
    x,
    xlabel,
    y,
    ylabel,
    yscale,
    color_key,
    color_scale,
    domain,
    labelExpr,
    detail,
    width=300,
    height=300,
):
    """
    Create a chart using Altair.

    Args:
        alt_results (pl.DataFrame): The results DataFrame.
        outcome_option       (str): The selected outcome option.
        x                    (str): The x-axis column name.
        xlabel               (str): The x-axis label.
        y                    (str): The y-axis column name.
        ylabel               (str): The y-axis label.
        yscale             (tuple): The y-axis scale limits.
        color_key            (str): The color key for the chart.
        color_scale    (alt.Scale): The color scale for the chart.
        domain              (list): The domain for the color scale.
        labelExpr            (str): The expression for the legend labels.
        detail               (str): Additional detail for the chart.
        width                (int): Width of the chart. Defaults to 300.
        height               (int): Height of the chart. Defaults to 300.

    Returns:
        alt.Chart: The Altair chart object.
    """
    chart = (
        alt.Chart(alt_results, title=outcome_option)
        .mark_line(opacity=0.4)
        .encode(
            x=alt.X(x, title=xlabel),
            y=alt.Y(y, title=ylabel).scale(domain=yscale),
            color=alt.Color(
                color_key,
                scale=color_scale,
                legend=alt.Legend(
                    title="Population",
                    values=domain,
                    labelExpr=labelExpr,
                ),
            ),
            detail=detail,
        )
        .properties(width=width, height=height)
    )
    return chart


### Methods to summarize data
def get_median_trajectory_from_episize(
    results: pl.DataFrame, base_group: int = 0
) -> pl.DataFrame:
    """
    Get the trajectory of the replicate whose final value of R in the
    `base_group` is closest to the median in that group.

    Args:
        results (pl.DataFrame): A Polars DataFrame containing simulation results from `get_scenario_results` with columns including 'group', 'replicate', 'I' (infected count), and 't' (time).
        base_group (int, optional): The group identifier to filter results by. Defaults to 0.

    Returns:
        int: The replicate identifier whose final size is closest to the median
        final size across all replicates in the specified group.
    """
    # Get the maximum time point
    max_t = results["t"].max()

    # Filter results for the last time point and group 0
    filtered_results = results.filter(
        (pl.col("t") == max_t) & (pl.col("group").cast(pl.UInt32) == pl.lit(base_group))
    )

    # Calculate the median value of R
    median_R = filtered_results["R"].median()

    # Find the replicate with the closest R value to the median
    closest_replicate = (
        filtered_results.with_columns((pl.col("R") - median_R).abs().alias("distance"))
        .sort("distance", "replicate")
        .select("replicate")
        .head(1)
        .item()
    )

    # Return the trajectory for the closest replicate
    return closest_replicate


def get_median_trajectory_from_peak_time(
    results: pl.DataFrame, base_group: int = 0
) -> int:
    """
    Selects the replicate whose infection peak timing is closest to the median
    peak time across all replicates for a given group.

    Args:
        results (pl.DataFrame): A Polars DataFrame containing simulation results from `get_scenario_results` with columns including 'group', 'replicate', 'I' (infected count), and 't' (time).
        base_group (int, optional): The group identifier to filter results by. Defaults to 0.

    Returns:
        int: The replicate identifier whose peak infection time is closest to the
        median peak time across all replicates in the specified group.
    """
    filtered_results = (
        results.filter(pl.col("group").cast(pl.UInt32) == pl.lit(base_group))
        .filter((pl.col("I") == pl.col("I").max()).over("replicate"))
        .group_by("replicate")
        .agg(pl.col("t").median().alias("peak_time"))
    )

    # Filter for median peak time point across replicates
    median_peak_time = filtered_results["peak_time"].median()

    closest_replicate = (
        filtered_results.with_columns(
            (pl.col("peak_time") - median_peak_time).abs().alias("distance")
        )
        .sort("distance", "replicate")
        .select("replicate")
        .head(1)
        .item()
    )

    # Return the trajectory for the closest replicate
    return closest_replicate


def totals_same_by_ks(
    combined_results: pl.DataFrame, scenario_names: list, p_threshold: float = 0.05
) -> bool:
    """
    Perform a 2 sample Kolmogorov-Smirnov test to compare the total infections
    between two scenarios.

    Args:
        combined_results (pl.DataFrame): The combined results DataFrame.
        scenario_names           (list): List of scenario name labels created by the simulation.
        p_threshold             (float): The p-value threshold for determining indistinguishable distributions. Default is 0.5.
        - In general, we want to be selective about when to throw the error, as similar, but different, distributions
            may not reject the null hypothesis but still be visually different for low sample sizes.

    Returns:
        bool: True if the p-value is greater than the specified threshold, indicating two indistinguishable distributions.
        - A value of 0.05 would reject the null hypothesis that the two distributions are independent samples of the same distribution
        - Higher threshold values will increase the confidence that the two distributions are identical.
    """
    # Get total infections for each scenario
    scenario_0 = combined_results.filter(pl.col("Scenario") == scenario_names[0])[
        "Total"
    ].to_numpy()
    scenario_1 = combined_results.filter(pl.col("Scenario") == scenario_names[1])[
        "Total"
    ].to_numpy()

    # Perform KS test
    _, p_value = stats.ks_2samp(scenario_0, scenario_1)

    return p_value > p_threshold


def img_to_bytes(img_path):
    """
    Convert an image file to a base64 encoded string.

    Args:
        img_path (str): The path to the image file.

    Returns:
        str: A base64 encoded string of the image.
    """
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()

    return encoded


def img_to_html(img_path, alt, width=30, vertical_align="middle", margin_right=8):
    """
    Convert an image file to an HTML image tag with base64 encoding.

    Args:
        img_path (str): The path to the image file.
        alt (str): Alt text for the image
        width (int): The width of the image in pixels. Defaults to 30.
        vertical_align (str): The vertical alignment of the image. Defaults to "middle".
        margin_right (int): The right margin in pixels. Defaults to 8.

    Returns:
        str: An HTML image tag with the base64 encoded image.
    """
    encoded = img_to_bytes(img_path)
    img_html = f"<img alt='{alt}' src='data:image/png;base64, {encoded}' width='{width}' style='vertical-align:{vertical_align}; margin-right:{margin_right}px' class='img-fluid'>"
    return img_html


def is_light_color():
    """
    Determine if the current theme is light or dark.

    Returns:
        bool: True if the theme is light, False if dark. Defaults to True if
        the theme is not recognized.
    """
    try:
        theme = st.context.theme.type
    except Exception:
        theme = "light"
    theme = st.context.theme.type

    if theme == "light":
        return True
    elif theme == "dark":
        return False
    return True  # Default to light if theme is not recognized


def get_github_logo_path(is_background_light):
    """
    Get the path to the GitHub logo image in app_assets based on the background
    color theme.

    Args:
        is_background_light (bool): True if the background is light, False if dark. Defaults to True if theme is not recognized.

    Returns:
        str: The path to the GitHub logo image in app_assets.
    """
    if is_background_light:
        image_path = os.path.join(
            os.path.dirname(__file__), "app_assets", "github-mark.png"
        )
    else:
        image_path = os.path.join(
            os.path.dirname(__file__), "app_assets", "github-mark-white.png"
        )
    return image_path


def render_chart_title(title, subtitle=None):
    subtitle_html = (
        f"""<div style="font-size: 0.8em; line-height: 1.1; margin-bottom: 1em;">{subtitle}</div>"""
        if subtitle
        else ""
    )
    return f"""<h4 style="font-size: inherit; padding:0; line-height: 1.1;margin-bottom: 0.2em;">{title}</h4>{subtitle_html}"""


def combine_incidence_results(alt_results1, alt_results2, combined_ave_results):
    return pl.concat(
        [
            alt_results1.with_columns(pl.col("replicate").cast(pl.String)),
            alt_results2.with_columns(pl.col("replicate").cast(pl.String)),
            combined_ave_results.with_columns(pl.lit("median").alias("replicate")),
        ]
    )


def csv_download_button(polars_df, link_text, filename_stem):
    csv_buffer = io.StringIO()
    polars_df.write_csv(csv_buffer, include_header=True)
    base64_encoded = base64.b64encode(csv_buffer.getvalue().encode("utf-8")).decode(
        "utf-8"
    )
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    data_uri = f'<div style="text-align:right;" class="st-emotion-cache-6l3wav es51y5e1"><a download="{filename_stem}-{time}.csv" href="data:text/csv;base64,{base64_encoded}">{link_text}</a></div>'
    st.markdown(data_uri, unsafe_allow_html=True)
