# This file contains helper functions for the metapop app.
import copy

import altair as alt
import griddler
import griddler.griddle
import numpy as np
import polars as pl
import streamlit as st

# import what's needed from other metapop modules
from .sim import simulate

# if you want to use methods from metapop in this file under
# if __name__ == "__main__": you'll need to import them as:
# from metapop.sim import simulate
### note: this is not recommended use within a file that is imported as a package module, but it can be useful for testing purposes

__all__ = [
    "get_scenario_results",
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
    "get_base_widget_idkeys",
    "get_widget_idkeys",
    "get_parms_from_table",
    "update_parms_from_table",
    "correct_parameter_types",
    "add_daily_incidence",
    "get_interval_cumulative_incidence",
    "get_interval_results",
    "create_chart",
    "calculate_outbreak_summary",
    "get_table",
    "set_parms_to_zero",
    "rescale_prop_vax",
    "get_median_trajectory",
]


### Methods to simulate the model for the app ###
def get_scenario_results(parms):
    """
    Run simulations for a grid set of parameters and return the combined results Dataframe.

    Args:
        parms (list): List of dictionaries containing model parameters.

    Returns:

    """
    results = griddler.run_squash(griddler.replicated(simulate), parms)
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
            "E1",
            "E2",
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

    Returns:
        dict: Dictionary of parameters for the metapopulation model.
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
        pl.DataFrame: DataFrame containing the default parameters and their values for two scenarios.
    """
    # read in parms, some of which are lists
    parms = read_parameters()

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
            "No Interventions": values,
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
        pl.DataFrame: DataFrame containing the default parameters and their values for two scenarios.
    """

    full_defaults = get_default_full_parameters()
    show_parameter_mapping = get_show_parameter_mapping()
    show_defaults = full_defaults.filter(
        pl.col("Parameter").is_in(show_parameter_mapping.keys())
    )

    # replace specific values with integers
    for key in ["No Interventions", "Interventions"]:
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
        pl.col("No Interventions").cast(pl.Float64)
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
    for key in ["No Interventions", "Interventions"]:
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
        pl.col("No Interventions").cast(pl.Float64)
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
        I0_0="Initial infections in large population",
        I0_1="Initial infections in small population 1",
        I0_2="Initial infections in small population 2",
        # vaccine_uptake = "Enable vaccine uptake",
        total_vaccine_uptake_doses="% unvaccinated individuals that get vaccinated",
        vaccine_uptake_start_day="Active vaccination start day",
        vaccine_uptake_duration_days="Active vaccination duration days",
        vaccinated_group="Vaccinated group",
        # symptomatic_isolation = "Enable symptomatic isolation",
        isolation_success="Symptomatic individuals isolating",
        symptomatic_isolation_start_day="Symptomatic isolation start day",
        symptomatic_isolation_duration_days="Symptomatic isolation duration days",
        pre_rash_isolation_success="Stay-at-home",
        pre_rash_isolation_start_day="Pre-rash isolation start day",
        pre_rash_isolation_duration_days="Pre-rash isolation duration days",
        tf="Time steps",
        # n_replicates = "Number of replicates",
        # seed = "Random seed",
        initial_vaccine_coverage_0="Baseline vaccination in large population",
        initial_vaccine_coverage_1="Baseline vaccination in small population 1",
        initial_vaccine_coverage_2="Baseline vaccination in small population 2",
    )

    if parms is not None and isinstance(parms, dict):
        if parms["n_groups"] == 1:
            show_mapping["pop_sizes_0"] = "Population Size"
            show_mapping["I0_0"] = "Initial infections"
            show_mapping["initial_vaccine_coverage_0"] = "Baseline vaccination"

    return show_mapping


def get_advanced_parameter_mapping():
    """
    Get a mapping of advanced parameter names to their display names.

    Returns:
        dict: A dictionary mapping advanced parameter names to their display names.
    """
    # Define the mapping of advanced parameter names to display names
    advanced_mapping = dict(
        desired_r0="R0",
        n_groups="Number of groups",
        infectious_duration="Infectious period (days)",
        latent_duration="Latent period (days)",
        pre_rash_isolation_success="Daily proportion of infectious individuals who stay-at-home following exposure prior to rash onset",
        isolation_success="Daily proportion of infectious individuals who isolate after rash onset",
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
    )
    return advanced_mapping


def get_outcome_options():
    """
    Get the available outcome options for the app.

    Returns:
        tuple: A tuple containing the available outcome options.
    """
    return (
        "Weekly Incidence",
        "Weekly Cumulative Incidence",
        "Daily Infections",
        "Daily Incidence",
        "Daily Cumulative Incidence",
    )


def get_outcome_mapping():
    """
    Get a mapping of outcome names to their corresponding codes.

    Returns:
        dict: A dictionary mapping outcome names to their corresponding output column name.
    """
    # Define the mapping of outcome names to their corresponding codes
    return {
        "Daily Infections": "I",
        "Daily Incidence": "inc",
        "Daily Cumulative Incidence": "Y",
        "Weekly Incidence": "Winc",
        "Weekly Cumulative Incidence": "WCI",
    }


### Methods to handle parameter keys based on their value types ###
def get_list_keys(parms):
    """
    Get the keys of parameters that have list values.

    Args:
        parms (dict): The parameters dictionary.

    Returns:
        list: The keys of the parameters that have lists values.
    """
    list_keys = [key for key, value in parms.items() if isinstance(value, list)]
    return list_keys


def get_keys_in_list(parms, updated_parms):
    """
    Get the expanded keys of parameters from updated_parms that map to keys that have list values in the parms dictionary.

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
def set_parms_to_zero(parms, parms_to_set):
    edited_parms = copy.deepcopy(parms)

    for key in parms_to_set:
        edited_parms[key] = 0.0

    return edited_parms


def rescale_prop_vax(edited_parms):
    pop_sizes = np.array(edited_parms["pop_sizes"])
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
    Create the sidebar for editing parameters.

    Args:
        element (st container object): The Streamlit element to place the sidebar in.
        scenario_name           (str): The name of the scenario.
        parms                  (dict): The parameters to edit.
        ordered_keys           (list): An ordered list of the parameters to edit.
        list_keys              (list): The keys of the parameters that are lists.
        widget_types           (dict): The types of widgets for the parameters.
        show_parameter_mapping (dict): The mapping of parameter names to display names.
        min_values             (dict): The minimum values for the parameters.
        max_values             (dict): The maximum values for the parameters.
        steps                  (dict): The step sizes for the parameters.
        helpers                (dict): The help text for the parameters.
        formats                (dict): The formats for the parameters.
        element_keys           (dict): The keys for the Streamlit elements.
        disabled               (bool): Whether the widgets should be disabled. Defaults to False.

    Returns:
        edited_parms: The edited parameters.

    """
    edited_parms = copy.deepcopy(parms)

    with element:
        st.subheader(scenario_name)

        for key in ordered_keys:
            if key not in list_keys:
                # if key in slider_keys:
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
                        disabled=disabled,
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
                    value = st.toggle(
                        show_parameter_mapping[key],
                        value=False,
                        help=helpers[key],
                        key=element_keys[key],
                        disabled=disabled,
                    )
                    # if toggle is turned on, set value to the original value for the model
                    if value is True:
                        value = parms[key]
                    if value is False:
                        value = 0
                else:
                    pass
                edited_parms[key] = value
            if key in list_keys:
                for index in range(len(parms[key])):
                    # if key in slider_keys:
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
                    # else:
                    elif widget_types[key] == "number_input":
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
                        # if toggle is turned on, set value to the original value for the model
                        if value is True:
                            value = parms[key][index]
                        if value is False:
                            value = 0
                    else:
                        pass
                    edited_parms[key][index] = value
    return edited_parms


def get_widget_types(widget_types=None):
    """
    Get the types of widgets for each of the app parameters.

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
        vaccine_uptake_start_day="slider",
        vaccine_uptake_duration_days="slider",
        total_vaccine_uptake_doses="slider",
        vaccinated_group="number_input",
        isolation_success="toggle",
        symptomatic_isolation_start_day="slider",
        symptomatic_isolation_duration_days="slider",
        pre_rash_isolation_success="toggle",
        pre_rash_isolation_start_day="slider",
        pre_rash_isolation_duration_days="slider",
        tf="number_input",
    )
    if widget_types is not None and isinstance(widget_types, dict):
        # update with parms if provided
        defaults.update(widget_types)
    return defaults


def get_min_values(parms=None):
    """
    Get the minimum values for the app parameters.

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
        vaccine_uptake_start_day=0,
        vaccine_uptake_duration_days=0,
        total_vaccine_uptake_doses=0.0,
        vaccinated_group=0,
        isolation_success=0.0,
        symptomatic_isolation_start_day=0,
        symptomatic_isolation_duration_days=0,
        pre_rash_isolation_success=0.0,
        pre_rash_isolation_start_day=0,
        pre_rash_isolation_duration_days=0,
        tf=30,
    )
    # update with parms if provided
    if parms is not None and isinstance(parms, dict):
        defaults.update(parms)
    return defaults


def get_max_values(parms=None):
    """
    Get the maximum values for the app parameters.

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
        vaccine_uptake_start_day=365,
        vaccine_uptake_duration_days=365,
        total_vaccine_uptake_doses=100.0,
        vaccinated_group=2,
        isolation_success=0.75,
        symptomatic_isolation_start_day=365,
        symptomatic_isolation_duration_days=365,
        pre_rash_isolation_success=1.0,
        pre_rash_isolation_start_day=365,
        pre_rash_isolation_duration_days=365,
        tf=400,
    )
    # update with parms if provided
    if parms is not None and isinstance(parms, dict):
        defaults.update(parms)
    return defaults


def get_step_values(parms=None):
    """
    Get the step or increment values for the app parameters.

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
        vaccine_uptake_start_day=1,
        vaccine_uptake_duration_days=7,
        total_vaccine_uptake_doses=5.0,
        vaccinated_group=1,
        isolation_success=0.01,
        symptomatic_isolation_start_day=1,
        symptomatic_isolation_duration_days=1,
        pre_rash_isolation_success=0.01,
        pre_rash_isolation_start_day=1,
        pre_rash_isolation_duration_days=1,
        tf=1,
    )
    # update with parms if provided
    if parms is not None and isinstance(parms, dict):
        defaults.update(parms)
    return defaults


def get_helpers(parms=None):
    """
    Get the help text for the app parameters.

    Args:
        parms (dict): Optional parameters dictionary.

    Returns:
        dict: A dictionary of help text for the app parameters.
    """
    defaults = dict(
        desired_r0="Basic reproduction number R0. R0 cannot be negative",
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
        latent_duration="Latent period (days)",
        infectious_duration="Infectious period (days)",
        I0=[
            "Initial infections in large population",
            "Initial infections in small population 1",
            "Initial infections in small population 2",
        ],
        initial_vaccine_coverage=[
            "Baseline vaccination coverage in large population",
            "Baseline vaccination coverage in small population 1",
            "Baseline vaccination coverage in small population 2",
        ],
        vaccine_uptake_start_day="Day vaccination starts",
        vaccine_uptake_duration_days="Days vaccines are administered",
        total_vaccine_uptake_doses="Percent of unvaccinated individuals that get vaccinated",
        vaccinated_group="Population receiving the vaccine",
        isolation_success="If turned on, 75% of symptomatic individuals isolate",
        symptomatic_isolation_start_day="Day symptomatic isolation starts",
        symptomatic_isolation_duration_days="Duration of symptomatic isolation",
        pre_rash_isolation_success="If turned on, 10% of individuals stay at home after being exposed",
        pre_rash_isolation_start_day="Day pre-rash isolation starts",
        pre_rash_isolation_duration_days="Duration of pre-rash isolation",
        tf="Number of time steps to simulate",
    )
    if parms is not None and isinstance(parms, dict):
        # update with parms if provided
        defaults.update(parms)
    return defaults


def get_formats(parms=None):
    """
    Get the formats for the app parameters.

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
        vaccine_uptake_start_day="%.0d",
        vaccine_uptake_duration_days="%.0d",
        total_vaccine_uptake_doses="%.1f",
        vaccinated_group="%.0d",
        isolation_success="%.2f",
        symptomatic_isolation_start_day="%.0d",
        symptomatic_isolation_duration_days="%.0d",
        pre_rash_isolation_success="%.2f",
        pre_rash_isolation_start_day="%.0d",
        pre_rash_isolation_duration_days="%.0d",
        tf="%.0d",
    )
    if parms is not None and isinstance(parms, dict):
        # update with parms if provided
        defaults.update(parms)
    return defaults


def get_base_widget_idkeys(parms=None):
    widget_idkeys = dict(
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
        vaccine_uptake_start_day="vaccine_uptake_start_day",
        vaccine_uptake_duration_days="vaccine_uptake_duration_days",
        total_vaccine_uptake_doses="total_vaccine_uptake_doses",
        vaccinated_group="vaccinated_group",
        isolation_success="isolation_success",
        symptomatic_isolation_start_day="symptomatic_isolation_start_day",
        symptomatic_isolation_duration_days="symptomatic_isolation_duration_days",
        pre_rash_isolation_success="pre_rash_isolation_success",
        pre_rash_isolation_start_day="pre_rash_isolation_start_day",
        pre_rash_isolation_duration_days="pre_rash_isolation_duration_days",
        tf="tf",
    )
    if parms is not None and isinstance(parms, dict):
        # update with parms if provided
        widget_idkeys.update(parms)
    return widget_idkeys


def get_widget_idkeys(widget_no):
    widget_idkeys = get_base_widget_idkeys()
    for key, value in widget_idkeys.items():
        if isinstance(value, list):
            widget_idkeys[key] = [
                f"{widget_idkeys[key][i]}_{widget_no}"
                for i in range(len(widget_idkeys[key]))
            ]
        else:
            widget_idkeys[key] = f"{widget_idkeys[key]}_{widget_no}"

    return widget_idkeys


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
    if len(split_key) > 1:
        # remove the value at the end of the key - this is used for naming
        # purposes to make each key unique
        split_key = split_key[:-1]

        # check if the last element of the split key is a number - this means
        # the value of this parameter was stored in a list in the model parameter
        if split_key[-1].isdigit():
            index = int(split_key[-1])
            split_key = split_key[:-1]
        key = "_".join(split_key)

    return key, index


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

        if key == "":
            continue
        if index == "":
            value = defaults[key]
        elif isinstance(index, int):
            value = defaults[key][index]
        else:
            raise ValueError(f"Invalid index type: {type(index)} for key: {key}")

        # by default, turn toggles off
        if widget_types[key] == "toggle":
            value = False

        # set the session state value to the default value
        st.session_state[session_key] = value

    # reset the session state for the app
    st.session_state["reset"] = True


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
    # Parameter, No Interventions, Interventions
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
    results = results.with_columns(pl.lit(None).alias("inc"))
    unique_replicates = results.select("replicate").unique().to_series().to_list()
    updated_rows = []

    for replicate in unique_replicates:
        tempdf = results.filter(pl.col("replicate") == replicate)
        for group in groups:
            group_data = tempdf.filter(pl.col("group") == group)
            group_data = group_data.sort("t")
            inc = group_data["Y"] - group_data["Y"].shift(1)
            group_data = group_data.with_columns(inc.alias("inc"))
            updated_rows.append(group_data)

    results = pl.concat(updated_rows, how="vertical")
    return results


def get_interval_cumulative_incidence(results, groups, interval=7):
    """
    Calculate cumulative incidence over specified intervals.

    Args:
        results (pl.DataFrame): The results DataFrame.
        groups          (list): List of group indices.
        interval         (int): The interval in days.

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
        interval         (int): The interval in days

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
    interval_results = interval_results.drop("inc")
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
def calculate_outbreak_summary(combined_results, threshold):
    """
    Calculate the outbreak summary based on the given threshold.

    Args:
        combined_results (pl.DataFrame): The combined results DataFrame.
        threshold (int): The threshold for filtering replicates.

    Returns:
        pl.DataFrame: A DataFrame containing the outbreak summary.
    """
    # Filter combined_results for replicates where Total >= threshold
    filtered_results = combined_results.filter(pl.col("Total") >= threshold)

    # Group by Scenario and count unique replicates
    outbreak_summary = filtered_results.group_by("Scenario").agg(
        pl.col("replicate").n_unique().alias("outbreaks")
    )

    # Ensure both scenarios are present in the summary
    scenarios = ["No Interventions", "Interventions"]
    for scenario in scenarios:
        if scenario not in outbreak_summary["Scenario"].to_list():
            # Add missing scenario with outbreaks = 0
            outbreak_summary = outbreak_summary.vstack(
                pl.DataFrame({"Scenario": [scenario], "outbreaks": [0]}).with_columns(
                    pl.col("outbreaks").cast(
                        outbreak_summary.schema["outbreaks"]
                    )  # Match the type
                )
            )
    return outbreak_summary


def get_table(combined_results, IHR, edited_parms):
    """
    Calculate the hospitalization summary based on the given IHR.

    Args:
        combined_results (pl.DataFrame): The combined results DataFrame.
        IHR (float): The infection hospitalization rate.

    Returns:
        pl.DataFrame: A DataFrame containing the hospitalization summary.
    """
    # Calculate hospitalizations
    combined_results = combined_results.with_columns(
        pl.Series(
            name="Hospitalizations",
            values=np.random.binomial(combined_results["Total"].to_numpy(), IHR),
        )
    )

    # Group by Scenario and get mean hospitalizations
    hospitalization_summary = (
        combined_results.group_by("Scenario")
        .mean()
        .with_columns(
            [
                pl.col("Hospitalizations").round_sig_figs(2),
                pl.col("Total").round_sig_figs(2),
            ]
        )
        .drop("replicate")
        .rename(
            {
                "Total": "Mean Outbreak Size",
                "Hospitalizations": "Mean Number of Hospitalizations",
            }
        )
    )

    # Ensure the order of scenarios
    scenario_names = ["No Interventions", "Interventions"]

    hospitalization_summary = (
        hospitalization_summary.with_columns(
            pl.when(pl.col("Scenario") == scenario_names[0])
            .then(0)
            .when(pl.col("Scenario") == scenario_names[1])
            .then(1)
            .otherwise(2)
            .alias("_sort_order")
        )
        .sort("_sort_order")
        .drop("_sort_order")
    )

    dose_vec = [0, edited_parms["total_vaccine_uptake_doses"]]
    isolation_vec = [0, int(edited_parms["pre_rash_isolation_success"] * 100)]
    symp_vec = [0, int(edited_parms["isolation_success"] * 100)]

    intervention_summary = pl.DataFrame(
        {
            "Scenario": scenario_names,
            "Vaccines Administered": dose_vec,
            "Stay-at-home Success (%)": isolation_vec,
            "Symptomatic Isolation Success (%)": symp_vec,
        }
    )

    outbreak_summary = (
        intervention_summary.join(hospitalization_summary, on="Scenario", how="inner")
        .drop("Scenario")
        .transpose(include_header=True)
        .rename(
            {"column": "", "column_0": scenario_names[0], "column_1": scenario_names[1]}
        )
        .with_columns(
            (
                (pl.col(scenario_names[0]) - pl.col(scenario_names[1]))
                / pl.col(scenario_names[0])
                * 100
            )
            .round_sig_figs(3)
            .alias("Relative Difference (%)")
        )
    )
    return outbreak_summary


def get_median_trajectory(results):
    # data at the end of simulation
    max_t = results.select(pl.col("t").max()).item()  # Get the maximum value of t
    filtered_results = results.filter(pl.col("t") == max_t)
    median_R = filtered_results.select(
        pl.col("R").median()
    ).item()  # Get the median value of R

    # get replicate closest to the median value of R
    closest_replicate = (
        filtered_results.with_columns(
            (pl.col("R") - median_R).abs().alias("distance")
        )  # Calculate the absolute difference
        .sort("distance")  # Sort by the distance
        .select("replicate")  # Select the replicate column
        .head(1)  # Get the first (closest) replicate
        .item()
    )

    median_trajectory = results.filter(pl.col("replicate") == closest_replicate)
    return median_trajectory
