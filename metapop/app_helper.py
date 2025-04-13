# This file contains helper functions for the metapop app.
import streamlit as st
import numpy as np
import polars as pl
import altair as alt
import griddler
import griddler.griddle
import copy
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
    "get_parms_from_table",
    "update_parms_from_table",
    "correct_parameter_types",
    "add_daily_incidence",
    "get_interval_cumulative_incidence",
    "get_interval_results",
    "create_chart",
    "calculate_outbreak_summary",
    "get_hospitalizations",
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
    results = results.select([
        "k_21", "t", "group",
        "S", "V", "E1", "E2", "I1", "I2", "R", "Y", "X", "replicate"
    ])
    # add a column for total infections
    results = results.with_columns((pl.col("I1") + pl.col("I2")).alias("I"))
    return results


### Methods to read in default parameters ###
def read_parameters(filepath="scripts/app/app_config.yaml"):
    """"
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
            parms['{}_{}'.format(key, i)] = value

        del parms[key]

    keys = [key for key in parms.keys()]
    values = [parms[key] for key in keys]

    defaults = pl.DataFrame(
        {
            "Parameter": keys,
            "Scenario 1": values,
            "Scenario 2": values,
        },
        strict=False
    )
    return defaults


def get_default_show_parameters_table():
    """"
    Get a Dataframe of the default simulation parameters that users always see
    in the app sidebar. This Dataframe contains default values for two
    scenarios that can be updated by the user through other methods.

    Returns:
        pl.DataFrame: DataFrame containing the default parameters and their values for two scenarios.
    """

    full_defaults = get_default_full_parameters()
    show_parameter_mapping = get_show_parameter_mapping()
    show_defaults = full_defaults.filter(pl.col("Parameter").is_in(show_parameter_mapping.keys()))

    # replace specific values with integers
    for key in ['Scenario 1', 'Scenario 2']:
        show_defaults = show_defaults.with_columns(
            pl.when(pl.col(key).str.to_lowercase() == "true")
            .then(1)
            .when(pl.col(key).str.to_lowercase() == "false")
            .then(0)
            .otherwise(pl.col(key))
            .alias(key)
        )

    # cast to float
    show_defaults = show_defaults.with_columns(pl.col("Scenario 1").cast(pl.Float64))
    show_defaults = show_defaults.with_columns(pl.col("Scenario 2").cast(pl.Float64))

    # renaming keys with longer names
    show_defaults = show_defaults.with_columns(
        pl.Series(name="Parameter",
                  values=[show_parameter_mapping.get(key) for key in show_defaults["Parameter"].to_list()]))

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
    for key in ['Scenario 1', 'Scenario 2']:
        advanced_defaults = advanced_defaults.with_columns(
            pl.when(pl.col(key).str.to_lowercase() == "true")
            .then(1)
            .when(pl.col(key).str.to_lowercase() == "false")
            .then(0)
            .otherwise(pl.col(key))
            .alias(key)
        )

    # cast to float
    advanced_defaults = advanced_defaults.with_columns(pl.col("Scenario 1").cast(pl.Float64))
    advanced_defaults = advanced_defaults.with_columns(pl.col("Scenario 2").cast(pl.Float64))

    # renaming keys with longer names
    advanced_defaults = advanced_defaults.with_columns(
        pl.Series(name="Parameter",
                  values=[advanced_parameter_mapping.get(key) for key in advanced_defaults["Parameter"].to_list()]))
    return advanced_defaults



### Methods to handle how parameters are displayed in the app ###
def get_show_parameter_mapping():
    """
    Get a mapping of parameter names to their display names.

    Returns:
        dict: A dictionary mapping parameter names to their display names.
    """
    # Define the mapping of parameter names to display names
    show_mapping = dict(
        # n_groups = "number of groups",
        # desired_r0="R0",
        # k = "average degree",
        # k_i_0 = "average degree per person in large population",
        # k_i_1 = "average degree per person in small population 1",
        # k_i_2 = "average degree per person in small population 2",
        # k_g1 = "average degree of small population 1 connecting to large population",
        # k_g2 = "average degree of small population 2 connecting to large population",
        # k_21 = "average degree between small populations",
        # connectivity_scenario = "connectivity scenario",
        # n_e_compartments = "number of exposed compartments",
        # latent_duration = "latent period",
        # n_i_compartments = "number of infectious compartments",
        # infectious_duration = "infectious period",
        pop_sizes_0 = "size of large population",
        pop_sizes_1 = "size of small population 1",
        pop_sizes_2 = "size of small population 2",
        I0_0="initial infections in large population",
        I0_1="initial infections in small population 1",
        I0_2="initial infections in small population 2",
        # vaccine_uptake = "Enable vaccine uptake",
        total_vaccine_uptake_doses="vaccine doses total",
        vaccine_uptake_start_day="active vaccination start day",
        vaccine_uptake_duration_days="active vaccination duration days",
        vaccinated_group = "vaccinated group",
        # symptomatic_isolation = "Enable symptomatic isolation",
        isolation_success = "Symptomatic isolation proportion",
        symptomatic_isolation_start_day = "Symptomatic isolation start day",
        symptomatic_isolation_duration_days = "Symptomatic isolation duration days",
        pre_rash_isolation_success = "Pre-rash isolation proportion",
        pre_rash_isolation_start_day = "Pre-rash isolation start day",
        pre_rash_isolation_duration_days = "Pre-rash isolation duration days",
        tf = "time steps",
        # n_replicates = "number of replicates",
        # seed = "random seed",
        initial_vaccine_coverage_0 = "baseline vaccination in large population",
        initial_vaccine_coverage_1 = "baseline vaccination in small population 1",
        initial_vaccine_coverage_2 = "baseline vaccination in small population 2",
    )
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
        n_groups="number of groups",
        infectious_duration="infectious period",
        latent_duration="latent period",
        # n_e_compartments="number of exposed compartments",
        # n_i_compartments="number of infectious compartments",
        # tf="number of time steps",
        # pop_sizes_0="size of large population",
        # pop_sizes_1="size of small population 1",
        # pop_sizes_2="size of small population 2",
        # k_i = "average degree",
        k_i_0 = "average degree for large population",
        k_i_1 = "average degree for small population 1",
        k_i_2 = "average degree for small population 2",
        k_g1 = "average degree of small population 1 connecting to large population",
        k_g2 = "average degree of small population 2 connecting to large population",
        k_21 = "connectivity between smaller populations",
        # vaccine_uptake="vaccine uptake",
        # vaccine_uptake_doses="vaccine doses",
        # isolation_percentage="isolation percentage",
        # isolation_effectiveness="isolation effectiveness",
    )
    return advanced_mapping


def get_outcome_options():
    """
    Get the available outcome options for the app.

    Returns:
        tuple: A tuple containing the available outcome options.
    """
    return (
            # "Weekly Infections",
            "Weekly Incidence", "Weekly Cumulative Incidence",
            "Daily Infections", "Daily Incidence", "Cumulative Daily Incidence",
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
        "Cumulative Daily Incidence": "Y",
        # "Weekly Infections": "WI",
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
    keys_in_list = [key for key in updated_parms.keys() if any(key.startswith(list_key) for list_key in list_keys)]
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
        if isinstance(parms[list_key][0], int) and not isinstance(parms[list_key][0], bool):
            updated_parms[list_key].append(int(updated_parms[key]))
        elif isinstance(parms[list_key][0], bool):
            updated_parms[list_key].append(True if updated_parms[key] in [True, 'TRUE', 'True', 'true', '1'] else False)
        elif isinstance(parms[list_key][0], float):
            updated_parms[list_key].append(float(updated_parms[key]))
        elif isinstance(parms[list_key][0], str):
            updated_parms[list_key].append(str(updated_parms[key]))

    for key in keys_in_list:
        del updated_parms[key]

    return updated_parms




### Methods to create user inputs interfaces ###
def app_editors(element, scenario_name, parms,
                ordered_keys, list_keys, slider_keys,
                widget_types,
                show_parameter_mapping,
                min_values, max_values, steps, helpers, formats, element_keys,
                disabled=False):
    """
    Create the sidebar for editing parameters.

    Args:
        element (st container object): The Streamlit element to place the sidebar in.
        scenario_name (str): The name of the scenario.
        parms (dict): The parameters to edit.
        ordered_keys (list): An ordered list of the parameters to edit.
        list_keys (list): The keys of the parameters that are lists.
        slider_keys (list): The keys of the parameters that are sliders.
        widget_types (dict): The types of widgets for the parameters.
        show_parameter_mapping (dict): The mapping of parameter names to display names.
        min_values (dict): The minimum values for the parameters.
        max_values (dict): The maximum values for the parameters.
        steps (dict): The step sizes for the parameters.
        helpers (dict): The help text for the parameters.
        formats (dict): The formats for the parameters.
        element_keys (dict): The keys for the Streamlit elements.
        disabled (bool): Whether the widgets should be disabled. Defaults to False.

    Returns:
        edited_parms: The edited parameters.

    """
    edited_parms = copy.deepcopy(parms)

    with element:
        st.subheader(scenario_name)

        for key in ordered_keys:
            if key not in list_keys:
                # if key in slider_keys:
                if widget_types[key] == 'slider':
                    value = st.slider(show_parameter_mapping[key],
                                      min_value=min_values[key], max_value=max_values[key],
                                      value=parms[key],
                                      step=steps[key],
                                      help=helpers[key],
                                      format=formats[key],
                                      key=element_keys[key],
                                      disabled=disabled
                                      )
                elif widget_types[key] == 'number_input':
                    value = st.number_input(show_parameter_mapping[key],
                                        min_value=min_values[key], max_value=max_values[key],
                                        value=parms[key],
                                        step=steps[key],
                                        help=helpers[key],
                                        format=formats[key],
                                        key=element_keys[key],
                                        disabled=disabled)
                else:
                    pass
                edited_parms[key] = value
            if key in list_keys:
                for index in range(len(parms[key])):
                    # if key in slider_keys:
                    if widget_types[key] == 'slider':
                        value = st.slider(show_parameter_mapping[f"{key}_{index}"],
                                            min_value=min_values[key][index], max_value=max_values[key][index],
                                            value=parms[key][index],
                                            step=steps[key],
                                            help=helpers[key][index],
                                            format=formats[key],
                                            key=element_keys[key][index],
                                            disabled=disabled)
                    # else:
                    elif widget_types[key] == 'number_input':
                        value = st.number_input(show_parameter_mapping[f"{key}_{index}"],
                                            min_value=min_values[key][index], max_value=max_values[key][index],
                                            value=parms[key][index],
                                            step=steps[key],
                                            help=helpers[key][index],
                                            format=formats[key],
                                            key=element_keys[key][index],
                                            disabled=disabled)
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
            k_i = "slider",
            k_g1 = "number_input",
            k_g2 = "number_input",
            k_21 = "number_input",
            pop_sizes="slider",
            latent_duration="slider",
            infectious_duration="slider",
            I0="slider",
            initial_vaccine_coverage = "slider",
            vaccine_uptake_start_day="slider",
            vaccine_uptake_duration_days="slider",
            total_vaccine_uptake_doses="slider",
            vaccinated_group="number_input",
            isolation_success = "slider",
            symptomatic_isolation_start_day = "slider",
            symptomatic_isolation_duration_days = "slider",
            pre_rash_isolation_success = "slider",
            pre_rash_isolation_start_day = "slider",
            pre_rash_isolation_duration_days = "slider",
            tf="number_input"
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
            desired_r0=0.,
            k_i = [0., 0., 0.],
            k_g1 = 0.,
            k_g2 = 0.,
            k_21 = 0.,
            pop_sizes=[15000, 100, 100],
            latent_duration=6.,
            infectious_duration=5.,
            I0=[0, 0, 0],
            initial_vaccine_coverage=[0., 0., 0.],
            vaccine_uptake_start_day=0,
            vaccine_uptake_duration_days=0,
            total_vaccine_uptake_doses=0,
            vaccinated_group=0,
            isolation_success = 0.,
            symptomatic_isolation_start_day = 0,
            symptomatic_isolation_duration_days = 0,
            pre_rash_isolation_success = 0.,
            pre_rash_isolation_start_day = 0,
            pre_rash_isolation_duration_days = 0,
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
            desired_r0=30.,
            k_i = [50., 50., 50.],
            k_g1 = 50.,
            k_g2 = 50.,
            k_21 = 50.,
            pop_sizes=[100_000, 15_000, 15_000],
            latent_duration=18.,
            infectious_duration=11.,
            I0=[100, 100, 100],
            initial_vaccine_coverage=[1., 1., 1.],
            vaccine_uptake_start_day=365,
            vaccine_uptake_duration_days=365,
            total_vaccine_uptake_doses=1000,
            vaccinated_group=2,
            isolation_success = 0.75,
            symptomatic_isolation_start_day = 365,
            symptomatic_isolation_duration_days = 365,
            pre_rash_isolation_success = 1.,
            pre_rash_isolation_start_day = 365,
            pre_rash_isolation_duration_days = 365,
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
            k_i = 0.1,
            k_g1 = 0.01,
            k_g2 = 0.01,
            k_21 = 0.01,
            pop_sizes=100,
            latent_duration=0.1,
            infectious_duration=0.1,
            I0=1,
            initial_vaccine_coverage = 0.01,
            vaccine_uptake_start_day = 1,
            vaccine_uptake_duration_days = 1,
            total_vaccine_uptake_doses = 1,
            vaccinated_group = 1,
            isolation_success = 0.01,
            symptomatic_isolation_start_day = 1,
            symptomatic_isolation_duration_days = 1,
            pre_rash_isolation_success = 0.01,
            pre_rash_isolation_start_day = 1,
            pre_rash_isolation_duration_days = 1,
            tf = 1,
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
            k_i=["Average daily contacts for large population",
                 "Average daily contacts for small population 1",
                 "Average daily contacts for small population 2"],
            k_g1 = "Average daily contact per person in small population 1 with people in the large population",
            k_g2 = "Average daily contact per person in small population 2 with people in the large population",
            k_21 = "Average daily contact between people in the small populations",
            pop_sizes=["Size of the large population",
                       "Size of the small population 1",
                       "Size of the small population 2"],
            latent_duration = "Latent period (days)",
            infectious_duration = "Infectious period (days)",
            I0=["Initial infections in large population",
                "Initial infections in small population 1",
                "Initial infections in small population 2"],
            initial_vaccine_coverage=["Baseline vaccination coverage in large population",
                                      "Baseline vaccination coverage in small population 1",
                                      "Baseline vaccination coverage in small population 2"],
            vaccine_uptake_start_day="Day vaccination starts",
            vaccine_uptake_duration_days="Days vaccines are administered",
            total_vaccine_uptake_doses="Total vaccine doses administered during the vaccination campaign",
            vaccinated_group="Population receiving the vaccine",
            isolation_success = "Proportion reduction in contacts due to symptomatic isolation",
            symptomatic_isolation_start_day = "Day symptomatic isolation starts",
            symptomatic_isolation_duration_days = "Duration of symptomatic isolation",
            pre_rash_isolation_success = "Proportion reduction in contacts due to pre-rash isolation",
            pre_rash_isolation_start_day = "Day pre-rash isolation starts",
            pre_rash_isolation_duration_days = "Duration of pre-rash isolation",
            tf = "Number of time steps to simulate",
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
            k_i = "%.1f",
            k_g1 = "%.2f",
            k_g2 = "%.2f",
            k_21 = "%.2f",
            pop_sizes="%.0d",
            latent_duration = "%.1f",
            infectious_duration = "%.1f",
            I0="%.0d",
            initial_vaccine_coverage="%.2f",
            vaccine_uptake_start_day="%.0d",
            vaccine_uptake_duration_days="%.0d",
            total_vaccine_uptake_doses="%.0d",
            vaccinated_group="%.0d",
            isolation_success = "%.2f",
            symptomatic_isolation_start_day = "%.0d",
            symptomatic_isolation_duration_days = "%.0d",
            pre_rash_isolation_success = "%.2f",
            pre_rash_isolation_start_day = "%.0d",
            pre_rash_isolation_duration_days = "%.0d",
            tf="%.0d",
            )
    if parms is not None and isinstance(parms, dict):
        # update with parms if provided
        defaults.update(parms)
    return defaults


def get_base_widget_idkeys(parms=None):
    widget_idkeys = dict(
        desired_r0="R0",
        k_i=["k_i_0", "k_i_1", "k_i_2"],
        k_g1 = "k_g1",
        k_g2 = "k_g2",
        k_21 = "k_21",
        pop_sizes=["pop_size_0",
                   "pop_size_1",
                   "pop_size_2"],
        latent_duration = "latent_duration",
        infectious_duration = "infectious_duration",
        I0=["I0_0", "I0_1", "I0_2"],
        initial_vaccine_coverage=["initial_vaccine_coverage_0",
                                  "initial_vaccine_coverage_1",
                                  "initial_vaccine_coverage_2"],
        vaccine_uptake_start_day="vaccine_uptake_start_day",
        vaccine_uptake_duration_days="vaccine_uptake_duration_days",
        total_vaccine_uptake_doses="total_vaccine_uptake_doses",
        vaccinated_group = "vaccinated_group",
        isolation_success = "isolation_success",
        symptomatic_isolation_start_day = "symptomatic_isolation_start_day",
        symptomatic_isolation_duration_days = "symptomatic_isolation_duration_days",
        pre_rash_isolation_success = "pre_rash_isolation_success",
        pre_rash_isolation_start_day = "pre_rash_isolation_start_day",
        pre_rash_isolation_duration_days = "pre_rash_isolation_duration_days",
        tf = "tf",
        )
    if parms is not None and isinstance(parms, dict):
        # update with parms if provided
        widget_idkeys.update(parms)
    return widget_idkeys

def get_widget_idkeys(widget_no):
    widget_idkeys = get_base_widget_idkeys()
    for key, value in widget_idkeys.items():
        if isinstance(value, list):
            widget_idkeys[key] = [f"{widget_idkeys[key][i]}_{widget_no}" for i in range(len(widget_idkeys[key]))]
        else:
            widget_idkeys[key] = f"{widget_idkeys[key]}_{widget_no}"

    return widget_idkeys


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
    # Parameter, Scenario 1, Scenario 2
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
            if parms_from_table[key] in [True, 'TRUE', 'True', 'true', '1']:
                parms_from_table[key] = True
            else:
                parms_from_table[key] = False
        elif isinstance(value, float):
            parms_from_table[key] = float(parms_from_table[key])
        elif isinstance(value, str):
            parms_from_table[key] = str(parms_from_table[key])
    return parms_from_table


### Methods to calculate different metrics from simulation results ###
def add_daily_incidence(results, replicate_inds, groups):
    """
    Add daily incidence to the results DataFrame."

    Args:
        results (pl.DataFrame): The results DataFrame.
        replicate_inds  (list): List of replicate indices.
        groups          (list): List of group indices.

    Returns:
        pl.DataFrame: The updated results DataFrame with daily incidence added.
    """
    # add a column for daily incidence
    results = results.with_columns(pl.lit(None).alias("inc"))
    updated_rows = []

    for replicate in replicate_inds:
        tempdf = results.filter(pl.col("replicate") == replicate)
        for group in groups:
            group_data = tempdf.filter(pl.col("group") == group)
            group_data = group_data.sort("t")
            inc = group_data["Y"] - group_data["Y"].shift(1)
            group_data = group_data.with_columns(inc.alias("inc"))
            updated_rows.append(group_data)

    results = pl.concat(updated_rows, how="vertical")
    return results


def get_interval_cumulative_incidence(results, replicate_inds, groups, interval=7):
    """"
    Calculate cumulative incidence over specified intervals.

    Args:
        results (pl.DataFrame): The results DataFrame.
        replicate_inds  (list): List of replicate indices.
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
    repeated_interval_points = np.tile(interval_points, len(groups) * len(replicate_inds))
    # add the interval time points to the interval results table
    interval_results = interval_results.with_columns(pl.Series(name="interval_t", values=repeated_interval_points))

    # now Y is the cumulative incidence at each time point and interval_t is the interval time point
    return interval_results


def get_interval_results(results, replicate_inds, groups, interval=7):
    """"
    Calculate interval results for cumulative incidence.

    Args:
        results (pl.DataFrame): The results DataFrame.
        replicate_inds  (list): List of replicate indices.
        groups          (list): List of group indices.
        interval         (int): The interval in days

    Returns:
         pl.DataFrame: The updated results DataFrame with interval cumulative incidence added.""
    """
    # get a table with results, in particular cumulative incidence at each interval time point
    # in this table, Y is the cumulative incidence
    interval_results = get_interval_cumulative_incidence(results, replicate_inds, groups, interval)
    # now process this table to get the interval incidence
    updated_rows = []
    for replicate in replicate_inds:
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
def create_chart(alt_results, outcome_option, x, xlabel, y, ylabel, yscale, color_key, color_scale, domain, labelExpr, detail, width=300, height=300):
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
    chart = alt.Chart(
        alt_results,
        title=outcome_option
        ).mark_line(opacity=0.4).encode(
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
        ).properties(width=width, height=height)
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
    outbreak_summary = (
        filtered_results
        .group_by("Scenario")
        .agg(pl.col("replicate").n_unique().alias("outbreaks"))
    )

    # Ensure both scenarios are present in the summary
    scenarios = ["Scenario 1 (Baseline)", "Scenario 2"]
    for scenario in scenarios:
        if scenario not in outbreak_summary["Scenario"].to_list():
            # Add missing scenario with outbreaks = 0
            outbreak_summary = outbreak_summary.vstack(
                pl.DataFrame({"Scenario": [scenario], "outbreaks": [0]}).with_columns(
                    pl.col("outbreaks").cast(outbreak_summary.schema["outbreaks"])  # Match the type
                )
            )
    return outbreak_summary

def get_hospitalizations(combined_results, IHR):
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
            name = "Hospitalizations",
            values = np.random.binomial(combined_results["Total"].to_numpy(), IHR)))

    # Group by Scenario and get mean hospitalizations
    hospitalization_summary = (
        combined_results
        .group_by("Scenario")
        .mean()
        .with_columns([pl.col("Hospitalizations").cast(pl.Int64), pl.col("Total").cast(pl.Int64)])
        .drop("replicate")
        .rename({"Total": "Total Infections"})
    )

    # Ensure the order of scenarios
    scenario_order = ["Scenario 1 (Baseline)", "Scenario 2"]
    hospitalization_summary = hospitalization_summary.with_columns(
        pl.when(pl.col("Scenario") == scenario_order[0])
        .then(0)
        .when(pl.col("Scenario") == scenario_order[1])
        .then(1)
        .otherwise(2)
        .alias("_sort_order")
    ).sort("_sort_order").drop("_sort_order")

    return hospitalization_summary
