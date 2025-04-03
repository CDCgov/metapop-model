import streamlit as st
import numpy as np
import polars as pl
import altair as alt
import copy
from metapop.model import *
from metapop.helper import *
from metapop.app_helper import *

### Methods to create user inputs interfaces ###
def app_editors(element, scenario_name, parms, ordered_keys, list_keys, slide_keys, show_parameter_mapping, min_values, max_values, steps, helpers, formats, element_keys):
    """
    Create the sidebar for editing parameters.

    Args:
        element: The Streamlit element to place the sidebar in.
        scenario_name: The name of the scenario.
        parms: The parameters to edit.
        ordered_keys: The keys of the parameters to edit.
        list_keys: The keys of the parameters that are lists.
        slide_keys: The keys of the parameters that are sliders.
        show_parameter_mapping: The mapping of parameter names to display names.
        min_values: The minimum values for the parameters.
        max_values: The maximum values for the parameters.
        steps: The step sizes for the parameters.
        helpers: The help text for the parameters.
        formats: The formats for the parameters.
        element_keys: The keys for the Streamlit elements.

    Returns:
        edited_parms: The edited parameters.

    """
    edited_parms = copy.deepcopy(parms)

    with element:
        st.subheader(scenario_name)

        for key in ordered_keys:
            if key not in list_keys:
                if key in slide_keys:
                    value = st.slider(show_parameter_mapping[key],
                                      min_value=min_values[key], max_value=max_values[key],
                                      value=parms[key],
                                      step=steps[key],
                                      help=helpers[key],
                                      format=formats[key],
                                      key=element_keys[key],
                                      )
                else:
                    value = st.number_input(show_parameter_mapping[key],
                                        min_value=min_values[key], max_value=max_values[key],
                                        value=parms[key],
                                        step=steps[key],
                                        help=helpers[key],
                                        format=formats[key],
                                        key=element_keys[key])
                edited_parms[key] = value
            if key in list_keys:
                for index in range(len(parms[key])):
                    if key in slide_keys:
                        value = st.slider(show_parameter_mapping[f"{key}_{index}"],
                                            min_value=min_values[key][index], max_value=max_values[key][index],
                                            value=parms[key][index],
                                            step=steps[key],
                                            help=helpers[key][index],
                                            format=formats[key],
                                            key=element_keys[key][index])
                    else:
                        value = st.number_input(show_parameter_mapping[f"{key}_{index}"],
                                            min_value=min_values[key][index], max_value=max_values[key][index],
                                            value=parms[key][index],
                                            step=steps[key],
                                            help=helpers[key][index],
                                            format=formats[key],
                                            key=element_keys[key][index])
                    edited_parms[key][index] = value

    return edited_parms


def get_min_values():
    return dict(
            desired_r0=0.,
            pop_sizes=[15000, 100, 100],
            vaccine_uptake_start_day=0,
            vaccine_uptake_duration_days=0,
            total_vaccine_uptake_doses=0,
            I0=[0, 0, 0],
            initial_vaccine_coverage=[0., 0., 0.],
            k_i=[0., 0., 0.],
            latent_duration = 6.,
            infectious_duration = 5.,
            k_g1 = 0.,
            k_21 = 0.,
            isolation_success = 0.,
            symptomatic_isolation_start_day = 0,
            symptomatic_isolation_duration_days = 0,
            )

def get_max_values():
    return dict(
            desired_r0=20.0,
            pop_sizes=[100000, 15000, 15000],
            vaccine_uptake_start_day=400,
            vaccine_uptake_duration_days=100,
            total_vaccine_uptake_doses=1000,
            I0=[100, 100, 100],
            initial_vaccine_coverage=[1.00, 1.00, 1.00],
            k_i=[30., 30., 30.],
            latent_duration = 18.,
            infectious_duration = 10.,
            k_g1 = 30.,
            k_21 = 30.,
            isolation_success = 1.,
            symptomatic_isolation_start_day = 365,
            symptomatic_isolation_duration_days = 365,
            )

def get_steps():
    return dict(
            desired_r0=0.1,
            pop_sizes=100,
            vaccine_uptake_start_day=1,
            vaccine_uptake_duration_days=1,
            total_vaccine_uptake_doses=1,
            I0=1,
            initial_vaccine_coverage=0.01,
            k_i = 0.1,
            latent_duration = 0.1,
            infectious_duration = 0.1,
            k_g1 = 0.01,
            k_21 = 0.01,
            isolation_success = 0.01,
            symptomatic_isolation_start_day = 1,
            symptomatic_isolation_duration_days = 10,
            )

def get_helpers():
    return dict(
            desired_r0="Basic reproduction number (R0) can not be negative",
            pop_sizes=["Size of the large population (15,000 - 100,000)",
                       "Size of the small population 1 (0 - 15,000)",
                       "Size of the small population 2 (0 - 15,000)"],
            vaccine_uptake_start_day="Day vaccination starts",
            vaccine_uptake_duration_days="Duration of vaccination uptake",
            total_vaccine_uptake_doses="Total vaccine doses",
            I0=["Initial infections in large population",
                "Initial infections in small population 1",
                "Initial infections in small population 2"],
            initial_vaccine_coverage=["Baseline vaccination in large population",
                                      "Baseline vaccination in small population 1",
                                      "Baseline vaccination in small population 2"],
            k_i=["Average degree for large population",
                 "Average degree for small population 1",
                 "Average degree for small population 2"],
            latent_duration = "Latent period (days)",
            infectious_duration = "Infectious period (days)",
            k_g1 = "Average degree of small population 1 connecting to large population",
            k_21 = "Average degree between small populations",
            isolation_success = "Percentage of symptomatic cases isolated",
            symptomatic_isolation_start_day = "Day symptomatic isolation starts",
            symptomatic_isolation_duration_days = "Duration of symptomatic isolation",
            )

def get_formats():
    return  dict(
            desired_r0="%.1f",
            pop_sizes="%.0d",
            vaccine_uptake_start_day="%.0d",
            vaccine_uptake_duration_days="%.0d",
            total_vaccine_uptake_doses="%.0d",
            I0="%.0d",
            initial_vaccine_coverage="%.2f",
            k_i="%.1f",
            latent_duration = "%.1f",
            infectious_duration = "%.1f",
            k_g1 = "%.2f",
            k_21 = "%.2f",
            isolation_success = "%.2f",
            symptomatic_isolation_start_day = "%.0d",
            symptomatic_isolation_duration_days = "%.0d",
            )


def app(replicates=20):
    st.title("Metapopulation Model")
    parms = read_parameters()

    show_parameter_mapping = get_show_parameter_mapping()
    advanced_parameter_mapping = get_advanced_parameter_mapping()

    with st.sidebar:
        st.header(
            "Scenario parameters",
            help="Enter model parameters for each scenario. Hover over the ? for more information about each parameter.",
        )

        min_values = mt.get_min_values()
        max_values = mt.get_max_values()
        steps = mt.get_step_values()
        helpers = mt.get_helpers()
        formats = mt.get_formats()
        # this can likely be done more programmatically but works for now
        keys1 = dict(
            desired_r0="R0_1",
            pop_sizes=["pop_size_0_1", "pop_size_1_1", "pop_size_2_1"],
            vaccine_uptake_start_day="vaccine_uptake_days_0_1",
            vaccine_uptake_duration_days="vaccine_uptake_duration_days_0_1",
            total_vaccine_uptake_doses="total_vaccine_uptake_doses_1",
            I0=["I0_0_1", "I0_1_1", "I0_2_1"],
            initial_vaccine_coverage=["initial_vaccine_coverage_0_1", "initial_vaccine_coverage_1_1", "initial_vaccine_coverage_2_1"],
            k_i=["k_i_0_1", "k_i_1_1", "k_i_2_1"],
            latent_duration = "latent_duration_1",
            infectious_duration = "infectious_duration_1",
            k_g1 = "k_g1_1",
            k_21 = "k_21_1",
            isolation_success = "isolation_success_1",
            symptomatic_isolation_start_day = "symptomatic_isolation_start_day_1",
            symptomatic_isolation_duration_days = "symptomatic_isolation_duration_days_1",
            )
        keys2 = dict(
            desired_r0="R0_2",
            pop_sizes=["pop_size_0_2", "pop_size_1_2", "pop_size_2_2"],
            vaccine_uptake_start_day="vaccine_uptake_days_0_2",
            vaccine_uptake_duration_days="vaccine_uptake_duration_days_0_2",
            total_vaccine_uptake_doses="total_vaccine_uptake_doses_2",
            I0=["I0_0_2", "I0_1_2", "I0_2_2"],
            initial_vaccine_coverage=["initial_vaccine_coverage_0_2", "initial_vaccine_coverage_1_2", "initial_vaccine_coverage_2_2"],
            k_i=["k_i_0_2", "k_i_1_2", "k_i_2_2"],
            latent_duration = "latent_duration_2",
            infectious_duration= "infectious_duration_2",
            k_g1 = "k_g1_2",
            k_21 = "k_21_2",
            isolation_success = "isolation_success_2",
            symptomatic_isolation_start_day = "symptomatic_isolation_start_day_2",
            symptomatic_isolation_duration_days = "symptomatic_isolation_duration_days_2",
            )
        # order of parameters in the sidebar
        ordered_keys = [
                        # 'desired_r0',
                        'pop_sizes',
                        'I0',
                        'initial_vaccine_coverage',
                        'total_vaccine_uptake_doses',
                        'vaccine_uptake_start_day',
                        'vaccine_uptake_duration_days',
                        'isolation_success',
                        'symptomatic_isolation_start_day',
                        'symptomatic_isolation_duration_days',
                        ]

        list_parameter_keys = [
                              'pop_sizes',
                              'I0',
                              'initial_vaccine_coverage',
                              'vaccine_uptake_doses',
                              ]

        slide_keys = ['desired_r0','pop_sizes', 'I0']
        # number of scenarios
        col1, col2 = st.columns(2)

        edited_parms1 = mt.app_editors(
            col1, "Scenario 1", parms, ordered_keys, list_parameter_keys,
            slide_keys, show_parameter_mapping, min_values, max_values,
            steps, helpers, formats, keys1
        )

        edited_parms2 = mt.app_editors(
            col2, "Scenario 2", parms, ordered_keys, list_parameter_keys,
            slide_keys, show_parameter_mapping, min_values, max_values,
            steps, helpers, formats, keys2
        )

        with st.expander("Advanced options"):
            # try to place two sliders side by side
            advanced_ordered_keys = ["desired_r0", "latent_duration", "infectious_duration", "k_g1", "k_21", "k_i"]
            advanced_list_keys = ["k_i"]
            advanced_slide_keys = ["latent_duration", "infectious_duration"]

            adv_col1, adv_col2 = st.columns(2)

            edited_advanced_parms1 = mt.app_editors(
                adv_col1, "Scenario 1", edited_parms1, advanced_ordered_keys,
                advanced_list_keys, advanced_slide_keys, advanced_parameter_mapping,
                min_values, max_values, steps, helpers, formats, keys1
            )

            edited_advanced_parms2 = mt.app_editors(
                adv_col2, "Scenario 2", edited_parms2, advanced_ordered_keys,
                advanced_list_keys, advanced_slide_keys, advanced_parameter_mapping,
                min_values, max_values, steps, helpers, formats, keys2
            )
    # get the selected outcome from the sidebar
    outcome_option = st.selectbox(
        "Metric",
        get_outcome_options(),
        index=0,  # by default display weekly incidence
        placeholder="Select an outcome to plot",
    )

    # Map the selected option to the outcome variable
    outcome_mapping = get_outcome_mapping()
    outcome = outcome_mapping[outcome_option]

    full_defaults = get_default_full_parameters()

    # get updated parameter dictionaries
    updated_parms1 = get_parms_from_table(full_defaults, value_col="Scenario 1")
    updated_parms2 = get_parms_from_table(full_defaults, value_col="Scenario 2")

    # get updated values from user through the sidebar
    for key, value in edited_parms1.items():
        updated_parms1[key] = value
    for key, value in edited_parms2.items():
        updated_parms2[key] = value

    for key, value in edited_advanced_parms1.items():
        updated_parms1[key] = value
    for key, value in edited_advanced_parms2.items():
        updated_parms2[key] = value

    # correct types for single values
    updated_parms1 = correct_parameter_types(parms, updated_parms1)
    updated_parms2 = correct_parameter_types(parms, updated_parms2)

    scenario1 = [updated_parms1]
    scenario2 = [updated_parms2]

    # run the model with the updated parameters
    results1 = get_scenario_results(scenario1)
    results2 = get_scenario_results(scenario2)

    # extract groups
    groups = results1["group"].unique().to_list()

    # filter for a sample of replicates
    replicate_inds = np.random.choice(results1["replicate"].unique().to_numpy(), replicates, replace=False)
    results1 = results1.filter(pl.col("replicate").is_in(replicate_inds))
    results2 = results2.filter(pl.col("replicate").is_in(replicate_inds))

    # do some processing here to get daily incidence
    results1 = add_daily_incidence(results1, replicate_inds, groups)
    results2 = add_daily_incidence(results2, replicate_inds, groups)

    # create tables with interval results - weekly incidence, weekly cumulative incidence
    interval = 7

    interval_results1 = get_interval_results(results1, replicate_inds, groups, interval)
    interval_results2 = get_interval_results(results2, replicate_inds, groups, interval)

    # rename columns for the app
    app_column_mapping = {f"inc_{interval}": "Winc", "Y": "WCI"}
    interval_results1 = interval_results1.rename(app_column_mapping)
    interval_results2 = interval_results2.rename(app_column_mapping)

    # set up the color scale
    domain = [str(i) for i in range(len(groups))]
    group_labels = ["General population", "Small population 1", "Small population 2"]

    # plot with Altair
    color_scale = alt.Scale(
        domain=[str(i) for i in range(len(results1["group"].unique()))],
        range = [
            "#20419a", # blue
            "#cf4828", # red
            "#f78f47", # orange
        ]
    )
    if outcome not in ["I", "Y", "inc", "Winc", "WCI"]:
        print("outcome not available yet, defaulting to Cumulative Daily Incidence")
        outcome = "Y"

    if outcome_option in ['Daily Infections', 'Daily Incidence', 'Cumulative Daily Incidence']:
        alt_results1 = results1
        alt_results2 = results2
        min_y, max_y = 0, max(results1[outcome].max(), results2[outcome].max())
        x = "t:Q"
        time_label = "Time (days)"
    elif outcome_option in ['Weekly Incidence', 'Weekly Cumulative Incidence']:
        alt_results1 = interval_results1
        alt_results2 = interval_results2
        min_y, max_y = 0, max(interval_results1[outcome].max(), interval_results2[outcome].max())
        x = "interval_t:Q"
        time_label = "Time (weeks)"

    y = f"{outcome}:Q"
    yscale = [min_y, max_y]
    color_key = "group"
    labelExpr=f"datum.value == '0' ? '{group_labels[0]}' : datum.value == '1' ? '{group_labels[1]}' : '{group_labels[2]}'"
    detail="replicate:N"

    chart1 = create_chart(alt_results1, outcome_option,
                          x, time_label,
                          y, outcome_option, yscale,
                          color_key, color_scale, domain,
                          labelExpr,
                          detail)
    chart2 = create_chart(alt_results2, outcome_option,
                          x, time_label,
                          y, outcome_option, yscale,
                          color_key, color_scale, domain,
                          labelExpr,
                          detail)
    st.altair_chart(chart1 | chart2, use_container_width=True)



if __name__ == "__main__":
    app()
