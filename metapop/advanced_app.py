import os

import altair as alt
import numpy as np
import polars as pl
import streamlit as st

# import what's needed from other metapop modules
from .app_helper import (
    add_daily_incidence,
    app_editors,
    create_chart,
    get_advanced_parameter_mapping,
    get_formats,
    get_helpers,
    get_interval_results,
    get_max_values,
    get_min_values,
    get_outcome_mapping,
    get_outcome_options,
    get_scenario_results,
    get_show_parameter_mapping,
    get_step_values,
    get_widget_idkeys,
    get_widget_types,
    read_parameters,
    update_intervention_parameters_from_widget,
)
from .helper import seed_from_string

__all__ = [
    "advanced_app",
]


def advanced_app(replicates=20):
    st.title("Measles Outbreak Simulator")
    st.text(
        "This interactive tool illustrates the impact of vaccination and isolation on the probability and size of measles outbreaks following introduction of measles into different connected communities."
    )
    filepath = os.path.join(os.path.dirname(__file__), "app_assets", "app_config.yaml")
    parms = read_parameters(filepath)
    plot_rng = np.random.default_rng([parms["seed"], seed_from_string("plot")])

    show_parameter_mapping = get_show_parameter_mapping()
    advanced_parameter_mapping = get_advanced_parameter_mapping()

    with st.sidebar:
        st.header(
            "Scenario parameters",
            help="Enter model parameters for each scenario. Hover over the ? for more information about each parameter.",
        )

        widget_types = (
            get_widget_types()
        )  # defines the type of widget for each parameter
        min_values = get_min_values()
        max_values = get_max_values()
        steps = get_step_values()
        helpers = get_helpers()
        formats = get_formats()
        keys1 = get_widget_idkeys(1)
        keys2 = get_widget_idkeys(2)

        # order of parameters in the sidebar
        ordered_keys = [
            "pop_sizes",
            "I0",
            "initial_vaccine_coverage",
            "total_vaccine_uptake_doses",
            "vaccine_uptake_start_day",
            "vaccine_uptake_duration_days",
            "isolation_adherence",
            "symptomatic_isolation_start_day",
            "symptomatic_isolation_duration_days",
        ]

        # list of parameters that are lists or arrays
        list_parameter_keys = [
            "pop_sizes",
            "I0",
            "initial_vaccine_coverage",
            "vaccine_uptake_doses",
        ]

        # number of scenarios
        col1, col2 = st.columns(2)

        edited_parms1 = app_editors(
            col1,
            "No Interventions",
            parms,
            ordered_keys,
            list_parameter_keys,
            show_parameter_mapping,
            widget_types,
            min_values,
            max_values,
            steps,
            helpers,
            formats,
            keys1,
        )

        edited_parms2 = app_editors(
            col2,
            "Interventions",
            parms,
            ordered_keys,
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

        with st.expander("Advanced options"):
            # try to place two sliders side by side
            advanced_ordered_keys = [
                "desired_r0",
                "latent_duration",
                "infectious_duration",
                "k_g1",
                "k_21",
                "k_i",
            ]
            advanced_list_keys = ["k_i"]

            adv_col1, adv_col2 = st.columns(2)

            edited_advanced_parms1 = app_editors(
                adv_col1,
                "No Interventions",
                edited_parms1,
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
            )

            edited_advanced_parms2 = app_editors(
                adv_col2,
                "Interventions",
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
                keys2,
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

    edited_intervention_parms1 = update_intervention_parameters_from_widget(
        edited_advanced_parms1
    )
    edited_intervention_parms2 = update_intervention_parameters_from_widget(
        edited_advanced_parms2
    )

    updated_parms1 = edited_intervention_parms1.copy()
    updated_parms2 = edited_intervention_parms2.copy()

    scenario1 = [updated_parms1]
    scenario2 = [updated_parms2]

    # run the model with the updated parameters
    results1 = get_scenario_results(scenario1)
    results2 = get_scenario_results(scenario2)

    # extract groups
    groups = results1["group"].unique().to_list()

    # do some processing here to get daily incidence
    results1 = add_daily_incidence(results1, groups)
    results2 = add_daily_incidence(results2, groups)

    # create tables with interval results - weekly incidence, weekly cumulative incidence
    interval = 7

    interval_results1 = get_interval_results(results1, groups, interval)
    interval_results2 = get_interval_results(results2, groups, interval)

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
        range=[
            "#20419a",  # blue
            "#cf4828",  # red
            "#f78f47",  # orange
        ],
    )
    if outcome not in ["I", "Y", "inc", "Winc", "WCI"]:
        print("outcome not available yet, defaulting to Cumulative Daily Incidence")
        outcome = "Y"

    if outcome_option in [
        "Daily Infections",
        "Daily Incidence",
        "Daily Cumulative Incidence",
    ]:
        alt_results1 = results1
        alt_results2 = results2
        min_y, max_y = 0, max(results1[outcome].max(), results2[outcome].max())
        x = "t:Q"
        time_label = "Time (days)"
    elif outcome_option in ["Weekly Incidence", "Weekly Cumulative Incidence"]:
        alt_results1 = interval_results1
        alt_results2 = interval_results2
        min_y, max_y = (
            0,
            max(interval_results1[outcome].max(), interval_results2[outcome].max()),
        )
        x = "interval_t:Q"
        time_label = "Time (weeks)"

    # filter for a sample of replicates
    replicate_inds = plot_rng.choice(
        results1["replicate"].unique().to_numpy(), replicates, replace=False
    )
    alt_results1 = alt_results1.filter(pl.col("replicate").is_in(replicate_inds))
    alt_results2 = alt_results2.filter(pl.col("replicate").is_in(replicate_inds))

    y = f"{outcome}:Q"
    yscale = [min_y, max_y]
    color_key = "group"
    labelExpr = f"datum.value == '0' ? '{group_labels[0]}' : datum.value == '1' ? '{group_labels[1]}' : '{group_labels[2]}'"
    detail = "replicate:N"

    chart1 = create_chart(
        alt_results1,
        outcome_option,
        x,
        time_label,
        y,
        outcome_option,
        yscale,
        color_key,
        color_scale,
        domain,
        labelExpr,
        detail,
    )
    chart2 = create_chart(
        alt_results2,
        outcome_option,
        x,
        time_label,
        y,
        outcome_option,
        yscale,
        color_key,
        color_scale,
        domain,
        labelExpr,
        detail,
    )
    st.altair_chart(chart1 | chart2, use_container_width=True)
