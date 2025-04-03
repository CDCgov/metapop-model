import streamlit as st
import numpy as np
import polars as pl
import metapop as mt
import altair as alt


def app(replicates=5):
    st.title("Metapopulation Model")
    parms = mt.read_parameters()

    show_parameter_mapping = mt.get_show_parameter_mapping()
    advanced_parameter_mapping = mt.get_advanced_parameter_mapping()

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
        keys0 = mt.get_widget_idkeys(0)
        keys1 = mt.get_widget_idkeys(1)
        keys2 = mt.get_widget_idkeys(2)

        shared_keys = [
            "pop_sizes",
            "I0",
            "initial_vaccine_coverage",
        ]
        shared_list_keys = [
            "pop_sizes",
            "I0",
            "initial_vaccine_coverage",
        ]
        shared_slide_keys = [
            'pop_sizes',
            'I0',
            'initial_vaccine_coverage']

        col0 = st.columns(1)
        col0 = col0[0]

        subheader = "Shared parameters"

        edited_parms = mt.app_editors(
            col0, subheader, parms, shared_keys, shared_list_keys,
            shared_slide_keys, show_parameter_mapping, min_values, max_values,
            steps, helpers, formats, keys0
        )

        # order of parameters in the sidebar
        ordered_keys = [
                        # 'desired_r0',
                        # 'pop_sizes',
                        # 'I0',
                        # 'initial_vaccine_coverage',
                        'total_vaccine_uptake_doses',
                        'vaccine_uptake_range',
                        'isolation_success',
                        'symptomatic_isolation_day'
                        ]

        list_parameter_keys = [
                              #'pop_sizes',
                              #'I0',
                              'vaccine_uptake_range',
                              # 'initial_vaccine_coverage',
                              ]

        slide_keys = [
            'vaccine_uptake_range',
            'total_vaccine_uptake_doses',
            'isolation_success',
            'symptomatic_isolation_day',
            #'desired_r0',
            #'pop_sizes',
            #'I0'
            ]
        # number of scenarios
        col1, col2 = st.columns(2)

        # show the parameters for scenario 1 but do not allow editing
        edited_parms1 = mt.app_editors(
            col1, "Scenario 1", edited_parms, ordered_keys, list_parameter_keys,
            slide_keys, show_parameter_mapping, min_values, max_values,
            steps, helpers, formats, keys1, disabled=True
        )

        edited_parms2 = mt.app_editors(
            col2, "Scenario 2", edited_parms, ordered_keys, list_parameter_keys,
            slide_keys, show_parameter_mapping, min_values, max_values,
            steps, helpers, formats, keys2
        )

        with st.expander("Advanced options"):
            # try to place two sliders side by side
            advanced_ordered_keys = [
                # "desired_r0",
                "latent_duration",
                "infectious_duration",
                "k_g1",
                "k_g2",
                "k_21",
                "k_i",
                ]
            advanced_list_keys = ["k_i"]
            advanced_slide_keys = ["latent_duration", "infectious_duration"]

            adv_col1, adv_col2 = st.columns(2)

            # show the parameters for scenario 1 but do not allow editing
            edited_advanced_parms1 = mt.app_editors(
                adv_col1, "Scenario 1", edited_parms1, advanced_ordered_keys,
                advanced_list_keys, advanced_slide_keys, advanced_parameter_mapping,
                min_values, max_values, steps, helpers, formats, keys1,
                disabled=True
            )

            edited_advanced_parms2 = mt.app_editors(
                adv_col2, "Scenario 2", edited_parms2, advanced_ordered_keys,
                advanced_list_keys, advanced_slide_keys, advanced_parameter_mapping,
                min_values, max_values, steps, helpers, formats, keys2
            )
    # get the selected outcome from the sidebar
    outcome_option = st.selectbox(
        "Metric",
        mt.get_outcome_options(),
        index=0,  # by default display weekly incidence
        placeholder="Select an outcome to plot",
    )

    # Map the selected option to the outcome variable
    outcome_mapping = mt.get_outcome_mapping()
    outcome = outcome_mapping[outcome_option]

    full_defaults = mt.get_default_full_parameters()

    # get updated parameter dictionaries
    updated_parms1 = mt.get_parms_from_table(full_defaults, value_col="Scenario 1")
    updated_parms2 = mt.get_parms_from_table(full_defaults, value_col="Scenario 2")

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
    updated_parms1 = mt.correct_parameter_types(parms, updated_parms1)
    updated_parms2 = mt.correct_parameter_types(parms, updated_parms2)

    scenario1 = [updated_parms1]
    scenario2 = [updated_parms2]

    # run the model with the updated parameters
    results1 = mt.get_scenario_results(scenario1)
    results2 = mt.get_scenario_results(scenario2)

    # extract groups
    groups = results1["group"].unique().to_list()

    # filter for a sample of replicates
    replicate_inds = np.random.choice(results1["replicate"].unique().to_numpy(), replicates, replace=False)
    results1 = results1.filter(pl.col("replicate").is_in(replicate_inds))
    results2 = results2.filter(pl.col("replicate").is_in(replicate_inds))

    # do some processing here to get daily incidence
    results1 = mt.add_daily_incidence(results1, replicate_inds, groups)
    results2 = mt.add_daily_incidence(results2, replicate_inds, groups)

    # create tables with interval results - weekly incidence, weekly cumulative incidence
    interval = 7

    interval_results1 = mt.get_interval_results(results1, replicate_inds, groups, interval)
    interval_results2 = mt.get_interval_results(results2, replicate_inds, groups, interval)

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

    chart1 = mt.create_chart(alt_results1, outcome_option,
                          x, time_label,
                          y, outcome_option, yscale,
                          color_key, color_scale, domain,
                          labelExpr,
                          detail)
    chart2 = mt.create_chart(alt_results2, outcome_option,
                          x, time_label,
                          y, outcome_option, yscale,
                          color_key, color_scale, domain,
                          labelExpr,
                          detail)
    st.altair_chart(chart1 | chart2, use_container_width=True)


if __name__ == "__main__":
    app()
