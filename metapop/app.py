# flake8: noqa
import streamlit as st
import numpy as np
import polars as pl
import altair as alt
from metapop.model import *
from metapop.helper import *
from metapop.app_helper import *


def app(replicates=20):
    st.title("Metapopulation Model")
    parms = read_parameters("scripts/app/onepop_config.yaml")

    show_parameter_mapping = get_show_parameter_mapping()
    advanced_parameter_mapping = get_advanced_parameter_mapping()

    with st.sidebar:
        st.header(
            "Scenario parameters",
            help="Enter model parameters for each scenario. Hover over the ? for more information about each parameter.",
        )

        min_values = get_min_values()
        max_values = get_max_values()
        steps = get_step_values()
        helpers = get_helpers()
        formats = get_formats()
        keys0 = get_widget_idkeys(0) # keys for the shared parameters
        keys1 = get_widget_idkeys(1) # keys for the parameters for scenario 1
        keys2 = get_widget_idkeys(2) # keys for the parameters for scenario 2

        # define parameters to be shared between scenarios that are shown in the sidebar by default
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
        shared_slider_keys = [
            'pop_sizes',
            'I0',
            'initial_vaccine_coverage']

        col0 = st.columns(1)
        col0 = col0[0]

        subheader = "Shared parameters"

        edited_parms = app_editors(
            col0, subheader, parms, shared_keys, shared_list_keys,
            shared_slider_keys, show_parameter_mapping, min_values, max_values,
            steps, helpers, formats, keys0
        )

        # parameters for each scenario separately
        # order of parameters in the sidebar
        ordered_keys = [
                        # 'desired_r0',
                        # 'pop_sizes',
                        # 'I0',
                        # 'initial_vaccine_coverage',
                        'total_vaccine_uptake_doses',
                        'vaccine_uptake_start_day',
                        'vaccine_uptake_duration_days',
                        'isolation_success',
                        'symptomatic_isolation_start_day',
                        'symptomatic_isolation_duration_days',
                        'pre_rash_isolation_success',
                        'pre_rash_isolation_start_day',
                        'pre_rash_isolation_duration_days',
                        ]

        list_parameter_keys = [
                              #'pop_sizes',
                              #'I0',
                              #'vaccine_uptake_start_day',
                              # 'initial_vaccine_coverage',
                              ]

        slider_keys = [
            'vaccine_uptake_start_day',
            'vaccine_uptake_duration_days'
            'total_vaccine_uptake_doses',
            'isolation_success',
            'symptomatic_isolation_start_day',
            'symptomatic_isolation_duration_days',
            #'desired_r0',
            #'pop_sizes',
            #'I0'
            ]

        # make a section for each scenario
        col1, col2 = st.columns(2)

        # show the parameters for scenario 1 but do not allow editing
        edited_parms1 = app_editors(
            col1, "Scenario 1 (Baseline)", edited_parms, ordered_keys, list_parameter_keys,
            slider_keys, show_parameter_mapping, min_values, max_values,
            steps, helpers, formats, keys1, disabled=True
        )

        edited_parms2 = app_editors(
            col2, "Scenario 2", edited_parms, ordered_keys, list_parameter_keys,
            slider_keys, show_parameter_mapping, min_values, max_values,
            steps, helpers, formats, keys2
        )

        with st.expander("Advanced options"):
            # try to place two sliders side by side
            advanced_ordered_keys = [
                "desired_r0",
                "latent_duration",
                "infectious_duration",
                "k_i",
                ]
            # show additional advanced parameters if there are multiple population groups
            if parms['n_groups'] > 1:
                advanced_ordered_keys = advanced_ordered_keys + ["k_g1", "k_g2", "k_21"]

            advanced_list_keys = [
                "k_i"
                ]
            advanced_slider_keys = ["latent_duration", "infectious_duration"]

            adv_col1, adv_col2 = st.columns(2)

            # show the parameters for scenario 1 but do not allow editing
            edited_advanced_parms1 = app_editors(
                adv_col1, "Scenario 1 (Baseline)", edited_parms1, advanced_ordered_keys,
                advanced_list_keys, advanced_slider_keys, advanced_parameter_mapping,
                min_values, max_values, steps, helpers, formats, keys1,
                disabled=True
            )

            edited_advanced_parms2 = app_editors(
                adv_col2, "Scenario 2", edited_parms2, advanced_ordered_keys,
                advanced_list_keys, advanced_slider_keys, advanced_parameter_mapping,
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

    updated_parms1 = edited_advanced_parms1.copy()
    updated_parms2 = edited_advanced_parms2.copy()

    scenario1 = [updated_parms1]
    scenario2 = [updated_parms2]

    # run the model with the updated parameters
    results1 = get_scenario_results(scenario1)
    results2 = get_scenario_results(scenario2)

    # fullresults for later
    fullresults1 = results1
    fullresults2 = results2

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
    if len(groups) == 3:
        color_scale = alt.Scale(
            domain=[str(i) for i in range(len(results1["group"].unique()))],
            range = [
                "#20419a", # blue
                "#cf4828", # red
                "#f78f47", # orange
            ]
        )
    elif len(groups) == 1:
        color_scale = alt.Scale(
            domain=[str(i) for i in range(len(results1["group"].unique()))],
            range = [
                "#068482", # green
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

    # y = f"{outcome}:Q"
    # yscale = [min_y, max_y]
    # color_key = "group"
    # labelExpr=f"datum.value == '0' ? '{group_labels[0]}' : datum.value == '1' ? '{group_labels[1]}' : '{group_labels[2]}'"
    # detail="replicate:N"

    # chart1 = create_chart(alt_results1, outcome_option,
    #                       x, time_label,
    #                       y, outcome_option, yscale,
    #                       color_key, color_scale, domain,
    #                       labelExpr,
    #                       detail)
    # chart2 = create_chart(alt_results2, outcome_option,
    #                       x, time_label,
    #                       y, outcome_option, yscale,
    #                       color_key, color_scale, domain,
    #                       labelExpr,
    #                       detail)
    # st.altair_chart(chart1 | chart2, use_container_width=True)
    alt_results1 = alt_results1.with_columns(pl.lit("Scenario 1 (Baseline)").alias("scenario"))
    alt_results2 = alt_results2.with_columns(pl.lit("Scenario 2").alias("scenario"))
    combined_alt_results = alt_results1.vstack(alt_results2)
    chart = alt.Chart(combined_alt_results.to_pandas()).mark_line(opacity=0.4).encode(
        x=alt.X(x, title=time_label),
        y=alt.Y(outcome, title=outcome_option),
        color=alt.Color(
            "scenario",
            title="Scenario",
            scale=alt.Scale(
                domain=["Scenario 1 (Baseline)", "Scenario 2"],  # Scenarios
                range=["#cf4828","#20419a"]  # Corresponding colors (blue, red)
            )
        ),
        detail = "replicate",
        tooltip=["scenario", "t", outcome]
    ).properties(
        title="Outcome Comparison by Scenario",
        width=800,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)


    ### Summary stats based on outbreak sizes
    st.subheader("Outbreak summary statistics")

    threshold = st.selectbox(
        label = "Outbreak threshold:",
        options=[50, 100, 200, 300, 500],
        index=2,  # Default selected option (index of the options list)
    )
    fullresults1 = fullresults1.with_columns(
        pl.lit("Scenario 1 (Baseline)").alias("Scenario")
    )
    fullresults2 = fullresults2.with_columns(
        pl.lit("Scenario 2").alias("Scenario")
    )

    combined_results = fullresults2.vstack(fullresults1).with_columns(
        pl.col("t").cast(pl.Int64),
        pl.col("group").cast(pl.Int64),
        pl.col("Y").cast(pl.Int64),
    ).filter(pl.col("t") == 365) \
     .group_by("Scenario", "replicate"
     ).agg(pl.col("Y").sum().alias("Total"))

    outbreak_summary = calculate_outbreak_summary(combined_results, threshold)

    hospitalization_summary = get_hospitalizations(combined_results, parms["IHR"])

    columns = st.columns(len(outbreak_summary)+ len(hospitalization_summary))

    n_reps = parms["n_replicates"]

    for idx, row in enumerate(outbreak_summary.iter_rows(named=True)):
        scenario = row["Scenario"]
        outbreaks = row["outbreaks"]
        outbreak_prop = f"{(outbreaks / n_reps) * 100:.2f}%"

        # Use st.error for the first column, st.success for the second
        if scenario == "Scenario 1 (Baseline)":
            columns[0].error(f"{scenario}: {outbreaks}/{n_reps} ({outbreak_prop}) simulations had >= {threshold} cases total ")
        else:
            columns[1].info(f"{scenario}: {outbreaks}/{n_reps} ({outbreak_prop}) simulations had >= {threshold} cases total ")


    for idx, row in enumerate(hospitalization_summary.iter_rows(named=True)):
        scenario = row["Scenario"]
        hospitalizations = row["hospitalizations"]

        # Use st.error for the first column, st.success for the second
        if scenario == "Scenario 1 (Baseline)":
            columns[2].error(f"{scenario}: simulations had an average {hospitalizations} hospitalizations ")
        else:
            columns[3].info(f"{scenario}: simulations had an average {hospitalizations} hospitalizations ")


if __name__ == "__main__":
    app()
