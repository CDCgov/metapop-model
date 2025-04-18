# flake8: noqa
# This file is part of the metapop package. It contains the Streamlit app for
# the metapopulation model
import streamlit as st
import numpy as np
import polars as pl
import altair as alt
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
    add_daily_incidence,
    get_interval_results,
    calculate_outbreak_summary,
    get_hospitalizations,
)

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
#     get_hospitalizations,
# )
### note: this is not recommended use within a file that is imported as a package module, but it can be useful for testing purposes

__all__ = [
    "app",
]


def app(replicates=20):
    st.set_page_config(layout="wide")
    st.title("Measles Outbreak Simulator")
    st.text("This interactive tool illustrates the impact of " \
    "vaccination, isolation, and stay-at-home measures on the probability "
    "and size of measles outbreaks over 365 days following introduction of measles into " \
    "a community, by comparing scenarios with and without interventions.")
    parms = read_parameters("scripts/app/onepop_config.yaml")

    scenario_names = ["No Interventions", "Interventions"]
    show_parameter_mapping = get_show_parameter_mapping(parms)

    advanced_parameter_mapping = get_advanced_parameter_mapping()

    with st.sidebar:
        st.header(
            "Model Inputs",
            help="Enter the population size, overall vaccine coverage, and number of people " \
            "initially infected with measles in a community. " \
            "This tool is meant to demonstrate the dynamics following initial introductions " \
            "of cases into a community. Hover over the ? for more information about each parameter.",
        )

        widget_types = get_widget_types() # defines the type of widget for each parameter
        min_values = get_min_values()
        max_values = get_max_values()
        steps = get_step_values()
        helpers = get_helpers()
        formats = get_formats()
        keys0 = get_widget_idkeys(0) # keys for the shared parameters
        keys1 = get_widget_idkeys(1) # keys for the parameters for scenario 1
        keys2 = get_widget_idkeys(2) # keys for the parameters for Interventions

        # define parameters to be shared between scenarios that are shown in the sidebar by default
        # order of shared parameters in the sidebar
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

        col0 = st.columns(1)
        col0 = col0[0]

        subheader = "Shared parameters"

        edited_parms = app_editors(
            col0, subheader, parms, shared_keys, shared_list_keys,
            show_parameter_mapping, widget_types,
            min_values, max_values, steps, helpers, formats, keys0
        )

        # Set parameters for each scenario separately
        # order of parameters in the sidebar
        ordered_keys = [
                        'total_vaccine_uptake_doses',
                        'vaccine_uptake_start_day',
                        'vaccine_uptake_duration_days',
                        'pre_rash_isolation_success',
                        'isolation_success',
                        ]
        # parameters that are lists or arrays
        list_parameter_keys = []

        # Scenario parameters
        # parameters for scenario 1 are not shown
        st.header(
            "Interventions scenario",
            help="Choose strategies to implement for the 'intervention' scenario. " \
            "Interventions can be applied independently or in combination with each other. " \
            "The results are compared to an 'unmitigated' scenario which does not have any vaccine " \
            "uptake, isolation, or stay at home incorporated.",
        )

        # For the no intervention scenario, intervention parameters are set to 0
        edited_parms1 = set_parms_to_zero(edited_parms, ["pre_rash_isolation_success", "isolation_success"])

        # For the intervention scenario, user defines values
        edited_parms2 = app_editors(
            st.container(), "", edited_parms, ordered_keys, list_parameter_keys,
            show_parameter_mapping, widget_types,
            min_values, max_values,
            steps, helpers, formats, keys2
        )

        # if total uptake doses is a proportion, scale to a number of doses
        if(parms["use_prop_vaccine_uptake"]):
            edited_parms2 = rescale_prop_vax(edited_parms2)


        with st.expander("Advanced options"):
            st.text("These options allow changes to parameter assumptions including "
            "measles natural history parameters as well as parameters governing "
            "intervention efficacy.")

            advanced_ordered_keys = [
                "desired_r0",
                "latent_duration",
                "infectious_duration",
                ]

            # show additional advanced parameters if there are multiple population groups
            if parms['n_groups'] > 1:
                advanced_ordered_keys = advanced_ordered_keys + ["k_i_0", "k_g1", "k_g2", "k_21"]

            # advanced parameters that are lists or arrays
            advanced_list_keys = [
                "k_i"
                ]

            # set the parameters for scenario 2
            edited_advanced_parms2 = app_editors(
                st.container(), "", edited_parms2, advanced_ordered_keys,
                advanced_list_keys, advanced_parameter_mapping, widget_types,
                min_values, max_values, steps, helpers, formats, keys1,
                disabled=False
            )

            # shared advanced parameters to scenario 1
            edited_advanced_parms1 = edited_parms1
            for key in advanced_ordered_keys:
                edited_advanced_parms1[key] = edited_advanced_parms2[key]

    #### Intervention scenarios:
    # instead of expander, can use st.radio:
    with st.expander("Show intervention strategy info.", expanded=False):
        columns = st.columns(2)

        columns[0].error("No Interventions:\n"
                        " - Vaccine doses administered: 0\n"
                        " - % of infectious individuals isolating before rash onset: 0\n"
                        " - % of infectious individuals isolating after rash onset: 0")

        columns[1].info("Interventions:\n"
                        f" - Vaccine doses administered: {edited_parms2['total_vaccine_uptake_doses']} between day {edited_parms2['vaccine_uptake_start_day']} and day {edited_parms2['vaccine_uptake_start_day'] + edited_parms2['vaccine_uptake_duration_days']}\n"
                        f" - % of infectious individuals isolating before rash onset: {edited_parms2['pre_rash_isolation_success']*100} between day {edited_parms2['pre_rash_isolation_start_day']} and day {edited_parms2['pre_rash_isolation_start_day'] + edited_parms2['pre_rash_isolation_duration_days']}\n"
                        f" - % of infectious individuals isolating after rash onset: {edited_parms2['isolation_success']*100} between day {edited_parms2['symptomatic_isolation_start_day']} and day {edited_parms2['symptomatic_isolation_start_day'] + edited_parms2['symptomatic_isolation_duration_days']}\n")

    #### Plot Options:
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

    # # set up the color scale
    # domain = [str(i) for i in range(len(groups))]
    # group_labels = ["General population", "Small population 1", "Small population 2"]

    # # plot with Altair
    # if len(groups) == 3:
    #     color_scale = alt.Scale(
    #         domain=[str(i) for i in range(len(results1["group"].unique()))],
    #         range = [
    #             "#20419a", # blue
    #             "#cf4828", # red
    #             "#f78f47", # orange
    #         ]
    #     )
    # elif len(groups) == 1:
    #     color_scale = alt.Scale(
    #         domain=[str(i) for i in range(len(results1["group"].unique()))],
    #         range = [
    #             "#068482", # green
    #         ]
    #     )
    if outcome not in ["I", "Y", "inc", "Winc", "WCI"]:
        print("outcome not available yet, defaulting to Cumulative Daily Incidence")
        outcome = "Y"

    if outcome_option in ['Daily Infections', 'Daily Incidence', 'Daily Cumulative Incidence']:
        alt_results1 = results1
        alt_results2 = results2
        # min_y, max_y = 0, max(results1[outcome].max(), results2[outcome].max())
        x = "t:Q"
        time_label = "Time (days)"
    elif outcome_option in ['Weekly Incidence', 'Weekly Cumulative Incidence']:
        alt_results1 = interval_results1
        alt_results2 = interval_results2
        # min_y, max_y = 0, max(interval_results1[outcome].max(), interval_results2[outcome].max())
        x = "interval_t:Q"
        time_label = "Time (weeks)"


    alt_results1 = alt_results1.with_columns(pl.lit(scenario_names[0]).alias("scenario"))
    alt_results2 = alt_results2.with_columns(pl.lit(scenario_names[1]).alias("scenario"))
    combined_alt_results = alt_results1.vstack(alt_results2)

    title = (
        f"Outcome Comparison by Scenario: Reactive Vaccination "
        f"({edited_parms2['total_vaccine_uptake_doses']} doses), "
        f"{int(edited_parms2['pre_rash_isolation_success'] * 100)}% isolation, "
        f"{int(edited_parms2['isolation_success'] * 100)}% stay-at-home"
    )


    chart = alt.Chart(combined_alt_results.to_pandas()).mark_line(opacity=0.4).encode(
        x=alt.X(x, title=time_label),
        y=alt.Y(outcome, title=outcome_option),
        color=alt.Color(
            "scenario",
            title="Scenario",
            scale=alt.Scale(
                domain=[scenario_names[0], scenario_names[1]],  # Scenarios
                range=["#cf4828","#20419a"]  # Corresponding colors (blue, red)
            )
        ),
        detail = "replicate",
        tooltip=["scenario", "t", outcome]
    ).properties(
        title=title,
        width=800,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)


    ### Outbreak Summary Stats
    st.subheader("Outbreak summary statistics")
    fullresults1 = fullresults1.with_columns(
        pl.lit(scenario_names[0]).alias("Scenario")
    )
    fullresults2 = fullresults2.with_columns(
        pl.lit(scenario_names[1]).alias("Scenario")
    )

    threshold_percent = st.slider(
        label = "Outbreak threshold (% of population that gets infected):",
        min_value=1,
        max_value=10,
        step=1,
        help = "Sum the number of simulations that have more than this percentage of the population infected.",
    )

    threshold = np.sum(edited_parms2["pop_sizes"]) * threshold_percent / 100.0

    combined_results = fullresults2.vstack(fullresults1).with_columns(
        pl.col("t").cast(pl.Int64),
        pl.col("group").cast(pl.Int64),
        pl.col("Y").cast(pl.Int64),
    ).filter(pl.col("t") == 365) \
     .group_by("Scenario", "replicate"
     ).agg(pl.col("Y").sum().alias("Total"))

    outbreak_summary = calculate_outbreak_summary(combined_results, threshold)
    hospitalization_summary = get_hospitalizations(combined_results, parms["IHR"])

    # Merge hospitalization_summary and outbreak_summary by the "Scenario" column
    merged_summary = outbreak_summary.join(
        hospitalization_summary,
        on="Scenario",
        how="inner"  # Use "inner" to keep only matching rows
    )

    transposed = merged_summary.select([
        pl.col("outbreaks").alias(f"Number of outbreaks with more than > {threshold} infections"),
        pl.col("Total Infections").alias("Average Outbreak Size"),
        pl.col("Hospitalizations").alias("Average Number of Hospitalizations")
    ]).transpose(include_header=True)

    transposed = transposed.rename({"column": "Metric", "column_0": scenario_names[0], "column_1": scenario_names[1]})
    transposed = transposed.with_columns(
        (
            (pl.col(scenario_names[0]) - pl.col(scenario_names[1])) / pl.col(scenario_names[0])
        ).alias("Relative Difference")
    )

    st.dataframe(transposed)

if __name__ == "__main__":
    app()
