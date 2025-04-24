# This file is part of the metapop package. It contains the streamlit app with
# a table interface for users
import streamlit as st
import polars as pl
import numpy as np
import altair as alt

# import what's needed from other metapop modules
from .app_helper import (
    get_scenario_results,
    read_parameters,
    get_default_full_parameters,
    get_default_show_parameters_table,
    get_show_parameter_mapping,
    get_outcome_options,
    get_outcome_mapping,
    get_parms_from_table,
    update_parms_from_table,
    correct_parameter_types,
    get_keys_in_list,
    repack_list_parameters,
    add_daily_incidence,
    get_interval_results,
    create_chart,
)

# if you want to use the methods from metapop in this file under
# if __name__ == "__main__": you'll need to import them as:
# from metapop.app_helper import (
#     get_scenario_results,
#     read_parameters,
#     get_default_full_parameters,
#     get_default_show_parameters_table,
#     get_show_parameter_mapping,
#     get_outcome_options,
#     get_outcome_mapping,
#     get_parms_from_table,
#     correct_parameter_types,
#     add_daily_incidence,
#     get_interval_results,
#     create_chart,
# )
### note: this is not recommended use within a file that is imported as a package modules, but it can be useful for testing

__all__ = [
    "app_with_table",
]


# This method is currently deprecated, but does look prettier
def app_with_table(replicates=20):
    st.title("Measles Outbreak Simulator")
    st.text(
        "This interactive tool illustrates the impact of vaccination and isolation on the probability and size of measles outbreaks following introduction of measles into different connected communities."
    )

    parms = read_parameters()

    default_table = get_default_show_parameters_table()
    show_parameter_mapping = get_show_parameter_mapping(parms)
    # advanced_table = get_advanced_parameters_table()
    # advanced_parameter_mapping = get_advanced_parameter_mapping()

    with st.sidebar:
        st.header(
            "Scenario parameters",
            help="TODO",
        )
        # st.subheader(
        #     "Table subheader",
        #     help="subhead TODO",
        # )

        edited_table = st.sidebar.data_editor(
            default_table,
            hide_index=True,
            disabled=["Parameter"],  # prevent editing of the parameter names
        )

        with st.expander("Advanced options"):
            # try to place two sliders side by side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("No Interventions")
                n_groups_slider1 = st.slider(  # noqa
                    "Number of groups",
                    min_value=1,
                    max_value=3,  # noqa
                    value=parms["n_groups"],
                    step=1,
                    help="At most 3 groups can be modeled",
                    key="n_groups_slider1",
                )
            with col2:
                st.subheader("Interventions")
                n_groups_slider2 = st.slider(  # noqa
                    "Number of groups",
                    min_value=1,
                    max_value=3,  # noqa
                    value=parms["n_groups"],
                    step=1,
                    help="At most 3 groups can be modeled",
                    key="n_groups_slider2",
                )
            # # TODO: add advanced parameters as a table that hides under expander, doesn't seem to work
            # advanced_edited_table = st.sidebar.data_editor( # noqa
            #     advanced_table,
            #     hide_index=True,
            #     disabled=["Parameter"],
            #     )

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
    updated_parms1 = get_parms_from_table(full_defaults, value_col="No Interventions")
    updated_parms2 = get_parms_from_table(full_defaults, value_col="Interventions")

    # get updated values from user through the sidebar
    updated_parms1 = update_parms_from_table(
        updated_parms1,
        edited_table,
        show_parameter_mapping,
        value_col="No Interventions",
    )
    updated_parms2 = update_parms_from_table(
        updated_parms2, edited_table, show_parameter_mapping, value_col="Interventions"
    )

    # correct types for single values
    updated_parms1 = correct_parameter_types(parms, updated_parms1)
    updated_parms2 = correct_parameter_types(parms, updated_parms2)

    # find all the values that should be arrays or lists
    keys_in_list = get_keys_in_list(parms, updated_parms1)

    # find the original keys and repack the list values in each updated parameters dictionary
    updated_parms1 = repack_list_parameters(parms, updated_parms1, keys_in_list)
    updated_parms2 = repack_list_parameters(parms, updated_parms2, keys_in_list)

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
        print("outcome not available yet, defaulting to Y")
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

    # for altair, plot only a subset
    replicate_inds = np.random.choice(
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
