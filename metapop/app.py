import streamlit as st
import numpy as np
import polars as pl
import griddler
import griddler.griddle
import metapop as mt
import altair as alt


def simulate(parms):

    #### Set beta matrix based on desired R0 and connectivity scenario ###

    parms["beta"] = mt.construct_beta(parms)

    #### set up the model time steps
    steps = parms["tf"]
    t = np.linspace(1, steps, steps)

    #### Initialize population
    groups = parms["n_groups"]
    S, V, E1, E2, I1, I2, R, Y, u = mt.initialize_population(steps, groups, parms)

    #### Run the model
    model = mt.SEIRModel(parms)
    S, V, E1, E2, I1, I2, R, Y, u = mt.run_model(model, u, t, steps, groups, S, V, E1, E2, I1, I2, R, Y)

    #### Flatten into a dataframe
    df = pl.DataFrame({
        't': np.repeat(t, groups),
        'group': np.tile(np.arange(groups), steps),
        'S': S.flatten(),
        'V': V.flatten(),
        'E1': E1.flatten(),
        'E2': E2.flatten(),
        'I1': I1.flatten(),
        'I2': I2.flatten(),
        'R': R.flatten(),
        'Y': Y.flatten()
    })

    return df


def get_scenario_results(parms):
    results = griddler.run_squash(griddler.replicated(simulate), parms)
    # cast group to string
    results = results.with_columns(pl.col("group").cast(pl.Utf8))
    # select subset of values to return
    results = results.select(
        [
            "initial_coverage_scenario",
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
            "replicate"
        ]
    )
    # add a column for total infections
    results = results.with_columns((pl.col("I1") + pl.col("I2")).alias("I"))
    return results

def read_parameters():
    parameter_sets = griddler.griddle.read("scripts/app_config.yaml")
    parms = parameter_sets[0]
    return parms


def get_list_keys(parms):
    list_keys = [key for key, value in parms.items() if isinstance(value, list)]
    return list_keys


def get_default_full_parameters_v2():
    parms = read_parameters()

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


def get_show_parameter_mapping2():
    show_mapping = dict(
        desired_r0="R0",
        I0_1="initial infections in small population 1",
        I0_2="initial infections in small population 2",
        vaccine_uptake_doses_0="vaccine doses per day",
        isolation_percentage_1="percent isolation in population 1",
        isolation_percentage_2="percent isolation in population 2",
        isolation_effectiveness="isolation effectiveness",
        gamma="gamma"
    )
    return show_mapping

def get_show_keys2():
    show_keys = [key for key in get_show_parameter_mapping2().keys()]
    return show_keys

def get_show_names2():
    show_keys = get_show_keys2()
    show_parameter_mapping = get_show_parameter_mapping2()
    show_names = [show_parameter_mapping[key] for key in show_keys]
    return show_names


def get_default_show_parameters_table():
    full_defaults = get_default_full_parameters_v2()
    show_keys = get_show_keys2()
    # show_names = get_show_names2()
    show_defaults = full_defaults.filter(pl.col("Parameter").is_in(show_keys))

    # casting values to float
    show_defaults = show_defaults.with_columns(pl.col("Scenario 1").cast(pl.Float64))
    show_defaults = show_defaults.with_columns(pl.col("Scenario 2").cast(pl.Float64))

    # renaming keys
    # show_defaults = show_defaults.with_columns(pl.Series(name="Parameter", values=show_names))

    return show_defaults

def get_advanced_parameters_table():
    full_defaults = get_default_full_parameters_v2()
    show_keys = get_show_keys2()
    advanced_defaults = full_defaults.filter(~pl.col("Parameter").is_in(show_keys))
    advanced_defaults = advanced_defaults.with_columns(
        pl.when(pl.col("Scenario 1").cast(pl.Float64, strict=False).is_not_null())
        .then(pl.col("Scenario 1").cast(pl.Float64, strict=False))
        .otherwise(pl.col("Scenario 1"))
        .alias("Scenario 1")
    )
    advanced_defaults = advanced_defaults.with_columns(
        pl.when(pl.col("Scenario 2").cast(pl.Float64, strict=False).is_not_null())
        .then(pl.col("Scenario 2").cast(pl.Float64, strict=False))
        .otherwise(pl.col("Scenario 2"))
        .alias("Scenario 2")
    )
    return advanced_defaults



def app_old():
    st.title("Metapopulation Model")

    # some default dataframe

    full_defaults = get_default_full_parameters_v2() # noqa
    show_table = get_default_show_parameters_table()
    advanced_table = get_advanced_parameters_table() # noqa

    # print("advanced_table")
    # print(advanced_table["Parameter"].to_list())
    with st.sidebar:
        st.header(
            "Scenario parameters",
            help="TODO",
        )

        st.subheader(
            "Table subheader",
            help="subhead TODO",
        )
        new_params = st.sidebar.data_editor(show_table, hide_index=True) #noqa

        with st.expander("Advanced options"):
            st.write("TODO")

    print(new_params)

    # new_params_full = pl.concat(
    #     [
    #         new_params.with_columns(
    #             pl.col("Scenario 1").cast(pl.Utf8, strict=False),
    #             pl.col("Scenario 2").cast(pl.Utf8, strict=False)
    #         ),
    #         advanced_table.with_columns(
    #             pl.col("Scenario 1").cast(pl.Utf8, strict=False),
    #             pl.col("Scenario 2").cast(pl.Utf8, strict=False)
    #         )
    #     ],
    #     how="vertical"
    # )
    # new_scenario1 = dict(
    #     (key, float(value) if value.replace('.', '', 1).isdigit() else value)
    #     for key, value in zip(new_params_full["Parameter"].to_list(), new_params_full["Scenario 1"].to_list())
    # )
    # new_scenario2 = dict(
    #     (key, float(value) if value.replace('.', '', 1).isdigit() else value)
    #     for key, value in zip(new_params_full["Parameter"].to_list(), new_params_full["Scenario 2"].to_list())
    # )


    # keys_in_list = [key for key in new_scenario1.keys() if any(key.startswith(list_key) for list_key in list_keys)]
    # print("keys_in_list")
    # print(keys_in_list)

    # recombined_parms1 = dict()
    # for key in keys_in_list:
    #     key_split = key.split("_")
    #     list_key = "_".join(key_split[:-1])
    #     if list_key not in recombined_parms1:
    #         recombined_parms1[list_key] = []
    #     recombined_parms1[list_key].append(float(new_scenario1[key]))


    # recombined_parms2 = dict()
    # for key in keys_in_list:
    #     key_split = key.split("_")
    #     list_key = "_".join(key_split[:-1])
    #     if list_key not in recombined_parms2:
    #         recombined_parms2[list_key] = []
    #     recombined_parms2[list_key].append(float(new_scenario2[key]))


    # scenario1 = {**new_scenario1, **recombined_parms1}
    # scenario2 = {**new_scenario2, **recombined_parms2}

    # diff_keys2 = set(scenario1.keys()).difference(parms.keys())


    # # Remove keys in diff_keys2 from scenario1
    # for key in diff_keys2:
    #     if key in scenario1:
    #         del scenario1[key]
    #         del scenario2[key]

    # # correct the types
    # for key, value in scenario1.items():
    #     if type(parms[key]) == int:
    #         scenario1[key] = int(value)
    #         scenario2[key] = int(scenario2[key])
    #     elif type(parms[key]) == float:
    #         scenario1[key] = float(value)
    #         scenario2[key] = float(scenario2[key])
    #     elif type(parms[key]) == list:
    #         scenario1[key] = [float(i) for i in value]
    #         scenario2[key] = [float(i) for i in scenario2[key]]


    # scenario1 = [scenario1]

    # scenario1_copy = scenario1.copy()
    # print("scenario1")


    # define scenarios
    parameter_sets = griddler.griddle.read("scripts/app_config.yaml")
    scenario1 = parameter_sets[0:1]
    # scenario2 = parameter_sets[1:2]

    # for k in scenario1[0].keys():
    #     print(k, scenario1[0][k], scenario1_copy[0][k])

    results1 = griddler.run_squash(griddler.replicated(simulate), scenario1)
    results1 = results1.select([
        "initial_coverage_scenario", "k_21", "t", "group", "S", "V",
        "E1", "E2", "I1", "I2", "R", "Y", "replicate"
    ])

    results1 = results1.with_columns(pl.col("group").cast(pl.Utf8))

    seed = 0 # should be a parameter
    np.random.seed(seed)

    replicates = 3 # should be fixed value or at least a slider with a fixed range
    # choose replicates to show
    replicate_inds = np.random.choice(results1["replicate"].unique().to_numpy(), replicates, replace=False)

    results1 = results1.filter(pl.col("replicate").is_in(replicate_inds))
    results1 = results1.with_columns((pl.col("I1") + pl.col("I2")).alias("I"))

    print("how many replicates")
    print(len(results1["replicate"].unique().to_list()))

    groups = results1["group"].unique().to_list()
    domain = [str(i) for i in range(len(groups))]
    group_labels = ["General population", "Small population 1", "Small population 2"]

    # Plotting with Altair
    # Define a color scale for groups

    color_scale = alt.Scale(
        domain=[str(i) for i in range(len(results1["group"].unique()))],
        range=[
            "#20419a",  # Group 0
            "#cf4828",  # Group 1
            "#f78f47",  # Group 2
        ]
    )

    chart1 = (
        alt.Chart(
        results1,
        title="Daily Infections")
        .mark_line(opacity=0.5)  # Set line opacity for partial transparency
        .encode(
            x=alt.X("t:Q", title="Time (days)"),  # Updated x-axis label
            y=alt.X("I1:Q", title="Daily Infections"),  # Updated y-axis label
            color=alt.Color(
                "group",
                scale=color_scale,
                legend=alt.Legend(
                    title="Population",
                    values=domain,  # Specify the group values to show in the legend
                    labelExpr=f"datum.value == '0' ? '{group_labels[0]}' : datum.value == '1' ? '{group_labels[1]}' : '{group_labels[2]}'",  # Rename legend labels
                ),
            ),  # Specify color scale for groups
            detail="replicate:N",  # Separate line for each replicate
        )
    )


    st.altair_chart(chart1, use_container_width=True)



def get_outcome_options():
    return ("Daily Infections", "Daily Incidence", "Cumulative Daily Incidence", "Weekly Infections", "Weekly Incidence", "Weekly Cumulative Incidence")


def get_outcome_mapping():
    return {
        "Daily Infections": "I",
        "Daily Incidence": "inc",
        "Cumulative Daily Incidence": "Y",
        "Weekly Infections": "WI",
        "Weekly Incidence": "inc_7",
        "Weekly Cumulative Incidence": "WCI",
    }

def get_parms_from_table(table, value_col="Scenario 1"):
    # get updated parameter dictionaries
    parms = dict()
    # expect the table to have the following columns
    # Parameter, Scenario 1, Scenario 2
    for key, value in zip(table["Parameter"].to_list(), table[value_col].to_list()):
        parms[key] = value
    return parms


def update_parms_from_table(parms, table, value_col="Scenario 1"):
    # get updated values from user throught the sidebar
    for key, value in zip(table["Parameter"].to_list(), table[value_col].to_list()):
        parms[key] = value
    return parms

def update_column_ranges(table):
    for row in table.iter_rows(named=True):
        parameter = row["Parameter"]
        if parameter == "R0":
            row["Scenario 1"].min_value = 0.5
            row["Scenario 1"].max_value = 5.0
            row["Scenario 2"].min_value = 0.5
            row["Scenario 2"].max_value = 5.0
        elif parameter == "gamma":
            row["Scenario 1"].min_value = 0.1
            row["Scenario 1"].max_value = 1.0
            row["Scenario 2"].min_value = 0.1
            row["Scenario 2"].max_value = 1.0
            # Add more conditions for other parameters as needed

def correct_parameter_types(original_parms, parms_from_table):
    for key, value in original_parms.items():
        if isinstance(value, int) and not isinstance(value, bool):
            parms_from_table[key] = int(parms_from_table[key])
        elif isinstance(value, bool):
            if parms_from_table[key] in [True, 'True', 'true', '1']:
                parms_from_table[key] = True
            else:
                parms_from_table[key] = False
        elif isinstance(value, float):
            parms_from_table[key] = float(parms_from_table[key])
        elif isinstance(value, str):
            parms_from_table[key] = str(parms_from_table[key])
    return parms_from_table

def get_keys_in_list(parms, updated_parms):
    list_keys = get_list_keys(parms)
    keys_in_list = [key for key in updated_parms.keys() if any(key.startswith(list_key) for list_key in list_keys)]
    keys_in_list = [key for key in sorted(keys_in_list)]
    return keys_in_list

def repack_list_parameters(parms, updated_parms, keys_in_list):
    for key in keys_in_list:
        key_split = key.split("_")
        list_key = "_".join(key_split[:-1])

        if list_key not in updated_parms:
            updated_parms[list_key] = []
        if isinstance(parms[list_key][0], int) and not isinstance(parms[list_key][0], bool):
            updated_parms[list_key].append(int(updated_parms[key]))
        elif isinstance(parms[list_key][0], bool):
            updated_parms[list_key].append(True if updated_parms[key] in [True, 'True', 'true', '1'] else False)
        elif isinstance(parms[list_key][0], float):
            updated_parms[list_key].append(float(updated_parms[key]))
        elif isinstance(parms[list_key][0], str):
            updated_parms[list_key].append(str(updated_parms[key]))

    for key in keys_in_list:
        del updated_parms[key]

    return updated_parms


def add_daily_incidence(results, replicate_inds, groups):
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

# def add_interval_incidence(results, replicate_inds, groups, interval=7):
#     # add a column for interval incidence
#     if interval == 1:
#         metric = "inc"
#     else:
#         metric = f"inc_{interval}"
#     results = results.with_columns(pl.lit(None).alias(metric))
#     updated_rows = []
#     for replicate in replicate_inds:
#         tempdf = results.filter(pl.col("replicate") == replicate)
#         for group in groups:
#             group_data = tempdf.filter(pl.col("group") == group)
#             group_data = group_data.sort("t")
#             inc = group_data["Y"] - group_data["Y"].shift(interval)
#             group_data = group_data.with_columns(inc.alias(metric))
#             updated_rows.append(group_data)

#     results = pl.concat(updated_rows)
#     return results


def get_interval_cumulative_incidence(results, replicate_inds, groups, interval=7):
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
    # interval_results = interval_results.with_columns(f"inc_{interval}".alias("inc"))
    return interval_results




def app(replicates=3):
    st.title("Metapopulation Model")
    parms = read_parameters()

    default_table = get_default_show_parameters_table()
    default_table_2 = get_default_show_parameters_table()
    # show_parameters_mapping = get_show_parameter_mapping()
    show_parameters_mapping = get_show_parameter_mapping2()
    default_table_2 = default_table_2.with_columns(pl.Series(name="Parameter", values=[show_parameters_mapping.get(key) for key in default_table_2["Parameter"].to_list()]))

    with st.sidebar:
        st.header(
            "Scenario parameters",
            help="TODO",
        )
        st.subheader(
            "Table subheader",
            help="subhead TODO",
        )

        # column_config = {
        #     "Scenario 1": st.column_config.NumberColumn(
        #         min_value=0,  # default minimum value
        #         max_value=100,  # default maximum value
        #         help="Adjust values for Scenario 1"
        #     ),
        #     "Scenario 2": st.column_config.NumberColumn(
        #         min_value=0,  # default minimum value
        #         max_value=100,  # default maximum value
        #         help="Adjust values for Scenario 2"
        #     ),
        # }

        edited_table = st.sidebar.data_editor(
            default_table,
            hide_index=True,
            disabled=["Parameter"],  # prevent editing of the parameter names
            # column_config=column_config,
            # on_change=update_column_ranges  # dynamically update ranges
        )

        with st.expander("Advanced options"):
            st.write("TODO")

    with st.sidebar:
        st.header(
            "Scenario parameters 2",
            help="TODO",
        )
        st.subheader(
            "Table subheader 2",
            help="subhead TODO",
        )
        edited_parms_2 = st.sidebar.data_editor(default_table_2, hide_index=True)


    outcome_option = st.selectbox(
        "Metric",
        get_outcome_options()
        ,
        index=2,  # Corrected index for "Daily Incidence"
        placeholder="Select an outcome to plot",
    )

    # Map the selected option to the outcome variable
    outcome_mapping = get_outcome_mapping()
    outcome = outcome_mapping[outcome_option]


    full_defaults = get_default_full_parameters_v2()

    # get updated parameter dictionaries
    updated_parms1 = get_parms_from_table(full_defaults, value_col="Scenario 1")
    updated_parms2 = get_parms_from_table(full_defaults, value_col="Scenario 2")


    # get updated values from user throught the sidebar
    updated_parms1 = update_parms_from_table(updated_parms1, edited_table)
    updated_parms2 = update_parms_from_table(updated_parms2, edited_table, value_col="Scenario 2")

    # dummy parms to see if we can repack to the correct format
    dummy_parms = dict()
    for key, value in zip(full_defaults["Parameter"].to_list(), full_defaults["Scenario 1"].to_list()):
        # original_key = next((k for k, v in show_parameters_mapping.items() if v == key), key)
        original_key = key
        dummy_parms[original_key] = value
        print(f"adding {key} with value {value}")
    for key, value in zip(edited_parms_2["Parameter"].to_list(), edited_parms_2["Scenario 1"].to_list()):
        original_key = next((k for k, v in show_parameters_mapping.items() if v == key), key)
        dummy_parms[original_key] = value
        print(f"updating {original_key} with value {value}")

    print("dummy_parms")
    print(dummy_parms)

    # correct types for single values
    updated_parms1 = correct_parameter_types(parms, updated_parms1)
    updated_parms2 = correct_parameter_types(parms, updated_parms2)
    dummy_parms = correct_parameter_types(parms, dummy_parms)

    # find all the values that should be arrays or lists

    keys_in_list = get_keys_in_list(parms, updated_parms1)


    # find the original keys and repack the list values in each updated parameters dictionary
    updated_parms1 = repack_list_parameters(parms, updated_parms1, keys_in_list)
    updated_parms2 = repack_list_parameters(parms, updated_parms2, keys_in_list)
    dummy_parms = repack_list_parameters(parms, dummy_parms, keys_in_list)

    scenario1 = [updated_parms1]
    scenario2 = [updated_parms2]
    dummy_scenario = [dummy_parms]

    print(f"there are {len(dummy_parms)} dummy parms")
    print(f"there are {len(updated_parms1)} updated parms")

    # run the model with the updated parameters

    results1 = get_scenario_results(scenario1)
    results2 = get_scenario_results(scenario2)
    dummy_results = get_scenario_results(dummy_scenario) # noqa

    # filter for a sample of replicates
    replicate_inds = np.random.choice(results1["replicate"].unique().to_numpy(), replicates, replace=False)
    results1 = results1.filter(pl.col("replicate").is_in(replicate_inds))
    results2 = results2.filter(pl.col("replicate").is_in(replicate_inds))

    # extract groups
    groups = results1["group"].unique().to_list()

    # do some processing here to get daily incidence
    results1 = add_daily_incidence(results1, replicate_inds, groups)
    results2 = add_daily_incidence(results2, replicate_inds, groups)

    # create tables with interval results - weekly incidence, weekly cumulative incidence
    interval = 7

    interval_results1 = get_interval_results(results1, replicate_inds, groups, interval)
    interval_results2 = get_interval_results(results2, replicate_inds, groups, interval)

    print("interval results 2")
    print(interval_results2.head(n=20)[["t", "group", "replicate","Y", "inc_7", "interval_t"]])

    print("interval results")
    print(interval_results1.head(n=20)[["t", "group", "replicate","Y", "inc_7", "interval_t"]])


    domain = [str(i) for i in range(len(groups))]
    group_labels = ["General population", "Small population 1", "Small population 2"]

    # plot with Altair
    color_scale = alt.Scale(
        domain=[str(i) for i in range(len(results1["group"].unique()))],
        range = [
            "#20419a",
            "#cf4828",
            "#f78f47",
        ]
    )
    if outcome not in ["I", "Y", "inc", "inc_7"]:
        print("outcome not available yet, defaulting to Y")

    min_y, max_y = 0, max(results1[outcome].max(), results2[outcome].max())

    chart1 = alt.Chart(
        results1,
        title="Daily Infections"
        ).mark_line(opacity=0.5).encode(
            x=alt.X("t:Q", title="Time (days)"),
            y=alt.Y(f"{outcome}:Q", title="Daily Infections").scale(domain=[min_y, max_y]),
            color=alt.Color(
                "group",
                scale=color_scale,
                legend=alt.Legend(
                    title="Population",
                    values=domain,
                    labelExpr=f"datum.value == '0' ? '{group_labels[0]}' : datum.value == '1' ? '{group_labels[1]}' : '{group_labels[2]}'",
                ),
            ),
            detail="replicate:N",
        ).properties(width=300, height=300)


    chart2 = alt.Chart(
        results2,
        title="Daily Infections"
        ).mark_line(opacity=0.5).encode(
            x=alt.X("t:Q", title="Time (days)"),
            y=alt.X(f"{outcome}:Q", title="Daily Infections").scale(domain=[min_y, max_y]),
            color=alt.Color(
                "group",
                scale=color_scale,
                legend=alt.Legend(
                    title="Population",
                    values=domain,
                    labelExpr=f"datum.value == '0' ? '{group_labels[0]}' : datum.value == '1' ? '{group_labels[1]}' : '{group_labels[2]}'",
            ),
        ),
        detail="replicate:N",
    ).properties(width=300, height=300)

    st.altair_chart(chart1 | chart2, use_container_width=True)


if __name__ == "__main__":
    app()

    # default_table = get_default_show_parameters_table()
    # print(default_table)
