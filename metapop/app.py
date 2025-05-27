# flake8: noqa
# This file is part of the metapop package. It contains the Streamlit app for
# the metapopulation model

import pandas as pd
import streamlit as st
import numpy as np
import polars as pl
import altair as alt
from st_flexible_callout_elements import flexible_callout
import io
import sys
import base64
import datetime

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
    update_intervention_parameters_from_widget,
    reset,
    add_daily_incidence,
    get_interval_results,
    get_table,
    get_median_trajectory_from_episize,
    get_median_trajectory_from_peak_time,
)
from .helper import build_vax_schedule

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
#     get_table,
# )
### note: this is not recommended use within a file that is imported as a package module, but it can be useful for testing purposes

__all__ = [
    "app",
]


def app(replicates=20):
    st.set_page_config(layout="wide")

    st.title("Measles Outbreak Simulator")

    st.text(
        "This interactive tool illustrates the impact of "
        "vaccination, isolation, and quarantine measures on the "
        "size of measles outbreaks following introduction of measles into "
        "a community, by comparing scenarios with and without interventions."
    )

    parms = read_parameters("scripts/app/onepop_config.yaml")

    scenario_names = ["No Interventions", "Interventions"]
    show_parameter_mapping = get_show_parameter_mapping(parms)

    advanced_parameter_mapping = get_advanced_parameter_mapping()

    with st.sidebar:
        st.text(
            "This tool is meant for use at the beginning of an outbreak at the county level or finer geographic scale. "
            "It is not intended to provide an exact forecast of cases in any community. "
            "Hover over the ? for more information about each parameter."
        )
        st.header(
            "Model Inputs",
            help="Enter the population size, baseline immunity, and number of people "
            "initially infected with measles in a community. ",
        )

        widget_types = (
            get_widget_types()
        )  # defines the type of widget for each parameter
        min_values = dict(pop_sizes=[1000, 100, 100])
        min_values = get_min_values(min_values)
        max_values = get_max_values()
        steps = get_step_values()
        helpers = get_helpers()
        formats = get_formats()
        keys0 = get_widget_idkeys(0)  # keys for the shared parameters
        keys1 = get_widget_idkeys(1)  # keys for the parameters for scenario 1
        keys2 = get_widget_idkeys(2)  # keys for the parameters for Interventions

        # some customization of the helper texts in the sidebar
        helpers["I0"][0] = "The model currently has a maximum of 10 initial infections."
        helpers["pop_sizes"][0] = (
            "The model currently has a minimum of 1,000 people and a maximum of 100,000 people. As population sizes get larger, the assumption of a well-mixed population becomes less valid."
        )
        helpers["initial_vaccine_coverage"][0] = (
            "The percent of the population with any immunity against measles, including both through MMR vaccination and through past infection."
        )

        # add a section in the sidebar panel to reset the app
        col_reset = st.columns(1)[0]

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

        # add a section for the shared parameters in the sidebar panel
        col0 = st.columns(1)[0]

        col0.subheader(
            "Shared parameters",
        )

        col0.text(
            "Type in a population size and baseline immunity, as "
            "well as the number of people initially infected with measles in the population. "
        )

        subheader = ""
        edited_parms = app_editors(
            col0,
            subheader,
            parms,
            shared_keys,
            shared_list_keys,
            show_parameter_mapping,
            widget_types,
            min_values,
            max_values,
            steps,
            helpers,
            formats,
            keys0,
        )

        # Set parameters for each scenario separately
        # order of parameters in the sidebar
        ordered_keys = [
            "isolation_on",
            "pre_rash_isolation_on",
            "total_vaccine_uptake_doses",
            "vaccine_uptake_start_day",
            "vaccine_uptake_duration_days",
        ]
        # parameters that are lists or arrays
        list_parameter_keys = []

        # Scenario parameters
        # parameters for scenario 1 are not shown
        st.header(
            "Interventions scenario",
            help="The adherence to both isolation and quarantine, "
            "as well as the vaccine uptake and start time and duration of the vaccine campaign, "
            "can be specified.",
        )
        st.text(
            "Choose interventions to simulate and compare with a "
            "scenario with no active intervenions. Interventions can be applied "
            "independently or in combination with each other. "
            "The results are compared to a baseline scenario that does not "
            "have any vaccine uptake, isolation, or quarantine incorporated."
        )

        st.text(
            "When quarantine and isolation are turned on, they are applied to the entire duration of the simulation."
        )

        # For the no intervention scenario, intervention parameters are set to 0
        edited_parms1 = set_parms_to_zero(
            edited_parms, ["pre_rash_isolation_reduction", "isolation_reduction"]
        )

        col_intervention = st.columns(1)[0]

        # For the intervention scenario, user defines values
        edited_parms2 = app_editors(
            # st.container(),
            col_intervention,
            "",
            edited_parms,
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

        # if total uptake doses is a proportion, scale to a number of doses
        if parms["use_prop_vaccine_uptake"]:
            edited_parms2 = rescale_prop_vax(edited_parms2)

        # placeholder for text about the vaccine campaign doses and other intervention effects in the siderbar
        # defining this here allows us to place it above the advanced options section
        col_intervention_text = st.columns(1)[0]

        with st.expander("Disease parameters"):
            st.text(
                "These options allow changes to parameter assumptions including "
                "measles natural history parameters as well as parameters governing "
                "intervention efficacy."
            )

            advanced_ordered_keys = [
                "desired_r0",
                "latent_duration",
                "infectious_duration",
                "pre_rash_isolation_adherence",
                "isolation_adherence",
                "IHR",
            ]

            # show additional advanced parameters if there are multiple population groups
            if parms["n_groups"] > 1:
                advanced_ordered_keys = advanced_ordered_keys + [
                    "k_i_0",
                    "k_g1",
                    "k_g2",
                    "k_21",
                ]

            # advanced parameters that are lists or arrays
            advanced_list_keys = ["k_i"]

            # set the parameters for scenario 2
            edited_advanced_parms2 = app_editors(
                st.container(),
                "",
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
                keys1,
                disabled=False,
            )

            # shared advanced parameters to scenario 1
            edited_advanced_parms1 = edited_parms1
            for key in advanced_ordered_keys:
                edited_advanced_parms1[key] = edited_advanced_parms2[key]

    #### Intervention scenarios:
    # instead of expander, can use st.radio:
    if (
        (
            edited_parms2["total_vaccine_uptake_doses"] == 0
            or edited_parms2["vaccine_uptake_duration_days"] == 0
        )
        and edited_parms2["pre_rash_isolation_on"] == False
        and edited_parms2["isolation_on"] == False
    ):
        interventions = "Off"
    else:
        interventions = "On"

    # before running the model, reset the parameters to their original values if the user clicks the reset button
    with col_reset:
        reset_button = st.button(
            "Reset parameters",
            on_click=reset,
            args=(
                parms,
                widget_types,
            ),
        )

    # set model parameters based on app inputs
    edited_intervention_parms1 = update_intervention_parameters_from_widget(
        edited_advanced_parms1
    )
    edited_intervention_parms2 = update_intervention_parameters_from_widget(
        edited_advanced_parms2
    )

    # Display number of doses administered now that use has finished selecting parameters
    with col_intervention_text:
        st.text(
            f"Total vaccines administered during campaign: {edited_intervention_parms2['total_vaccine_uptake_doses']}",
            help="This number is calculated based on user input for the percentage of the non-immune population that gets vaccinated during the vaccine campaign.",
        )

    ### Dictate that isolation > quarantine
    if (
        edited_parms2["pre_rash_isolation_on"] == True
        and edited_parms2["isolation_on"] == False
    ):
        st.error(
            "Isolation must be activated for quarantine to be activated. "
            "Please adjust the intervention toggles"
        )
        return

    if (
        edited_intervention_parms2["isolation_adherence"]
        < edited_intervention_parms2["pre_rash_isolation_adherence"]
        and edited_parms2["pre_rash_isolation_on"] == True
        and edited_parms2["isolation_on"] == True
    ):
        st.error(
            "Isolation adherence should be greater than or equal to quarantine adherence. "
            "Please adjust the parameters."
        )
        return

    updated_parms1 = edited_intervention_parms1.copy()
    updated_parms2 = edited_intervention_parms2.copy()

    scenario1 = [updated_parms1]
    scenario2 = [updated_parms2]

    #### Plot Options:
    # get the selected outcome from the sidebar
    outcome_option = st.selectbox(
        "Metric",
        get_outcome_options(),
        index=0,  # by default display weekly incidence
        placeholder="Select an outcome to plot",
    )

    ### Run computations and display results
    chart_placeholder = st.empty()

    # Map the selected option to the outcome variable
    outcome_mapping = get_outcome_mapping()
    outcome = outcome_mapping[outcome_option]

    # run the model with the updated parameters
    chart_placeholder.text("Running scenarios...")
    results1 = get_scenario_results(scenario1)
    results2 = get_scenario_results(scenario2)

    # fullresults for later
    fullresults1 = results1
    fullresults2 = results2

    # extract groups
    groups = results1["group"].unique().to_list()

    # do some processing here to get daily incidence
    chart_placeholder.text("Adding daily incidence...")
    results1 = add_daily_incidence(results1, groups)
    results2 = add_daily_incidence(results2, groups)

    # create tables with interval results - weekly incidence, weekly cumulative incidence
    interval = 7

    chart_placeholder.text("Getting interval results...")
    interval_results1 = get_interval_results(results1, groups, interval)
    interval_results2 = get_interval_results(results2, groups, interval)

    # rename columns for the app
    app_column_mapping = {f"inc_{interval}": "Winc", "Y": "WCI"}
    interval_results1 = interval_results1.rename(app_column_mapping)
    interval_results2 = interval_results2.rename(app_column_mapping)

    # vax schedule for plotting
    sched = build_vax_schedule(edited_parms2)

    if outcome not in ["Y", "inc", "Winc", "WCI"]:
        print("outcome not available yet, defaulting to Cumulative Daily Incidence")
        outcome = "Y"

    if outcome_option in [
        "Daily Incidence",
        "Daily Cumulative Incidence",
    ]:
        alt_results1 = results1
        alt_results2 = results2
        # min_y, max_y = 0, max(results1[outcome].max(), results2[outcome].max())
        x = "t:Q"
        time_label = "Time (days)"
        vax_start = min(sched.keys())
        vax_end = max(sched.keys())
    elif outcome_option in ["Weekly Incidence", "Weekly Cumulative Incidence"]:
        alt_results1 = interval_results1
        alt_results2 = interval_results2
        # min_y, max_y = 0, max(interval_results1[outcome].max(), interval_results2[outcome].max())
        x = "interval_t:Q"
        time_label = "Time (weeks)"
        vax_start = min(sched.keys()) / interval
        vax_end = max(sched.keys()) / interval

    # get median line for each scenario (based on ALL sims, not just smaller sample)
    ave_results1 = get_median_trajectory_from_peak_time(alt_results1).with_columns(
        pl.lit(scenario_names[0]).alias("scenario")
    )
    ave_results2 = get_median_trajectory_from_peak_time(alt_results2).with_columns(
        pl.lit(scenario_names[1]).alias("scenario")
    )
    combined_ave_results = ave_results1.vstack(ave_results2)

    # Get smaller subset of results for plotting
    alt_results1 = alt_results1.with_columns(
        pl.lit(scenario_names[0]).alias("scenario")
    )
    alt_results2 = alt_results2.with_columns(
        pl.lit(scenario_names[1]).alias("scenario")
    )
    replicate_inds = np.random.choice(
        results1["replicate"].unique().to_numpy(), replicates, replace=False
    )
    combined_alt_results = alt_results1.vstack(alt_results2).filter(
        pl.col("replicate").is_in(replicate_inds)
    )

    # create chart title, depending on whether interventions are on/off
    if interventions == "On":
        pre_rash_isolation_adherance = 0
        isolation_adherance = 0
        if edited_intervention_parms2["pre_rash_isolation_on"]:
            pre_rash_isolation_adherance = edited_intervention_parms2[
                "pre_rash_isolation_adherence"
            ]
        if edited_intervention_parms2["isolation_on"]:
            isolation_adherance = edited_intervention_parms2["isolation_adherence"]
        pre_rash_isolation_adherance_pct = int(pre_rash_isolation_adherance * 100)
        isolation_adherance_pct = int(isolation_adherance * 100)
        title = alt.TitleParams(
            "Outcome Comparison by Scenario",
            subtitle=[
                (
                    f"Vaccine campaign: {edited_intervention_parms2['total_vaccine_uptake_doses']} "
                    "doses administered"
                ),
                f"Quarantine adherence: {pre_rash_isolation_adherance_pct}%",
                f"Isolation adherence: {isolation_adherance_pct}%",
            ],
        )
    else:
        combined_alt_results = alt_results1.filter(
            pl.col("replicate").is_in(replicate_inds)
        )
        combined_ave_results = ave_results1
        title = "No Intervention Scenario"

    chart_placeholder.text("Building charts...")
    chart = (
        alt.Chart(combined_alt_results)
        .mark_line(opacity=0.3, strokeWidth=0.75)
        .encode(
            x=alt.X(x, title=time_label),
            y=alt.Y(outcome, title=outcome_option),
            color=alt.Color(
                "scenario",
                title="Scenario",
                scale=alt.Scale(
                    domain=[scenario_names[0], scenario_names[1]],  # Scenarios
                    range=["#FB7E38", "#0057b7"],  # Corresponding colors (orange, blue)
                ),
            ),
            detail="replicate",
            tooltip=["scenario", "t", outcome],
        )
        .properties(title=title, width=800, height=400)
    )

    # If vaccines administered > 0, add vax schedule to plot
    if edited_intervention_parms2["total_vaccine_uptake_doses"] > 0:
        vax = (
            alt.Chart(
                pd.DataFrame(
                    {
                        "x_start": [vax_start],  # Start of the box
                        "x_end": [vax_end],  # End of the box
                    }
                )
            )
            .mark_rect(opacity=0.1, color="grey")
            .encode(
                x=alt.X("x_start:Q", title=time_label),
                x2="x_end:Q",  # End position of the box
            )
        )

        chart = chart + vax

    ave_line = (
        alt.Chart(combined_ave_results.to_pandas())
        .mark_line(opacity=1.0, strokeWidth=3.0)
        .encode(
            x=alt.X(x, title=time_label),
            y=alt.Y(outcome, title=outcome_option),
            color=alt.Color(
                "scenario",
                title="Scenario",
                scale=alt.Scale(
                    domain=[scenario_names[0], scenario_names[1]],  # Scenarios
                    range=["#cf4828", "#20419a"],  # Corresponding colors (red, blue)
                ),
            ),
            tooltip=["scenario", "t", outcome],
        )
    )

    chart = chart + ave_line

    if interventions == "Off":
        # Create a text annotation
        annotation = (
            alt.Chart(
                pd.DataFrame(
                    {"text": ["Use at least one intervention to compare scenarios"]}
                )
            )
            .mark_text(align="center", baseline="top", color="grey", fontSize=18)
            .encode(text="text:N", y=alt.value(10))
        )

        # Add the annotation to the chart
        chart = chart + annotation
    else:
        chart = chart

    chart = chart.properties(padding={"top": 10, "bottom": 30, "left": 30, "right": 40})
    chart_placeholder.altair_chart(chart, use_container_width=True)

    # Text below the chart
    st.markdown(
        '<p style="font-size:14px;">'
        "Each thin line represents an individual simulation of the stochastic "
        "model. All simulations within a given scenario (i.e., shown with "
        "the same color) are run under the same set of parameters, and "
        "differences between each individual simulation are due to random "
        "variation in contact rates. Bolded lines show the simulation that possessed "
        "the median time of peak prevalence across all epidemic trajectories for "
        "each scenario."
        "</p>",
        unsafe_allow_html=True,
    )

    ### Outbreak Summary Stats
    st.subheader("Simulation summary")

    with st.expander("Show intervention strategy info.", expanded=False):
        columns = st.columns(2)

        flexible_callout(
            (
                "No Interventions:<br><ul>"
                "<li> Vaccines administered during campaign: 0</li>"
                "<li> Adherence to quarantine among pre-symptomatic infectious individuals: 0%</li>"
                "<li> Adherence to isolation among symptomatic infectious individuals: 0%</li></ul>"
            ),
            background_color="#feeadf",
            font_color="#8f3604",
            container=columns[0],
        )
        if interventions == "On":
            pre_rash_isolation_adherance = 0
            isolation_adherance = 0
            if edited_intervention_parms2["pre_rash_isolation_on"]:
                pre_rash_isolation_adherance = edited_intervention_parms2[
                    "pre_rash_isolation_adherence"
                ]
            if edited_intervention_parms2["isolation_on"]:
                isolation_adherance = edited_intervention_parms2["isolation_adherence"]

            callout_text = "Interventions:<br><ul>"
            if (
                edited_intervention_parms2["total_vaccine_uptake_doses"] == 0
                or edited_intervention_parms2["vaccine_uptake_duration_days"] == 0
            ):
                callout_text += "<li> Vaccines administered during campaign: 0</li>"
            else:
                callout_text += f"<li> Vaccines administered during campaign: {edited_intervention_parms2['total_vaccine_uptake_doses']} between day {edited_intervention_parms2['vaccine_uptake_start_day']} and day {edited_intervention_parms2['vaccine_uptake_start_day'] + edited_intervention_parms2['vaccine_uptake_duration_days']}</li>"
            callout_text += f"<li> Adherence to quarantine among pre-symptomatic infectious individuals: {int(pre_rash_isolation_adherance*100)}%</li>"
            callout_text += f"<li> Adherence to isolation among symptomatic infectious individuals: {int(isolation_adherance*100)}%</li></ul>"

            flexible_callout(
                callout_text,
                background_color="#cbe4ff",
                font_color="#001833",
                container=columns[1],
            )

    fullresults1 = fullresults1.with_columns(
        pl.lit(scenario_names[0]).alias("Scenario")
    )
    fullresults2 = fullresults2.with_columns(
        pl.lit(scenario_names[1]).alias("Scenario")
    )

    combined_results = (
        fullresults2.vstack(fullresults1)
        .with_columns(
            pl.col("t").cast(pl.Int64),
            pl.col("group").cast(pl.Int64),
            pl.col("Y").cast(pl.Int64),
        )
        .filter(pl.col("t") == pl.col("t").max())
        .group_by("Scenario", "replicate")
        .agg(pl.col("Y").sum().alias("Total"))
    )

    outbreak_summary = get_table(combined_results, edited_intervention_parms2["IHR"])

    if interventions == "Off":
        outbreak_summary = outbreak_summary.select("", scenario_names[0])

    # add highlight of the outbreak summary
    if interventions == "On":
        relative_difference = (
            outbreak_summary.filter(pl.col("") == "Infections, mean (95% CI)")
            .select("Relative Difference (%)")
            .to_numpy()[0][0]
        )

        intervention_text = f"Adding "
        interventions = []
        if edited_intervention_parms2["total_vaccine_uptake_doses"] > 0:
            interventions.append("vaccination")
        if edited_intervention_parms2["pre_rash_isolation_success"] > 0:
            interventions.append("quarantine")
        if edited_intervention_parms2["isolation_success"] > 0:
            interventions.append("isolation")

        if len(interventions) > 1:
            intervention_text += (
                ", ".join(interventions[:-1]) + " and " + interventions[-1]
            )
        elif interventions:
            intervention_text += interventions[0]

        st.text(
            f"{intervention_text} decreases total cases by {float(relative_difference):.0f}% "
            f"in a population of size {edited_parms2['pop_sizes'][0]} "
            f"with baseline immunity of {round(edited_parms2['initial_vaccine_coverage'][0]*100)}%."
        )

        # if the Relative Difference is NaN, set it to ""
        # outbreak_summary = outbreak_summary.with_columns(
        #     pl.col("Relative Difference (%)").fill_nan("")
        # )

    if "pyodide" in sys.modules:
        # Workaround for stlite not displying st.dataframe correctly in this case.
        # Unclear if this is a stlite pyarrow problem (see notes at
        # https://github.com/whitphx/stlite?tab=readme-ov-file#limitations
        # about `st.dataframe()`.)
        st.markdown(outbreak_summary.to_pandas().to_markdown(index=False))
        csv_buffer = io.StringIO()
        outbreak_summary.to_pandas().to_csv(csv_buffer, index=False)

        base64_encoded = base64.b64encode(csv_buffer.getvalue().encode("utf-8")).decode(
            "utf-8"
        )
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        data_uri = f'<p style="font-size:10px;text-align:right;"><a download="measles-sim-{time}.csv" href="data:text/csv;base64,{base64_encoded}">Download as CSV</a></p>'
        st.markdown(data_uri, unsafe_allow_html=True)
    else:
        st.dataframe(outbreak_summary)

    # add a section on the detailed methods
    with st.expander("Detailed methods", expanded=False):
        st.markdown(
            """
            <p style="font-size:14px;">
            This model examines measles transmission in a population after
            introduction of measles cases. This is a stochastic SVEIR model,
            with built-in random variation. We run 200 individual
            simulations to produce a range of possible outcomes and estimates
            of associated uncertainty. People who are immune at the beginning
            of the simulation, either through past vaccination or previous
            infection, begin in the "Vaccinated" compartment.<br><br>

            <p style="font-size:14px;">
            Users can explore the impact of interventions, including vaccination,
            isolation, and quarantine measures ("interventions" scenario)
            compared to a baseline scenario without active interventions ("No
            Interventions"). The start and end time of the vaccine campaign
            can be specified.<br><br>

            <b style="font-size:14px;">Assumptions</b>
            <p style="font-size:14px;">We note that this modeling approach
            makes several simplifying assumptions, including the following:</p>
            <ul>
            <li style="font-size:14px;">This is a compartmental SVEIR model of
            a well-mixed population, which means individuals have the same
            probability of contact with each other. At the county scale, this
            may underestimate the risk of an outbreak if unvaccinated people
            are more likely to come into contact with each other.</li>

            <b style="font-size:14px;">Model Parameters</b>
            <ul>
            <li style="font-size:14px;"> The basic reproductive number (R<sub>0</sub>), captures contact rates and the probability of infection given contact with an infectious individual. R<sub>0</sub> for measles is generally estimated
            to be between 12 and 18  <a href='https://www.ecdc.europa.eu/en/measles/facts' target='_blank'>[Factsheet about measles]</a>. Communities with higher contact rates — for example populations with higher population density
            or larger households <a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC8765757/' target='_blank'>[Social contact patterns and implications for infectious disease transmission – a systematic review and meta-analysis of contact surveys | eLife]</a> — may have higher contact rates and higher R<sub>0</sub>. The probability of infection given contact with an infectious individuals is very high for measles;
            the household attack rate is estimated to be 90% <a href='https://www.cdc.gov/yellow-book/hcp/travel-associated-infections-diseases/measles-rubeola.html#:~:text=Measles%20is%20among%20the%20most,global%20eradication%20of%20measles%20feasible' target='_blank'>[CDC Yellow Book: Measles (Rubeola)]</a>.
           </a></li>
            <li style="font-size:14px;">The latent period is generally
            estimated to be around 11 days
            <a href='https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html' target='_blank'>[Measles Clinical Diagnosis Fact Sheet | Measles (Rubeola) | CDC]</a>.</li>
            <li style="font-size:14px;">The infectious period is generally
            estimated to be around 9 days
            <a href='https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html' target='_blank'>[Measles Clinical Diagnosis Fact Sheet | Measles (Rubeola) | CDC]</a>.</li>
            <li style="font-size:14px;">Measles rash onset is generally
            estimated to be on day 5 of this infectious period
            <a href='https://www.cdc.gov/measles/hcp/communication-resources/clinical-diagnosis-fact-sheet.html' target='_blank'>[Measles Clinical Diagnosis Fact Sheet | Measles (Rubeola) | CDC]</a>.
            In this model, isolation when sick is assumed to start halfway through the infectious period.
            </li>
            <li style="font-size:14px;"> We assume vaccine efficacy for individuals vaccinated during the campaign is 93%,
            the estimate for one dose of MMR
            <a href='https://www.cdc.gov/measles/vaccines/index.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fvaccines%2Fvpd%2Fmmr%2Fpublic%2Findex.html' target='_blank'>[MMR Vaccine Information]</a>.
            </li>
            </li>
            <li style="font-size:14px;"> We assume vaccine efficacy for individuals vaccinated prior
            to the campaign is 97%, the estimate for two doses of MMR
            <a href='https://www.cdc.gov/measles/vaccines/index.html?CDC_AA_refVal=https%3A%2F%2Fwww.cdc.gov%2Fvaccines%2Fvpd%2Fmmr%2Fpublic%2Findex.html' target='_blank'>[MMR Vaccine Information]</a>.
            </li>
            <li style="font-size:14px;"> Individuals with immunity prior to introduction are assumed to have one or two doses of MMR or have had a prior measles infection.
            </li>
            <li style="font-size:14px;">Isolation when sick is estimated to be
            approximately 75% effective at reducing transmission when comparing
            people who do isolate when sick to people who do not
            <a href='https://academic.oup.com/cid/article/75/1/152/6424734' target='_blank'>[Impact of Isolation and Exclusion as a Public Health Strategy to Contain Measles Virus Transmission During a Measles Outbreak | Clinical Infectious Diseases | Oxford Academic]</a>
            . In this model, since isolation starts only at rash onset,
            isolation reduces transmission by 100% during the second half of the infectious period,
            leading to a reduction of 50% overall.
            <li style="font-size:14px;">Quarantine for people who are unvaccinated but have been
            exposed is estimated to be 44-76% effective at reducing transmission
            when comparing those who do quarantine to those who do not.
            We assume a 60% reduction in transmission, which is the mean of this range.
            <a href='https://academic.oup.com/cid/article/75/1/152/6424734' target='_blank'>[Impact of Isolation and Exclusion as a Public Health Strategy to Contain Measles Virus Transmission During a Measles Outbreak | Clinical Infectious Diseases | Oxford Academic]</a>
            </li>

            </ul>
            </p>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    app()
