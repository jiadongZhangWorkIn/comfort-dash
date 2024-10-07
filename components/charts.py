import base64
import io
from copy import deepcopy

import dash_mantine_components as dmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv, two_nodes, set_tmp
from pythermalcomfort.utilities import v_relative, clo_dynamic
from scipy import optimize
import math

from components.drop_down_inline import generate_dropdown_inline
from utils.my_config_file import ElementsIDs, Models, Functionalities
from utils.website_text import TextHome
import matplotlib
from pythermalcomfort.models import adaptive_en
from pythermalcomfort.psychrometrics import t_o, psy_ta_rh

matplotlib.use("Agg")

import plotly.graph_objects as go
from dash import dcc


def chart_selector(selected_model: str, function_selection: str):

    list_charts = deepcopy(Models[selected_model].value.charts)
    if function_selection == Functionalities.Compare.value:
        if selected_model == Models.PMV_ashrae.name:
            list_charts = deepcopy(Models[selected_model].value.charts_compare)

    list_charts = [chart.name for chart in list_charts]
    drop_down_chart_dict = {
        "id": ElementsIDs.chart_selected.value,
        "question": TextHome.chart_selection.value,
        "options": list_charts,
        "multi": False,
        "default": list_charts[0],
    }

    return generate_dropdown_inline(
        drop_down_chart_dict, value=drop_down_chart_dict["default"], clearable=False
    )


def pmv_en_psy_chart(
    inputs: dict = None,
    model="iso",
    function_selection: str = Functionalities.Default,
    use_to: bool = False,
):
    traces = []

    category_3_up = np.linspace(20.5, 27.1, 100)
    category_2_up = np.linspace(21.4, 26.2, 100)
    category_1_up = np.linspace(22.7, 24.7, 100)
    category_3_low = np.array([33.3, 24.2])
    category_2_low = np.array([32, 25.5])
    category_1_low = np.array([30, 27.4])
    category_1_x = np.concatenate((category_1_up, category_1_low))
    category_2_x = np.concatenate((category_2_up, category_2_low))
    category_3_x = np.concatenate((category_3_up, category_3_low))

    # Category III
    category_3_y = []
    for t in category_3_up:
        category_3_y.append(psy_ta_rh(tdb=t, rh=100, p_atm=101325)["hr"] * 1000)
    category_3_y = np.concatenate((category_3_y, [0] * 2))
    traces.append(
        go.Scatter(
            x=category_3_x,
            y=category_3_y,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="toself",
            fillcolor="rgba(0,255,0,0.2)",
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Category II
    category_2_y = []
    for t in category_2_up:
        category_2_y.append(psy_ta_rh(tdb=t, rh=100, p_atm=101325)["hr"] * 1000)
    category_2_y = np.concatenate((category_2_y, [0] * 2))
    traces.append(
        go.Scatter(
            x=category_2_x,
            y=category_2_y,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="toself",
            fillcolor="rgba(0,255,0,0.3)",
            showlegend=False,
            hoverinfo="none",
        )
    )

    # Category I
    category_1_y = []
    for t in category_1_up:
        category_1_y.append(psy_ta_rh(tdb=t, rh=100, p_atm=101325)["hr"] * 1000)
    category_1_y = np.concatenate((category_1_y, [0] * 2))
    traces.append(
        go.Scatter(
            x=category_1_x,
            y=category_1_y,
            mode="lines",
            line=dict(color="rgba(0,0,0,0)"),
            fill="toself",
            fillcolor="rgba(0,255,0,0.4)",
            showlegend=False,
            hoverinfo="none",
        )
    )

    rh_list = np.arange(0, 101, 10)
    tdb = np.linspace(10, 36, 500)
    for rh in rh_list:
        hr_list = np.array(
            [psy_ta_rh(tdb=t, rh=rh, p_atm=101325)["hr"] * 1000 for t in tdb]
        )
        trace = go.Scatter(
            x=tdb,
            y=hr_list,
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="x+y",
            name=f"{rh}% RH",
            showlegend=False,
        )
        traces.append(trace)

    tdb = inputs[ElementsIDs.t_db_input.value]
    rh = inputs[ElementsIDs.rh_input.value]
    tr = inputs[ElementsIDs.t_r_input.value]
    psy_results = psy_ta_rh(tdb, rh)

    if use_to:
        x_value = t_o(tdb=tdb, tr=tr, v=inputs[ElementsIDs.v_input.value])
        x_label = "Operative Temperature [°C]"
    else:
        x_value = tdb
        x_label = "Dry-bulb Temperature [°C]"

    red_point = [x_value, psy_ta_rh(tdb, rh, p_atm=101325)["hr"] * 1000]
    traces.append(
        go.Scatter(
            x=[red_point[0]],
            y=[red_point[1]],
            mode="markers",
            marker=dict(
                color="red",
                size=4,
            ),
            showlegend=False,
        )
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = red_point[0] + 0.6 * np.cos(theta)
    circle_y = red_point[1] + 1.2 * np.sin(theta)
    traces.append(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="red", width=1.5),
            showlegend=False,
        )
    )

    layout = go.Layout(
        xaxis=dict(title=x_label, showgrid=False),
        yaxis=dict(
            title="Humidity Ratio [g<sub>w</sub>/kg<sub>da</sub>]", showgrid=False
        ),
        showlegend=True,
        plot_bgcolor="white",
        annotations=[
            dict(
                x=14,
                y=28,
                xref="x",
                yref="y",
                text=(
                    f"t<sub>db</sub>: {tdb:.1f} °C<br>"
                    f"rh: {rh:.1f} %<br>"
                    f"W<sub>a</sub>: {psy_results.hr * 1000:.1f} g<sub>w</sub>/kg<sub>da</sub><br>"
                    f"t<sub>wb</sub>: {psy_results.t_wb:.1f} °C<br>"
                    f"t<sub>dp</sub>: {psy_results.t_dp:.1f} °C<br>"
                    f"h: {psy_results.h / 1000:.1f} kJ/kg"
                ),
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0)",
                font=dict(size=14),
            )
        ],
    )

    fig = go.Figure(data=traces, layout=layout)

    return fig


def generate_adaptive_en_chart(
    inputs: dict = None, model="iso", function_selection: str = Functionalities.Default
):
    traces = []

    tdb = inputs[ElementsIDs.t_db_input.value]
    tr = inputs[ElementsIDs.t_r_input.value]
    v = inputs[ElementsIDs.v_input.value]
    t_running_mean = inputs[ElementsIDs.t_rm_input.value]

    x_values = np.array([10, 30])
    results_min = adaptive_en(tdb=tdb, tr=tr, t_running_mean=x_values[0], v=v)
    results_max = adaptive_en(tdb=tdb, tr=tr, t_running_mean=x_values[1], v=v)

    y_values_cat_iii_up = [
        results_min["tmp_cmf_cat_iii_up"],
        results_max["tmp_cmf_cat_iii_up"],
    ]
    y_values_cat_iii_low = [
        results_min["tmp_cmf_cat_iii_low"],
        results_max["tmp_cmf_cat_iii_low"],
    ]

    y_values_cat_ii_up = [
        results_min["tmp_cmf_cat_ii_up"],
        results_max["tmp_cmf_cat_ii_up"],
    ]
    y_values_cat_ii_low = [
        results_min["tmp_cmf_cat_ii_low"],
        results_max["tmp_cmf_cat_ii_low"],
    ]

    y_values_cat_i_up = [
        results_min["tmp_cmf_cat_i_up"],
        results_max["tmp_cmf_cat_i_up"],
    ]
    y_values_cat_i_low = [
        results_min["tmp_cmf_cat_i_low"],
        results_max["tmp_cmf_cat_i_low"],
    ]

    category_3_x = np.concatenate((x_values, x_values[::-1]))
    category_2_x = np.concatenate((x_values, x_values[::-1]))
    category_1_x = np.concatenate((x_values, x_values[::-1]))

    # traces[0]
    traces.append(
        go.Scatter(
            x=category_3_x,
            y=np.concatenate([y_values_cat_iii_up, y_values_cat_iii_low[::-1]]),
            fill="toself",
            fillcolor="rgba(144, 238, 144, 0.3)",
            line=dict(color="rgba(144, 238, 144, 0)", shape="linear"),
            name="Category III",
            mode="lines",
        )
    )
    # traces[1]
    traces.append(
        go.Scatter(
            x=category_2_x,
            y=np.concatenate([y_values_cat_ii_up, y_values_cat_ii_low[::-1]]),
            fill="toself",
            fillcolor="rgba(34, 139, 34, 0.5)",
            line=dict(color="rgba(34, 139, 34, 0)", shape="linear"),
            name="Category II",
            mode="lines",
        )
    )
    # traces[2]
    traces.append(
        go.Scatter(
            x=category_1_x,
            y=np.concatenate([y_values_cat_i_up, y_values_cat_i_low[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 100, 0, 0.7)",
            line=dict(color="rgba(0, 100, 0, 0)", shape="linear"),
            name="Category I",
            mode="lines",
        )
    )

    # Red point
    x = t_running_mean
    y = t_o(tdb=tdb, tr=tr, v=v)
    red_point = [x, y]
    # traces[3]
    traces.append(
        go.Scatter(
            x=[red_point[0]],
            y=[red_point[1]],
            mode="markers",
            marker=dict(
                color="red",
                size=6,
            ),
            showlegend=False,
        )
    )
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = red_point[0] + 0.5 * np.cos(theta)
    circle_y = red_point[1] + 0.7 * np.sin(theta)
    # traces[4]
    traces.append(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="red", width=2.5),
            showlegend=False,
        )
    )

    layout = go.Layout(
        title="Adaptive Chart",
        xaxis=dict(
            title="Outdoor Running Mean Temperature [℃]",
            range=[10, 30],
            dtick=2,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1.5,
            ticks="outside",
            ticklen=5,
            showline=True,
            linewidth=1.5,
            linecolor="black",
        ),
        yaxis=dict(
            title="Operative Temperature [℃]",
            range=[14, 36],
            dtick=2,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1.5,
            ticks="outside",
            ticklen=5,
            showline=True,
            linewidth=1.5,
            linecolor="black",
        ),
        legend=dict(x=0.8, y=1),
        showlegend=False,
        plot_bgcolor="white",
    )
    fig = go.Figure(data=traces, layout=layout)
    return fig


def t_rh_pmv(
    inputs: dict = None,
    model: str = "iso",
    function_selection: str = Functionalities.Default,
):
    results = []
    pmv_limits = [-0.5, 0.5]
    # todo determine if the value is IP unit , transfer to SI
    clo_d = clo_dynamic(
        clo=inputs[ElementsIDs.clo_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    vr = v_relative(
        v=inputs[ElementsIDs.v_input.value], met=inputs[ElementsIDs.met_input.value]
    )

    if function_selection == Functionalities.Compare.value:
        try:
            clo_d_compare = clo_dynamic(
                clo=inputs.get(ElementsIDs.clo_input_input2.value),
                met=inputs.get(ElementsIDs.met_input_input2.value),
            )
            vr_compare = v_relative(
                v=inputs.get(ElementsIDs.v_input_input2.value),
                met=inputs.get(ElementsIDs.met_input_input2.value),
            )
        except KeyError as e:
            print(f"KeyError: {e}. Skipping comparison plotting.")
            clo_d_compare, vr_compare = None, None

    def calculate_pmv_results(tr, vr, met, clo):
        results = []
        for pmv_limit in pmv_limits:
            for rh in np.arange(0, 110, 10):

                def function(x):
                    return (
                        pmv(
                            x,
                            tr=tr,
                            vr=vr,
                            rh=rh,
                            met=met,
                            clo=clo,
                            wme=0,
                            standard=model,
                            limit_inputs=False,
                        )
                        - pmv_limit
                    )

                temp = optimize.brentq(function, 10, 100)
                results.append(
                    {
                        "rh": rh,
                        "temp": temp,
                        "pmv_limit": pmv_limit,
                    }
                )
        return pd.DataFrame(results)

    df = calculate_pmv_results(
        tr=inputs[ElementsIDs.t_r_input.value],
        vr=vr,
        met=inputs[ElementsIDs.met_input.value],
        clo=clo_d,
    )

    # Create the Plotly figure
    fig = go.Figure()

    # Add the filled area between PMV limits
    t1 = df[df["pmv_limit"] == pmv_limits[0]]
    t2 = df[df["pmv_limit"] == pmv_limits[1]]
    fig.add_trace(
        go.Scatter(
            x=t1["temp"],
            y=t1["rh"],
            fill=None,
            mode="lines",
            line=dict(color="rgba(59, 189, 237, 0.7)"),
            name=f"{model} Lower Limit",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t2["temp"],
            y=t2["rh"],
            fill="tonexty",
            mode="lines",
            fillcolor="rgba(59, 189, 237, 0.7)",
            line=dict(color="rgba(59, 189, 237, 0.7)"),
            name=f"{model} Upper Limit",
        )
    )

    # Add scatter point for the current input
    fig.add_trace(
        go.Scatter(
            x=[inputs[ElementsIDs.t_db_input.value]],
            y=[inputs[ElementsIDs.rh_input.value]],
            mode="markers",
            marker=dict(color="red", size=8),
            name="Current Input",
            # hoverinfo='skip',
        )
    )

    # Add hover area to allow hover interaction
    # todo: the interaction area should not the whole chart, at least should not include the while area (e.g. blue only)
    x_range = np.linspace(10, 40, 100)
    y_range = np.linspace(0, 100, 100)
    xx, yy = np.meshgrid(x_range, y_range)
    fig.add_trace(
        go.Scatter(
            x=xx.flatten(),
            y=yy.flatten(),
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)"),
            hoverinfo="x+y",
            name="Interactive Hover Area",
        )
    )

    if (
        function_selection == Functionalities.Compare.value
        and clo_d_compare is not None
    ):
        df_compare = calculate_pmv_results(
            tr=inputs[ElementsIDs.t_r_input_input2.value],
            vr=vr_compare,
            met=inputs[ElementsIDs.met_input_input2.value],
            clo=clo_d_compare,
        )
        t1_compare = df_compare[df_compare["pmv_limit"] == pmv_limits[0]]
        t2_compare = df_compare[df_compare["pmv_limit"] == pmv_limits[1]]
        fig.add_trace(
            go.Scatter(
                x=t1_compare["temp"],
                y=t1_compare["rh"],
                fill=None,
                mode="lines",
                line=dict(color="rgba(30,70,100,0.5)"),
                name=f"{model} Compare Lower Limit",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t2_compare["temp"],
                y=t2_compare["rh"],
                fill="tonexty",
                mode="lines",
                fillcolor="rgba(30,70,100,0.5)",
                line=dict(color="rgba(30,70,100,0.5)"),
                name=f"{model} Compare Upper Limit",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[inputs[ElementsIDs.t_db_input_input2.value]],
                y=[inputs[ElementsIDs.rh_input_input2.value]],
                mode="markers",
                marker=dict(color="blue", size=8),
                name="Compare Input",
            )
        )

    # todo add mouse x,y axis parameter to here

    annotation_text = (
        f"t<sub>db</sub>   {inputs[ElementsIDs.t_db_input.value]:.1f} °C<br>"
    )

    fig.add_annotation(
        x=32,
        y=96,
        xref="x",
        yref="y",
        text=annotation_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=14),
    )
    # Update layout
    fig.update_layout(
        yaxis=dict(title="RH (%)", range=[0, 100], dtick=10),
        xaxis=dict(title="Temperature (°C)", range=[10, 40], dtick=2),
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        hoverdistance=5,
    )

    # Add grid lines and make the spines invisible
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.2)")

    return fig


def SET_outputs_chart(
    inputs: dict = None, calculate_ce: bool = False, p_atmospheric: int = 101325
):
    # Dry-bulb air temperature (x-axis)
    tdb_values = np.arange(10, 40, 0.5, dtype=float).tolist()

    # Prepare arrays for the outputs we want to plot
    set_temp = []  # set_tmp()
    skin_temp = []  # t_skin
    core_temp = []  # t_core
    clothing_temp = []  # t_cl
    mean_body_temp = []  # t_body
    total_skin_evaporative_heat_loss = []  # e_skin
    sweat_evaporation_skin_heat_loss = []  # e_rsw
    vapour_diffusion_skin_heat_loss = []  # e_diff
    total_skin_senesible_heat_loss = []  # q_sensible
    total_skin_heat_loss = []  # q_skin
    heat_loss_respiration = []  # q_res
    skin_wettedness = []  # w

    # Extract common input values
    tr = float(inputs[ElementsIDs.t_r_input.value])
    vr = float(
        v_relative(  # Ensure vr is scalar
            v=inputs[ElementsIDs.v_input.value], met=inputs[ElementsIDs.met_input.value]
        )
    )
    rh = float(inputs[ElementsIDs.rh_input.value])  # Ensure rh is scalar
    met = float(inputs[ElementsIDs.met_input.value])  # Ensure met is scalar
    clo = float(
        clo_dynamic(  # Ensure clo is scalar
            clo=inputs[ElementsIDs.clo_input.value], met=met
        )
    )

    # Iterate through each temperature value and call set_tmp
    for tdb in tdb_values:
        set = set_tmp(
            tdb=tdb,
            tr=tr,
            v=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=0,
            limit_inputs=False,
        )
        set_temp.append(float(set))  # Convert np.float64 to float

    # Iterate through each temperature value and call `two_nodes`
    for tdb in tdb_values:
        results = two_nodes(
            tdb=tdb,
            tr=tr,
            v=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=0,
        )
        # Collect relevant data for each variable, converting to float
        skin_temp.append(float(results["t_skin"]))  # Convert np.float64 to float
        core_temp.append(float(results["t_core"]))  # Convert np.float64 to float
        total_skin_evaporative_heat_loss.append(
            float(results["e_skin"])
        )  # Convert np.float64 to float
        sweat_evaporation_skin_heat_loss.append(
            float(results["e_rsw"])
        )  # Convert np.float64 to float
        vapour_diffusion_skin_heat_loss.append(
            float(results["e_skin"] - results["e_rsw"])
        )  # Convert np.float64 to float
        total_skin_senesible_heat_loss.append(
            float(results["q_sensible"])
        )  # Convert np.float64 to float
        total_skin_heat_loss.append(
            float(results["q_skin"])
        )  # Convert np.float64 to float
        heat_loss_respiration.append(
            float(results["q_res"])
        )  # Convert np.float64 to float
        skin_wettedness.append(
            float(results["w"]) * 100
        )  # Convert to percentage and float

        # calculate clothing temperature t_cl
        pressure_in_atmospheres = float(p_atmospheric / 101325)
        r_clo = 0.155 * clo
        f_a_cl = 1.0 + 0.15 * clo
        h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
        h_fc = 8.600001 * pow((vr * pressure_in_atmospheres), 0.53)
        h_cc = max(h_cc, h_fc)
        if not calculate_ce and met > 0.85:
            h_c_met = 5.66 * (met - 0.85) ** 0.39
            h_cc = max(h_cc, h_c_met)
        h_r = 4.7
        h_t = h_r + h_cc
        r_a = 1.0 / (f_a_cl * h_t)
        t_op = (h_r * tr + h_cc * tdb) / h_t
        clothing_temp.append(
            float((r_a * results["t_skin"] + r_clo * t_op) / (r_a + r_clo))
        )
        # calculate mean body temperature t_body
        alfa = 0.1
        mean_body_temp.append(
            float(alfa * results["t_skin"] + (1 - alfa) * results["t_core"])
        )
    # df = pd.DataFrame(results)
    fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #     x=tdb_values,
    #     y=set_temp,
    #     mode='lines',
    #     name='SET temperature',
    #     line=dict(color='blue')
    # ))

    # Added SET temperature curve
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=set_temp,
            mode="lines",
            name="SET temperature",
            line=dict(color="blue"),
            yaxis="y1",  # Use a  y-axis
        )
    )

    # Adding skin temperature curve
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=skin_temp,
            mode="lines",
            name="Skin temperature",
            line=dict(color="cyan"),
        )
    )

    # Added core temperature curve
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=core_temp,
            mode="lines",
            name="Core temperature",
            line=dict(color="limegreen"),
            yaxis="y1",  # Use a second y-axis
        )
    )

    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=clothing_temp,
            mode="lines",
            name="Clothing temperature",
            line=dict(color="lightgreen"),
            yaxis="y1",  # Use a second y-axis
        )
    )

    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=mean_body_temp,
            mode="lines",
            name="Mean body temperature",
            visible="legendonly",
            line=dict(color="green"),
            yaxis="y1",  # Use a second y-axis
        )
    )
    # total skin evaporative heat loss
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=total_skin_evaporative_heat_loss,
            mode="lines",
            name="Total skin evaporative heat loss",
            visible="legendonly",
            line=dict(color="lightgrey"),
            yaxis="y2",  # Use a second y-axis
        )
    )
    # sweat evaporation skin heat loss
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=sweat_evaporation_skin_heat_loss,
            mode="lines",
            name="Sweat evaporation skin heat loss ",
            visible="legendonly",
            line=dict(color="orange"),
            yaxis="y2",  # Use a second y-axis
        )
    )

    # vapour diffusion skin heat loss
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=vapour_diffusion_skin_heat_loss,
            mode="lines",
            name="Vapour diffusion skin heat loss ",
            visible="legendonly",
            line=dict(color="darkorange"),
            yaxis="y2",  # Use a second y-axis
        )
    )

    # total skin sensible heat loss
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=total_skin_heat_loss,
            mode="lines",
            name="Total skin sensible heat loss ",
            visible="legendonly",
            line=dict(color="darkgrey"),
            yaxis="y2",  # Use a second y-axis
        )
    )

    # Added  total skin heat loss curve
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=total_skin_heat_loss,
            mode="lines",
            name="Total skin heat loss",
            line=dict(color="black"),
            yaxis="y2",  # Use a second y-axis
        )
    )

    #  heat loss respiration curve
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=heat_loss_respiration,
            mode="lines",
            name="Heat loss respiration",
            line=dict(color="black", dash="dash"),
            yaxis="y2",  # Use a second y-axis
        )
    )

    # Added skin moisture curve
    fig.add_trace(
        go.Scatter(
            x=tdb_values,
            y=skin_wettedness,
            mode="lines",
            name="Skin wettedness",
            visible="legendonly",
            line=dict(color="yellow"),
            yaxis="y2",  # Use a second y-axis
        )
    )

    # Set the layout of the chart and adjust the legend position
    fig.update_layout(
        title="Temperature and Heat Loss",
        xaxis=dict(
            title="Dry-bulb Air Temperature [°C]",
            showgrid=False,
            range=[10, 40],
            dtick=2,
        ),
        yaxis=dict(title="Temperature [°C]", showgrid=False, range=[18, 38], dtick=2),
        yaxis2=dict(
            title="Heat Loss [W] / Skin Wettedness [%]",
            showgrid=False,
            overlaying="y",
            side="right",
            range=[0, 70],
            # title_standoff=50  # Increase the distance between the Y axis title and the chart
        ),
        legend=dict(
            x=0.5,  # Adjust the horizontal position of the legend
            y=-0.2,  # Move the legend below the chart
            orientation="h",  # Display the legend horizontally
            traceorder="normal",
            xanchor="center",
            yanchor="top",
        ),
        template="plotly_white",
        autosize=False,
        width=700,  # 3:4
        height=700,  # 3:4
    )

    # show
    return fig


def speed_temp_pmv(inputs: dict = None, model: str = "iso"):
    results = []
    pmv_limits = [-0.5, 0.5]
    clo_d = clo_dynamic(
        clo=inputs[ElementsIDs.clo_input.value], met=inputs[ElementsIDs.met_input.value]
    )

    for pmv_limit in pmv_limits:
        for vr in np.arange(0.1, 1.3, 0.1):

            def function(x):
                return (
                    pmv(
                        x,
                        tr=inputs[ElementsIDs.t_r_input.value],
                        vr=vr,
                        rh=inputs[ElementsIDs.rh_input.value],
                        met=inputs[ElementsIDs.met_input.value],
                        clo=clo_d,
                        wme=0,
                        standard=model,
                        limit_inputs=False,
                    )
                    - pmv_limit
                )

            temp = optimize.brentq(function, 10, 40)
            results.append(
                {
                    "vr": vr,
                    "temp": temp,
                    "pmv_limit": pmv_limit,
                }
            )
    df = pd.DataFrame(results)
    fig = go.Figure()

    # Define trace1
    fig.add_trace(
        go.Scatter(
            x=df[df["pmv_limit"] == pmv_limits[0]]["temp"],
            y=df[df["pmv_limit"] == pmv_limits[0]]["vr"],
            mode="lines",
            # fill='tozerox',
            # fillcolor='rgba(123, 208, 242, 0.5)',
            name=f"PMV {pmv_limits[0]}",
            showlegend=False,
            line=dict(color="rgba(0,0,0,0)"),
        )
    )

    # Define trace2
    fig.add_trace(
        go.Scatter(
            x=df[df["pmv_limit"] == pmv_limits[1]]["temp"],
            y=df[df["pmv_limit"] == pmv_limits[1]]["vr"],
            mode="lines",
            fill="tonextx",
            fillcolor="rgba(123, 208, 242, 0.5)",
            name=f"PMV {pmv_limits[1]}",
            showlegend=False,
            line=dict(color="rgba(0,0,0,0)"),
        )
    )

    # Define input point
    fig.add_trace(
        go.Scatter(
            x=[inputs[ElementsIDs.t_db_input.value]],
            y=[inputs[ElementsIDs.v_input.value]],
            mode="markers",
            marker=dict(color="red"),
            name="Input",
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis_title="Operative Temperature [°C]",  # x title
        yaxis_title="Relative Air Speed [m/s]",  # y title
        template="plotly_white",
        width=700,
        height=525,
        xaxis=dict(
            range=[20, 34],  # x range
            tickmode="linear",
            tick0=20,
            dtick=2,
            gridcolor="lightgrey",
        ),
        yaxis=dict(
            range=[0.0, 1.2],  # 设置y轴范围
            tickmode="linear",
            tick0=0.0,
            dtick=0.1,
            gridcolor="lightgrey",
        ),
    )
    # Return the figure
    return fig


def get_heat_losses(inputs: dict = None, model: str = "ashrae"):
    tr = inputs[ElementsIDs.t_r_input.value]
    print(tr)
    met = inputs[ElementsIDs.met_input.value]
    print(met)
    vel = v_relative(
        v=inputs[ElementsIDs.v_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    print(vel)
    clo_d = clo_dynamic(
        clo=inputs[ElementsIDs.clo_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    print(clo_d)
    rh = inputs[ElementsIDs.rh_input.value]

    ta_range = np.arange(10, 41)
    results = {
        "h1": [],  # Water vapor diffusion through the skin
        "h2": [],  # Evaporation of sweat
        "h3": [],  # Respiration latent
        "h4": [],  # Respiration sensible
        "h5": [],  # Radiation from clothing surface
        "h6": [],  # Convection from clothing surface
        "h7": [],  # Total latent heat loss
        "h8": [],  # Total sensible heat loss
        "h9": [],  # Total heat loss
        "h10": [],  # Metabolic rate
    }

    for ta in ta_range:
        heat_losses = pmv_origin(
            ta=ta, tr=tr, vel=vel, rh=rh, met=met, clo=clo_d, wme=0
        )
        print(heat_losses)
        results["h1"].append(round(heat_losses["hl1"], 1))
        results["h2"].append(round(heat_losses["hl2"], 1))
        results["h3"].append(round(heat_losses["hl3"], 1))
        results["h4"].append(round(heat_losses["hl4"], 1))
        results["h5"].append(round(heat_losses["hl5"], 1))
        results["h6"].append(round(heat_losses["hl6"], 1))
        results["h7"].append(
            round(heat_losses["hl1"] + heat_losses["hl2"] + heat_losses["hl3"], 1)
        )
        results["h8"].append(
            round(heat_losses["hl4"] + heat_losses["hl5"] + heat_losses["hl6"], 1)
        )
        results["h9"].append(
            round(
                heat_losses["hl1"]
                + heat_losses["hl2"]
                + heat_losses["hl3"]
                + heat_losses["hl4"]
                + heat_losses["hl5"]
                + heat_losses["hl6"],
                1,
            )
        )
        results["h10"].append(round(met * 58.15, 1))
    # df = pd.DataFrame(results)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h1",
            mode="lines",
            name="Water vapor diffusion through the skin",
            visible="legendonly",
            line=dict(color="darkgreen"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h2",
            mode="lines",
            name="Evaporation of sweat",
            visible="legendonly",
            line=dict(color="lightgreen"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h3",
            mode="lines",
            name="Respiration latent",
            visible="legendonly",
            line=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h4",
            mode="lines",
            name="WRespiration sensible",
            visible="legendonly",
            line=dict(color="darkred"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h5",
            mode="lines",
            name="Radiation from clothing surface",
            visible="legendonly",
            line=dict(color="darkorange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h6",
            mode="lines",
            name="Convection from clothing surface",
            visible="legendonly",
            line=dict(color="orange"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h7",
            mode="lines",
            name="Total latent heat loss",
            line=dict(color="grey"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h8",
            mode="lines",
            name="Total sensible heat loss",
            line=dict(color="lightgrey"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h9",
            mode="lines",
            name="Total heat loss",
            line=dict(color="black"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ta_range,
            y="h10",
            mode="lines",
            name="Metabolic rate",
            line=dict(color="purple"),
        )
    )
    fig.update_layout(
        title="Temperature and Heat Loss",
        xaxis=dict(
            title="Dry-bulb Air Temperature [°C]",
            showgrid=False,
            range=[10, 40],
            dtick=2,
        ),
        yaxis=dict(title="Heat Loss[W/m] ", showgrid=False, range=[10, 120], dtick=20),
        legend=dict(
            x=0.5,  # Adjust the horizontal position of the legend
            y=-0.2,  # Move the legend below the chart
            orientation="h",  # Display the legend horizontally
            traceorder="normal",  # 按顺序显示
            xanchor="center",
            yanchor="top",
        ),
        template="plotly_white",
        autosize=False,
        width=700,  # 3:4
        height=700,  # 3:4
    )

    # show
    return fig


# what is this?
def pmv_origin(ta, tr, vel, rh, met, clo, wme=0):

    pa = rh * 10 * math.exp(16.6536 - 4030.183 / (ta + 235))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    mw = m - w

    if icl <= 0.078:
        fcl = 1 + 1.29 * icl
    else:
        fcl = 1.05 + 0.645 * icl

    hcf = 12.1 * math.sqrt(vel)
    taa = ta + 273
    tra = tr + 273

    t_cla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * math.pow(tra / 100, 4)
    xn = t_cla / 100
    xf = t_cla / 50
    eps = 0.00015

    n = 0
    while abs(xn - xf) > eps:
        xf = (xf + xn) / 2
        hcn = 2.38 * math.pow(abs(100.0 * xf - taa), 0.25)
        hc = hcf if hcf > hcn else hcn
        xn = (p5 + p4 * hc - p2 * math.pow(xf, 4)) / (100 + p3 * hc)
        n += 1
        if n > 150:
            raise ValueError("Max iterations exceeded")

    tcl = 100 * xn - 273

    hl1 = 3.05 * 0.001 * (5733 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - ta)
    hl5 = 3.96 * fcl * (math.pow(xn, 4) - math.pow(tra / 100, 4))
    hl6 = fcl * hc * (tcl - ta)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

    ppd = 100.0 - 95.0 * math.exp(
        -0.03353 * math.pow(pmv, 4.0) - 0.2179 * math.pow(pmv, 2.0)
    )

    return {
        "pmv": pmv,
        "ppd": ppd,
        "hl1": hl1,
        "hl2": hl2,
        "hl3": hl3,
        "hl4": hl4,
        "hl5": hl5,
        "hl6": hl6,
    }

def t_rh_pmv_category(inputs: dict = None, model: str = "iso", function_selection: str = Functionalities.Default):
    results = []
    # Specifies the category of the PMV interval
    pmv_limits = [-0.7, -0.5, -0.2, 0.2, 0.5, 0.7]
    colors = [
        "rgba(168,204,162,0.9)",  # Light green
        "rgba(114,174,106,0.9)",  # Medium green
        "rgba(78,156,71,0.9)",  # Dark green
        "rgba(114,174,106,0.9)",  # Medium green
        "rgba(168,204,162,0.9)",  # Light green
    ]
    clo_d = clo_dynamic(
        clo=inputs[ElementsIDs.clo_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    vr = v_relative(
        v=inputs[ElementsIDs.v_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    for i in range(len(pmv_limits) - 1):
        lower_limit = pmv_limits[i]
        upper_limit = pmv_limits[i + 1]
        color = colors[i]  # Corresponding color

        for rh in np.arange(0, 110, 10):
            # Find the upper and lower limits of temperature
            def function(x):
                return (
                    pmv(
                        x,
                        tr=inputs[ElementsIDs.t_r_input.value],
                        vr=vr,
                        rh=rh,
                        met=inputs[ElementsIDs.met_input.value],
                        clo=clo_d,
                        wme=0,
                        standard=model,
                        limit_inputs=False,
                    )
                    - lower_limit
                )

            temp_lower = optimize.brentq(function, 10, 40)

            def function_upper(x):
                return (
                    pmv(
                        x,
                        tr=inputs[ElementsIDs.t_r_input.value],
                        vr=vr,
                        rh=rh,
                        met=inputs[ElementsIDs.met_input.value],
                        clo=clo_d,
                        wme=0,
                        standard=model,
                        limit_inputs=False,
                    )
                    - upper_limit
                )

            temp_upper = optimize.brentq(function_upper, 10, 40)
            # Record RH and temperature upper and lower limits for each interval
            results.append(
                {
                    "rh": rh,
                    "temp_lower": temp_lower,
                    "temp_upper": temp_upper,
                    "pmv_lower_limit": lower_limit,
                    "pmv_upper_limit": upper_limit,
                    "color": color,  # Use the specified color
                }
            )
    df = pd.DataFrame(results)

    if df.empty:
        print("No data available for plotting.")
    # Visualization: Create a chart with multiple fill areas
    fig = go.Figure()
    for i in range(len(pmv_limits) - 1):
        region_data = df[
            (df["pmv_lower_limit"] == pmv_limits[i])
            & (df["pmv_upper_limit"] == pmv_limits[i + 1])
        ]
        # Draw the temperature line at the bottom
        fig.add_trace(
            go.Scatter(
                x=region_data["temp_lower"],
                y=region_data["rh"],
                fill=None,
                mode="lines",
                line=dict(color="rgba(255,255,255,0)"),
            )
        )
        # Draw the temperature line at the top and fill in the color
        if colors[i]:
            fig.add_trace(
                go.Scatter(
                    x=region_data["temp_upper"],
                    y=region_data["rh"],
                    fill="tonexty",
                    fillcolor=colors[i],  # Use defined colors
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                )
            )
    # Add red dots to indicate the current input temperature and humidity
    fig.add_trace(
        go.Scatter(
            x=[inputs[ElementsIDs.t_db_input.value]],
            y=[inputs[ElementsIDs.rh_input.value]],
            mode="markers",
            marker=dict(color="red", size=12),
            name="Current Condition",
        )
    )

    annotation_text = (
        f"t<sub>db</sub>   {inputs[ElementsIDs.t_db_input.value]:.1f} °C<br>"
    )

    fig.add_annotation(
        x=32,
        y=96,
        xref="x",
        yref="y",
        text=annotation_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(size=14),
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Temperature (°C)",
        yaxis_title="Relative Humidity (%)",
        showlegend=False,
        template="simple_white",
        xaxis=dict(
            range=[10, 40],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            dtick=2,  # Set the horizontal scale interval to 2
        ),
        yaxis=dict(
            range=[0, 100],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            dtick=10,  # Set the ordinate scale interval to 10
        ),
    )
    # return dcc.Graph(figure=fig)
    return fig