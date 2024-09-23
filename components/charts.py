import base64
import io
from copy import deepcopy
import dash_mantine_components as dmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pythermalcomfort.models import pmv, adaptive_ashrae
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.psychrometrics import psy_ta_rh
from scipy import optimize
import math
from components.drop_down_inline import generate_dropdown_inline
from utils.my_config_file import ElementsIDs, Models, UnitSystem, UnitConverter
from utils.website_text import TextHome
import matplotlib

matplotlib.use("Agg")

import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
import dash_core_components as dcc


def chart_selector(selected_model: str):
    list_charts = deepcopy(Models[selected_model].value.charts)
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


# fig example
def t_rh_pmv(inputs: dict = None, model: str = "iso"):
    results = []
    pmv_limits = [-0.5, 0.5]
    clo_d = clo_dynamic(
        clo=inputs[ElementsIDs.clo_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    vr = v_relative(
        v=inputs[ElementsIDs.v_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    for pmv_limit in pmv_limits:
        for rh in np.arange(0, 110, 10):

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
                    - pmv_limit
                )

            temp = optimize.brentq(function, 10, 40)
            results.append(
                {
                    "rh": rh,
                    "temp": temp,
                    "pmv_limit": pmv_limit,
                }
            )

    df = pd.DataFrame(results)
    f, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
    t1 = df[df["pmv_limit"] == pmv_limits[0]]
    t2 = df[df["pmv_limit"] == pmv_limits[1]]
    axs.fill_betweenx(
        t1["rh"], t1["temp"], t2["temp"], alpha=0.5, label=model, color="#7BD0F2"
    )
    axs.scatter(
        inputs[ElementsIDs.t_db_input.value],
        inputs[ElementsIDs.rh_input.value],
        color="red",
    )
    axs.set(
        ylabel="RH (%)",
        xlabel="Temperature (°C)",
        ylim=(0, 100),
        xlim=(10, 40),
    )
    axs.legend(frameon=False).remove()
    axs.grid(True, which="both", linestyle="--", linewidth=0.5)
    axs.spines["top"].set_visible(False)
    axs.spines["right"].set_visible(False)
    plt.tight_layout()

    my_stringIObytes = io.BytesIO()
    plt.savefig(
        my_stringIObytes,
        format="png",
        transparent=True,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    plt.close("all")
    return dmc.Image(
        src=f"data:image/png;base64, {my_base64_jpgData}",
        alt="Heat stress chart",
        py=0,
    )


def t_rh_pmv_category(inputs: dict = None, model: str = "iso"):
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
    return dcc.Graph(figure=fig)


def pmot_ot_adaptive_ashrae(inputs: dict = None, model: str = "ashrae"):
    # Input parameter
    air_temperature = inputs[ElementsIDs.t_db_input.value]  # Air Temperature
    mean_radiant_temp = inputs[ElementsIDs.t_r_input.value]  # Mean Radiant Temperature
    prevailing_mean_outdoor_temp = inputs[
        ElementsIDs.t_rm_input.value
    ]  # Prevailing Mean Outdoor Temperature
    air_speed = inputs[ElementsIDs.v_input.value]  # Air Speed
    operative_temperature = air_temperature  # 计算 Operative Temperature
    units = inputs[ElementsIDs.UNIT_TOGGLE.value]  # unit (IP or SI)
    # Calculate the values for the special points t_running_mean = 10 and t_running_mean = 33.5
    t_running_means = [10, 33.5]  # special points
    results = []
    for t_running_mean in t_running_means:
        adaptive = adaptive_ashrae(
            tdb=air_temperature,
            tr=mean_radiant_temp,
            t_running_mean=t_running_mean,
            v=air_speed,
        )
        if units == UnitSystem.IP.value:
            t_running_mean = UnitConverter.celsius_to_fahrenheit(t_running_mean)
            adaptive.tmp_cmf = UnitConverter.celsius_to_fahrenheit(adaptive.tmp_cmf)
            adaptive.tmp_cmf_80_low = UnitConverter.celsius_to_fahrenheit(
                adaptive.tmp_cmf_80_low
            )
            adaptive.tmp_cmf_80_up = UnitConverter.celsius_to_fahrenheit(
                adaptive.tmp_cmf_80_up
            )
            adaptive.tmp_cmf_90_low = UnitConverter.celsius_to_fahrenheit(
                adaptive.tmp_cmf_90_low
            )
            adaptive.tmp_cmf_90_up = UnitConverter.celsius_to_fahrenheit(
                adaptive.tmp_cmf_90_up
            )
        results.append(
            {
                "prevailing_mean_outdoor_temp": t_running_mean,
                "tmp_cmf_80_low": round(adaptive.tmp_cmf_80_low, 2),
                "tmp_cmf_80_up": round(adaptive.tmp_cmf_80_up, 2),
                "tmp_cmf_90_low": round(adaptive.tmp_cmf_90_low, 2),
                "tmp_cmf_90_up": round(adaptive.tmp_cmf_90_up, 2),
            }
        )

    # Convert the result to a DataFrame
    df = pd.DataFrame(results)
    # Create a Plotly graphics object
    fig = go.Figure()

    if units == UnitSystem.IP.value:
        air_temperature = UnitConverter.celsius_to_fahrenheit(air_temperature)
        mean_radiant_temp = UnitConverter.celsius_to_fahrenheit(mean_radiant_temp)
        prevailing_mean_outdoor_temp = UnitConverter.celsius_to_fahrenheit(
            prevailing_mean_outdoor_temp
        )
        operative_temperature = UnitConverter.celsius_to_fahrenheit(
            operative_temperature
        )

    # 80% acceptance zone
    fig.add_trace(
        go.Scatter(
            x=df["prevailing_mean_outdoor_temp"],
            y=df["tmp_cmf_80_up"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["prevailing_mean_outdoor_temp"],
            y=df["tmp_cmf_80_low"],
            fill="tonexty",  # Fill into the next trace
            fillcolor="lightblue",
            mode="lines",
            line=dict(width=0),
            name="80% Acceptability",
        )
    )

    # 90% acceptance zone
    fig.add_trace(
        go.Scatter(
            x=df["prevailing_mean_outdoor_temp"],
            y=df["tmp_cmf_90_up"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["prevailing_mean_outdoor_temp"],
            y=df["tmp_cmf_90_low"],
            fill="tonexty",  # Fill into the next trace
            fillcolor="blue",
            mode="lines",
            line=dict(width=0),
            name="90% Acceptability",
        )
    )

    # Red dot of the current condition
    fig.add_trace(
        go.Scatter(
            x=[prevailing_mean_outdoor_temp],
            y=[operative_temperature],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Current Condition",
        )
    )
    if units == UnitSystem.IP.value:
        xaxis_range = [
            UnitConverter.celsius_to_fahrenheit(10),
            UnitConverter.celsius_to_fahrenheit(33.5),
        ]
        xaxis_tick0 = UnitConverter.celsius_to_fahrenheit(10)
        # xaxis_tick0 = 50
        xaxis_dtick = UnitConverter.celsius_to_fahrenheit(
            2
        ) - UnitConverter.celsius_to_fahrenheit(
            0
        )  # calculate dtick
        # xaxis_dtick = 5
        xaxis_title = "Prevailing Mean Outdoor Temperature (°F)"
    else:
        xaxis_range = [10, 33.5]
        xaxis_tick0 = 10
        xaxis_dtick = 2
        xaxis_title = "Prevailing Mean Outdoor Temperature (°C)"
    # Set chart style
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=(
            "Operative Temperature (°C)"
            if units == UnitSystem.SI.value
            else "Operative Temperature (°F)"
        ),
        xaxis=dict(
            range=xaxis_range, tick0=xaxis_tick0, dtick=xaxis_dtick, showgrid=True
        ),  # Set the X-axis range and scale dynamically
        yaxis=dict(
            range=[df["tmp_cmf_80_low"].min(), df["tmp_cmf_80_up"].max()],
            showgrid=True,
        ),
        showlegend=True,
        template="simple_white",
    )

    return dmc.Paper(children=[dcc.Graph(figure=fig)])


def t_hr_pmv(inputs: dict = None, model: str = "ashrae"):
    results = []
    pmv_limits = [-0.5, 0.5]
    clo_d = clo_dynamic(
        clo=inputs[ElementsIDs.clo_input.value], met=inputs[ElementsIDs.met_input.value]
    )
    vr = v_relative(
        v=inputs[ElementsIDs.v_input.value], met=inputs[ElementsIDs.met_input.value]
    )

    current_tdb = inputs[ElementsIDs.t_db_input.value]
    current_rh = inputs[ElementsIDs.rh_input.value]
    psy_data = psy_ta_rh(current_tdb, current_rh)

    for pmv_limit in pmv_limits:
        for rh in np.arange(10, 110, 10):
            psy_data_rh = psy_ta_rh(current_tdb, rh)

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
                    - pmv_limit
                )

            temp = optimize.brentq(function, 10, 40)
            results.append(
                {
                    "rh": rh,
                    "hr": psy_data_rh["hr"] * 1000,
                    "temp": temp,
                    "pmv_limit": pmv_limit,
                }
            )

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(8, 6))

    for rh in np.arange(10, 110, 10):
        temp_range = np.arange(10, 40, 1)
        hr_values = [psy_ta_rh(t, rh)["hr"] * 1000 for t in temp_range]
        ax.plot(temp_range, hr_values, color="grey", linestyle="--")

    t1 = df[df["pmv_limit"] == pmv_limits[0]]
    t2 = df[df["pmv_limit"] == pmv_limits[1]]
    ax.fill_betweenx(t1["hr"], t1["temp"], t2["temp"], alpha=0.5, color="#7BD0F2")

    ax.scatter(
        current_tdb, psy_data["hr"] * 1000, color="red", edgecolor="black", s=100
    )

    ax.set_xlabel("Dry-bulb Temperature (°C)", fontsize=14)
    ax.set_ylabel("Humidity Ratio (g_water/kg_dry_air)", fontsize=14)
    ax.set_xlim(10, 40)
    ax.set_ylim(0, 30)

    label_text = (
        f"t_db: {current_tdb:.1f} °C\n"
        f"rh: {current_rh:.1f} %\n"
        f"Wa: {psy_data['hr'] * 1000:.1f} g_w/kg_da\n"
        f"twb: {psy_data['t_wb']:.1f} °C\n"
        f"tdp: {psy_data['t_dp']:.1f} °C\n"
        f"h: {psy_data['h'] / 1000:.1f} kJ/kg"
    )

    ax.text(
        0.05,
        0.95,
        label_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.6),
    )
    plt.tight_layout()
    my_stringIObytes = io.BytesIO()
    plt.savefig(
        my_stringIObytes,
        format="png",
        transparent=True,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    plt.close("all")

    return dmc.Image(
        src=f"data:image/png;base64, {my_base64_jpgData}",
        alt="Psychrometric chart",
        py=0,
    )


def speed_temp_pmv(inputs: dict = None, model: str = "ashrae"):
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
    df = pd.DataFrame(results)
    print(df)
    static_image_data = """
    iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/wcAAwAB
    /aEEBkAAAAAASUVORK5CYII=
    """

    return dmc.Image(
        src=f"data:image/png;base64,{static_image_data}",
        alt="Static Image",
        py=0,
    )


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
