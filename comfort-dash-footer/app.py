import os

import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
from dash import Dash, dcc, html
from icecream import install, ic

from components.footer import my_footer
from components.navbar import my_navbar
from utils.my_config_file import (
    Config,
    Stores,
    ElementsIDs,
    Dimensions,
)
from utils.website_text import app_name

from dash.dependencies import Input, Output
from components.dropdowns import dd_model
from components.input_environmental_personal import input_environmental_personal
from components.dropdowns import chart_selection
from utils.my_config_file import (
    MODELS,
    AdaptiveEN,
    AdaptiveAshrae,
    PmvAshraeResultCard,
    PmvENResultCard,
)

install()
# from components.dropdowns import Ash55_air_speed_selection
ic.configureOutput(includeContext=True)

# try:
#     cred = credentials.Certificate("secret.json")
# except FileNotFoundError:
#     cred = credentials.Certificate(json.loads(os.environ.get("firebase_secret")))
#
# firebase_admin.initialize_app(
#     cred,
#     {"databaseURL": FirebaseFields.database_url},
# )

# This is required by dash mantine components to work with react 18
dash._dash_renderer._set_react_version("18.2.0")


# Exposing the Flask Server to enable configuring it for logging in
app = Dash(
    __name__,
    title=app_name,
    update_title="Loading...",
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
        # include google fonts
        "https://fonts.googleapis.com/css2?family=Inter:wght@100;200;300;400;500;900&display=swap",
        # dash mantine stylesheets
        "https://unpkg.com/@mantine/dates@7/styles.css",
        "https://unpkg.com/@mantine/code-highlight@7/styles.css",
        "https://unpkg.com/@mantine/charts@7/styles.css",
        "https://unpkg.com/@mantine/carousel@7/styles.css",
        "https://unpkg.com/@mantine/notifications@7/styles.css",
        "https://unpkg.com/@mantine/nprogress@7/styles.css",
    ],
    external_scripts=["https://cdn.plot.ly/plotly-basic-2.26.2.min.js"],
    prevent_initial_callbacks=True,
    use_pages=True,
    serve_locally=False,
)
app.config.suppress_callback_exceptions = True

# app.index_string = """<!DOCTYPE html>
# <html lang="en-AU">
# <head>
#     <!-- Google tag (gtag.js) -->
#     <script async src="https://www.googletagmanager.com/gtag/js?id=G-MZFW54YKZ5"></script>
#     <script>
#       window.dataLayer = window.dataLayer || [];
#       function gtag(){dataLayer.push(arguments);}
#       gtag('js', new Date());
#
#       gtag('config', 'G-MZFW54YKZ5');
#     </script>
#     <meta charset="utf-8">
#     <link rel="apple-touch-icon" href="/assets/media/CBE-logo-2018.png"/>
#     <link rel="apple-touch-icon" sizes="180x180" href="/assets/media/CBE-logo-2018.png">
#     <link rel="icon" type="image/png" sizes="32x32" href="/assets/media/CBE-logo-2018.png">
#     <link rel="icon" type="image/png" sizes="16x16" href="/assets/media/CBE-logo-2018.png">
#     <link rel="manifest" href="./assets/manifest.json">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <meta name="author" content="Federico Tartarini">
#     <meta name="keywords" content="">
#     <meta name="description" content="">
#     <meta name="theme-color" content="#E64626" />
#     <title>HeatWatch</title>
#     <meta property="og:image" content="./assets/media/CBE-logo-2018.png">
#     <meta property="og:description" content="">
#     <meta property="og:title" content="">
#     {%favicon%}
#     {%css%}
# </head>
# <body>
# <script>
#   if ('serviceWorker' in navigator) {
#     window.addEventListener('load', ()=> {
#       navigator
#       .serviceWorker
#       .register("./assets/sw01.js")
#       .then(()=>console.log("Ready."))
#       .catch((e)=>console.log("Err...", e));
#     });
#   }
# </script>
# {%app_entry%}
# <footer>
# {%config%}
# {%scripts%}
# {%renderer%}
# </footer>
# </body>
# </html>
# """

app.layout = dmc.MantineProvider(
    defaultColorScheme="light",
    theme={
        "colorScheme": "dark",
        "fontFamily": "'Inter', sans-serif",
        "primaryColor": "indigo",
        "components": {
            "Button": {"styles": {"root": {"fontWeight": 400}}},
            "Alert": {"styles": {"title": {"fontWeight": 500}}},
            "AvatarGroup": {"styles": {"truncated": {"fontWeight": 500}}},
        },
    },
    children=html.Div(
        [
            my_navbar(),
            dcc.Location(id=ElementsIDs.URL.value),
            dcc.Store(id=Stores.INPUT_DATA.value, storage_type="local"),
            html.Div(
                dmc.Container(
                    dash.page_container,
                    p="xs",
                    size=Dimensions.default_container_width.value,
                ),
                style={
                    "paddingBottom": "6.5rem",
                },
            ),
            my_footer(),
        ],
        style={
            "minHeight": "100vh",
            "position": "relative",
        },
    ),
)


# @app.server.route("/sitemap/")
# def sitemap():
#     return Response(
#         """<?xml version="1.0" encoding="UTF-8"?>
#         <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
#           <url>
#             <loc>https://heatwatch.sydney.sydney.edu.au</loc>
#             <lastmod>2023-10-05T04:22:26+00:00</lastmod>
#           </url>
#           <url>
#             <loc>https://heatwatch.sydney.edu.au/settings</loc>
#             <lastmod>2023-10-05T04:22:26+00:00</lastmod>
#           </url>
#         </urlset>
#         """,
#         mimetype="text/xml",
#     )


@app.callback(
    Output("input_card", "children"),
    Output("graph-container", "children"),
    Output("chart-select", "children"),
    Output("graph-container", "cols"),
    Input(dd_model["id"], "value"),
)
def capture_selected_model(selected_model):
    print(selected_model)
    input_content = input_environmental_personal(selected_model)
    graph_content = update_graph_content(selected_model)
    chart_content = chart_selection(selected_model)
    result_content = change_cols(selected_model)

    return input_content, graph_content, chart_content, result_content


def change_cols(selected_model):
    if (
        selected_model == MODELS.Adaptive_EN.value
        or selected_model == MODELS.Adaptive_ashrae.value
    ):
        cols = 1
    elif selected_model == MODELS.PMV_ashrae.value or selected_model == MODELS.PMV_EN.value:
        cols = 3
    return cols


def update_graph_content(selected_model):

    if selected_model == MODELS.Adaptive_EN.value:
        grid_content = [
            dmc.Center(dmc.Text(AdaptiveEN.class_III.value)),
            dmc.Center(dmc.Text(AdaptiveEN.class_II.value)),
            dmc.Center(dmc.Text(AdaptiveEN.class_I.value)),
            dmc.Center(dmc.Text(AdaptiveEN.adaptive_chart.value)),
        ]
    elif selected_model == MODELS.Adaptive_ashrae.value:
        grid_content = [
            dmc.Center(dmc.Text(AdaptiveAshrae.acceptability_limits_80.value)),
            dmc.Center(dmc.Text(AdaptiveAshrae.acceptability_limits_90.value)),
            dmc.Center(dmc.Text(AdaptiveAshrae.adaptive_chart.value)),
        ]
    elif selected_model == MODELS.PMV_ashrae.value:
        grid_content = [
            dmc.Center(dmc.Text(PmvAshraeResultCard.pmv.value)),
            dmc.Center(dmc.Text(PmvAshraeResultCard.ppd.value)),
            dmc.Center(dmc.Text(PmvAshraeResultCard.sensation.value)),
            dmc.Center(dmc.Text(PmvAshraeResultCard.set.value)),
        ]
    elif selected_model == MODELS.PMV_EN.value:
        grid_content = [
            dmc.Center(dmc.Text(PmvENResultCard.pmv.value)),
            dmc.Center(dmc.Text(PmvENResultCard.ppd.value)),
            dmc.Center(dmc.Text(PmvENResultCard.set.value)),
        ]
    return grid_content

# def generate_buttons(selected_model):
    return generate_dropdown_inline()




if __name__ == "__main__":
    app.run_server(
        debug=Config.DEBUG.value,
        host="0.0.0.0",
        port=os.environ.get("PORT_APP", 9090),
        processes=1,
        threaded=True,
    )
