import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


MIN_DEATHS = 10

TOP_TEXT = """
### COVID-19 Growth Curves

These growth curves are lined up by date of first death
(or 2020-01-22 if deaths began before then) for a better
comparison between the growth rate in different countries.
Countries may vary in their patient testing policies and
procedures, but deaths offer a more policy-independent,
albeit low-sample-size, [proxy for the true number of cases]
(https://medium.com/@tomaspueyo/coronavirus-act-today-or-people-will-die-f4d3d9cd99ca).
See below for a description of user options.
"""

BOTTOM_TEXT1 = """
Usage:
- Choose a metric and normalization.
- Click on a curve to highlight it. Shift click to highlight multiple curves.
- Click on a country name in the legend to remove it.
- Double click legend to add countries to a blank plot.
"""

BOTTOM_TEXT2 = """
Normalize by:
- **population** to show the proportion of each country's population affected;
- **population density** to show the effective geographical area impacted by the spread;
- **hospital beds** to show the load of the infection on each country's healthcare capacity.
"""

BOTTOM_TEXT3 = """
Links:
- [Source code](https://github.com/pdqnguyen/covid-19)
- [COVID-19 data](https://github.com/datasets/covid-19)
- [Population data](https://worldpopulationreview.com/countries/countries-by-density/)
- [Hospital data](https://data.worldbank.org/indicator/sh.med.beds.zs)
"""


covid19 = pd.read_csv('countries-aggregated.csv')
covid19.columns = ['date', 'country', 'cases', 'recovered', 'deaths']

# Population data
popdens = pd.read_csv('popdens.csv')
popdens.index = popdens['name']
dens = popdens['density'].to_dict()
pop = (popdens['pop2020'] * 1000).to_dict()

# Diamond Princess data
pop['Diamond Princess'] = 3711 # total ship attendance
dens['Diamond Princess'] = 3711 / 11 # rectangular area from ship dimensions

# Hospital beds data
beds = pd.read_csv('hospitalbeds.csv')
beds.index = beds['Country Name']
beds = beds[beds.columns[4:]]
def most_recent(x):
    notna = x[~x.isnull()]
    if len(notna) > 0:
        return notna[-1]
    else:
        return None
beds = beds.apply(most_recent, axis=1).to_dict()

# Process data, create a dataframe for each country
countries = sorted(set(covid19['country']))
processed = {}
for c in countries:
    ts = covid19[covid19['country'] == c]
    pop_c = pop.get(c, np.inf)
    dens_c = dens.get(c, np.inf)
    beds_c = beds.get(c, np.inf) * pop_c / 1000
    if ts['deaths'].max() >= MIN_DEATHS:
        ts_crop = ts[ts['deaths'] > 0].reset_index(drop=True)
        ts_crop['day'] = range(len(ts_crop))
        ts_crop = ts_crop[['day', 'country', 'cases', 'recovered', 'deaths']]
        by_pop = ts_crop[['cases', 'recovered', 'deaths']] / pop_c * 100
        by_dens = ts_crop[['cases', 'recovered', 'deaths']] / dens_c
        by_beds = ts_crop[['cases', 'recovered', 'deaths']] / beds_c
        ts_crop[['cases_pop', 'recovered_pop', 'deaths_pop']] = by_pop
        ts_crop[['cases_dens', 'recovered_dens', 'deaths_dens']] = by_dens
        ts_crop[['cases_beds', 'recovered_beds', 'deaths_beds']] = by_beds
        processed[c] = ts_crop
countries = list(processed.keys())
columns = [
    'cases', 'recovered', 'deaths',
    'cases_pop', 'recovered_pop', 'deaths_pop',
    'cases_dens', 'recovered_dens', 'deaths_dens',
    'cases_beds', 'recovered_beds', 'deaths_beds'
]
data = [
    (column, c, ts['day'], ts[column])
    for column in columns
    for c, ts in processed.items()
]

# Plot initial traces
traces = []
for column, c, x, y in data:
    visible = (column == 'deaths')
    traces.append(go.Scatter(x=x, y=y, mode='lines+markers', name=c, visible=visible))
layout = go.Layout(
    width=1200,
    height=600,
    autosize=False,
    margin=dict(t=80, b=80, l=80, r=0),
    font=dict(size=16),
    xaxis=dict(title="Days since first death"),
    clickmode='event+select'
)
fig = go.Figure(data=traces, layout=layout)

# Dropdown menu options
metric_options = [
    dict(label="Confirmed cases", value='cases'),
    dict(label="Recovered cases", value='recovered'),
    dict(label="Deaths", value='deaths')
]
normalize_options = [
    dict(label="None", value=''),
    dict(label="Population", value='pop'),
    dict(label="Population density", value='dens'),
    dict(label="Hospital beds", value='beds')
]

# Create app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div([

    dcc.Markdown(children=TOP_TEXT, style={'width': '1200px'}),

    # Menu for choosing numberator
    html.Div([
            html.Label("Choose a metric:"),
            dcc.Dropdown(id='opt-metric',
                         options=metric_options,
                         value=metric_options[0]['value'],
                         searchable=False)
        ],
        style={'width': '300px',
               'fontSize': '20px',
               'marginRight': '2em',
               'display': 'inline-block'}
    ),

    # Menu for choosing denominator
    html.Div([
            html.Label("Normalize by:"),
            dcc.Dropdown(id='opt-normalize',
                         options=normalize_options,
                         value=normalize_options[0]['value'],
                         searchable=False)
        ],
        style={'width': '300px',
               'fontSize': '20px',
               'marginRight': '2em',
               'display': 'inline-block'}
    ),

    # Checkbox for log scale
    html.Div([
            html.Label("Y-scale:",
                       style={'marginRight': '1em',
                              'display': 'inline-block'}),
            dcc.Checklist(id="opt-scale",
                          options=[{'label': 'Logarithmic', 'value': 'log'}],
                          value=['log'],
                          style={'display': 'inline-block'})
        ],
        style={'width': '300px',
               'fontSize': '20px',
               'display': 'inline-block'}
    ),
    html.Button('Deselect traces', id='reset-button'),

    # Main figure
    dcc.Graph(id='plot', figure=fig),

    html.Div([
            html.Div(
                [dcc.Markdown(children=BOTTOM_TEXT1, style={'width': '500px'})],
                className="six columns"),
            html.Div(
                [dcc.Markdown(children=BOTTOM_TEXT2, style={'width': '500px'})],
                className="six columns"),
        ],
        className="row",
        style={'width': '1200px'}
    ),

    dcc.Markdown(children=BOTTOM_TEXT3, style={'width': '1200px'})
])


@app.callback(
    Output('plot', 'figure'),
    [
        Input('opt-metric', 'value'),
        Input('opt-normalize', 'value'),
        Input('opt-scale', 'value'),
        Input('plot', 'selectedData'),
    ]
)
def update_figure(input1, input2, input3, input4):
    """Callback function for updating figure from user actions
    """
    # Update figure data
    
    if input2 != "":
        selection = '_'.join((input1, input2))
    else:
        selection = input1
    if input2 == 'pop':
        yfmt = "%{y:,.4f}%"
    elif input2 == 'beds':
        yfmt = "%{y:,.4f}"
    else:
        yfmt = "%{y:,.0f}"
    if input4:
        names = [point['text'] for point in input4['points']]
        traces = [
            go.Scatter(x=x, y=y, mode='lines+markers', name=c, text=[c] * len(x),
                       hovertemplate="<b>%{text}</b><br><br>" +
                       "day: %{x:.0f}<br>" +
                       "value: " + yfmt + "<br><extra></extra>",
                       opacity=(1.0 if c in names else 0.1))
            for i, (column, c, x, y) in enumerate(data) if column == selection
        ]
    else:
        traces = [
            go.Scatter(x=x, y=y, mode='lines+markers', name=c, text=[c] * len(x),
                       hovertemplate="<b>%{text}</b><br><br>" +
                       "day: %{x:.0f}<br>" +
                       "value: " + yfmt + "<br><extra></extra>")
            for column, c, x, y in data if column == selection
        ]
    fig = go.Figure(
        data=traces,
        layout=layout
    )
    # Update axis and plot titles
    title = f"Cumulative COVID-19 {input1} in countries with over {MIN_DEATHS} deaths"
    yaxis_titles = {
        'cases': "Confirmed cases",
        'recovered': "Recovered cases",
        'deaths': "Fatal cases"
    }
    yaxis_title = yaxis_titles[input1]
    if input2 == 'pop':
        yaxis_title += " (% of population)"
    elif input2 == 'dens':
        yaxis_title += " effective area [sq km]"
    elif input2 == 'beds':
        yaxis_title += " per hospital bed"
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        yaxis=dict(title=yaxis_title)
    )
    # Toggle log scale
    if input3:
        fig.update_layout(yaxis_type='log')

    return fig


@app.callback(Output('plot', 'selectedData'), [Input('reset-button', 'n_clicks')])
def reset_figure(n_clicks):
    return None


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--debug", default=False, action="store_true",
        help="run in debug mode (for development only)"
    )
    args = parser.parse_args()
    app.run_server(debug=args.debug)
