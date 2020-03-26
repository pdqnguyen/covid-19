import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html


MIN_DEATHS = 50


raw = pd.read_csv('countries-aggregated.csv')
countries = sorted(set(raw['Country']))
processed = {}
for c in countries:
    ts = raw[raw['Country'] == c]
    if ts['Deaths'].max() > MIN_DEATHS:
        ts_crop = ts[ts['Deaths'] > 0].reset_index(drop=True)
        ts_crop['Day'] = range(len(ts_crop))
        ts_crop = ts_crop[['Day', 'Country', 'Confirmed', 'Recovered', 'Deaths']]
        processed[c] = ts_crop
data_confirmed = [('Confirmed', c, ts['Day'], ts['Confirmed']) for c, ts in processed.items()]
data_recovered = [('Recovered', c, ts['Day'], ts['Recovered']) for c, ts in processed.items()]
data_deaths = [('Deaths', c, ts['Day'], ts['Deaths']) for c, ts in processed.items()]
data = data_confirmed + data_recovered + data_deaths
show_confirmed = [(row[0] == 'Confirmed') for row in data]
show_recovered = [(row[0] == 'Recovered') for row in data]
show_deaths = [(row[0] == 'Deaths') for row in data]
fig = go.Figure()
for column, c, x, y in data:
    visible = (column == 'Deaths')
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=c, visible=visible))

# Appearance
fig.update_layout(
    width=1200,
    height=600,
    autosize=False,
    margin=dict(t=50, b=200, l=0, r=0),
    template='plotly_white',
    yaxis_type='log'
)

# Text
title_confirmed = f"Cumulative COVID-19 cases in countries with over {MIN_DEATHS} deaths"
title_recovered = f"Cumulative COVID-19 recoveries in countries with over {MIN_DEATHS} deaths"
title_deaths = f"Cumulative COVID-19 deaths in countries with over {MIN_DEATHS} deaths"
label_confirmed = "Cases"
label_recovered = "Recoveries"
label_deaths = "Deaths"
fig.update_layout(
    font=dict(size=16),
    title=dict(text=title_deaths, x=0.5, xanchor='center'),
    yaxis=dict(title=label_deaths),
    annotations=[
        dict(x=0.5, y=-0.15, showarrow=False, font=dict(size=18),
             text="Days since first death",
             xref="paper", yref="paper"),
        dict(text="Y-scale:", showarrow=False, x=0.02, y=-0.25,
             xref='paper', yref='paper'),
        dict(text="Data:", showarrow=False, x=0.36, y=-0.25,
             xref='paper', yref='paper')
    ]
)

# Buttons
fig.update_layout(
    updatemenus=[
        dict(
            type='buttons',
            direction='left',
            buttons=list([
                dict(
                    label='Logarithmic',
                    method='relayout',
                    args=[
                        {'yaxis': {'type': 'log'}}
                    ]
                ),
                dict(
                    label='Linear',
                    method='relayout',
                    args=[
                        {'yaxis': {'type': 'linear'}}
                    ]
                )
            ]),
            pad=dict(r=10, t=10),
            showactive=True,
            x=0.1,
            xanchor='left',
            y=-0.26,
            yanchor='bottom'
        ),
        dict(
            type='buttons',
            direction='left',
            buttons=list([
                dict(
                    label='Deaths',
                    method='update',
                    args=[
                        dict(visible=show_deaths),
                        dict(title_text=title_deaths,
                             yaxis_label=label_deaths)
                    ]
                ),
                dict(
                    label='Confirmed cases',
                    method='update',
                    args=[
                        dict(visible=show_confirmed),
                        dict(title_text=title_confirmed,
                             yaxis_label=label_confirmed)
                    ]
                ),
                dict(
                    label='Recoveries',
                    method='update',
                    args=[
                        dict(visible=show_recovered),
                        dict(title_text=title_recovered,
                             yaxis_label=label_recovered)
                    ]
                )
            ]),
            pad=dict(r=10, t=10),
            showactive=True,
            x=0.4,
            xanchor='left',
            y=-0.26,
            yanchor='bottom'
        ),
    ]
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
markdown_text = '''
### COVID-19 Growth Curves

Links:
[[Source code](https://github.com/pdqnguyen/covid-19)]
[[Full Data](https://github.com/datasets/covid-19)]

These growth curves are lined up by date of first death
(or 2020-01-22 if deaths began before then) for a better
comparison between the growth rate in different countries.
Countries may vary in their patient testing policies and
procedures, but deaths offer a more policy-independent,
albeit low-sample-size, [proxy for the true number of cases]
(https://medium.com/@tomaspueyo/coronavirus-act-today-or-people-will-die-f4d3d9cd99ca).
'''
app.layout = html.Div([
    dcc.Markdown(children=markdown_text),
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server()
