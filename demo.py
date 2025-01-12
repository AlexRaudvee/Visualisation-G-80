import dash
from dash import dcc, html, Input, Output, State
import dash_leaflet as dl
import dash_leaflet.express as dlx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from shapely.geometry import Point, Polygon

# Load dataset
df = pd.read_csv("./data/shark_data.csv")

# Convert 'Incident.year' to numeric, handle errors by coercing invalid values to NaN
df['Incident.year'] = pd.to_numeric(df['Incident.year'], errors='coerce')

# Drop rows with NaN values in 'Incident.year', 'latitude', or 'longitude'
df = df.dropna(subset=['Incident.year', 'latitude', 'longitude'])

# Calculate center of the map for initial display
center_lat = df["latitude"].mean()
center_lon = df["longitude"].mean()

# Determine the range of years in the dataset
min_year = int(df['Incident.year'].min())
max_year = int(df['Incident.year'].max())

app = dash.Dash(__name__)

app.layout = html.Div([
    # Year range slider
    dcc.RangeSlider(
        id='year-slider',
        min=min_year,
        max=max_year,
        value=[min_year, max_year],
        marks={str(year): str(year) for year in range(min_year, max_year + 1, 5)},
        step=1
    ),
    # Map component
    dl.Map(center=[center_lat, center_lon], zoom=4, id='map', style={'width': '100%', 'height': '500px'}),
    # Histograms
    html.Div([
        dcc.Dropdown(id='attr1-dropdown', options=[{'label': col, 'value': col} for col in df.columns], value=df.columns[0]),
        dcc.Graph(id='hist1')
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Dropdown(id='attr2-dropdown', options=[{'label': col, 'value': col} for col in df.columns], value=df.columns[1]),
        dcc.Graph(id='hist2')
    ], style={'width': '48%', 'display': 'inline-block'}),
    # Parallel Categories Plot
    dcc.Graph(id='pcp')
])

@app.callback(
    Output('map', 'children'),
    Input('year-slider', 'value')
)
def update_map(year_range):
    # Filter data based on year range
    filtered_df = df[(df['Incident.year'] >= year_range[0]) & (df['Incident.year'] <= year_range[1])]
    
    # Create markers for the map
    markers = [dl.Marker(position=[row['latitude'], row['longitude']], id={'type': 'marker', 'index': idx})
               for idx, row in filtered_df.iterrows()]
    
    return markers

@app.callback(
    Output('hist1', 'figure'),
    Input('attr1-dropdown', 'value'),
    Input('map', 'clickData')
)
def update_hist1(attr1, click_data):
    # Update histogram based on selected attribute and map click
    # Implement filtering logic based on click_data if necessary
    fig = px.histogram(df, x=attr1, title=f'Distribution of {attr1}')
    return fig

@app.callback(
    Output('hist2', 'figure'),
    Input('attr2-dropdown', 'value'),
    Input('hist1', 'selectedData')
)
def update_hist2(attr2, selected_data):
    # Update histogram based on selected attribute and histogram selection
    # Implement filtering logic based on selected_data if necessary
    fig = px.histogram(df, x=attr2, title=f'Distribution of {attr2}')
    return fig

@app.callback(
    Output('pcp', 'figure'),
    Input('map', 'clickData'),
    Input('hist1', 'selectedData'),
    Input('hist2', 'selectedData')
)
def update_pcp(map_click, hist1_select, hist2_select):
    # Update Parallel Categories Plot based on selections
    # Implement filtering logic based on map_click, hist1_select, and hist2_select
    categorical_dimensions = []
    dimensions = [dict(values=df[label], label=label) for label in categorical_dimensions]
    color = np.zeros(len(df), dtype='uint8')
    colorscale = [[0, 'gray'], [1, 'firebrick']]
    fig = go.Figure(data=[go.Parcats(dimensions=dimensions, line={'colorscale': colorscale, 'cmin': 0, 'cmax': 1, 'color': color})])
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
