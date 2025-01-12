# Import necessary libraries
import os
import folium                           # Folium for map-related functionalities (redundant import)
from streamlit_folium import st_folium  # Streamlit Folium for integrating folium maps in Streamlit app

import pandas as pd                     # Pandas for data manipulation and analysis
import seaborn as sns                   # Seaborn for statistical data visualization (not used in this code)
import streamlit as st                  # Streamlit for building the web app
import plotly.express as px             # Plotly Express for creating interactive charts
import matplotlib.pyplot as plt         # Matplotlib for static plots (not used in this code)
import plotly.graph_objects as go       # For advanced plotly visualizations

from folium.features import Icon
from folium.plugins import HeatMap, Draw
from streamlit_plotly_events import plotly_events  # For handling plotly events in Streamlit

st.set_page_config(layout='wide', 
                   initial_sidebar_state="collapsed")


# GLOBAL VARS
st.session_state["DF"] = pd.read_csv("./data/shark_data.csv")
center_lat = st.session_state["DF"]["latitude"].mean()
center_lon = st.session_state["DF"]["longitude"].mean()
print(st.session_state["DF"].columns)
min_year = int(st.session_state["DF"]['Incident.year'].min())
max_year = int(st.session_state["DF"]['Incident.year'].max())


# SIDEBAR AT THE LEFT
with st.sidebar:
    st.header("Controls")

    # Time range slider
    year_range = st.slider(
        "Select Year Range",
        min_value=int(st.session_state["DF"]['Incident.year'].min()),
        max_value=int(st.session_state["DF"]['Incident.year'].max()),
        value=(int(st.session_state["DF"]['Incident.year'].min()), int(st.session_state["DF"]['Incident.year'].max()))
    )

# Filter data based on time range
st.session_state['DF'] = st.session_state["DF"][(st.session_state["DF"]['Incident.year'] >= year_range[0]) &
                 (st.session_state["DF"]['Incident.year'] <= year_range[1])]


    
# PREPARATION OF DATA
st.session_state["DF"]['Incident.year'] = pd.to_numeric(st.session_state["DF"]['Incident.year'], errors='coerce')

# Filter the DataFrame based on the selected year range
st.session_state["DF"] = st.session_state["DF"][
    (st.session_state["DF"]['Incident.year'] >= year_range[0]) &
    (st.session_state["DF"]['Incident.year'] <= year_range[1])
]

# Prepare data for heatmap
heat_data = st.session_state["DF"][['latitude', 'longitude']].values



# BUILD THE MAP WITH LAYERS 
st.session_state['m'] = folium.Map(location=[center_lat, center_lon], zoom_start=8)

# Add markers for each point in the filtered DataFrame
kw = {"opacity": 0.1, "color": 'grey'}
st.session_state['markers'] = []
for _, row in st.session_state["DF"].iterrows():
    st.session_state["markers"].append(folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=1.5,  # Radius of the circle
        popup=f"Latitude: {row['latitude']}, Longitude: {row['longitude']}",
        **kw
        ))

fg = folium.FeatureGroup(name="Markers")
for marker in st.session_state["markers"]:
    fg.add_child(marker)

# Add heatmap layer
HeatMap(heat_data, radius=10).add_to(st.session_state["m"])

# Add drawing tools to the map
draw = Draw(export=True)
draw.add_to(st.session_state["m"])



# DISPLAYING AND INTERACTION

# THE MAP
st.subheader("Interactive Map")
map_select_data = st_folium(st.session_state['m'],
                    center=None,
                    zoom=4,
                    key="new",
                    returned_objects=["last_object_clicked_tooltip", "last_object_clicked", "last_object_clicked_popup", "last_active_drawing"],
                    feature_group_to_add=fg,
                    height=700,
                    width=1400)

points_in_region_map = None
selected_point_map = None
# Check if a region has been selected
if map_select_data and 'last_active_drawing' in map_select_data and map_select_data['last_active_drawing']:

    if map_select_data["last_active_drawing"]["geometry"]['type'] == 'Point':
        tolerance = 0.0001
        target_lat = map_select_data['last_active_drawing']['geometry']['coordinates'][1]
        target_lon = map_select_data['last_active_drawing']['geometry']['coordinates'][0]
        
    # Find rows matching the target coordinates within the tolerance
        selected_point_map = st.session_state["DF"][
            (st.session_state["DF"]["latitude"].between(target_lat - tolerance, target_lat + tolerance)) &
            (st.session_state["DF"]["longitude"].between(target_lon - tolerance, target_lon + tolerance))
        ]
        # Display the filtered points
        # st.write(f"Point clicked")
        # st.write(selected_point_map)
        
    else:
        selected_region_map = map_select_data['last_active_drawing']['geometry']['coordinates'][0]

        # Function to check if a point is inside the selected polygon
        from shapely.geometry import Point, Polygon
        polygon = Polygon(selected_region_map)
        st.session_state["DF"]['is_in_region'] = st.session_state["DF"].apply(lambda row: polygon.contains(Point(row['longitude'], row['latitude'])), axis=1)

        # Filter points within the selected region
        points_in_region_map = st.session_state["DF"][st.session_state["DF"]['is_in_region']]

        # Display the filtered points
        # st.write(f"Number of points in the selected region: {len(points_in_region_map)}")
        # st.write(points_in_region_map)


# THE BARS
# List of attributes to choose from
attributes = [col for col in st.session_state["DF"].columns if col != 'Incident.year']

# Create two columns in Streamlit
col1, col2 = st.columns(2)

# Store selected indices for interaction
if "selected_indices_1" not in st.session_state:
    st.session_state["selected_indices_1"] = []
if "selected_indices_2" not in st.session_state:
    st.session_state["selected_indices_2"] = []

# First column: Attribute selection and histogram
with col1:
    attr1 = st.selectbox("Select Attribute for Column 1", attributes, key="attr1", index=17)
    filtered_data_1 = (
        st.session_state["DF"].iloc[st.session_state["selected_indices_2"]]
        if st.session_state["selected_indices_2"]
        else st.session_state["DF"]
    )
    fig1 = px.histogram(filtered_data_1, x=attr1, title=f'Distribution of {attr1}')
    ret1 = st.plotly_chart(fig1, key="hist1", use_container_width=True, on_select='rerun')
    indices_bar_1 = ret1['selection']['point_indices']
    st.session_state["selected_indices_1"] = indices_bar_1

# Second column: Attribute selection and histogram
with col2:
    attr2 = st.selectbox("Select Attribute for Column 2", attributes, key="attr2", index=17)
    filtered_data_2 = (
        st.session_state["DF"].iloc[st.session_state["selected_indices_1"]]
        if st.session_state["selected_indices_1"]
        else st.session_state["DF"]
    )
    fig2 = px.histogram(filtered_data_2, x=attr2, title=f'Distribution of {attr2}')
    ret2 = st.plotly_chart(fig2, key="hist2", use_container_width=True, on_select='rerun')
    indices_bar_2 = ret2['selection']['point_indices']
    st.session_state["selected_indices_2"] = indices_bar_2
                
##TODO: PCP Is shit, update it.
# PCP Plot below 

