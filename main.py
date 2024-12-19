# Import necessary libraries
import os
import folium                           # Folium for map-related functionalities (redundant import)
from streamlit_folium import st_folium  # Streamlit Folium for integrating folium maps in Streamlit app

import pandas as pd                     # Pandas for data manipulation and analysis
import seaborn as sns                   # Seaborn for statistical data visualization (not used in this code)
import streamlit as st                  # Streamlit for building the web app
import plotly.express as px             # Plotly Express for creating interactive charts
import matplotlib.pyplot as plt         # Matplotlib for static plots (not used in this code)

from folium.features import Icon
from folium.plugins import HeatMap, Draw
from streamlit_plotly_events import plotly_events  # For handling plotly events in Streamlit

st.set_page_config(layout='wide')

# PREPARATION OF DATA

st.session_state["DF"] = pd.read_csv("./data/shark_data.csv")

# Initialize a Folium map centered at the mean latitude and longitude of the dataset
center_lat = st.session_state["DF"]["latitude"].mean()
center_lon = st.session_state["DF"]["longitude"].mean()
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

# Prepare data for heatmap
heat_data = st.session_state["DF"][['latitude', 'longitude']].values

# Add heatmap layer
HeatMap(heat_data, radius=10).add_to(st.session_state["m"])

# Add drawing tools to the map
draw = Draw(export=True)
draw.add_to(st.session_state["m"])

# Display the map in Streamlit
st.subheader("Interactive Map")
st_data = st_folium(st.session_state['m'],
                    center=None,
                    zoom=4,
                    key="new",
                    returned_objects=["last_object_clicked_tooltip", "last_object_clicked", "last_object_clicked_popup", "last_active_drawing"],
                    feature_group_to_add=fg,
                    height=700,
                    width=1300)

# st.write(st_data)
# Check if a region has been selected
if st_data and 'last_active_drawing' in st_data and st_data['last_active_drawing']:

    if st_data["last_active_drawing"]["geometry"]['type'] == 'Point':
        tolerance = 0.0001
        target_lat = st_data['last_active_drawing']['geometry']['coordinates'][1]
        target_lon = st_data['last_active_drawing']['geometry']['coordinates'][0]
        
       # Find rows matching the target coordinates within the tolerance
        matching_rows = st.session_state["DF"][
            (st.session_state["DF"]["latitude"].between(target_lat - tolerance, target_lat + tolerance)) &
            (st.session_state["DF"]["longitude"].between(target_lon - tolerance, target_lon + tolerance))
        ]
        # Display the filtered points
        st.write(f"Point clicked")
        st.write(matching_rows)
        
    else:
        selected_region = st_data['last_active_drawing']['geometry']['coordinates'][0]

        # Function to check if a point is inside the selected polygon
        from shapely.geometry import Point, Polygon
        polygon = Polygon(selected_region)
        st.session_state["DF"]['is_in_region'] = st.session_state["DF"].apply(lambda row: polygon.contains(Point(row['longitude'], row['latitude'])), axis=1)

        # Filter points within the selected region
        points_in_region = st.session_state["DF"][st.session_state["DF"]['is_in_region']]

        # Display the filtered points
        st.write(f"Number of points in the selected region: {len(points_in_region)}")
        st.write(points_in_region)
