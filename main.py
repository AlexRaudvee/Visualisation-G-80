# Import necessary libraries
import folium                           # Folium for map-related functionalities (redundant import)
from streamlit_folium import st_folium  # Streamlit Folium for integrating folium maps in Streamlit app

import pandas as pd                     # Pandas for data manipulation and analysis
import seaborn as sns                   # Seaborn for statistical data visualization (not used in this code)
import streamlit as st                  # Streamlit for building the web app
import plotly.express as px             # Plotly Express for creating interactive charts
import matplotlib.pyplot as plt         # Matplotlib for static plots (not used in this code)

from dataloader import df
from streamlit_plotly_events import plotly_events  # For handling plotly events in Streamlit

st.set_page_config(layout='wide')

# PREPARATION OF DATA

st.session_state["DF"] = df

# Initialize a Folium map centered at the mean latitude and longitude of the dataset
center_lat = st.session_state["DF"]["latitude"].mean()
center_lon = st.session_state["DF"]["longitude"].mean()
map_object = folium.Map(location=[center_lat, center_lon], zoom_start=8)
st.session_state['m'] = map_object

# Add markers for each point in the filtered DataFrame
st.session_state['markers'] = []
for _, row in df.iterrows():
    st.session_state["markers"].append(folium.Marker(
        location=[row["latitude"], row["longitude"]],
        popup=f"Latitude: {row['latitude']}, Longitude: {row['longitude']}",
        tooltip="Click for more info"
        ))


fg = folium.FeatureGroup(name="Markers")
for marker in st.session_state["markers"]:
    fg.add_child(marker)


# Display the map in Streamlit
st.subheader("Interactive Map")
st_data = st_folium(st.session_state['m'],
                    center=None,
                    zoom=8,
                    key="new",
                    returned_objects=['last_object_clicked'],
                    feature_group_to_add=fg,
                    height=400,
                    width=700)

st.write(st_data)