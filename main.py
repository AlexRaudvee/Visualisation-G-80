# Import necessary libraries
import os
import folium                           # Folium for map-related functionalities (redundant import)
from streamlit_folium import st_folium  # Streamlit Folium for integrating folium maps in Streamlit app

import pandas as pd                     # Pandas for data manipulation and analysis
import numpy as np                      # Numpy for manipulations with dataframes
import seaborn as sns                   # Seaborn for statistical data visualization (not used in this code)
import streamlit as st                  # Streamlit for building the web app
import plotly.express as px             # Plotly Express for creating interactive charts
import matplotlib.pyplot as plt         # Matplotlib for static plots (not used in this code)
import plotly.graph_objects as go       # For advanced plotly visualizations

from branca.colormap import LinearColormap
from folium.features import Icon
from folium.plugins import HeatMap, Draw
from streamlit_plotly_events import plotly_events  # For handling plotly events in Streamlit
from streamlit_js_eval import streamlit_js_eval

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
    # Initialize session state for the toggle switch
    if "REG_2" not in st.session_state:
        st.session_state.REG_2 = False  # Default state is off
        
    # Time range slider
    year_range = st.slider(
        "Select Year Range",
        min_value=int(st.session_state["DF"]['Incident.year'].min()),
        max_value=int(st.session_state["DF"]['Incident.year'].max()),
        value=(int(st.session_state["DF"]['Incident.year'].min()), int(st.session_state["DF"]['Incident.year'].max()))
    )

    col1_side, col2_side = st.columns(2)
    with col1_side:
        if st.button('Reset Interaction'):
            st.session_state['filtered_df'] = st.session_state['DF']
            # Execute JavaScript to reload the browser window
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
            
    with col2_side:
        if st.button("Select 2 Regions"):
            # Toggle the state
            st.session_state.REG_2 = not st.session_state.REG_2

            # Display the current state
            if st.session_state.REG_2:
                st.write("Selection for two regions is ON")
            else:
                st.write("Selection for two regions is OFF")
                
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
st.session_state["filtered_df"] = st.session_state["DF"]



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
    
# Create a colormap
# Define the colormap: Blue -> Green -> Red
colormap = LinearColormap(
    colors=['#3d59e3', '#3de35e', '#e33d3d'],
    vmin=1,  # Minimum value
    vmax=56  # Maximum value
)
colormap.caption = 'Heatmap Intensity'

# Add the colormap as a legend to the map
colormap.add_to(st.session_state["m"])

# Add heatmap layer
HeatMap(heat_data, radius=10).add_to(st.session_state["m"])

# Add drawing tools to the map
draw = Draw(export=False)
draw.add_to(st.session_state["m"])



# DISPLAYING AND INTERACTION

# THE MAP
map_select_data = st_folium(st.session_state['m'],
                    center=None,
                    zoom=4,
                    key="new",
                    returned_objects=["last_object_clicked_tooltip", "last_object_clicked", "last_object_clicked_popup", "last_active_drawing"],
                    feature_group_to_add=fg,
                    height=700,
                    width=1300)


points_in_region_map = None
selected_point_map = None
if "selected_regions" not in st.session_state:
    st.session_state["selected_regions"] = []  
if "selected_region_ids" not in st.session_state:
    st.session_state["selected_region_ids"] = set()
# Check if a region has been selected (strictly from the map)
if map_select_data and 'last_active_drawing' in map_select_data and map_select_data['last_active_drawing']:
    selected_geometry = map_select_data["last_active_drawing"]["geometry"]

    if selected_geometry['type'] == 'Point':
        # Handle point selection
        tolerance = 0.0001
        target_lat = selected_geometry['coordinates'][1]
        target_lon = selected_geometry['coordinates'][0]

        # Find rows matching the target coordinates within the tolerance
        selected_point_map = st.session_state["DF"][
            (st.session_state["DF"]["latitude"].between(target_lat - tolerance, target_lat + tolerance)) &
            (st.session_state["DF"]["longitude"].between(target_lon - tolerance, target_lon + tolerance))
        ]
        st.session_state["selected_regions"] = [selected_point_map]  # Reset to only one region
        st.session_state['filtered_df'] = selected_point_map  # Update filtered data to match the selected point

    elif selected_geometry['type'] == 'Polygon':
        # Handle polygon selection
        from shapely.geometry import Point, Polygon
        polygon = Polygon(selected_geometry['coordinates'][0])
        st.session_state["DF"]['is_in_region'] = st.session_state["DF"].apply(
            lambda row: polygon.contains(Point(row['longitude'], row['latitude'])), axis=1
        )

        # Filter points within the selected region
        points_in_region_map = st.session_state["DF"][st.session_state["DF"]['is_in_region']]

        # Append the dataframe to the selected_regions list
        region_id = hash(points_in_region_map.to_csv(index=False))

        if region_id not in st.session_state["selected_region_ids"]:
            st.session_state["selected_region_ids"].add(region_id)
            st.session_state["selected_regions"].append(points_in_region_map.copy())

        # Limit regions to the last two selected
        if len(st.session_state["selected_regions"]) > 2:
            st.session_state["selected_regions"] = st.session_state["selected_regions"][-2:]

        st.session_state['filtered_df'] = points_in_region_map  # Update filtered data



# THE BARS

if st.session_state.REG_2:
    # Remove 'Unnamed: 0' and 'UIN' columns from the selected regions
    st.session_state["selected_regions"][0] = st.session_state["selected_regions"][0].drop(columns=["Unnamed: 0", "UIN"], errors="ignore")
    st.session_state["selected_regions"][1] = st.session_state["selected_regions"][1].drop(columns=["Unnamed: 0", "UIN"], errors="ignore")

    # Display bar charts for two selected regions
    col1, col2 = st.columns(2)

    # First region bar chart
    with col1:
        attr1 = st.selectbox("Select Attribute for First Region", st.session_state["selected_regions"][0].columns, key="attr1")
        fig1 = px.histogram(st.session_state["selected_regions"][0], x=attr1,
                            title=f"First Region: {attr1} Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    # Second region bar chart
    with col2:
        attr2 = st.selectbox("Select Attribute for Second Region", st.session_state["selected_regions"][1].columns, key="attr2")
        fig2 = px.histogram(st.session_state["selected_regions"][1], x=attr2,
                            title=f"Second Region: {attr2} Distribution")
        st.plotly_chart(fig2, use_container_width=True)

else:
    # Single-region bar chart rendering
    col1, col2 = st.columns(2)
    
    # List of attributes to choose from
    attributes = [col for col in st.session_state["DF"].columns if col != 'Incident.year']
    if "selected_indices_2" not in st.session_state:
        st.session_state["selected_indices_2"] = []
    if "selected_indices_1" not in st.session_state:
        st.session_state["selected_indices_1"] = []
        
    # First column: Attribute selection and histogram
    with col1:
        attr1 = st.selectbox("Select Attribute for Column 1", attributes, key="attr1", index=17)
        filtered_data_1 = (
            st.session_state["DF"].iloc[st.session_state["selected_indices_2"]]
            if st.session_state["selected_indices_2"]
            else st.session_state["filtered_df"]
        )
        fig1 = px.histogram(filtered_data_1, x=attr1, title=f'Distribution of {attr1}')
        ret1 = st.plotly_chart(fig1, key="hist1", use_container_width=True, on_select='rerun')
        indices_bar_1 = ret1['selection']['point_indices'] if ret1 and 'selection' in ret1 else []
        st.session_state["selected_indices_1"] = indices_bar_1

    # Second column: Attribute selection and histogram
    with col2:
        attr2 = st.selectbox("Select Attribute for Column 2", attributes, key="attr2", index=3)
        filtered_data_2 = (
            st.session_state["DF"].iloc[st.session_state["selected_indices_1"]]
            if st.session_state["selected_indices_1"]
            else st.session_state["filtered_df"]
        )
        fig2 = px.histogram(filtered_data_2, x=attr2, title=f'Distribution of {attr2}')
        ret2 = st.plotly_chart(fig2, key="hist2", use_container_width=True, on_select='rerun')
        indices_bar_2 = ret2['selection']['point_indices'] if ret2 and 'selection' in ret2 else []
        st.session_state["selected_indices_2"] = indices_bar_2

            
##TODO: PCP Is shit, update it.
# PCP Plot below 

# Parallel Categories Plot preparation
def create_parallel_coordinates(dataframe, selected_columns):
    # Define the numerical and categorical columns for the PCP
    # selected_columns = ["Incident.year", "latitude", "longitude"]  # Add/modify relevant columns
    if 'Provoked/unprovoked' in selected_columns: 
        # Map values in the 'Provoked/unprovoked' column to 1 and 0
        mapping = {"provoked": 1, "unprovoked": 0}
        dataframe["Provoked/unprovoked"] = dataframe["Provoked/unprovoked"].str.lower().map(mapping)

    # Create the Parallel Coordinates Plot
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=dataframe["Incident.year"], colorscale="Viridis"),
            dimensions=[
                dict(range=[dataframe[col].min(), dataframe[col].max()],
                     label=col, values=dataframe[col]) for col in selected_columns
            ]
        )
    )
    fig.update_layout(height=300)
    return fig

# PCP Display

# Filter columns to include only numeric types (int and float)
columns_to_exclude = ["Unnamed: 0", "UIN"]
columns_for_pcp = [
    col for col in st.session_state["filtered_df"].select_dtypes(include=["float64", "int64"]).columns
    if col not in columns_to_exclude
]
columns_for_pcp.append("Provoked/unprovoked")

st.subheader("Parallel Coordinates Plot")

default_columns = ["Shark.length.m", "Victim.age", "Time.in.water.min"]
selected_columns = st.multiselect(
    "Select the attributes:",
    options=columns_for_pcp,  # Use filtered numeric columns for the Parallel Coordinates Plot
    default=[col for col in default_columns if col in columns_for_pcp]  # Preselect the default columns if they exist
)

pcp_fig = create_parallel_coordinates(st.session_state["filtered_df"], selected_columns)
st.plotly_chart(pcp_fig, use_container_width=True)

