import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
from folium.plugins import MarkerCluster
from dataloader import df

st.session_state['df'] = df

center_lat = st.session_state["df"]["latitude"].mean()
center_lon = st.session_state["df"]["longitude"].mean()

def create_map():
    if 'map' not in st.session_state or st.session_state.map is None:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
        
        marker_cluster = MarkerCluster().add_to(m)
        # Add markers for each point in the filtered DataFrame
        for _, row in st.session_state['df'].iterrows():
            folium.Marker(location=[row["latitude"], row["longitude"]],
                          popup=f"Latitude: {row['latitude']}, Longitude: {row['longitude']}",
                          tooltip="Click for more info").add_to(marker_cluster)
        
        st.session_state.map = m  # Save the map in the session state
        
    return st.session_state.map


def show_map():
    m = create_map()  # Get or create the map
    st_folium(m)

show_map()