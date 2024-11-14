import folium.map
import plotly.express as px
import streamlit as st
import folium 
import streamlit_folium

from streamlit_plotly_events import plotly_events

df = px.data.tips()

# Writes a component similar to st.write()
fig = px.histogram(df, x="total_bill", color="sex")

selected_points = st.plotly_chart(fig, on_select='rerun', selection_mode='box')
st.write(selected_points)

m = folium.Map()

check = streamlit_folium.st_folium(m)

st.write(check)