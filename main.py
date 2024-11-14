import folium.map
import plotly.express as px
import pandas as pd
import streamlit as st
import folium 
import streamlit_folium
import seaborn as sns
import matplotlib.pyplot as plt

from streamlit_plotly_events import plotly_events

csv_file_path = "data/shark_data.csv"

try:
    # Load the .csv file into a DataFrame
    df = pd.read_csv(csv_file_path)
    print("CSV file loaded successfully into DataFrame")
    # st.dataframe(df)  # Display the first few rows

except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")

# Title
st.title("DataFrame Statistics and Visualization")

# Brief statistics
st.header("DataFrame Overview")
st.write("**Shape of the DataFrame:**", df.shape)
st.write("**Data Types:**")
st.write(df.dtypes)

# NaN counts and summary statistics
st.subheader("Missing Values and Summary Statistics")
st.write("**Number of NaN values per column:**")
st.write(df.isna().sum())

st.write("**Summary Statistics:**")
st.write(df.describe(include='all'))

# FILTER OUT THE COLUMNS
df = df.loc[:, ~df.columns.isin(["Site.category.comment", "Shark.identification.source", "Tidal.cycle", "Weather.condition",
                                "Fish.speared?", "Commercial.dive.activity", "Object.of.bite", 
                                "Direction.first.strike", "Shark.captured", "Other.clothing.colour", 
                                "Clothing.pattern", "Fin.colour", "Diversionary.action.taken", "Diversionary.action.outcome",
                                "People <3m", "People 3-15m", "Unnamed: 59"])]

# FILTER OUT THE ROWS 
# Define the threshold
threshold = 0.075  # 7.5%

# Calculate the threshold count of missing values for each column
total_rows = len(df)
max_missing = total_rows * threshold

# Identify columns where the count of NaNs is less than the threshold
columns_to_filter = [col for col in df.columns if df[col].isna().sum() < max_missing]

# Drop rows with NaNs in columns where missing values are below the threshold
df = df.dropna(subset=columns_to_filter)

st.header("Cleaned a bit dataframe")
st.write(df)

# Filterable Bar Chart for Categorical Columns
st.subheader("Bar Chart for Categorical Columns")

# Dropdown to select categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
selected_cat_col = st.selectbox("Choose a Categorical Column for Bar Chart", options=categorical_cols)

# Plotly Express Bar Chart
if selected_cat_col:
    fig = px.histogram(df[selected_cat_col],
                 labels={'index': selected_cat_col, selected_cat_col: 'Count'},
                 title=f'Bar Chart of {selected_cat_col}')
    fig.update_layout(xaxis_title=selected_cat_col, yaxis_title='Count')
    st.plotly_chart(fig)

# Histogram for Selected Numeric Column
st.subheader("Histogram for Numeric Columns")

# Dropdown to select a numeric column
numeric_cols = df.select_dtypes(include='number').columns.tolist()
selected_num_col = st.selectbox("Choose a Numeric Column for Histogram", options=numeric_cols)

# Plotly Express Histogram
if selected_num_col:
    fig = px.histogram(df, x=selected_num_col, nbins=20,
                       title=f'Histogram of {selected_num_col}',
                       labels={selected_num_col: selected_num_col})
    fig.update_layout(xaxis_title=selected_num_col, yaxis_title='Frequency')
    st.plotly_chart(fig)

# Scatter Plot with Selectable Axes for Numeric Columns
st.subheader("Scatter Plot")

# Dropdowns to select numeric columns for X and Y axis
x_axis = st.selectbox("Choose X-axis (Numeric Column)", options=numeric_cols)
y_axis = st.selectbox("Choose Y-axis (Numeric Column)", options=numeric_cols)

# Plotting scatter plot based on selected X and Y using Plotly Express
if x_axis and y_axis:
    scatter_fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot of {x_axis} vs {y_axis}')
    scatter_fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
    st.plotly_chart(scatter_fig)