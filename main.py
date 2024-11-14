# Import necessary libraries
import folium                           # Folium for map-related functionalities (redundant import)
import folium.map                       # Folium for map-related functionalities
import streamlit_folium                 # Streamlit Folium for integrating folium maps in Streamlit app

import pandas as pd                     # Pandas for data manipulation and analysis
import seaborn as sns                   # Seaborn for statistical data visualization (not used in this code)
import streamlit as st                  # Streamlit for building the web app
import plotly.express as px             # Plotly Express for creating interactive charts
import matplotlib.pyplot as plt         # Matplotlib for static plots (not used in this code)

from streamlit_plotly_events import plotly_events  # For handling plotly events in Streamlit

# Define the path to the CSV file containing shark data
csv_file_path = "data/shark_data.csv"

try:
    # Attempt to load the .csv file into a DataFrame
    df = pd.read_csv(csv_file_path)
    print("CSV file loaded successfully into DataFrame")
    # st.dataframe(df)  # Display the first few rows of the DataFrame in Streamlit

except Exception as e:
    # Handle errors if loading the CSV fails
    print(f"An error occurred while loading the CSV: {e}")

# Title for the Streamlit application
st.title("DataFrame Statistics and Visualization")

# Overview section displaying basic DataFrame information
st.header("DataFrame Overview")
st.write("**Shape of the DataFrame:**", df.shape)  
st.write("**Data Types:**")  
st.write(df.dtypes)  

# Section for displaying missing values and summary statistics
st.subheader("Missing Values and Summary Statistics")
st.write("**Number of NaN values per column:**")  
st.write(df.isna().sum())  

st.write("**Summary Statistics:**")  
st.write(df.describe(include='all'))  

# Filter out unnecessary columns from the DataFrame
df = df.loc[:, ~df.columns.isin(["Site.category.comment", "Shark.identification.source", "Tidal.cycle", "Weather.condition",
                                "Fish.speared?", "Commercial.dive.activity", "Object.of.bite", 
                                "Direction.first.strike", "Shark.captured", "Other.clothing.colour", 
                                "Clothing.pattern", "Fin.colour", "Diversionary.action.taken", "Diversionary.action.outcome",
                                "People <3m", "People 3-15m", "Unnamed: 59"])]  

# Filter out rows based on the threshold for missing values
threshold = 0.075  # Define a threshold for the proportion of missing data (7.5%)

# Calculate the maximum number of rows allowed to have missing values based on the threshold
total_rows = len(df)
max_missing = total_rows * threshold

# Identify columns where the count of NaNs is below the threshold
columns_to_filter = [col for col in df.columns if df[col].isna().sum() < max_missing]

# Drop rows where selected columns have NaN values
df = df.dropna(subset=columns_to_filter)

# Display the cleaned DataFrame after filtering out missing data
st.header("Cleaned a bit dataframe")
st.write(df)  

# Section for displaying a filterable bar chart for categorical columns
st.subheader("Bar Chart for Categorical Columns")

# Dropdown menu for selecting a categorical column
categorical_cols = df.select_dtypes(include='object').columns.tolist()  # Get a list of categorical columns
selected_cat_col = st.selectbox("Choose a Categorical Column for Bar Chart", options=categorical_cols)  # Streamlit selectbox for column selection

# Generate a bar chart using Plotly Express if a column is selected
if selected_cat_col:
    fig = px.histogram(df[selected_cat_col],
                 labels={'index': selected_cat_col, selected_cat_col: 'Count'},
                 title=f'Bar Chart of {selected_cat_col}')
    fig.update_layout(xaxis_title=selected_cat_col, yaxis_title='Count')  
    st.plotly_chart(fig)  

# Section for displaying a histogram for selected numeric columns
st.subheader("Histogram for Numeric Columns")

# Dropdown menu for selecting a numeric column
numeric_cols = df.select_dtypes(include='number').columns.tolist()  # Get list of numeric columns
selected_num_col = st.selectbox("Choose a Numeric Column for Histogram", options=numeric_cols)  # Streamlit selectbox for column selection

# Generate a histogram using Plotly Express if a numeric column is selected
if selected_num_col:
    fig = px.histogram(df, x=selected_num_col, nbins=20,
                       title=f'Histogram of {selected_num_col}',
                       labels={selected_num_col: selected_num_col})  
    fig.update_layout(xaxis_title=selected_num_col, yaxis_title='Frequency')  
    st.plotly_chart(fig) 

# Section for displaying a scatter plot with selectable axes
st.subheader("Scatter Plot")

# Dropdown menus for selecting numeric columns for the X and Y axes
x_axis = st.selectbox("Choose X-axis (Numeric Column)", options=numeric_cols)  
y_axis = st.selectbox("Choose Y-axis (Numeric Column)", options=numeric_cols)  

# Generate a scatter plot using Plotly Express if both X and Y axes are selected
if x_axis and y_axis:
    scatter_fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot of {x_axis} vs {y_axis}')
    scatter_fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)  # Set axis titles
    st.plotly_chart(scatter_fig)  
