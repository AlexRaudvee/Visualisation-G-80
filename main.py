# Import necessary libraries
import folium                           # Folium for map-related functionalities (redundant import)
import folium.map                       # Folium for map-related functionalities
import streamlit_folium                 # Streamlit Folium for integrating folium maps in Streamlit app
from streamlit_folium import folium_static

import pandas as pd                     # Pandas for data manipulation and analysis
import seaborn as sns                   # Seaborn for statistical data visualization (not used in this code)
import streamlit as st                  # Streamlit for building the web app
import plotly.express as px             # Plotly Express for creating interactive charts
import plotly.graph_objects as go
import matplotlib.pyplot as plt         # Matplotlib for static plots (not used in this code)
from datetime import datetime
from streamlit_plotly_events import plotly_events  # For handling plotly events in Streamlit


# Define the path to the CSV file containing shark data
csv_file_path = "data/shark_data.csv"

def remove_high_nan_columns(df: pd.DataFrame, threshold: float = 0.92) -> pd.DataFrame:
    """Remove columns with NaN percentage above threshold."""
    nan_percentages = df.isna().mean()
    columns_to_keep = nan_percentages[nan_percentages < threshold].index
    return df[columns_to_keep]


# Set page config
st.set_page_config(layout="wide")


def clean_coordinates(x):
    """Clean coordinate strings and convert to float."""
    if pd.isna(x):
        return None
    try:
        # Remove °, ', " and convert to float
        return float(str(x).replace('°', '').replace("'", '').replace('"', '').strip())
    except:
        return None


# Initialize session state if not exists
if 'selected_location' not in st.session_state:
    st.session_state['selected_location'] = None
if 'selected_time_range' not in st.session_state:
    st.session_state['selected_time_range'] = None


# Data loading and cleaning
@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("data/shark_data.csv")

        # Filter for last 40 years
        current_year = datetime.now().year
        df = df[df['Incident.year'] > (current_year - 40)]

        # Remove high-NaN columns
        df = remove_high_nan_columns(df)

        # Clean and convert coordinates
        df['Latitude'] = df['Latitude'].apply(clean_coordinates)
        df['Longitude'] = df['Longitude'].apply(clean_coordinates)

        # Remove rows with invalid coordinates
        df = df.dropna(subset=['Latitude', 'Longitude'])

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Load data
df = load_and_clean_data()


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
# Calculate and display both count and percentage of NaN values
nan_counts = df.isna().sum()
nan_percentages = (df.isna().sum() / len(df) * 100).round(2)

st.write("**Number of NaN values per column:**")
nan_info = pd.DataFrame({
    'Count': nan_counts,
    'Percentage': nan_percentages.apply(lambda x: f"{x}%")
})
nan_info = nan_info.sort_values('Count')  # Sort ascending
st.write(nan_info)

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

if df is not None:
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # Time range slider
        year_range = st.slider(
            "Select Year Range",
            min_value=int(df['Incident.year'].min()),
            max_value=int(df['Incident.year'].max()),
            value=(int(df['Incident.year'].min()), int(df['Incident.year'].max()))
        )

        # Attribute selectors
        st.subheader("Bar Chart Attributes")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        selected_bar_attr = st.multiselect(
            "Select attributes for Bar Chart",
            categorical_cols,
            default=['Victim.activity', 'Injury.severity'] if 'Victim.activity' in categorical_cols else None
        )

        st.subheader("PCP Attributes")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_pcp_attr = st.multiselect(
            "Select attributes for Parallel Coordinates",
            numeric_cols,
            default=['Victim.age', 'Shark.length.m'] if 'Victim.age' in numeric_cols else None
        )

    # Filter data based on time range
    filtered_df = df[(df['Incident.year'] >= year_range[0]) &
                     (df['Incident.year'] <= year_range[1])]

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Shark Attack Locations")

        # Create base map centered on Australia
        m = folium.Map(
            location=[-25.2744, 133.7751],  # Center of Australia
            zoom_start=4
        )

        # Add heatmap layer
        heat_data = [[row['Latitude'], row['Longitude']] for index, row in filtered_df.iterrows()]
        folium.plugins.HeatMap(heat_data).add_to(m)

        # Add clickable markers
        for idx, row in filtered_df.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                popup=f"Location: {row['Location']}<br>Date: {row['Incident.year']}",
                color='red',
                fill=True
            ).add_to(m)

        # Display map
        folium_static(m)

    with col2:
        if selected_bar_attr:
            st.header("Distribution Analysis")
            for attr in selected_bar_attr:
                fig = px.histogram(filtered_df, x=attr, title=f'Distribution of {attr}')
                st.plotly_chart(fig, use_container_width=True)

    # PCP Plot below map
    if selected_pcp_attr:
        st.header("Parallel Coordinates Plot")
        fig = go.Figure(data=
        go.Parcoords(
            line=dict(color=filtered_df['Incident.year'],
                      colorscale='Viridis'),
            dimensions=[
                dict(range=[filtered_df[attr].min(), filtered_df[attr].max()],
                     label=attr,
                     values=filtered_df[attr])
                for attr in selected_pcp_attr
            ]
        )
        )
        st.plotly_chart(fig, use_container_width=True)

    # Display data statistics
    with st.expander("Show Data Statistics"):
        st.write("Data Shape:", filtered_df.shape)
        st.write("Missing Values:")
        st.write(filtered_df.isna().sum())