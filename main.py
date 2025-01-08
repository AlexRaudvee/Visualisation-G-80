# Import necessary libraries
import folium                           # Folium for map-related functionalities (redundant import)
import folium.map                       # Folium for map-related functionalities
import streamlit_folium                 # Streamlit Folium for integrating folium maps in Streamlit app
from streamlit_folium import folium_static  # For displaying folium maps in streamlit

import pandas as pd                     # Pandas for data manipulation and analysis
import seaborn as sns                   # Seaborn for statistical data visualization (not used in this code)
import streamlit as st                  # Streamlit for building the web app
import plotly.express as px             # Plotly Express for creating interactive charts
import plotly.graph_objects as go       # For advanced plotly visualizations
import matplotlib.pyplot as plt         # Matplotlib for static plots (not used in this code)
from datetime import datetime           # For temporal operations
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
##TODO: Map has data points in the middle of the map, investigate NaN of coordinates.
with col1:
    st.header("Shark Attack Locations")


    def validate_coordinates(lat: float, lon: float) -> bool:
        """
        Validate if coordinates fall within Australian waters boundary.
        Args:
            lat (float): Latitude coordinate
            lon (float): Longitude coordinate
        Returns:
            bool: True if coordinates are valid, False otherwise
        """
        # Define Australia's geographical bounding box
        AUS_LAT_MIN, AUS_LAT_MAX = -44, -10  # Latitude range from South to North
        AUS_LON_MIN, AUS_LON_MAX = 113, 154  # Longitude range from West to East

        return (AUS_LAT_MIN <= lat <= AUS_LAT_MAX and
                AUS_LON_MIN <= lon <= AUS_LON_MAX)


    # Filter dataframe to only include valid coordinates within Australian waters
    valid_coords_df = filtered_df[
        filtered_df.apply(
            lambda row: validate_coordinates(row['Latitude'], row['Longitude']),
            axis=1
        )
    ]

    # Initialize the map centered on Australia
    m = folium.Map(
        location=[-25.2744, 133.7751],  # Coordinates for center of Australia
        zoom_start=4,  # Initial zoom level
        tiles='cartodbpositron'  # Light map style that highlights water bodies
    )

    # Add heatmap layer if valid coordinates exist
    if not valid_coords_df.empty:
        # Extract coordinates for heatmap
        heat_data = valid_coords_df[['Latitude', 'Longitude']].values.tolist()

        # Create heatmap layer
        folium.plugins.HeatMap(
            heat_data,
            min_opacity=0.3,  # Minimum opacity for sparse areas
            max_opacity=0.8,  # Maximum opacity for dense areas
            radius=15  # Size of heatmap points
        ).add_to(m)

        # Add individual markers for each incident
        for _, row in valid_coords_df.iterrows():
            # Create detailed popup content
            popup_content = f"""
                <b>Location:</b> {row['Location']}<br>
                <b>Date:</b> {row['Incident.year']}<br>
                <b>Activity:</b> {row.get('Victim.activity', 'Unknown')}<br>
                <b>Injury:</b> {row.get('Injury.severity', 'Unknown')}
            """

            # Add circle marker with popup
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,  # Size of marker
                popup=folium.Popup(  # Popup configuration
                    popup_content,
                    max_width=300  # Maximum width of popup
                ),
                color='red',  # Marker color
                fill=True,  # Fill the circle
                fill_opacity=0.7  # Opacity of fill
            ).add_to(m)

    else:
        # Display warning if no valid coordinates are found
        st.warning("No valid coordinates found in the selected data range.")

    # Render the map in Streamlit
    folium_static(m, width=800)  # Set fixed width for consistent display

with col2:
    if selected_bar_attr:
        st.header("Distribution Analysis")
        for attr in selected_bar_attr:
            fig = px.histogram(filtered_df, x=attr, title=f'Distribution of {attr}')
            st.plotly_chart(fig, use_container_width=True)
##TODO: PCP Is shit, update it.
# PCP Plot below map
if selected_pcp_attr:
    st.header("Parallel Coordinates Plot")

    """
    Creates an interactive Parallel Coordinates Plot (PCP) for multivariate data analysis.
    The PCP allows for visualization and exploration of relationships between multiple variables
    simultaneously, with color encoding representing the temporal dimension (Incident.year).
    """

    # Create a copy for data manipulation to preserve original data
    normalized_df = filtered_df.copy()

    # Normalize data to [0,1] range for each attribute
    for attr in selected_pcp_attr:
        """
        Normalize each attribute to [0,1] scale for consistent visualization.
        Uses min-max normalization: (x - min(x)) / (max(x) - min(x))
        Includes handling for edge case where min = max to prevent division by zero.
        """
        denom = filtered_df[attr].max() - filtered_df[attr].min()
        if denom != 0:
            normalized_df[attr] = (filtered_df[attr] - filtered_df[attr].min()) / denom
        else:
            normalized_df[attr] = filtered_df[attr]

    # Create the Parallel Coordinates Plot
    fig = go.Figure(data=
    go.Parcoords(
        # Line properties
        line=dict(
            # Color lines by year to show temporal patterns
            color=filtered_df['Incident.year'],
            # Electric colorscale provides good visibility for temporal progression
            colorscale='Electric',
            showscale=True,
            # Set color range to match year range
            cmin=filtered_df['Incident.year'].min(),
            cmax=filtered_df['Incident.year'].max()
        ),
        # Define each parallel axis
        dimensions=[
            dict(
                # Use actual data range for each attribute
                range=[filtered_df[attr].min(), filtered_df[attr].max()],
                # Attribute name as axis label
                label=attr,
                # Original values for accurate representation
                values=filtered_df[attr],
                # Show three reference points: min, median, max
                tickvals=[
                    filtered_df[attr].min(),
                    filtered_df[attr].median(),
                    filtered_df[attr].max()
                ],
                # Format tick labels to one decimal place
                ticktext=[
                    f"{filtered_df[attr].min():.1f}",
                    f"{filtered_df[attr].median():.1f}",
                    f"{filtered_df[attr].max():.1f}"
                ]
            ) for attr in selected_pcp_attr
        ],
        # Properties for unselected lines during brushing/linking
        unselected=dict(
            line=dict(
                color='lightgray',  # Muted color for unselected lines
                opacity=0.5  # Semi-transparent for visual hierarchy
            )
        )
    )
    )

    # Configure the layout for optimal visualization
    fig.update_layout(
        # Set height for good visibility of patterns
        height=600,
        # Margins to prevent axis label clipping
        margin=dict(l=80, r=80, t=50, b=50),
        # White background for clarity
        plot_bgcolor='white',
        paper_bgcolor='white',
        # Centered title with year color encoding information
        title=dict(
            text="Parallel Coordinates Plot (Color: Incident Year)",
            x=0.5,  # Center horizontally
            y=0.95  # Position near top
        )
    )

    # Display the plot in Streamlit with configuration
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            'displayModeBar': True,  # Show the modebar for interactions
            'displaylogo': False,  # Remove Plotly logo
            'modeBarButtonsToRemove': [  # Remove unnecessary buttons
                'lasso2d',
                'select2d'
            ]
        }
    )

    """
    Interactive Features:
    - Drag along axes to filter data ranges
    - Drag axis labels to reorder dimensions
    - Hover over lines to see exact values
    - Use modebar for zoom and pan operations

    Visual Encoding:
    - Line color: Year of incident
    - Line opacity: Selected/unselected state
    - Axis position: Variable dimension
    - Tick marks: Min, median, and max values

    Data Handling:
    - Normalized internally for consistent visualization
    - Original values displayed on axes for interpretation
    - Missing values handled gracefully
    - Interactive filtering preserved through brushing
    """

# Display data statistics
# Display data statistics
with st.expander("Show Data Statistics"):
    # DataFrame Overview
    st.header("DataFrame Overview")
    st.write("**Shape of the DataFrame:**", filtered_df.shape)
    st.write("**Data Types:**")
    st.write(filtered_df.dtypes)

    # Missing Values and Summary Statistics
    st.subheader("Missing Values and Summary Statistics")
    st.write("**Number of NaN values per column:**")

    # Calculate and display both count and percentage of NaN values
    nan_counts = filtered_df.isna().sum()
    nan_percentages = (filtered_df.isna().sum() / len(filtered_df) * 100).round(2)
    nan_info = pd.DataFrame({
        'Count': nan_counts,
        'Percentage': nan_percentages.apply(lambda x: f"{x}%")
    })
    nan_info = nan_info.sort_values('Count')  # Sort ascending
    st.write(nan_info)

    st.write("**Summary Statistics:**")
    st.write(filtered_df.describe(include='all'))

    # Cleaned DataFrame
    st.header("Cleaned a bit dataframe")
    st.write(filtered_df)

    # Bar Chart for Categorical Columns
    st.subheader("Bar Chart for Categorical Columns")
    categorical_cols = filtered_df.select_dtypes(include='object').columns.tolist()
    selected_cat_col = st.selectbox("Choose a Categorical Column for Bar Chart",
                                    options=categorical_cols,
                                    key='stat_cat_col')
    if selected_cat_col:
        fig = px.histogram(filtered_df[selected_cat_col],
                           labels={'index': selected_cat_col, selected_cat_col: 'Count'},
                           title=f'Bar Chart of {selected_cat_col}')
        fig.update_layout(xaxis_title=selected_cat_col, yaxis_title='Count')
        st.plotly_chart(fig)

        # Histogram for Numeric Columns
    st.subheader("Histogram for Numeric Columns")
    numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
    selected_num_col = st.selectbox("Choose a Numeric Column for Histogram",
                                    options=numeric_cols,
                                    key='stat_num_col')
    if selected_num_col:
        fig = px.histogram(filtered_df, x=selected_num_col, nbins=20,
                           title=f'Histogram of {selected_num_col}',
                           labels={selected_num_col: selected_num_col})
        fig.update_layout(xaxis_title=selected_num_col, yaxis_title='Frequency')
        st.plotly_chart(fig)

        # Scatter Plot
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Choose X-axis (Numeric Column)",
                          options=numeric_cols,
                          key='stat_scatter_x')
    y_axis = st.selectbox("Choose Y-axis (Numeric Column)",
                          options=numeric_cols,
                          key='stat_scatter_y')

    if x_axis and y_axis:
        scatter_fig = px.scatter(filtered_df, x=x_axis, y=y_axis,
                                 title=f'Scatter Plot of {x_axis} vs {y_axis}')
        scatter_fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
        st.plotly_chart(scatter_fig)