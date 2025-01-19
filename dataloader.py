# imports
import os
import requests

import pandas as pd

from datetime import datetime           # For temporal operations



# GLOBAL VARIABLES
# Define the URL of the .xlsx file and the directory path to save the file
url = 'https://zenodo.org/records/11334212/files/Australian%20Shark-Incident%20Database%20Public%20Version.xlsx?download=1'  
save_directory = './data'  # Directory where the downloaded file will be saved
file_name = 'shark_data.xlsx'  # Name for the downloaded .xlsx file

# Create the /data folder if it doesn't exist to ensure a valid save location
os.makedirs(save_directory, exist_ok=True)

# Define the complete path where the file will be saved
file_path = os.path.join(save_directory, file_name)



### FUNCTIONS ###
def remove_high_nan_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove columns with NaN percentage above threshold."""
    nan_percentages = df.isna().mean()
    columns_to_keep = nan_percentages[nan_percentages < threshold].index
    return df[columns_to_keep]

def clean_coordinates(x):
    """Clean coordinate strings and convert to float."""
    if pd.isna(x):
        return None
    try:
        # Remove °, ', " and convert to float
        return float(str(x).replace('°', '').replace("'", '').replace('"', '').strip())
    except:
        return None
    
def load_and_clean_data(df):
    try:
        df = df

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
        print(f"Error loading data: {e}")
        return None

# Download the .xlsx file from the specified URL
try:
    response = requests.get(url)
    response.raise_for_status()  # Raises an error if the request was unsuccessful (e.g., 404 or 500 status codes)
    
    # Save the content of the response to the specified file path in binary mode
    with open(file_path, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded successfully and saved to {file_path}")

except requests.exceptions.RequestException as e:
    # Handle any exception that occurs during the download process, such as connection issues
    print(f"An error occurred: {e}")

# Define the path where the converted .csv file will be saved
csv_file_path = 'data/shark_data.csv'

# Convert the downloaded .xlsx file to a .csv file
try:
    # Load the .xlsx file into a pandas DataFrame
    # Specifying the engine as 'openpyxl' to ensure compatibility with .xlsx format
    df = pd.read_excel("data/" + file_name, engine='openpyxl')

    # Save the DataFrame to a .csv file without the index column
    df.to_csv(csv_file_path, index=False)
    print(f"File converted to CSV and saved at {csv_file_path}")

except Exception as e:
    # Handle any exceptions that occur during the conversion process
    print(f"An error occurred during conversion: {e}")


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



# FIRST STAGE CLEANING OF THE DATA
df = df.loc[:, ~df.columns.isin(["Site.category.comment", "Shark.identification.source", "Tidal.cycle", "Weather.condition",
                                "Fish.speared?", "Commercial.dive.activity", "Object.of.bite", 
                                "Direction.first.strike", "Shark.captured", "Other.clothing.colour", 
                                "Clothing.pattern", "Fin.colour", "Diversionary.action.taken", "Diversionary.action.outcome",
                                "People <3m", "People 3-15m", "Unnamed: 59"])]  

# Filter out rows based on the threshold for missing values
threshold = 0.1  # Define a threshold for the proportion of missing data (7.5%)

# Calculate the maximum number of rows allowed to have missing values based on the threshold
total_rows = len(df)
max_missing = total_rows * threshold

# Identify columns where the count of NaNs is below the threshold
columns_to_filter = [col for col in df.columns if df[col].isna().sum() < max_missing]

# Drop rows where selected columns have NaN values
df = df.dropna(subset=columns_to_filter)


# Convert latitude and longitude to numeric (float)
try:
    df["latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")  # Convert to float, invalid entries become NaN
    df["longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")  # Convert to float, invalid entries become NaN
except Exception as e:
    print(f"Error converting latitude/longitude to numeric: {e}")

# Drop rows with missing or invalid latitude/longitude values
df = df.dropna(subset=["latitude", "longitude"])

# Rename values in the 'Victim.injury' column
df['Victim.injury'] = df['Victim.injury'].replace(
    {'Injured': 'injured', 'injury': 'injured'}
)

# SECONDARY CLEANING OF THE DATA
df = load_and_clean_data(df)

df.to_csv("./data/shark_data.csv")

print("Process Exit Code: 1 (success)")