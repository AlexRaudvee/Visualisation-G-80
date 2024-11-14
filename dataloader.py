import os
import requests

import pandas as pd

# Define the URL of the .xlsx file and the directory path
url = 'https://zenodo.org/records/11334212/files/Australian%20Shark-Incident%20Database%20Public%20Version.xlsx?download=1'  
save_directory = './data'
file_name = 'shark_data.xlsx'

# Create the /data folder if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Define the complete file path
file_path = os.path.join(save_directory, file_name)

# Download the .xlsx file
try:
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    with open(file_path, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded successfully and saved to {file_path}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    
# Define file paths
csv_file_path = 'data/shark_data.csv'    # Path where the .csv file will be saved

# Convert the .xlsx file to .csv
try:
    # Load the .xlsx file as a DataFrame
    df = pd.read_excel("data/" + file_name, engine='openpyxl')

    # Save the DataFrame as a .csv file
    df.to_csv(csv_file_path, index=False)
    print(f"File converted to CSV and saved at {csv_file_path}")

except Exception as e:
    print(f"An error occurred during conversion: {e}")