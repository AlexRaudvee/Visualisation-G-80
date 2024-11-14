# imports
import os
import requests

import pandas as pd

# Define the URL of the .xlsx file and the directory path to save the file
url = 'https://zenodo.org/records/11334212/files/Australian%20Shark-Incident%20Database%20Public%20Version.xlsx?download=1'  
save_directory = './data'  # Directory where the downloaded file will be saved
file_name = 'shark_data.xlsx'  # Name for the downloaded .xlsx file

# Create the /data folder if it doesn't exist to ensure a valid save location
os.makedirs(save_directory, exist_ok=True)

# Define the complete path where the file will be saved
file_path = os.path.join(save_directory, file_name)

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
