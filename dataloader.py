import os
import requests

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

