import os
import json
from src.xror import XROR


# Function to extract frames data along with metadata
def extract_frames_data(json_data):
    # Extract metadata
    try:
        user_id = json_data["info"]["user"]["id"]
    except KeyError:
        user_id = "unknown"

    try:
        device_name_1 = json_data["info"]["hardware"]["devices"][0]["name"]
    except (KeyError, IndexError):
        device_name_1 = "unknown"

    try:
        device_name_2 = json_data["info"]["hardware"]["devices"][1]["name"]
    except (KeyError, IndexError):
        device_name_2 = "unknown"

    try:
        device_name_3 = json_data["info"]["hardware"]["devices"][2]["name"]
    except (KeyError, IndexError):
        device_name_3 = "unknown"

    try:
        app_name = json_data["info"]["software"]["app"]["name"]
    except KeyError:
        app_name = "unknown"

    # Extract frames data
    frames = json_data["frames"]

    # Return metadata and frames data
    return frames, user_id, device_name_1, device_name_2, device_name_3, app_name


# Function to write data to CSV file
def write_to_csv(frames, user_id, device_name_1, device_name_2, device_name_3, app_name, output_file):
    with open(output_file, mode='w') as file:
        # Write metadata to the CSV file
        file.write('Frame Data,User ID,Device 1,Device 2,Device 3, App Name\n')

        # Write frame data to the CSV file
        for frame in frames:
            frame_str = ','.join(map(str, frame))
            file.write(f'{frame_str},{user_id},{device_name_1},{device_name_2},{device_name_3}, {app_name}\n')


# Specify the folder containing XROR files and the output folder
folder_path = r'/Users/tanvirmahdad/downloads/group-01-data/user2'
output_folder = r'/Users/tanvirmahdad/downloads/group-01-data/user2/output'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through all XROR files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.xror'):
        file_path = os.path.join(folder_path, filename)

        # Read XROR file
        with open(file_path, 'rb') as f:
            file_content = f.read()
            xror_data = XROR.unpack(file_content)

        # Convert XROR data to dictionary
        dict_data = xror_data.to_dict()

        # Convert dictionary to JSON
        json_data = json.dumps(dict_data, indent=2)

        # Extract data from JSON
        frames, user_id, device_name_1, device_name_2, device_name_3, app_name = extract_frames_data(json.loads(json_data))

        # Write data to CSV file in the output folder
        output_csv_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.csv')
        write_to_csv(frames, user_id, device_name_1, device_name_2, device_name_3, app_name, output_csv_file)

        print(f"CSV file '{output_csv_file}' has been created successfully.")
