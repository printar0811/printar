import sys
import os
import json
from src.xror import XROR

# Function to extract frames data along with metadata
def extract_frames_data(json_data):
    # Extract metadata
    user_id = json_data["info"]["user"]["id"]
    device_name_1 = json_data["info"]["hardware"]["devices"][0]["name"]  # Assuming the first device is the HMD
    device_name_2 = json_data["info"]["hardware"]["devices"][1]["name"]  # Assuming the second device is the hand device
    device_name_3 = json_data["info"]["hardware"]["devices"][2]["name"]  # Assuming the third device is the hand 2 device
    app_name= json_data["info"]["software"]["app"]["name"] #App Name

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

# Add current directory to the Python path
dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir)
os.chdir(dir)

# Read XROR file
with open('./data/bsor/user5.xror', 'rb') as f:
    file = f.read()
    xror_data = XROR.unpack(file)

# Convert XROR data to dictionary
dict_data = xror_data.to_dict()

# Function to convert bytes to string in the dictionary
def bytes_to_str(obj):
    if isinstance(obj, bytes):
        return obj.decode(errors='replace')
    elif isinstance(obj, dict):
        return {k: bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [bytes_to_str(element) for element in obj]
    else:
        return obj

# Convert bytes to str in the dictionary
dict_data_str = bytes_to_str(dict_data)

# Convert dictionary to JSON
json_data = json.dumps(dict_data_str, indent=2)

# Write JSON data to a file
with open("user1.json", "w") as json_file:
    json_file.write(json_data)

# Extract data from JSON
frames, user_id, device_name_1, device_name_2, device_name_3, app_name = extract_frames_data(json.loads(json_data))

# Write data to CSV file
write_to_csv(frames, user_id, device_name_1, device_name_2, device_name_3, app_name, 'output.csv')

print("CSV file has been created successfully.")
