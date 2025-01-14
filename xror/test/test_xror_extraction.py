import sys
import os
import pymongo
import bson
import json



from deepdiff import DeepDiff

# Add current directory to the Python path
dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir)
os.chdir(dir)

from src.xror import XROR
import numpy as np



class XRORJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, XROR):
            # Convert XROR object to a dictionary (or modify as needed)
            return obj.to_dict()  # Replace with your actual method

        # For other types, use the default serialization
        return super().default(obj)


def bytes_to_str(obj):
    if isinstance(obj, bytes):
        return obj.decode(errors='replace')
    elif isinstance(obj, dict):
        return {k: bytes_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [bytes_to_str(element) for element in obj]
    else:
        return obj


with open('./data/bsor/user1.xror', 'rb') as f:
    file = f.read()
second = XROR.unpack(file)


#bson_bytes = bson.BSON.encode(second)

# Deserialize BSON data
#decoded_data = bson.decode(bson_bytes)

# Print original and decoded data
#print("Original BSON data:")
#print(bson)
#print("\nDecoded BSON data:")
#print(decoded_data)

#print(second)

# Convert bytes to str in the dictionary
dict_data = bytes_to_str(second.to_dict())



json_data = json.dumps(dict_data, cls=XRORJSONEncoder, indent=2)
#print(json_data)

# File path to save JSON data
file_path = "user1.json"

# Write JSON data to a file
with open(file_path, "w") as json_file:
    json_file.write(json_data)

print("JSON data has been written to:", file_path)
