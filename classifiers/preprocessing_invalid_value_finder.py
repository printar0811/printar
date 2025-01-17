import os
import csv
import re

def find_invalid_floats_in_csv(folder_path):
    # Regular expression to match invalid float values
    invalid_float_pattern = re.compile(r'\d+\.\d+\.\d+')

    # Dictionary to store the results
    results = {}

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                invalid_lines_count = 0

                # Iterate over each line in the CSV file
                for row in csv_reader:
                    # Check all values except the last one in the row
                    for value in row[:-1]:
                        if invalid_float_pattern.search(value):
                            invalid_lines_count += 1
                            break

                # Store the result for the current file
                if invalid_lines_count > 0:
                    results[file_name] = invalid_lines_count

    return results

# Define the folder path
#folder_path = '/'
folder_path = 'input_files/output_device_10000_device_2/'

# Find invalid float values in CSV files and print the results
invalid_floats_results = find_invalid_floats_in_csv(folder_path)
for file_name, count in invalid_floats_results.items():
    print(f"File: {file_name}, Invalid lines count: {count}")
