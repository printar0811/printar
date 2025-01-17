import os
import csv
import re

def remove_invalid_lines_from_csv(folder_path):
    # Regular expression to match invalid float values
    invalid_float_pattern = re.compile(r'\d+\.\d+\.\d+')

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            temp_file_path = file_path + '.tmp'

            with open(file_path, 'r') as csv_file, open(temp_file_path, 'w', newline='') as temp_file:
                csv_reader = csv.reader(csv_file)
                csv_writer = csv.writer(temp_file)

                # Iterate over each line in the CSV file
                for row in csv_reader:
                    # Check all values except the last one in the row
                    if not any(invalid_float_pattern.search(value) for value in row[:-1]):
                        csv_writer.writerow(row)

            # Replace the original file with the cleaned file
            os.replace(temp_file_path, file_path)

# Define the folder path
folder_path = 'input_files/output_device_10000_device_2/'

# Remove invalid lines from CSV files
remove_invalid_lines_from_csv(folder_path)
print("Invalid lines removed from CSV files.")
