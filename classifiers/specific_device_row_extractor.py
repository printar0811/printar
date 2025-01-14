import os
import csv
from collections import defaultdict

def extract_rows_and_balance(folder_path, output_file, target_values):
    """
    Extract rows from all CSV files in the folder where the last value matches any of the target values.
    Normalize the number of rows extracted for each target value to match the lowest count among them.
    If no match is found, print all unique last values.

    Args:
        folder_path (str): Path to the folder containing input CSV files.
        output_file (str): Path to the output CSV file.
        target_values (list): List of target values to match in the last column.
    """
    # Dictionary to store rows for each target value
    target_rows = defaultdict(list)
    unique_last_values = set()

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)

            # Open each CSV file for reading
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)

                # Iterate over each row in the CSV file
                for row in csv_reader:
                    if row:  # Ensure row is not empty
                        last_value = row[-1]
                        unique_last_values.add(last_value)  # Add to unique last values set

                        # Check if the last value matches any of the target values
                        if last_value in target_values:
                            target_rows[last_value].append(row)

    # Find the minimum count among all matched target values
    if target_rows:
        min_count = min(len(rows) for rows in target_rows.values())
        min_count=5605
        print(f"Lowest count among matching target values: {min_count}")

        # Balance rows by limiting to `min_count` for each target value
        balanced_rows = []
        for value in target_values:
            if value in target_rows:
                balanced_rows.extend(target_rows[value][:min_count])

        # Write balanced rows to output file
        with open(output_file, 'w', newline='') as output_csv:
            csv_writer = csv.writer(output_csv)
            csv_writer.writerows(balanced_rows)

        # Print counts of rows extracted for each target value
        print("Counts of rows matching each target value (balanced):")
        for value in target_values:
            count = len(target_rows[value]) if value in target_rows else 0
            print(f"{value}: {min_count if count > 0 else 0}")
    else:
        print("No rows matched the target values.")
        print("Unique last values found across all files:")
        print(unique_last_values)

# Define the folder path, output file, and target values
folder_path = 'input_files/output_device_10000_device_2/'
output_file = 'output_test5.csv'
target_values = ["Oculus Quest", "Meta Quest Pro"]  # Add more values if needed

# Extract and balance rows with specific last values
extract_rows_and_balance(folder_path, output_file, target_values)
