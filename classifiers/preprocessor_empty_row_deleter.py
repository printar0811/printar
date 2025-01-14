import os
import pandas as pd

# Define the path to the folder containing CSV files
folder_path = 'input_files/output_device_10000_device3_user/'

# List all CSV files in the specified folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Function to check if a row contains only a single element
def row_has_single_element(row):
    return sum(pd.notna(row)) == 1

# Traverse all CSV files
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    data = pd.read_csv(file_path)
    
    # Find rows that have only a single element and drop them
    rows_to_drop = data.apply(row_has_single_element, axis=1)
    if rows_to_drop.any():
        print(f"Removing rows with only a single element from {csv_file}.")
        data_cleaned = data[~rows_to_drop]
        
        # Save the cleaned data back to the same CSV file
        data_cleaned.to_csv(file_path, index=False)

print("Finished processing CSV files.")
