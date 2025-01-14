import os
import pandas as pd
import warnings

# Define the path to the folder containing CSV files
folder_path = 'input_files/output_device_10000_device3_user/'

# List all CSV files in the specified folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Function to check for mixed types in a dataframe
def check_mixed_types(df):
    mixed_type_columns = []
    for column in df.columns:
        if df[column].apply(type).nunique() > 1:
            mixed_type_columns.append(column)
    return mixed_type_columns

# Check each file for mixed types
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Identify columns with mixed types
        mixed_columns = check_mixed_types(df)
        
        # Check if there are any mixed types warnings
        if mixed_columns:
            print(f"File '{csv_file}' has mixed types in columns: {mixed_columns}")
        else:
            print(f"File '{csv_file}' has no mixed types.")
    
