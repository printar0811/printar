import os
import csv

def check_and_delete_csv_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    files_to_delete = []

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            
            # Read the first line (header) and the second line (first data row)
            header = next(reader, None)
            second_line = next(reader, None)

            if second_line:
                # Check if the third from the last column is "UnKnown"
                if second_line[-4] == "Unknown" or second_line[-4] == "unknown":
                    files_to_delete.append(file_path)
                    #print("Yes I found it!")

    # Delete the files that meet the condition
    print(files_to_delete)
    for file_path in files_to_delete:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")

def main():
    folder_path = '/media/mahdad/easystore/utility_codes/media/Downloads/output_10000'  # Update this to your folder path
    check_and_delete_csv_files(folder_path)

if __name__ == "__main__":
    main()
