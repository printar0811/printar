import os
import csv
import time

# Replacement dictionary
replacements = {
    " Oculus Quest": "Oculus Quest"
}

def process_files_in_batches(folder_path, batch_size):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    total_files = len(files)
    
    for start_idx in range(0, total_files, batch_size):
        yield files[start_idx:start_idx + batch_size], total_files, start_idx + batch_size

def update_device_name(device_name):
    return replacements.get(device_name, device_name)

def scan_and_update_csv_files(folder_path, batch_size=100):
    start_time = time.time()

    for batch_files, total_files, processed_files in process_files_in_batches(folder_path, batch_size):
        for filename in batch_files:
            file_path = os.path.join(folder_path, filename)
            temp_file_path = file_path + '.tmp'
            with open(file_path, 'r') as csvfile, open(temp_file_path, 'w', newline='') as tempfile:
                reader = csv.reader(csvfile)
                writer = csv.writer(tempfile)
                
                header = next(reader)
                writer.writerow(header)

                for row in reader:
                    if len(row) >= 4:  # Ensure there are at least 3 columns to access the third-to-last
                        row[-4] = update_device_name(row[-4])
                    writer.writerow(row)
            
            os.remove(file_path)
            os.rename(temp_file_path, file_path)

        # Calculate progress and time remaining
        progress = (processed_files / total_files) * 100
        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / processed_files
        remaining_files = total_files - processed_files
        remaining_time = avg_time_per_file * remaining_files

        # Print progress and estimated time remaining
        print(f"\rProgress: {progress:.2f}% - Processed {min(processed_files, total_files)} of {total_files} files - Time remaining: {remaining_time:.2f} seconds", end='')

        if processed_files >= 10000:
            break

    print()  # Move to the next line after progress bar completes

def main():
    folder_path = '/media/mahdad/easystore/utility_codes/media/Downloads/output_10000'  # Update this to your folder path
    scan_and_update_csv_files(folder_path, batch_size=100)  # Adjust batch_size as needed

if __name__ == "__main__":
    main()
