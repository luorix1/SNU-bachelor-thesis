import pandas as pd
import os
import sys
import time
from collections import defaultdict

def parse_imu_data(input_file, output_dir):
    # Define allowed keys
    allowed_keys = ['frame', 'date', 'time', 'qx', 'qy', 'qz', 'qw', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    
    # Initialize dictionaries to store data
    data = {foot: {key: [] for key in allowed_keys} for foot in ['left', 'right']}
    current_foot = None

    # Read the file and parse content
    with open(input_file, "r") as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue

            if '"FRAME"' in line:
                frame = int(line.split(",")[1].strip('"'))
            elif '"Date"' in line:
                date = line.split(",")[1].strip('"')
            elif '"Time"' in line:
                time_val = line.split(",")[1].strip('"')

            # Identify the start of data for each foot
            elif '"SENSOR","HX210.11.31.M7-LF S0041"' in line:
                current_foot = 'left'
                data[current_foot]['frame'].append(frame)
                data[current_foot]['date'].append(date)
                data[current_foot]['time'].append(time_val)
            elif '"SENSOR","HX210.11.31.M7-RF S0041"' in line:
                current_foot = 'right'
                data[current_foot]['frame'].append(frame)
                data[current_foot]['date'].append(date)
                data[current_foot]['time'].append(time_val)

            # Parse IMU data
            elif current_foot and "IMU" in line:
                imu_type, value = line.split(",")
                imu_type = imu_type.strip('"').split()[-1].lower()
                if imu_type in allowed_keys:
                    value = float(value.strip('"'))
                    data[current_foot][imu_type].append(value)

    # Filter 'gz' duplicate values since they are shown twice for each time step
    for foot in ['left', 'right']:
        gz_indices = [i for i, key in enumerate(data[foot]['gz']) if i % 2 == 1]
        data[foot]['gz'] = [data[foot]['gz'][i] for i in gz_indices]

    # Check and report on list lengths
    for foot in ['left', 'right']:
        lengths = {key: len(value) for key, value in data[foot].items()}
        print(f"\nLength of lists for {foot} foot:")
        
        if len(set(lengths.values())) != 1:
            print(f"\nWARNING: Not all lists for {foot} foot have the same length!")
            min_length = min(lengths.values())
            print(f"Truncating all lists to minimum length: {min_length}")
            for key in data[foot]:
                data[foot][key] = data[foot][key][:min_length]

    for foot in ['left', 'right']:
        print(f"\nFinal length of lists for {foot} foot:")
        for key, length in lengths.items():
            print(f"{key}: {length}")
        print("\n")

    # Convert dictionaries to DataFrames
    for foot in ['left', 'right']:
        try:
            df = pd.DataFrame(data[foot])

            # Change column names
            # a* to accel_*, g* to gyro_*, and q* to quaternion_*
            for key in df.columns:
                if key.startswith('a'):
                    df.rename(columns={key: key.replace('a', 'accel_')}, inplace=True)
                elif key.startswith('g'):
                    df.rename(columns={key: key.replace('g', 'gyro_')}, inplace=True)
                elif key.startswith('q'):
                    df.rename(columns={key: key.replace('q', 'quaternion_')}, inplace=True)
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save DataFrame to CSV
            df.to_csv(os.path.join(output_dir, f"{foot}_foot_data.csv"), index=False)
            print(f"\nSuccessfully created CSV for {foot} foot")
        except ValueError as e:
            print(f"\nError creating DataFrame for {foot} foot: {str(e)}")
            print("Data shape:")
            for key, value in data[foot].items():
                print(f"{key}: {len(value)}")

if __name__ == "__main__":
    start = time.time()
    if len(sys.argv) != 3:
        print("Usage: python parse_insole_csv.py <input-file> <output-dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    # Parse the data and save to the specified output directory
    parse_imu_data(input_file, output_dir)

    print("\nData parsing and export complete.")
    
    end = time.time()
    print(f"Time elapsed: {end - start:.2f} seconds.")