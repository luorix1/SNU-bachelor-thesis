import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--sample_rate", type=int, required=True, help="Sample rate of the data")
    return parser.parse_args()

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_gyro_data(time, data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=data, mode='lines', name='gyro_z'))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Angular Velocity (deg/s)')
    fig.show()

def find_index(array, value):
    try:
        return np.where(array == value)[0][0]
    except IndexError:
        return None

def detect_gait_events(gyro_z, time_array, threshold=80, max_distance=10):
    """
    Detect gait events based on sagittal plane gyroscope data.
    """
    # Step 1: Find threshold indices with negative slope
    threshold_indices = []
    for i in range(1, len(gyro_z) - 8):
        if gyro_z[i] >= threshold and gyro_z[i + 1] < threshold:  # Value above threshold, next value below threshold
            if gyro_z[i] > gyro_z[i + 8]: # To remove noise
                # If too close to previous threshold, skip
                if len(threshold_indices) > 0 and i - threshold_indices[-1] < 30:
                    continue
                threshold_indices.append(i)
    
    if len(threshold_indices) < 2:
        raise ValueError("Not enough threshold indices with negative slope found.")
    
    print("Threshold indices with negative slope:")
    for index in threshold_indices:
        print(f"Time: {time_array[index]}, Value: {gyro_z[index]:.2f}")
    
    # Filter out threshold indices that are too close to each other (minimum 15)
    filtered_threshold_indices = [threshold_indices[0]]
    for i in range(1, len(threshold_indices)):
        if threshold_indices[i] - filtered_threshold_indices[-1] >= 15:
            filtered_threshold_indices.append(threshold_indices[i])
    threshold_indices = filtered_threshold_indices

    # Print the threshold time_array values
    print("Threshold indices with negative slope:")
    for index in threshold_indices:
        print(f"Time: {time_array[index]}, Value: {gyro_z[index]:.2f}")

    # Show the thresholds on the gyro data plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_array, y=gyro_z, mode='lines', name='Gyro Z'))
    fig.add_trace(go.Scatter(
        x=[time_array[i] for i in threshold_indices], 
        y=[gyro_z[i] for i in threshold_indices], 
        mode='markers', 
        name='Threshold', 
        marker=dict(color='red', size=10, symbol='circle')
    ))
    fig.update_layout(
        title="Gyro Z Data with Thresholds",
        xaxis_title="Time (s)",
        yaxis_title="Angular Velocity (deg/s)",
        legend_title="Events"
    )
    fig.show()
    
    events = []
    for i in range(len(threshold_indices) - 1):
        start, end = threshold_indices[i], threshold_indices[i + 1]
        section = gyro_z[start:end]
        section_time = time_array[start:end]
        
        # Find all negative peaks in the section
        negative_peaks, _ = find_peaks(-section)
        if len(negative_peaks) > 0:
            # Ensure there are enough peaks for clustering
            sorted_peaks = sorted(negative_peaks, key=lambda p: section[p])
            
            # Cluster peaks into two groups based on max_distance
            clustered_peaks = [[], []]
            for peak in sorted_peaks:
                if len(clustered_peaks[0]) == 0 or abs(peak - clustered_peaks[0][-1]) < max_distance:
                    clustered_peaks[0].append(peak)
                else:
                    clustered_peaks[1].append(peak)

            # Ensure that clustered_peaks[1] contains the smaller indices compared to clustered_peaks[0]
            if clustered_peaks[0][0] < clustered_peaks[1][0]:
                clustered_peaks = clustered_peaks[::-1]
            
            if len(clustered_peaks[0]) > 0 and len(clustered_peaks[1]) > 0:
                # Heelstrike is the most negative peak in cluster with smaller indices
                heelstrike = min(clustered_peaks[1], key=lambda p: section[p])

                # Toeoff is the most negative peak in cluster with larger indices
                toeoff = min(clustered_peaks[0], key=lambda p: section[p])
                
                events.append({
                    "heelstrike_time": section_time[heelstrike],
                    "heelstrike_value": section[heelstrike],
                    "toeoff_time": section_time[toeoff],
                    "toeoff_value": section[toeoff]
                })
    return events

def plot_gait_events(time, gyro_z, events, title):
    """
    Plots the gyro data with heelstrike and toeoff events overlaid.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=gyro_z, mode='lines', name='Gyro Z'))
    
    heelstrike_times = [event["heelstrike_time"] for event in events]
    heelstrike_values = [event["heelstrike_value"] for event in events]
    fig.add_trace(go.Scatter(
        x=heelstrike_times, 
        y=heelstrike_values, 
        mode='markers', 
        name='Heelstrike', 
        marker=dict(color='blue', size=10, symbol='circle')
    ))
    
    toeoff_times = [event["toeoff_time"] for event in events]
    toeoff_values = [event["toeoff_value"] for event in events]
    fig.add_trace(go.Scatter(
        x=toeoff_times, 
        y=toeoff_values, 
        mode='markers', 
        name='Toeoff', 
        marker=dict(color='red', size=10, symbol='square')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Angular Velocity (deg/s)",
        legend_title="Events"
    )
    fig.show()

def edit_gait_events(events, time_array, gyro_z):
    """
    Allow user to edit the detected gait events by adding, replacing, or removing events interactively.
    """
    while True:
        print("\nSelect an option:")
        print("1: Add Missing Gait Cycle (HS and TO)")
        print("2: Replace Heelstrike (HS)")
        print("3: Replace Toeoff (TO)")
        print("4: Delete Gait Cycle Using Heelstrike (HS)")
        print("5: Delete Gait Cycle Using Toeoff (TO)")
        print("6: Done Editing")

        option = input("Enter your choice (1/2/3/4/5/6): ").strip()

        if option == '1':  # Add missing gait cycle
            print("Adding a missing gait cycle.")
            hs_timestamp = input("Enter the Heelstrike (HS) timestamp: ")
            to_timestamp = input("Enter the Toeoff (TO) timestamp: ")

            hs_index = find_index(time_array, hs_timestamp)
            to_index = find_index(time_array, to_timestamp)

            if hs_index is not None and to_index is not None:
                events.append({
                    "heelstrike_time": time_array[hs_index],
                    "heelstrike_value": gyro_z[hs_index],
                    "toeoff_time": time_array[to_index],
                    "toeoff_value": gyro_z[to_index]
                })
                print(f"Added gait cycle with HS at {hs_timestamp} and TO at {to_timestamp}.")
            else:
                print("Invalid timestamps. Ensure the timestamps match the available time data.")

        elif option == '2':  # Replace Heelstrike
            print("Replacing a Heelstrike (HS).")
            to_timestamp = input("Enter the Toeoff (TO) timestamp of the gait cycle to modify: ")
            new_hs_timestamp = input("Enter the new Heelstrike (HS) timestamp: ")

            to_index = find_index(time_array, to_timestamp)
            new_hs_index = find_index(time_array, new_hs_timestamp)

            if to_index is not None and new_hs_index is not None:
                for event in events:
                    if event.get("toeoff_time") == time_array[to_index]:
                        event["heelstrike_time"] = time_array[new_hs_index]
                        event["heelstrike_value"] = gyro_z[new_hs_index]
                        print(f"Replaced HS at {to_timestamp} with new HS at {new_hs_timestamp}.")
                        break
            else:
                print("Invalid timestamps. Ensure the timestamps match the available time data.")

        elif option == '3':  # Replace Toeoff
            print("Replacing a Toeoff (TO).")
            hs_timestamp = input("Enter the Heelstrike (HS) timestamp of the gait cycle to modify: ")
            new_to_timestamp = input("Enter the new Toeoff (TO) timestamp: ")

            hs_index = find_index(time_array, hs_timestamp)
            new_to_index = find_index(time_array, new_to_timestamp)

            if hs_index is not None and new_to_index is not None:
                for event in events:
                    if event.get("heelstrike_time") == time_array[hs_index]:
                        event["toeoff_time"] = time_array[new_to_index]
                        event["toeoff_value"] = gyro_z[new_to_index]
                        print(f"Replaced TO at {hs_timestamp} with new TO at {new_to_timestamp}.")
                        break
            else:
                print("Invalid timestamps. Ensure the timestamps match the available time data.")

        elif option == '4': # Remove gait cycle based on HS timestamp
            print("Removing a gait cycle.")
            hs_timestamp = input("Enter the Heelstrike (HS) timestamp of the gait cycle to remove: ")
            hs_index = find_index(time_array, hs_timestamp)

            if hs_index is not None:
                for event in events:
                    if event.get("heelstrike_time") == time_array[hs_index]:
                        events.remove(event)
                        print(f"Removed gait cycle with HS at {hs_timestamp}.")
                        break
            else:
                print("Invalid timestamp. Ensure the timestamp matches the available time data.")
        elif option == '5': # Remove gait cycle based on TO timestamp
            print("Removing a gait cycle.")
            to_timestamp = input("Enter the Heelstrike (HS) timestamp of the gait cycle to remove: ")
            to_index = find_index(time_array, to_timestamp)

            if to_index is not None:
                for event in events:
                    if event.get("toeoff_time") == time_array[to_index]:
                        events.remove(event)
                        print(f"Removed gait cycle with TO at {to_timestamp}.")
                        break
            else:
                print("Invalid timestamp. Ensure the timestamp matches the available time data.")
        elif option == '6': # Done editing
            print("Finishing editing...")
            break

        else:
            print("Invalid option. Please try again.")

    return events

def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for foot in ['left', 'right']:
        process_foot = input(f"Do you want to process the {foot} foot? (y/n): ").lower()
        if process_foot != 'y':
            print(f"Skipping {foot} foot processing.")
            continue

        data = load_data(input_dir / f'{foot}_foot_data.csv')
        
        if foot == 'left':
            # Adjust axes for the left foot
            data['accel_x'] = data['accel_x']
            data['accel_y'] = -data['accel_y']
            data['accel_z'] = -data['accel_z']
            data['gyro_x'] = data['gyro_x']
            data['gyro_y'] = -data['gyro_y']
            data['gyro_z'] = -data['gyro_z']
        
        gyro_z = data['gyro_z'].values
        time_array = data['time'].values
        
        plot_gyro_data(time_array, gyro_z, f'{foot.capitalize()} Foot Gyro Z Data')

        # Detect gait events
        events = detect_gait_events(gyro_z, time_array, threshold=80, max_distance=10)

        print(events)
        
        # Plot detected events
        plot_gait_events(time_array, gyro_z, events, f'{foot.capitalize()} Foot Gait Events')

        # Allow the user to edit the gait events
        events = edit_gait_events(events, time_array, gyro_z)

        print(events)
        
        # Plot the edited events
        plot_gait_events(time_array, gyro_z, events, f'Edited {foot.capitalize()} Foot Gait Events')
        
        # Annotate data with edited gait events
        data['HS'] = 0
        data['TO'] = 0
        for event in events:
            hs_index = find_index(time_array, event["heelstrike_time"])
            to_index = find_index(time_array, event["toeoff_time"])
            if hs_index is not None: data.loc[hs_index, 'HS'] = 1
            if to_index is not None: data.loc[to_index, 'TO'] = 1
        
        # Save annotated data
        output_file = output_dir / f'{foot}_foot_data_annotated.csv'
        data.to_csv(output_file, index=False)
        print(f"Annotated data with gait events saved to {output_file}")

if __name__ == "__main__":
    main()
