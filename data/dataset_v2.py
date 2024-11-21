import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset

class GaitDatasetV2(Dataset):
    def __init__(
        self,
        data_dirs,
        window_size=12,
        sample_rate=120,
        use_filtered=False,
        label="phase",
    ):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.use_filtered = use_filtered
        self.label = label

        self.sequences = []
        self.full_sequences = []
        self.time_labels = []
        self.phase_labels = []

        self._load_data()

    def _load_data(self):
        for data_dir in self.data_dirs:
            # Iterate through all CSV files named "synced_data_annotated.csv" in each subdirectory
            for csv_file in data_dir.glob("synced_data_annotated.csv"):
                df = pd.read_csv(csv_file)

                # Select filtered or unfiltered features
                if self.use_filtered:
                    features = df[
                        [
                            "filtered_quaternion_x_left", "filtered_quaternion_y_left", "filtered_quaternion_z_left", "filtered_quaternion_w_left",
                            "filtered_accel_x_left", "filtered_accel_y_left", "filtered_accel_z_left", "filtered_gyro_x_left", "filtered_gyro_y_left", "filtered_gyro_z_left",
                            "filtered_quaternion_x_right", "filtered_quaternion_y_right", "filtered_quaternion_z_right", "filtered_quaternion_w_right",
                            "filtered_accel_x_right", "filtered_accel_y_right", "filtered_accel_z_right", "filtered_gyro_x_right", "filtered_gyro_y_right", "filtered_gyro_z_right",
                        ]
                    ].values
                else:
                    features = df[
                        [
                            "quaternion_x_left", "quaternion_y_left", "quaternion_z_left", "quaternion_w_left",
                            "accel_x_left", "accel_y_left", "accel_z_left", "gyro_x_left", "gyro_y_left", "gyro_z_left",
                            "quaternion_x_right", "quaternion_y_right", "quaternion_z_right", "quaternion_w_right",
                            "accel_x_right", "accel_y_right", "accel_z_right", "gyro_x_right", "gyro_y_right", "gyro_z_right",
                        ]
                    ].values

                times = df["time"].values

                # Find the TO gait events: rows with TO_left = 1 or TO_right = 1
                to_left_indices = np.where(df["TO_left"] == 1)[0]
                to_right_indices = np.where(df["TO_right"] == 1)[0]

                # Find the HS gait events: rows with HS_left = 1 or HS_right = 1
                hs_left_indices = np.where(df["HS_left"] == 1)[0]
                hs_right_indices = np.where(df["HS_right"] == 1)[0]

                for i in range(len(to_left_indices) - 1):
                    to_left_index = to_left_indices[i]
                    next_to_left_index = to_left_indices[i + 1]

                    # Extract the gait cycle between two TO events for the left foot
                    cycle_data_left = features[to_left_index:next_to_left_index]

                    # Calculate stride time for the left foot
                    start_time = datetime.strptime(times[to_left_index], "%H:%M:%S.%f")
                    end_time = datetime.strptime(times[next_to_left_index], "%H:%M:%S.%f")
                    duration_left = (end_time - start_time).total_seconds()

                    # If the duration is more than 2 seconds or less than 0.6 seconds, discard
                    if duration_left > 2 or duration_left < 0.6:
                        continue
                    
                    try:
                        # Look for the last right foot TO event before the current left foot TO
                        closest_right_to_index = to_right_indices[to_right_indices < to_left_index][-1]
                        
                        # Find right HS event right after closest_right_to_index
                        right_hs_index = hs_right_indices[hs_right_indices > closest_right_to_index][0]
                    except:
                        continue
                    
                    # Extract the right foot swing phase data
                    cycle_data_right = features[closest_right_to_index:right_hs_index]
                    
                    # Combine the right foot swing phase and the start of the left foot swing phase
                    combined_data = np.concatenate([cycle_data_right[-self.window_size:], cycle_data_left[:self.window_size]], axis=0)

                    self.sequences.append(combined_data)
                    self.full_sequences.append(cycle_data_left)
                    self.time_labels.append(duration_left)

                    if self.label == "phase":
                        cycle_phase_left = np.linspace(0, 1, len(cycle_data_left))[:self.window_size]
                        self.phase_labels.append(cycle_phase_left)

                for i in range(len(to_right_indices) - 1):
                    to_right_index = to_right_indices[i]
                    next_to_right_index = to_right_indices[i + 1]

                    # Extract the gait cycle between two TO events for the right foot
                    cycle_data_right = features[to_right_index:next_to_right_index]

                    # Calculate stride time for the right foot
                    start_time = datetime.strptime(times[to_right_index], "%H:%M:%S.%f")
                    end_time = datetime.strptime(times[next_to_right_index], "%H:%M:%S.%f")
                    duration_right = (end_time - start_time).total_seconds()

                    # If the duration is more than 2 seconds or less than 0.6 seconds, discard
                    if duration_right > 2 or duration_right < 0.6:
                        continue

                    try:
                        # Look for the last left foot TO event before the current right foot TO
                        closest_left_to_index = to_left_indices[to_left_indices < to_right_index][-1]
                        
                        # Find left HS event right after closest_left_to_index
                        left_hs_index = hs_left_indices[hs_left_indices > closest_left_to_index][0]
                    except:
                        continue

                    # Extract the left foot swing phase data
                    cycle_data_left = features[closest_left_to_index:left_hs_index]

                    # Combine the left foot swing phase and the start of the right foot swing phase
                    combined_data = np.concatenate([cycle_data_left[-self.window_size:], cycle_data_right[:self.window_size]], axis=0)

                    self.sequences.append(combined_data)
                    self.full_sequences.append(cycle_data_right)
                    self.time_labels.append(duration_right)

                    if self.label == "phase":
                        cycle_phase_right = np.linspace(0, 1, len(cycle_data_right))[:self.window_size]
                        self.phase_labels.append(cycle_phase_right)

        print(f"Loaded {len(self.sequences)} sequences")
        print(f"Loaded {len(self.full_sequences)} full sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.label == "time":
            return (
                torch.tensor(self.sequences[idx], dtype=torch.float),
                torch.tensor(self.time_labels[idx], dtype=torch.float),
            )
        elif self.label == "phase":
            return (
                torch.tensor(self.sequences[idx], dtype=torch.float),
                torch.tensor(self.phase_labels[idx], dtype=torch.float),
            )
        else:
            raise ValueError("Invalid label type")
