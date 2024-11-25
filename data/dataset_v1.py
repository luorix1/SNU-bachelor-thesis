"""
Dataset class
"""

import pandas as pd
import numpy as np
import torch

from datetime import datetime
from pathlib import Path

from torch.utils.data import Dataset


class GaitDatasetV1(Dataset):
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
            for csv_file in data_dir.glob("*_foot_data_annotated.csv"):
                df = pd.read_csv(csv_file)

                if self.use_filtered:
                    features = df[
                        [
                            "filtered_quaternion_x",
                            "filtered_quaternion_y",
                            "filtered_quaternion_z",
                            "filtered_quaternion_w",
                            "filtered_accel_x",
                            "filtered_accel_y",
                            "filtered_accel_z",
                            "filtered_gyro_x",
                            "filtered_gyro_y",
                            "filtered_gyro_z",
                        ]
                    ].values
                else:
                    features = df[
                        [
                            "quaternion_x",
                            "quaternion_y",
                            "quaternion_z",
                            "quaternion_w",
                            "accel_x",
                            "accel_y",
                            "accel_z",
                            "gyro_x",
                            "gyro_y",
                            "gyro_z",
                        ]
                    ].values

                times = df["time"].values

                # Find the TO gait events: rows with TO = 1
                to_indices = np.where(df["TO"] == 1)[0]

                for i in range(len(to_indices) - 1):
                    to_index = to_indices[i]
                    next_to_index = to_indices[i + 1]

                    # Extract the gait cycle between two TO events
                    cycle_data = features[to_index:next_to_index]

                    # Calculate uniformly increasing phase for the gait cycle with to_index as 0 and next_to_index as 1
                    cycle_phase = np.linspace(0, 1, next_to_index - to_index)

                    # If the cycle is shorter than window_size, discard
                    if len(cycle_data) < self.window_size:
                        continue

                    # Calculate stride time
                    start_time = datetime.strptime(times[to_index], "%H:%M:%S.%f")
                    end_time = datetime.strptime(times[next_to_index], "%H:%M:%S.%f")
                    duration = (end_time - start_time).total_seconds()

                    # If duration is more than 2 seconds or less than 0.6 seconds, discard
                    if duration > 2 or duration < 0.6:
                        continue

                    self.sequences.append(cycle_data[: self.window_size])
                    self.full_sequences.append(cycle_data)
                    if self.label == "time":
                        self.time_labels.append(duration)
                    elif self.label == "phase":
                        self.phase_labels.append(cycle_phase[:self.window_size])

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