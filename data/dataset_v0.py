import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path

from torch.utils.data import Dataset


class GaitDatasetV0(Dataset):
    def __init__(
        self,
        data_dirs,
        sample_rate=120,
        use_filtered=False,
    ):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.sample_rate = sample_rate
        self.use_filtered = use_filtered

        self.durations = []  # Store gait cycle durations
        self.features = []  # Store features for completeness

        self._load_data()

    def _load_data(self):
        for data_dir in self.data_dirs:
            for csv_file in data_dir.glob("*_foot_data_annotated.csv"):
                df = pd.read_csv(csv_file)

                # Select filtered or unfiltered features
                imu_columns = (
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
                    if self.use_filtered
                    else [
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
                )

                features = df[imu_columns].values
                times = df["time"].values

                # Find TO events
                to_indices = np.where(df["TO"] == 1)[0]

                for i in range(len(to_indices) - 1):
                    to_index = to_indices[i]
                    next_to_index = to_indices[i + 1]

                    # Calculate TO-TO duration
                    start_time = datetime.strptime(times[to_index], "%H:%M:%S.%f")
                    end_time = datetime.strptime(times[next_to_index], "%H:%M:%S.%f")
                    duration = (end_time - start_time).total_seconds()

                    # Discard cycles outside expected duration range
                    if duration > 2 or duration < 0.6:
                        continue

                    self.durations.append(duration)
                    self.features.append(features[to_index:next_to_index])

        print(f"Loaded {len(self.durations)} gait cycles.")

    def __len__(self):
        # We can only start predicting from the 4th cycle
        return len(self.durations) - 3

    def __getitem__(self, idx):
        # Calculate the average duration of the last three cycles
        avg_last_3 = np.mean(self.durations[idx : idx + 3])
        current_duration = self.durations[idx + 3]
        return avg_last_3, current_duration
