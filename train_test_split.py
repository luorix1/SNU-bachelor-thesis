import numpy as np
import os
import torch

from data.dataset_v2 import GaitDatasetV2


# Parse args
def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data_ema_swing_phase_train",
        help="Directory containing the data files",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=120,
        help="Sample rate of the data",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Percentage of data to use for training",
    )
    parser.add_argument(
        "--use_filtered",
        action="store_true",
        help="Use filtered data",
    )
    return parser.parse_args()


# Save train/test split indices
def split_data(data_dirs, args):
    # Set up dataset and dataloader
    dataset = GaitDatasetV2(
        data_dirs,
        window_size=12,
    )
    train_size = int(args.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Convert train/test indices to numpy arrays and save to h5 file
    train_indices = np.array(train_dataset.indices).astype(np.int32)
    test_indices = np.array(test_dataset.indices).astype(np.int32)

    print(f"Train size: {len(train_indices)}")
    print(f"Test size: {len(test_indices)}")

    print(train_indices.dtype)
    print(test_indices.dtype)

    # Save to .txt files
    np.savetxt(os.path.join(args.data_dir, "train_indices.txt"), train_indices, fmt="%d")
    np.savetxt(os.path.join(args.data_dir, "test_indices.txt"), test_indices, fmt="%d")
    

if __name__ == "__main__":
    args = parse_args()

    # Get all subdirectories in the data_dir that start with "seq_"
    data_dirs = [
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("seq_")
    ]

    split_data(data_dirs, args)