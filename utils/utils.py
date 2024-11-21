import os


# Function to load indices from .h5 file
def load_split_indices(data_dir):
    with open(os.path.join(data_dir, "train_indices.txt"), "r") as f:
        train_indices = [int(x) for x in f.readlines()]

    with open(os.path.join(data_dir, "test_indices.txt"), "r") as f:
        test_indices = [int(x) for x in f.readlines()]

    return train_indices, test_indices