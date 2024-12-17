import time
import os
import torch
import argparse
from data.dataset_v0 import GaitDatasetV0
from data.dataset_v1 import GaitDatasetV1
from run_validation import run_baseline_inference, run_indirect_inference, run_direct_inference, parse_args

def measure_inference_fps(inference_function, dataset, *args):
    """
    Measures the FPS of a given inference function.
    :param inference_function: Function to run inference (e.g., run_baseline_inference)
    :param dataset: Dataset to run inference on
    :param args: Additional arguments for the inference function
    :return: FPS (frames per second)
    """
    start_time = time.time()
    predictions, ground_truths = inference_function(dataset, *args)
    end_time = time.time()

    total_time = end_time - start_time
    num_samples = len(dataset)
    fps = num_samples / total_time

    return fps


def main():
    args = parse_args()
    device = torch.device(args.device)

    # List the subdirectories in the data directory
    data_dirs = [
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]

    for data_dir in data_dirs:
        print(f"Testing inference FPS for dataset: {data_dir}")

        # Initialize datasets
        baseline_dataset = GaitDatasetV0(
            [data_dir],
            sample_rate=120,
            use_filtered=False,
        )
        method_dataset = GaitDatasetV1(
            [data_dir],
            window_size=12,
            sample_rate=120,
            use_filtered=False,
            label="time",
        )

        # Measure FPS for baseline inference
        baseline_fps = measure_inference_fps(run_baseline_inference, baseline_dataset, device)
        print(f"Baseline Inference FPS: {baseline_fps:.2f}")

        # Measure FPS for indirect inference
        indirect_fps = measure_inference_fps(run_indirect_inference, method_dataset, args.indirect_model_dir, device)
        print(f"Indirect Inference FPS: {indirect_fps:.2f}")

        # Measure FPS for direct inference
        direct_fps = measure_inference_fps(run_direct_inference, method_dataset, args.direct_model_dir, device)
        print(f"Direct Inference FPS: {direct_fps:.2f}")

        print()

if __name__ == "__main__":
    main()
