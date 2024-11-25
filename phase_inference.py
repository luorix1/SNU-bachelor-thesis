import argparse
import os
import json
import logging
import torch
import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Subset

from data.dataset_v1 import GaitDatasetV1
from model.phase_estimator import ImprovedGaitLSTM
from utils.utils import load_split_indices


def load_args(experiment_dir, experiment_name):
    args_path = os.path.join(experiment_dir, experiment_name, "args.json")
    with open(args_path, "r") as f:
        return argparse.Namespace(**json.load(f))
    

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the gait cycle regressor")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments",
        help="Directory of the experiment",
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Name of the experiment"
    )

    return parser.parse_args()


def evaluate_regressor(args):
    # Set up logging
    log_file = os.path.join(
        args.experiment_dir, args.experiment_name, "regressor_evaluation.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the model
    model = ImprovedGaitLSTM(
        args.input_size, args.hidden_size, args.num_layers, args.dropout
    )

    model.load_state_dict(
        torch.load(
            os.path.join(args.experiment_dir, args.experiment_name, "best_model.pth"),
            map_location=device,
        )
    )
    model.to(device)
    model.eval()

    # Set up dataset and dataloader
    data_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    dataset = GaitDatasetV1(data_dirs, window_size=args.window_size, label="time")

    # Load test indices
    _, test_indices = load_split_indices(args.data_dir)

    # Create a subset of the dataset using the test indices
    test_dataset = Subset(dataset, test_indices)

    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predicted_stride_times = []
    actual_stride_times = []

    with torch.no_grad():
        for inputs, actual_time in dataloader:
            inputs = inputs.to(device)
            phase_predictions = model.get_phase(inputs)
            
            # Convert to numpy for linear regression
            phase_predictions = phase_predictions.cpu().numpy().flatten()

            # FIXME: Since we're using the start of each gait cycle, values over 0.5 should subtract 1
            phase_predictions = np.where(
                phase_predictions > 0.5, phase_predictions - 1, phase_predictions
            )

            # Perform linear regression
            X = np.arange(len(phase_predictions)).reshape(-1, 1)
            y = phase_predictions
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            intercept = reg.intercept_

            # Calculate predicted stride time
            predicted_stride_time = (
                (1 - intercept) / slope if slope != 0 else float("inf")
            )
            predicted_stride_time /= args.sample_rate  # Convert from frames to seconds
            predicted_stride_times.append(predicted_stride_time)
            actual_stride_times.append(actual_time.item())

    # Calculate metrics
    mse = np.mean(
        (np.array(predicted_stride_times) - np.array(actual_stride_times)) ** 2
    )
    mae = np.mean(
        np.abs(np.array(predicted_stride_times) - np.array(actual_stride_times))
    )
    r2 = r2_score(actual_stride_times, predicted_stride_times)

    logging.info(f"Mean Squared Error: {mse:.4f}")
    logging.info(f"Mean Absolute Error: {mae:.4f}")
    logging.info(f"R-squared: {r2:.4f}")

    # Set font sizes globally for all plots
    plt.rcParams.update({
        "font.size": 14,  # Default font size
        "axes.titlesize": 18,  # Title font size
        "axes.labelsize": 16,  # Axis labels font size
        "xtick.labelsize": 14,  # X-tick labels font size
        "ytick.labelsize": 14,  # Y-tick labels font size
        "legend.fontsize": 14,  # Legend font size
        "figure.titlesize": 20,  # Figure title font size
    })

    # Plot a histogram of the errors
    errors = np.array(actual_stride_times) - np.array(predicted_stride_times)
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Errors")
    plt.show()

    # Save metrics to JSON
    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
    }
    with open(
        os.path.join(
            args.experiment_dir, args.experiment_name, "regressor_metrics.json"
        ),
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_stride_times, predicted_stride_times, alpha=0.5)
    plt.plot(
        [min(actual_stride_times), max(actual_stride_times)],
        [min(actual_stride_times), max(actual_stride_times)],
        "r--",
        lw=2,
    )
    plt.xlabel("Actual Stride Time (s)")
    plt.ylabel("Predicted Stride Time (s)")
    plt.title(f"Predicted vs Actual Stride Times\nR² = {r2:.4f}")
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(
        args.experiment_dir,
        args.experiment_name,
        f"regression_stride_time_plot.png",
    )
    plt.savefig(plot_path)
    logging.info(f"Stride time plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    args = parse_args()

    # Load the original arguments
    original_args = load_args(args.experiment_dir, args.experiment_name)

    # Update the original arguments with the new test-specific arguments
    vars(original_args).update(vars(args))

    # Run the evaluation
    evaluate_regressor(original_args)

    print(
        f"Evaluation completed for {os.path.join(args.experiment_dir, args.experiment_name)}"
    )