import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from data.dataset_v0 import GaitDatasetV0
from data.dataset_v1 import GaitDatasetV1
from model.phase_estimator import ImprovedGaitLSTM
from model.time_predictor import StrideTimeEstimatorLSTM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indirect_model_dir", type=str, required=True)
    parser.add_argument("--direct_model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_args(experiment_dir):
    args_path = os.path.join(experiment_dir, "args.json")
    with open(args_path, "r") as f:
        return argparse.Namespace(**json.load(f))


def run_baseline_inference(dataset, device):
    """
    Baseline method: Average the last three gait cycles to predict the current gait cycle duration.
    """
    predictions = []
    ground_truths = []

    # Temporary list to store stride times for averaging
    stride_times = []

    # DataLoader setup for inference
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for sequences, current_duration in dataloader:
        stride_times.append(current_duration.item())

        # If there are at least 3 stride times, make a prediction
        if len(stride_times) >= 3:
            # Take the average of the last 3 stride times
            predicted_stride_time = np.mean(stride_times[-3:])
            predictions.append(predicted_stride_time)
            ground_truths.append(current_duration.item())

    # Convert to numpy arrays
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    return predictions, ground_truths


def run_indirect_inference(dataset, indirect_model_dir, device):
    """
    Indirect method: Predict the phase of the gait cycle and perform linear regression to predict stride time.
    """
    # Load model arguments and the indirect model itself
    model_args = load_args(indirect_model_dir)
    indirect_model = ImprovedGaitLSTM(
        model_args.input_size,
        model_args.hidden_size,
        model_args.num_layers,
        model_args.dropout,
    )
    indirect_model.load_state_dict(
        torch.load(
            os.path.join(indirect_model_dir, "best_model.pth"), map_location=device
        )
    )
    indirect_model.to(device)
    indirect_model.eval()

    predictions = []
    ground_truths = []

    # Assuming the dataset is already loaded, process it
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for inputs, actual_time in dataloader:
            inputs = inputs.to(device)
            phase_predictions = indirect_model.get_phase(inputs)

            # Convert phase predictions for regression
            phase_predictions = phase_predictions.cpu().numpy().flatten()
            phase_predictions = np.where(
                phase_predictions > 0.5, phase_predictions - 1, phase_predictions
            )

            # Perform linear regression to predict stride time
            X = np.arange(len(phase_predictions)).reshape(-1, 1)
            y = phase_predictions
            reg = LinearRegression().fit(X, y)
            slope = reg.coef_[0]
            intercept = reg.intercept_

            predicted_stride_time = (
                (1 - intercept) / slope if slope != 0 else float("inf")
            )
            predicted_stride_time /= model_args.sample_rate
            predictions.append(predicted_stride_time)
            ground_truths.append(actual_time.item())

    return np.array(predictions), np.array(ground_truths)


def run_direct_inference(dataset, direct_model_dir, device):
    """
    Direct method: Directly predict the stride time using a neural network.
    """
    # Load model arguments
    model_args = load_args(direct_model_dir)

    # Initialize the model
    model = StrideTimeEstimatorLSTM(
        model_args.input_size,
        model_args.hidden_size,
        model_args.num_layers,
        model_args.dropout,
    )

    # Load trained model checkpoint
    checkpoint_path = os.path.join(direct_model_dir, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # DataLoader setup for inference
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    predictions, actual_times = [], []

    # Run inference
    with torch.no_grad():
        for sequences, time_labels in test_loader:
            sequences = sequences.to(device)
            predicted_times = model(sequences)

            predictions.append(predicted_times.squeeze().cpu().numpy())
            actual_times.append(time_labels.numpy())

    return np.array(predictions), np.array(actual_times)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # List the subdirectories in the data directory
    data_dirs = [
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]

    # Test for each scenario
    for data_dir in data_dirs:
        # Initialize the dataset
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

        # Run baseline inference
        baseline_predictions, baseline_ground_truths = run_baseline_inference(
            baseline_dataset, device
        )

        # Run indirect inference
        indirect_predictions, indirect_ground_truths = run_indirect_inference(
            method_dataset, args.indirect_model_dir, device
        )

        # Run direct inference
        # direct_predictions, direct_ground_truths = run_direct_inference(
        #     method_dataset, args.direct_model_dir, device
        # )

        # Calculate R^2 scores
        baseline_r2 = r2_score(baseline_ground_truths, baseline_predictions)
        indirect_r2 = r2_score(indirect_ground_truths, indirect_predictions)
        # direct_r2 = r2_score(direct_ground_truths, direct_predictions)

        # Calculate MAE scores
        baseline_mae = np.mean(np.abs(baseline_ground_truths - baseline_predictions))
        indirect_mae = np.mean(np.abs(indirect_ground_truths - indirect_predictions))
        # direct_mae = np.mean(np.abs(direct_ground_truths - direct_predictions))

        # Compare baseline, indirect, and direct method R^2 scores and MAE scores
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].bar(
            np.arange(2),
            [baseline_r2, indirect_r2],
            tick_label=["Baseline", "Indirect"],
        )
        ax[0].set_title("R^2 Scores")

        ax[1].bar(
            np.arange(2),
            [baseline_mae, indirect_mae],
            tick_label=["Baseline", "Indirect"],
        )
        ax[1].set_title("Mean Absolute Error")

        # Print the relative improvement in R^2 score and MAE
        mae_reduction = (baseline_mae - indirect_mae) / baseline_mae * 100
        r2_improvement = (indirect_r2 - baseline_r2) / baseline_r2 * 100

        print(data_dir)
        print(f"MAE Reduction: {mae_reduction:.2f}%")
        print(f"R^2 Improvement: {r2_improvement:.2f}%")

        # Save the plot
        plt.savefig(os.path.join(data_dir, "validation_results.png"))


if __name__ == "__main__":
    main()