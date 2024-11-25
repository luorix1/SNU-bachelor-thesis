import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.dataset_v0 import GaitDatasetV0

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="gait_ema_merged",
        help="Directory containing the data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="baseline_results",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--use_filtered",
        action="store_true",
        help="Use filtered IMU data instead of raw",
    )
    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get all subdirectories in the data_dir that start with "seq_"
    data_dirs = [
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and d.startswith("seq_")
    ]

    performance_metrics = {}

    # Lists to collect data for the global scatter plot
    all_predictions = []
    all_ground_truths = []

    for sub_dir in data_dirs:
        print(f"Processing {sub_dir}...")

        # Create a subdirectory in the output directory
        os.makedirs(os.path.join(args.output_dir, os.path.basename(sub_dir)), exist_ok=True)

        # Load the dataset for the current subdirectory
        dataset = GaitDatasetV0(data_dirs=[sub_dir], use_filtered=args.use_filtered)

        try:
            predictions = []
            ground_truths = []

            for i in range(len(dataset)):
                avg_last_3, current_duration = dataset[i]
                predictions.append(avg_last_3)
                ground_truths.append(current_duration)

            predictions = np.array(predictions)
            ground_truths = np.array(ground_truths)

            # Append to global lists
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)

            # Calculate metrics
            mse = mean_squared_error(ground_truths, predictions)
            mae = mean_absolute_error(ground_truths, predictions)
            r2 = r2_score(ground_truths, predictions)

            abs_diff = np.abs(predictions - ground_truths)
            mae_excluding_severe = np.mean(abs_diff[abs_diff <= 0.15])
            success_percentage = len(abs_diff[abs_diff <= 0.05]) / len(abs_diff) * 100

            # Save metrics for the current subdirectory
            performance_metrics[sub_dir] = {
                "Mean Squared Error": mse,
                "Mean Absolute Error": mae,
                "R-squared": r2,
                "MAE Excluding Severe Cases": mae_excluding_severe,
                "Success Percentage (<50ms Error)": success_percentage,
            }

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

            # Plot predicted vs actual gait cycle durations
            plt.figure(figsize=(10, 5))
            plt.scatter(ground_truths, predictions, alpha=0.5, label="Predictions")
            plt.plot(
                [min(ground_truths), max(ground_truths)],
                [min(ground_truths), max(ground_truths)],
                "r--",
                label="Ideal Fit",
            )
            plt.xlabel("Actual Gait Cycle Duration (s)")
            plt.ylabel("Predicted Gait Cycle Duration (s)")
            plt.title(f"Predicted vs Actual Gait Cycle Duration - {os.path.basename(sub_dir)}")
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, os.path.basename(sub_dir), "predicted_vs_actual_duration.png"))
            plt.close()

            # Plot predicted and ground truth durations for the entire sequence
            plt.figure(figsize=(15, 5))
            plt.plot(ground_truths, label="Ground Truth", alpha=0.7)
            plt.plot(predictions, label="Predicted", alpha=0.7)
            plt.xlabel("Sample")
            plt.ylabel("Gait Cycle Duration (s)")
            plt.title(f"Predicted and Ground Truth Gait Cycle Duration - {os.path.basename(sub_dir)}")
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, os.path.basename(sub_dir), "duration_sequence.png"))
            plt.close()
        except Exception as e:
            print(f"Error processing {sub_dir}: {e}")
            continue

    # Save all performance metrics to a JSON file
    with open(os.path.join(args.output_dir, "performance_metrics.json"), "w") as f:
        # Order the dictionary in order of largest MAE
        performance_metrics = dict(
            sorted(performance_metrics.items(), key=lambda item: item[1]["Mean Absolute Error"], reverse=True)
        )
        json.dump(performance_metrics, f, indent=4)

    # Calculate global metrics
    all_ground_truths = np.array(all_ground_truths)
    all_predictions = np.array(all_predictions)

    mse = mean_squared_error(all_ground_truths, all_predictions)
    mae = mean_absolute_error(all_ground_truths, all_predictions)
    r2 = r2_score(all_ground_truths, all_predictions)

    abs_diff = np.abs(all_predictions - all_ground_truths)
    mae_excluding_severe = np.mean(abs_diff[abs_diff <= 0.15])
    success_percentage = len(abs_diff[abs_diff <= 0.05]) / len(abs_diff) * 100

    print(f"Global Mean Squared Error: {mse:.4f}")
    print(f"Global Mean Absolute Error: {mae:.4f}")
    print(f"Global R-squared: {r2:.4f}")
    print(f"Global MAE Excluding Severe Cases: {mae_excluding_severe:.4f}")
    print(f"Global Success Percentage (<50ms Error): {success_percentage:.2f}%")

    # Save global metrics to a JSON file
    global_metrics = {
        "Mean Squared Error": mse,
        "Mean Absolute Error": mae,
        "R-squared": r2,
        "MAE Excluding Severe Cases": mae_excluding_severe,
        "Success Percentage (<50ms Error)": success_percentage,
    }

    with open(os.path.join(args.output_dir, "global_performance_metrics.json"), "w") as f:
        json.dump(global_metrics, f, indent=4)

    # Global scatter plot for all data
    plt.figure(figsize=(10, 8))
    plt.scatter(all_ground_truths, all_predictions, alpha=0.5, label="Predictions")
    plt.plot(
        [min(all_ground_truths), max(all_ground_truths)],
        [min(all_ground_truths), max(all_ground_truths)],
        "r--",
        label="y = x (Ideal Fit)",
    )
    plt.xlabel("Actual Gait Cycle Duration (s)")
    plt.ylabel("Predicted Gait Cycle Duration (s)")
    plt.title(f"Predicted vs Actual Gait Cycle Durations\nR² = {r2:.4f}")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "global_predicted_vs_actual_duration.png"))
    plt.close()

    print(f"Performance metrics saved to performance_metrics.json")