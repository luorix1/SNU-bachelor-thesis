import argparse
import json
import logging
import numpy as np
import os
import shutil
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score

from data.dataset_v1 import GaitDatasetV1
from model.time_predictor import StrideTimeEstimatorLSTM
from utils.utils import load_split_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Stride Time Prediction")
    parser.add_argument("--experiment_dir", type=str, default="experiments", help="Directory to save experiments")
    parser.add_argument("--experiment_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Experiment name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing data")
    parser.add_argument("--window_size", type=int, default=12, help="Window size for each sequence")
    parser.add_argument("--sample_rate", type=int, default=120, help="Sampling rate of the data")
    parser.add_argument("--input_size", type=int, default=10, help="Input size of the model")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    return parser.parse_args()


def save_metrics(metrics, experiment_path):
    # Save metrics to JSON file
    with open(os.path.join(experiment_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def train_model(args):
    # Logging setup
    experiment_path = os.path.join(args.experiment_dir, args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    log_file = os.path.join(experiment_path, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # Dataset setup
    data_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    dataset = GaitDatasetV1(data_dirs, window_size=args.window_size, sample_rate=args.sample_rate, label="time")
    train_indices, test_indices = load_split_indices(args.data_dir)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model setup
    model = StrideTimeEstimatorLSTM(
        args.input_size,
        args.hidden_size,
        args.num_layers,
        args.dropout,
    )
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    train_losses, test_losses = [], []
    best_test_loss = float("inf")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, time_labels in train_loader:
            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions.squeeze(), time_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Test loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sequences, time_labels in test_loader:
                predictions = model(sequences)
                loss = criterion(predictions.squeeze(), time_labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        logging.info(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(experiment_path, "best_model.pth"))
            logging.info(f"Best model saved with test loss: {best_test_loss:.4f}")

    # Plot and save training/test loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Test Loss")
    plt.savefig(os.path.join(experiment_path, "loss_curve.png"))
    
    # Calculate metrics for the test set
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for sequences, time_labels in test_loader:
            predicted_times = model(sequences)
            predictions.append(predicted_times.squeeze().numpy())
            labels.append(time_labels.numpy())

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Calculate MSE, MAE, R²
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    r2 = r2_score(labels, predictions)

    metrics = {
        "train_loss": float(np.mean(train_losses)),
        "test_loss": float(np.mean(test_losses)),
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2)
    }
    
    logging.info(f"Mean Squared Error: {mse:.4f}")
    logging.info(f"Mean Absolute Error: {mae:.4f}")
    logging.info(f"R-squared: {r2:.4f}")

    # Save metrics to file
    save_metrics(metrics, experiment_path)

    # Test and Scatter Plot with y=x line
    test_model_and_plot(args, model, test_loader, experiment_path)


def test_model_and_plot(args, model, test_loader, experiment_path):
    # Test the model and generate predictions
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for sequences, time_labels in test_loader:
            predicted_times = model(sequences)
            predictions.append(predicted_times.squeeze().numpy())
            labels.append(time_labels.numpy())

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Plot predicted vs actual stride times in a scatter plot with y=x line
    plt.figure(figsize=(6, 6))
    plt.scatter(labels, predictions, s=1)
    plt.plot([0, max(labels.max(), predictions.max())], [0, max(labels.max(), predictions.max())], color="red", linestyle="--")
    plt.xlabel("Actual Stride Time")
    plt.ylabel("Predicted Stride Time")
    plt.title("Actual vs Predicted Stride Time\nR² = {:.4f}".format(r2_score(labels, predictions)))
    plt.savefig(os.path.join(experiment_path, "stride_time_scatter.png"))
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    experiment_path = os.path.join(args.experiment_dir, args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    with open(os.path.join(experiment_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    shutil.copy2(__file__, experiment_path)
    train_model(args)
