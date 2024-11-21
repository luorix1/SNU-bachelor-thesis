import argparse
import json
import logging
import numpy as np
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from datetime import datetime
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score

from data.dataset_v2 import GaitDatasetV2
from model.time_predictor_ipsilateral import AdjustedGaitModel as GaitModel
from utils.utils import load_split_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Cycle Duration Prediction")
    parser.add_argument("--experiment_dir", type=str, default="experiments", help="Directory to save experiments")
    parser.add_argument("--experiment_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Experiment name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing data")

    # Dataset hyperparameters
    parser.add_argument("--window_size", type=int, default=12, help="Window size for input sequences")
    parser.add_argument("--sample_rate", type=int, default=120, help="Sample rate of the data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    
    # Model hyperparameters
    parser.add_argument("--input_size", type=int, default=20, help="Input size of the model")
    parser.add_argument("--embed_size", type=int, default=256, help="Embedding size of the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

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
    dataset = GaitDatasetV2(data_dirs, window_size=args.window_size, sample_rate=args.sample_rate, label="time")
    train_indices, test_indices = load_split_indices(args.data_dir)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Model setup
    model = GaitModel(
        input_size=args.input_size,
        embed_size=args.embed_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    train_losses, test_losses = [], []
    best_test_loss = float("inf")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for _, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float()
            targets = targets.float()
            optimizer.zero_grad()
            outputs = model(inputs[:, :args.window_size], inputs[:, args.window_size:])
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Test loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.float()
                targets = targets.float()
                outputs = model(inputs[:, :args.window_size], inputs[:, args.window_size:])
                loss = criterion(outputs.squeeze(), targets)
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

    # Evaluate performance metrics on test set
    model.eval()
    predictions, ground_truths = [], []
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.float()
            targets = targets.float()
            outputs = model(inputs[:, :args.window_size], inputs[:, args.window_size:])
            predictions.extend(outputs.cpu().numpy())
            ground_truths.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Calculate MSE, MAE, and R²
    mse = np.mean((predictions - ground_truths) ** 2)
    mae = np.mean(np.abs(predictions - ground_truths))
    r2 = r2_score(ground_truths, predictions)

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

    # Test and generate scatter plot
    test_model_and_plot(predictions, ground_truths, experiment_path)

def test_model_and_plot(predictions, ground_truths, experiment_path):
    # Plot predicted vs actual gait cycle durations
    plt.figure(figsize=(6, 6))
    plt.scatter(ground_truths, predictions, s=1)
    plt.plot([min(ground_truths), max(ground_truths)], [min(ground_truths), max(ground_truths)], "r--")
    plt.xlabel("Actual Gait Cycle Duration (s)")
    plt.ylabel("Predicted Gait Cycle Duration (s)")
    plt.title("Predicted vs Actual Gait Cycle Duration")
    plt.savefig(os.path.join(experiment_path, "predicted_vs_actual.png"))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
