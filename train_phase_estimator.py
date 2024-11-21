import argparse
import json
import logging
import numpy as np
import os
import shutil
import torch

import matplotlib.pyplot as plt
import torch.optim as optim

from datetime import datetime
from torch.utils.data import DataLoader, Subset

from data.dataset_v1 import GaitDatasetV1
from model.phase_estimator import ImprovedGaitLSTM
from utils.loss import CircularLoss
from utils.utils import load_split_indices


def parse_args():
    parser = argparse.ArgumentParser(description="Gait Cycle Phase Estimation")

    ### Experiment ###
    parser.add_argument("--experiment_dir", type=str, default="experiments", help="Directory to save experiments")
    parser.add_argument("--experiment_name", type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Experiment name")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing data")

    ### Dataset ###
    parser.add_argument("--window_size", type=int, default=12, help="Window size for each sequence")
    parser.add_argument("--sample_rate", type=int, default=120, help="Sampling rate of the data")
    
    ### Model and Training ###
    parser.add_argument("--input_size", type=int, default=10, help="Input size of the model")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    
    return parser.parse_args()


def train_model(args):
    # Set up logging
    experiment_path = os.path.join(args.experiment_dir, args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)
    log_file = os.path.join(experiment_path, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    
    # Prepare dataset
    data_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    dataset = GaitDatasetV1(data_dirs, window_size=args.window_size, sample_rate=args.sample_rate, label="phase")
    
    # Load train/test indices
    train_indices, test_indices = load_split_indices(args.data_dir)

    # Use Subset to create datasets for training and testing
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = ImprovedGaitLSTM(
        args.input_size,
        args.hidden_size,
        args.num_layers,
        args.dropout,
    )
    criterion = CircularLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, phase_labels in train_loader:
            optimizer.zero_grad()
            phase_outputs = model.get_phase(sequences)
            loss = criterion(phase_outputs, phase_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for sequences, phase_labels in test_loader:
                phase_outputs = model.get_phase(sequences)
                loss = criterion(phase_outputs.squeeze(), phase_labels)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        logging.info(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Save the best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(experiment_path, "best_model.pth"))
            logging.info(f"New best model saved with test loss: {best_test_loss:.4f}")

    # Save training plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Test Loss")
    plt.savefig(os.path.join(experiment_path, "loss_curve.png"))
    
    return model


def test_model(args):
    # Load the trained model
    experiment_path = os.path.join(args.experiment_dir, args.experiment_name)
    model_path = os.path.join(experiment_path, "best_model.pth")
    model = ImprovedGaitLSTM(
        args.input_size,
        args.hidden_size,
        args.num_layers,
        args.dropout,
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load dataset
    data_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
    dataset = GaitDatasetV1(data_dirs, window_size=args.window_size, sample_rate=args.sample_rate, label="phase")
    
    # Load test indices
    _, test_indices = load_split_indices(args.data_dir)

    # Use Subset to create datasets for testing
    test_dataset = Subset(dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate model
    predictions, labels = [], []
    with torch.no_grad():
        for sequences, phase_labels in test_loader:
            phase_outputs = model.get_phase(sequences)
            predictions.append(phase_outputs.squeeze().numpy())
            labels.append(phase_labels.numpy())

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Plot predicted phase against actual phase on a scatter plot with the y=x line for comparison
    plt.figure(figsize=(6, 6))
    plt.scatter(labels, predictions, s=1)
    plt.plot([0, 1], [0, 1], color="red")
    plt.xlabel("Actual Phase")
    plt.ylabel("Predicted Phase")
    plt.title("Actual vs Predicted Phase")
    plt.savefig(os.path.join(experiment_path, "phase_scatter.png"))


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Create experiment directory
    experiment_path = os.path.join(args.experiment_dir, args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    # Save arguments to JSON file
    with open(os.path.join(experiment_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Copy current script to experiment directory
    shutil.copy2(__file__, experiment_path)

    # Train model
    trained_model = train_model(args)
    
    # Test model
    test_model(args)