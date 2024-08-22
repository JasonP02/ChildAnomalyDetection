# param_tune

import mlflow
import torch
import torch.nn as nn
import optuna
from torch_geometric.loader import DataLoader
from config import Config
from data_processing import YOLO_model, get_adjacency_matrix
from graph_dataset import VideoGraphDataset
from utils import calculate_metrics, EarlyStopping
from model import GCN_Autoencoder
from sklearn.model_selection import train_test_split


import optuna
from torch.utils.data import Subset

def objective(trial):
    config = Config()
    # Suggest hyperparameters
    hidden_channels = trial.suggest_int('hidden_channels', 64, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    yolo_model = YOLO_model()
    adj_matrix = get_adjacency_matrix()

    # Use a smaller subset of the dataset for tuning
    dataset = VideoGraphDataset(config.data_path)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_subset = Subset(dataset, train_idx[:500])  # Use only 500 training samples
    val_subset = Subset(dataset, val_idx[:100])  # Use only 100 validation samples
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    model = GCN_Autoencoder(num_node_features=config.num_node_features, hidden_channels=hidden_channels)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = config.loss_metric

    early_stopping = EarlyStopping(patience=5, delta=0.01)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch[0].to(config.device)
            optimizer.zero_grad()
            output = model(batch)
            loss = loss(output, batch.x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate the model
        val_loss, val_mse, val_mae = calculate_metrics(model, val_loader, config.device)
        trial.report(val_mse, epoch)

        if trial.should_prune() or early_stopping(val_loss):
            raise optuna.exceptions.TrialPruned()

    return val_mse

def tune_hyperparameters(n_trials=100, timeout=600):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    # Rest of the code remains the same