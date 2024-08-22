# utils


import torch
import torch.nn as nn

import mlflow
from config import Config
from graph_dataset import VideoGraphDataset
from model import GCN_Autoencoder
from torch_geometric.data import DataLoader

# Model Initalization
#########################################################
def initialize_model():
    config = Config()
    model = GCN_Autoencoder(num_node_features=config.num_node_features, hidden_channels=config.hidden_channels)
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return model, optimizer





def calculate_metrics(model, data_loader, device):
    config = Config()
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0


    with torch.no_grad():
        for batch in data_loader:
            batch = batch[0].to(device)
            output = model(batch)
            loss = config.loss_metric(output, batch.x).mean(dim=1)
            total_loss += loss.item()
            total_mse += torch.mean((output - batch.x) ** 2).item()
            total_mae += torch.mean(torch.abs(output - batch.x)).item()

    avg_loss = total_loss / len(data_loader)
    avg_mse = total_mse / len(data_loader)
    avg_mae = total_mae / len(data_loader)

    return avg_loss, avg_mse, avg_mae


def log_metrics(metrics, step=None):
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)


def evaluate_and_log(model, data_loader, device, epoch=None):
    avg_loss, avg_mse, avg_mae = calculate_metrics(model, data_loader, device)
    metrics = {
        "loss": avg_loss,
        "mse": avg_mse,
        "mae": avg_mae
    }
    log_metrics(metrics, step=epoch)
    return metrics


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True



