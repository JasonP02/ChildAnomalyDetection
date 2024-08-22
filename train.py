import mlflow
import torch
from config import Config
from utils import calculate_metrics, initialize_model, EarlyStopping, evaluate_and_log
import numpy as np
from data_processing import preprocess_training, load_data
import torch.nn as nn


def compute_graph_level_loss(output, batch):
    config = Config()
    node_losses = config.loss_metric(output, batch.x).mean(dim=1)
    graph_losses = torch.zeros(batch.num_graphs, device=config.device)

    for i in range(batch.num_graphs):
        mask = batch.batch == i
        graph_losses[i] = node_losses[mask].mean()

    return graph_losses.mean(), node_losses

def train_epoch(model, optimizer, train_loader, config):
    model.train()
    total_graph_loss = 0
    for batch in train_loader:
        batch = batch[0].to(config.device)
        optimizer.zero_grad()
        output = model(batch)
        graph_loss, node_losses = compute_graph_level_loss(output, batch)
        graph_loss.backward()
        optimizer.step()
        total_graph_loss += graph_loss.item()

    return total_graph_loss / len(train_loader)

def validate_epoch(model, val_loader, config):
    model.eval()
    total_graph_loss = 0
    total_node_losses = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch[0].to(config.device)
            output = model(batch)
            graph_loss, node_losses = compute_graph_level_loss(output, batch)
            total_graph_loss += graph_loss.item()

    return total_graph_loss / len(val_loader)


def train_model():
    config = Config()
    train_loader, val_loader = load_data(validation=True)
    model, optimizer = initialize_model()
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run():
        config.log_hyperparameters()
        early_stopping = EarlyStopping(patience=config.early_stopping_patience, delta=config.early_stopping_delta)
        for epoch in range(config.epochs):
            avg_train_loss = train_epoch(model, optimizer, train_loader, config)
            print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss:.4f}")
            mlflow.log_metric('average_training_loss', avg_train_loss, step=epoch)
            avg_val_loss = validate_epoch(model, val_loader, config)
            print(f"Epoch {epoch + 1}, Average Validation Loss: {avg_val_loss:.4f}")
            mlflow.log_metric('average_validation_loss', avg_val_loss, step=epoch)

            # Check for early stopping
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        # Log the model using MLflow
        mlflow.pytorch.log_model(model, "model")






def train_standard_model(preprocess):
    if preprocess:
        preprocess_training()
    train_model()