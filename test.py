# test

import mlflow
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from config import Config
from graph_dataset import VideoGraphDataset
from utils import calculate_metrics
from data_processing import preprocess_testing, preprocess_training



def test_model(preprocess):
    config = Config()
    if preprocess:
        preprocess_training()
        preprocess_testing()

    # Retrieve the latest run ID for the experiment (if not known)
    experiment_id = mlflow.get_experiment_by_name(config.experiment_name).experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time desc"], max_results=2)
    run_id = runs.iloc[0].run_id

    # Load the most recent model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    model.to(config.device)

    try:
        mse_95th = runs.iloc[0].data.metrics['mse_95th_percentile']
        print(f"Loaded 95th Percentile MSE: {mse_95th:.4f}")
    except KeyError as e:
        print(f"Error retrieving 95th Percentile MSE metric: {e}")
        print("Skipping 95th Percentile MSE check.")
        mse_95th = float('inf')  # Set a default value to continue the processing

    # Load the test data loader
    test_dataset = VideoGraphDataset('/home/j/Desktop/Training_Data/processed_testing_graphs/')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"Total number of samples: {len(test_dataset)}")

    for i, batch in enumerate(test_loader):
        print(f"Batch {i + 1}: {batch[0].num_graphs} samples")

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        anomalies = []
        for batch_idx, batch in enumerate(test_loader):
            batch = batch[0].to(config.device)
            output = model(batch)

            # Compute the MSE loss per node and then aggregate them per graph
            node_losses = config.loss_metric(output, batch.x).mean(dim=1)
            graph_losses = torch.zeros(batch.num_graphs, device=config.device)

            # Aggregate node losses to compute graph losses
            for i in range(batch.num_graphs):
                mask = batch.batch == i
                graph_losses[i] = node_losses[mask].mean()  # or .sum() if you prefer summing losses

            # Convert to numpy for easier handling
            graph_losses = graph_losses.cpu().numpy()

            # Print MSE per graph
            for graph_idx, loss in enumerate(graph_losses):
                print(f"Graph {batch_idx * config.batch_size + graph_idx + 1}: MSE = {loss:.4f}")

                # Check each graph for anomalies
                if loss > mse_95th:
                    anomalies.append(loss)

        print(f"Detected {len(anomalies)} anomalous graphs out of {len(test_dataset)} samples.")

        avg_loss, avg_mse, avg_mae = calculate_metrics(model, test_loader, config.device)
        print(f"Test Loss: {avg_loss:.4f}, Test MSE: {avg_mse:.4f}, Test MAE: {avg_mae:.4f}")
