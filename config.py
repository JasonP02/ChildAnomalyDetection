import torch
from datetime import datetime
import mlflow
import torch.nn as nn

class Config:
    def __init__(self):
        self.num_node_features = 3
        self.hidden_channels = 116
        self.learning_rate = 0.0033390237448091553
        self.weight_decay = 1.0574209568504245e-06  # Add this line
        self.batch_size = 16
        self.epochs = 1
        self.early_stopping_patience = 10
        self.early_stopping_delta = 1e-3
        self.validation_split = 0.2
        self.loss_metric = nn.MSELoss(reduction='none')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_path = '/home/j/Desktop/Training_Data/processed_graphs'
        self.random_seed = 5
        self.experiment_name = f'gcn_autoencoder'#_{datetime.now().strftime("%Y_%m_%d")}''
        self.training_videos = '/home/j/Desktop/Training_Data/videos'
        self.testing_videos = '/home/j/Desktop/Training_Data/testing_videos'
        self.training_graphs = '/home/j/Desktop/Training_Data/processed_graphs'
        self.testing_graphs = '/home/j/Desktop/Training_Data/processed_testing_graphs'

    def log_hyperparameters(self):
        mlflow.log_param('model_type', 'GCN_Autoencoder')
        mlflow.log_param('num_node_features', self.num_node_features)
        mlflow.log_param('hidden_channels', self.hidden_channels)
        mlflow.log_param('learning_rate', self.learning_rate)
        mlflow.log_param('weight_decay', self.weight_decay)  # Add this line
        mlflow.log_param('batch_size', self.batch_size)
        mlflow.log_param('epochs', self.epochs)