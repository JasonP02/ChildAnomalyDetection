import torch
from torch_geometric.nn import GCNConv

class GCN_Autoencoder(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN_Autoencoder, self).__init__()
        self.encoder_conv1 = GCNConv(num_node_features, hidden_channels)
        self.encoder_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.decoder_conv1 = GCNConv(hidden_channels, hidden_channels)
        self.decoder_conv2 = GCNConv(hidden_channels, num_node_features)

    def encode(self, data):
        # print(f"encode edge_index shape: {data.edge_index.shape}, x shape: {data.x.shape}")
        x = self.encoder_conv1(data.x, data.edge_index).relu()
        x = self.encoder_conv2(x, data.edge_index).relu()
        # print(x.shape)
        return x

    def decode(self, x, edge_index):
        x = self.decoder_conv1(x, edge_index).relu()
        x = self.decoder_conv2(x, edge_index).relu()
        return x

    def forward(self, batch):
        encoded = self.encode(batch)
        decoded = self.decode(encoded, batch.edge_index)
        return decoded