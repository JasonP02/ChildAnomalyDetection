# graph_dataset

import os
import torch
from torch_geometric.data import Dataset

class VideoGraphDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None, pre_filter=None):
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.pt')]
        self.graphs = []
        for file in self.file_list:
            file_path = os.path.join(self.root_dir, file)
            graphs_in_file = torch.load(file_path)
            self.graphs.extend([graph for graph in graphs_in_file if graph is not None])
        super(VideoGraphDataset, self).__init__(root_dir, transform, pre_transform, pre_filter)


    @property
    def raw_file_names(self):
        return self.file_list

    @property
    def processed_file_names(self):
        return self.file_list

    def download(self):
        # Implement download logic if necessary
        pass

    def process(self):
        # Implement process logic if necessary
        pass

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @property
    def num_features(self):
        return self._get_num_features()

    def _get_num_features(self):
        if len(self.graphs) > 0:
            return 18  # self.graphs[0].x.shape[1] # Get the feature dimension from the first graph
        return 0