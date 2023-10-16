import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch_geometric.utils as utils

class SoftLayer(nn.Module):
    def __init__(self, threshold=0.3):
        super(SoftLayer, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        x = torch.sigmoid(x)
        return (x > self.threshold).float()


class EnhancedGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(EnhancedGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 1)
        
        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.soft_layer = SoftLayer()

    def forward(self, data):
        x, _ = data.x, data.edge_index

        # Node classification
        x = self.conv1(x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        # Fully connect nodes for edge classification
        num_nodes = x.size(0)
        full_edge_index = torch.combinations(torch.arange(num_nodes), 2).T
        start, end = full_edge_index
        edge_features = torch.cat([x[start], x[end]], dim=1)
        edge_out = self.edge_classifier(edge_features).squeeze()

        # If you want edge_out to be of shape (num_nodes, num_nodes)
        adjacency_matrix = torch.zeros((num_nodes, num_nodes))
        adjacency_matrix[start, end] = edge_out
        adjacency_matrix[end, start] = edge_out  # considering undirected graph
        
        adjacency_matrix = self.soft_layer(adjacency_matrix)

        return adjacency_matrix
