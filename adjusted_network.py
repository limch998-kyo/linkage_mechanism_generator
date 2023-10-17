import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
from points_generator import generate_coordinates
from torch_geometric.data import Data

class SoftLayer(nn.Module):
    def __init__(self, threshold=0.5):
        super(SoftLayer, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        x = torch.sigmoid(x)
        return (x > self.threshold).float()

class EnhancedGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(EnhancedGNN, self).__init__()
        self.transform = GCNConv(num_node_features, 16)
        
        self.edge_classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.soft_layer = SoftLayer()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Feature transformation using GCN
        x = self.transform(x, edge_index)
        x = F.relu(x)

        # Fully connect nodes for edge classification
        num_nodes = x.size(0)
        full_edge_index = torch.combinations(torch.arange(num_nodes), 2).T
        start, end = full_edge_index
        edge_features = torch.cat([x[start], x[end]], dim=1)
        edge_out = self.edge_classifier(edge_features).squeeze()

        # Reshape to adjacency matrix
        adjacency_matrix = torch.zeros((num_nodes, num_nodes))
        adjacency_matrix[start, end] = edge_out
        adjacency_matrix[end, start] = edge_out  # considering undirected graph
        
        adjacency_matrix = self.soft_layer(adjacency_matrix)

        return adjacency_matrix

# coordinates = generate_coordinates(num_points=4)

# # Convert to PyTorch tensor and setup a PyTorch Geometric Data instance
# coordinates_tensor = torch.tensor(coordinates, dtype=torch.float)
# edge_index = torch.tensor([[i, i+1] for i in range(len(coordinates)-1)], dtype=torch.long).t().contiguous()
# data = Data(x=coordinates_tensor, edge_index=edge_index)

# # Setup and run the GNN
# model = EnhancedGNN(num_node_features=2)
# adjacency_matrix= model(data)

# print(adjacency_matrix)