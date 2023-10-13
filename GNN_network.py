import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

class SimpleGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 2)  # for fixed or moving axis classification
        
        # Edge classifier
        self.edge_classifier = nn.Sequential(
            nn.Linear(32, 16),  # 16 * 2 = 32, as we're considering embeddings of two nodes
            nn.ReLU(),
            nn.Linear(16, 1),  # Binary classification: connected or not
            nn.Sigmoid()
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Node classification
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        node_out = self.conv2(x, edge_index)
        
        # Edge classification
        start, end = edge_index
        edge_features = torch.cat([x[start], x[end]], dim=1)
        edge_out = self.edge_classifier(edge_features)
        
        return node_out, edge_out.squeeze()
