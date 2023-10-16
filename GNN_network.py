import torch
import torch.nn as nn

class CoordinateNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_nodes):
        super(CoordinateNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2 * output_nodes)
        self.output_nodes = output_nodes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).view(-1, self.output_nodes, 2)

class AdjacencyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_nodes):
        super(AdjacencyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Exclude diagonal for symmetric matrix since diagonal will be zeros
        self.fc3 = nn.Linear(hidden_dim, output_nodes * (output_nodes - 1) // 2)
        self.output_nodes = output_nodes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        adj_vector = torch.sigmoid(self.fc3(x))

        # Create the adjacency matrix
        indices = torch.triu_indices(self.output_nodes, self.output_nodes, offset=1, device=x.device)
        adjacency_matrix = torch.zeros(x.shape[0], self.output_nodes, self.output_nodes, device=x.device)
        adjacency_matrix[:, indices[0], indices[1]] = adj_vector
        adjacency_matrix[:, indices[1], indices[0]] = adj_vector  # Make it symmetric
        
        # Convert to binary
        adjacency_matrix = (adjacency_matrix > 0.5).float()
        
        return adjacency_matrix


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_nodes):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_nodes = output_nodes

        super(GNN, self).__init__()
        self.coordinate_net = CoordinateNetwork(self.input_dim, self.hidden_dim, self.output_nodes)
        self.adjacency_net = AdjacencyNetwork(self.input_dim, self.hidden_dim, self.output_nodes)

    def forward(self, x):
        coordinates = self.coordinate_net(x)
        adjacency_matrix = self.adjacency_net(x)
        return coordinates, adjacency_matrix

# # Usage example
# input_dim = 10 * 2  # 4 2D coordinates
# hidden_dim = 128
# output_nodes = 10  # For instance, outputting 10 nodes

# gnn = GNN(input_dim, hidden_dim, output_nodes)
# inputs = torch.rand(1, input_dim)
# coordinates, adjacency_matrix = gnn(inputs)

# print(coordinates[0])
# print(adjacency_matrix[0])