import torch
import torch.nn as nn
import torch.nn.functional as F
class CombinedNetwork(nn.Module):
    def __init__(self):
        super(CombinedNetwork, self).__init__()

        # First stage layers
        self.fc1 = nn.Linear(12, 32)
        self.fc2_binary = nn.Linear(32, 2)
        self.fc2_coords = nn.Linear(32, 8)

        # Second stage layers
        self.fc3 = nn.Linear(10, 64)  # 8 (coords) + 2 (binary)
        self.fc4_binary = nn.Linear(64, 4)
        self.fc4_indices = nn.Linear(64, 16)  # Producing (4,4) tensor 
        self.fc4_coords = nn.Linear(64, 8)  # Producing (4,2) coordinates

    def forward(self, x):
        # First stage
        x1 = x.view(x.size(0), -1)
        x1 = F.relu(self.fc1(x1))
        
        out1_binary = torch.round(torch.sigmoid(self.fc2_binary(x1)))
        out1_coords = self.fc2_coords(x1)
        out1_coords = torch.reshape(out1_coords, (-1, 4, 2))

        # Second stage
        x2 = torch.cat((out1_binary, out1_coords.view(out1_coords.size(0), -1)), dim=1)
        x2 = F.relu(self.fc3(x2))
        
        out2_binary = torch.round(torch.sigmoid(self.fc4_binary(x2)))
        
        # Generating unique adjacency values using softmax and argmax (assuming two largest values are selected)
        out2_adj_softmax = F.softmax(self.fc4_indices(x2).view(-1, 4, 4), dim=2)
        _, top2_indices = torch.topk(out2_adj_softmax, 2, dim=2)
        out2_adjacency = top2_indices.sort(dim=2).values
        
        out2_coords = self.fc4_coords(x2)
        out2_coords = torch.reshape(out2_coords, (-1, 4, 2))

        return out1_binary, out1_coords, out2_binary, out2_adjacency, out2_coords

# Test
net = CombinedNetwork()
sample_input = torch.rand((1, 6, 2))
out1_binary, out1_coords, out2_binary, out2_adjacency, out2_coords = net(sample_input)

print(out1_binary.shape)
print(out1_binary)
print(out1_coords.shape)
print(out1_coords)
print(out2_binary.shape)
print(out2_binary)
print(out2_adjacency.shape)
print(out2_adjacency)
print(out2_coords.shape)
print(out2_coords)
