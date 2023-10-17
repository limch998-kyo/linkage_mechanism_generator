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

        # Third stage layers
        self.fc5 = nn.Linear(24, 128)  # 8 (binary) + 16 (coords)
        # self.fc6_binary = nn.Linear(128, 2)
        self.fc6_indices = nn.Linear(128, 8)
        self.fc6_coords = nn.Linear(128, 2) 

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

        # Prepare inputs for the third stage
        stretched_out1_binary = out1_binary.repeat_interleave(2, dim=1)

        final_binary_input = torch.cat((stretched_out1_binary, out2_binary), dim=1)
        
        output_coords = torch.cat((out1_coords, out2_coords), dim=1)

        all_coordinates = torch.cat((out1_coords, out2_coords), dim=1).view(-1, 16)

        # Third stage
        x3 = torch.cat((final_binary_input, all_coordinates), dim=1)
        x3 = F.relu(self.fc5(x3))
        
        # out3_binary = torch.round(torch.sigmoid(self.fc6_binary(x3)))
        
        # Ensuring unique values for indices
        out3_adj_softmax = F.softmax(self.fc6_indices(x3), dim=1)
        _, top2_indices = torch.topk(out3_adj_softmax, 2, dim=1)
        out3_adjacency = top2_indices.sort(dim=1).values
        
        out3_coords = self.fc6_coords(x3)

        return final_binary_input, out2_adjacency, output_coords, out3_adjacency, out3_coords
    
if __name__ == '__main__':
    # Test
    net = CombinedNetwork()
    sample_input = torch.rand((1, 6, 2))
    coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = net(sample_input)

    print(coor_val)
    print(stage2_adjacency)
    print(all_coords)
    print(target_adjacency)
    print(target_coords)
