import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class CombinedNetwork(nn.Module):
    def __init__(self):
        super(CombinedNetwork, self).__init__()

        # First stage layers
        self.fc1 = nn.Linear(12, 64)
        self.fc1_2 = nn.Linear(64, 128)  # New hidden layer
        self.fc2_binary = nn.Linear(128, 2)
        self.fc2_coords = nn.Linear(128, 8)

        # Second stage layers
        self.fc3 = nn.Linear(10, 128)  # 8 (coords) + 2 (binary)
        self.fc3_2 = nn.Linear(128, 256)  # New hidden layer
        self.fc4_binary = nn.Linear(256, 4)
        self.fc4_indices = nn.Linear(256, 16)  # Producing (4,4) tensor
        self.fc4_coords = nn.Linear(256, 8)  # Producing (4,2) coordinates

        # Third stage layers
        self.fc5 = nn.Linear(24, 256)  # 8 (binary) + 16 (coords)
        self.fc5_2 = nn.Linear(256, 512)  # New hidden layer
        self.fc6_indices = nn.Linear(512, 8)
        self.fc6_coords = nn.Linear(512, 2)

        # Dropout layers
        self.dropout = nn.Dropout(0.5)

    def custom_sign(self, input, threshold=0.1):
        input[input.abs() < threshold] = 0
        return input.sign()

    def check_for_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")

    def forward(self, x):
        # First stage
        x1 = F.relu(self.fc1(x.view(x.size(0), -1)))
        x1 = self.dropout(F.relu(self.fc1_2(x1)))

        out1_binary = torch.round(torch.sigmoid(self.fc2_binary(x1)))
        out1_coords = torch.reshape(self.fc2_coords(x1), (-1, 4, 2))

        # Second stage
        x2 = torch.cat((out1_binary, out1_coords.view(out1_coords.size(0), -1)), dim=1)
        x2 = self.dropout(F.relu(self.fc3(x2)))
        x2 = self.dropout(F.relu(self.fc3_2(x2)))

        out2_binary = torch.round(torch.sigmoid(self.fc4_binary(x2)))
        out2_adj_softmax = F.softmax(self.fc4_indices(x2).view(-1, 4, 4), dim=2)
        _, top2_indices = torch.topk(out2_adj_softmax, 2, dim=2)
        out2_adjacency = top2_indices.sort(dim=2).values
        out2_coords = torch.reshape(self.fc4_coords(x2), (-1, 4, 2))

        # Third stage
        stretched_out1_binary = out1_binary.repeat_interleave(2, dim=1)
        final_binary_input = torch.cat((stretched_out1_binary, out2_binary), dim=1)

        output_coords = torch.cat((out1_coords, out2_coords), dim=1)
        all_coordinates = torch.cat((out1_coords, out2_coords), dim=1).view(-1, 16)
        x3 = torch.cat((final_binary_input, all_coordinates), dim=1)
        x3 = self.dropout(F.relu(self.fc5(x3)))
        x3 = self.dropout(F.relu(self.fc5_2(x3)))

        out3_adj_softmax = F.softmax(self.fc6_indices(x3), dim=1)
        _, top2_indices = torch.topk(out3_adj_softmax, 2, dim=1)
        out3_adjacency = top2_indices.sort(dim=1).values
        out3_coords = self.fc6_coords(x3)

        # Perform NaN checks
        self.check_for_nan(final_binary_input, "final_binary_input")
        self.check_for_nan(out2_adjacency, "out2_adjacency")
        self.check_for_nan(out2_coords, "out2_coords")
        self.check_for_nan(out3_adjacency, "out3_adjacency")
        self.check_for_nan(out3_coords, "out3_coords")

        return final_binary_input[0], out2_adjacency[0], output_coords[0], out3_adjacency[0], out3_coords[0]

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import random
# import numpy as np

# class CombinedNetwork(nn.Module):
#     def __init__(self):
#         super(CombinedNetwork, self).__init__()

#         # First stage layers
#         self.fc1 = nn.Linear(12, 64)
#         self.fc2_binary = nn.Linear(64, 2)
#         self.fc2_coords = nn.Linear(64, 8)

#         # Second stage layers
#         self.fc3 = nn.Linear(10, 128)  # 8 (coords) + 2 (binary)
#         self.fc4_binary = nn.Linear(128, 4)
#         self.fc4_indices = nn.Linear(128, 16)  # Producing (4,4) tensor 
#         self.fc4_coords = nn.Linear(128, 8)  # Producing (4,2) coordinates

#         # Third stage layers
#         self.fc5 = nn.Linear(24, 256)  # 8 (binary) + 16 (coords)
#         # self.fc6_binary = nn.Linear(128, 2)
#         self.fc6_indices = nn.Linear(256, 8)
#         self.fc6_coords = nn.Linear(256, 2) 

#     def custom_sign(self,input, threshold=0.1):
#         # Values within [-threshold, threshold] are mapped to 0
#         input[input.abs() < threshold] = 0
#         return input.sign()

#     def check_for_nan(self, tensor, name):
#         if torch.isnan(tensor).any():
#             print(f"NaN detected in {name}")

#     def forward(self, x):
#         # First stage
#         x1 = x.view(x.size(0), -1)
#         x1 = F.relu(self.fc1(x1))
        
#         out1_binary = torch.round(torch.sigmoid(self.fc2_binary(x1)))
#         self.check_for_nan(out1_binary, "out1_binary")

#         out1_coords = self.fc2_coords(x1)
#         out1_coords = torch.reshape(out1_coords, (-1, 4, 2))
#         self.check_for_nan(out1_coords, "out1_coords")

#         # Second stage
#         x2 = torch.cat((out1_binary, out1_coords.view(out1_coords.size(0), -1)), dim=1)
#         x2 = F.relu(self.fc3(x2))
        
#         out2_binary = torch.round(torch.sigmoid(self.fc4_binary(x2)))
        
#         # Generating unique adjacency values using softmax and argmax (assuming two largest values are selected)
#         out2_adj_softmax = F.softmax(self.fc4_indices(x2).view(-1, 4, 4), dim=2)
#         _, top2_indices = torch.topk(out2_adj_softmax, 2, dim=2)
#         out2_adjacency = top2_indices.sort(dim=2).values
        
#         out2_coords = self.fc4_coords(x2)
#         out2_coords = torch.reshape(out2_coords, (-1, 4, 2))

#         # Prepare inputs for the third stage
#         stretched_out1_binary = out1_binary.repeat_interleave(2, dim=1)

#         final_binary_input = torch.cat((stretched_out1_binary, out2_binary), dim=1)
        
#         output_coords = torch.cat((out1_coords, out2_coords), dim=1)

#         all_coordinates = torch.cat((out1_coords, out2_coords), dim=1).view(-1, 16)

#         # Third stage
#         x3 = torch.cat((final_binary_input, all_coordinates), dim=1)
#         x3 = F.relu(self.fc5(x3))
        
#         # out3_binary = torch.round(torch.sigmoid(self.fc6_binary(x3)))
        
#         # Ensuring unique values for indices
#         out3_adj_softmax = F.softmax(self.fc6_indices(x3), dim=1)
#         _, top2_indices = torch.topk(out3_adj_softmax, 2, dim=1)
#         out3_adjacency = top2_indices.sort(dim=1).values
        
#         out3_coords = self.fc6_coords(x3)

#         # Before returning the final values, add checks for NaNs
#         self.check_for_nan(final_binary_input, "final_binary_input")
#         self.check_for_nan(out2_adjacency, "out2_adjacency")
#         self.check_for_nan(output_coords, "output_coords")
#         self.check_for_nan(out3_adjacency, "out3_adjacency")
#         self.check_for_nan(out3_coords, "out3_coords")

#         return final_binary_input[0], out2_adjacency[0], output_coords[0], out3_adjacency[0], out3_coords[0]

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    seed_everything(42)
    net = CombinedNetwork()
    sample_input = torch.rand((1, 6, 2))
    coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = net(sample_input)



    print("Binary Output:", coor_val)
    print("Stage 2 Adjacency:", stage2_adjacency)
    print("All Coordinates:", all_coords)
    print("Target Adjacency:", target_adjacency)
    print("Target Coordinates:", target_coords)
    print(all_coords.shape)