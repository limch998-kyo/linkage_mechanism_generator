import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedNetwork(nn.Module):
    def __init__(self, input_size=62):
        super(CombinedNetwork, self).__init__()

        # First stage layers
        self.fc1 = nn.Linear(input_size, 64)  # Input to first hidden layer
        self.fc1_1 = nn.Linear(64, 64)  # Additional hidden layer
        self.fc2_coords = nn.Linear(64, 8)  # Output of first stage

        # Second stage layers
        self.fc3 = nn.Linear(8, 128)  # Input to second hidden layer
        self.fc3_1 = nn.Linear(128, 128)  # Additional hidden layer
        self.fc4_coords = nn.Linear(128, 8)  # Output of second stage

        # Third stage layers
        self.fc5 = nn.Linear(16, 64)  # Input to third stage
        self.fc6_coords = nn.Linear(64, 2)  # Output of third stage

    def custom_sign(self, input, threshold=0.1):
        # Values within [-threshold, threshold] are mapped to 0
        input[input.abs() < threshold] = 0
        return input.sign()

    def forward(self, x):
        device = x.device  # Get the device from the input tensor
        
        # First stage
        x1 = x.view(x.size(0), -1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc1_1(x1))  # Passing through the additional hidden layer

        x2 = self.fc2_coords(x1)
        out1_coords = torch.reshape(x2, (-1, 4, 2))

        # Second stage
        x3 = F.relu(self.fc3(x2))
        x3 = F.relu(self.fc3_1(x3))  # Passing through the additional hidden layer
        
        x3 = self.fc4_coords(x3)
        out2_coords = torch.reshape(x3, (-1, 4, 2))

        # Prepare inputs for the third stage
        output_coords = torch.cat((out1_coords, out2_coords), dim=1)
        all_coordinates = torch.cat((out1_coords, out2_coords), dim=1).view(-1, 16)

        # Third stage
        x3 = all_coordinates
        x3 = F.relu(self.fc5(x3))
        
        out3_coords = self.fc6_coords(x3)

        return output_coords[0], out3_coords[0]

    
if __name__ == '__main__':
    # Test
    net = CombinedNetwork()
    sample_input = torch.rand((1, 4, 2))
    print(sample_input)
    print(sample_input.shape)
    all_coords, target_coords = net(sample_input)

    print(all_coords)
    print(target_coords)
