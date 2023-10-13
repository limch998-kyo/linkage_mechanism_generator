from points_generator import generate_coordinates
from GNN_network import SimpleGNN
from visiualizer import visualize_with_rotation
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import imageio
import numpy as np

# Generate random coordinates
coordinates = generate_coordinates()

# Convert to PyTorch tensor and setup a PyTorch Geometric Data instance
x = torch.tensor(coordinates, dtype=torch.float)
edge_index = torch.tensor([[i, i+1] for i in range(len(coordinates)-1)], dtype=torch.long).t().contiguous()
data = Data(x=x, edge_index=edge_index)

# Setup and run the GNN
model = SimpleGNN(num_node_features=2)
node_out, edge_out = model(data)

# Classify nodes and edges
node_predictions = F.softmax(node_out, dim=1)
node_predictions[0] = torch.tensor([1.0, 0.0])  # Setting the first coordinate as rotating

# Threshold edge_out to decide connections
edge_threshold = 0.5
predicted_connections = (edge_out > edge_threshold).float()

print("Node Predictions:", node_predictions)
print("Edge Predictions:", predicted_connections)

# Number of frames and rotation for each frame
num_frames = 36
rotation_per_frame = 2 * np.pi / num_frames

frames = []
for frame in range(num_frames):
    rotation_angle = frame * rotation_per_frame
    image = visualize_with_rotation(coordinates, node_predictions, rotation_angle)
    frames.append(image)

# Save the GIF
imageio.mimsave('rotating_simulation.gif', frames, fps=10)
