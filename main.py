from points_generator import generate_coordinates
from GNN_network import EnhancedGNN
from visiualizer import visualize_with_rotation
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import imageio
import numpy as np

# Generate random coordinates
coordinates = generate_coordinates(num_points=10)

# Convert to PyTorch tensor and setup a PyTorch Geometric Data instance
x = torch.tensor(coordinates, dtype=torch.float)
edge_index = torch.tensor([[i, i+1] for i in range(len(coordinates)-1)], dtype=torch.long).t().contiguous()
data = Data(x=x, edge_index=edge_index)

# Setup and run the GNN
model = EnhancedGNN(num_node_features=2)
node_out, edge_out, full_edge_index = model(data)

# Classify nodes and edges
node_predictions = F.softmax(node_out, dim=1)
node_predictions[0] = torch.tensor([1.0, 0.0])  # Setting the first coordinate as rotating

# Threshold edge_out to decide connections
edge_threshold = 0.2
predicted_connections = (edge_out > edge_threshold).float()
# print(predicted_connections)

# print(predicted_connections)
print(predicted_connections)

print("Node Predictions:", node_predictions.shape)
print("Edge Predictions:", predicted_connections.shape)
print("Full Edge Index:", full_edge_index.shape)
# # Number of frames and rotation for each frame
# num_frames = 36
# rotation_per_frame = 2 * np.pi / num_frames

# frames = []
# for frame in range(num_frames):
#     rotation_angle = frame * rotation_per_frame
#     image = visualize_with_rotation(coordinates, node_predictions, rotation_angle)
#     frames.append(image)

# # Save the GIF
# imageio.mimsave('rotating_simulation.gif', frames, fps=10)


import matplotlib.pyplot as plt

def visualize_graph(coordinates, node_labels, edge_probs, edge_index, threshold=0.5):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot edges
    for i in range(edge_index.shape[0]):
        start, end = edge_index[i]
        if edge_probs[i] > threshold:
            ax.plot([coordinates[start][0], coordinates[end][0]],
                    [coordinates[start][1], coordinates[end][1]], 
                    color='gray', alpha=edge_probs[i].item())  # transparency based on probability
    
    # Plot nodes
    for i, coord in enumerate(coordinates):
        if i == 0:
            ax.scatter(*coord, color='red', label='Rotating', s=100)
        elif node_labels[i] == 0:  # Assuming 0 is fixed axis and 1 is moving axis
            ax.scatter(*coord, color='blue', label='Fixed Axis', s=50)
        else:
            ax.scatter(*coord, color='green', label='Moving Axis', s=50)

    ax.set_title("Graph Visualization")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.legend()

    plt.show()

node_out, edge_out, full_edge_index = model(data)

node_labels = torch.argmax(F.softmax(node_out, dim=1), dim=1).numpy()

visualize_graph(data.x.numpy(), node_labels, edge_out.detach().numpy(), full_edge_index.t().numpy())
