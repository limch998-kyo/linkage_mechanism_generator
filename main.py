from points_generator import generate_coordinates
from GNN_network import EnhancedGNN
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import imageio
import numpy as np
import os


def share_adjoining_point(adj_matrix, fixed_point_1, fixed_point_2):
    # Loop through each node (excluding the fixed points)
    for node in range(adj_matrix.shape[0]):
        if node != fixed_point_1 and node != fixed_point_2:
            if adj_matrix[fixed_point_1][node] == 1 and adj_matrix[fixed_point_2][node] == 1:
                # Both fixed points have a connection to this node
                return True
    return False
attempt = 0

def draw_mechanism(coordinates, fixed_point_1, fixed_point_2, angle):
    plt.figure(figsize=(5,5))
    
    rotated_coordinates = np.dot(coordinates, rotation_matrix(angle))
    
    for i in range(len(coordinates)-1):
        plt.plot([rotated_coordinates[i][0], rotated_coordinates[i+1][0]], 
                 [rotated_coordinates[i][1], rotated_coordinates[i+1][1]], '-o', color='blue')
    
    plt.scatter(rotated_coordinates[fixed_point_1][0], rotated_coordinates[fixed_point_1][1], color='red', s=100)
    plt.scatter(rotated_coordinates[fixed_point_2][0], rotated_coordinates[fixed_point_2][1], color='red', s=100)
    
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.savefig(f"frame_{angle}.png")
    plt.close()

def rotation_matrix(angle):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

while True:

    attempt += 1
    # Generate random coordinates
    coordinates = generate_coordinates(num_points=4)

    # Convert to PyTorch tensor and setup a PyTorch Geometric Data instance
    coordinates_tensor = torch.tensor(coordinates, dtype=torch.float)
    edge_index = torch.tensor([[i, i+1] for i in range(len(coordinates)-1)], dtype=torch.long).t().contiguous()
    data = Data(x=coordinates_tensor, edge_index=edge_index)

    # Setup and run the GNN
    model = EnhancedGNN(num_node_features=2)

    node_out, adjacency_matrix= model(data)
    # node_labels = torch.argmax(F.softmax(node_out, dim=1), dim=1).numpy()



    row_sums = torch.sum(adjacency_matrix, dim=1)
    if not (row_sums == 0).any().item() and not (row_sums == 1).any().item():
        
        #number of link
        sum_adj_matrix = torch.sum(adjacency_matrix)
        num_link = sum_adj_matrix.item()/2

        num_ones = int(torch.sum(node_out).item())
        num_zeros = int(node_out.numel() - num_ones)
        calculate_freedom = 3*(num_link-1) - 2*(num_ones + num_zeros)
        fixed_points = []
        if calculate_freedom==1 and num_ones==2:
            count = 0

            for i in range(node_out.size(0)):
                if count == 2:
                    break
                if node_out[i]==1:
                    fixed_points.append(i)
                    count += 1

            fixed_point_1, fixed_point_2 = fixed_points
            if not share_adjoining_point(adjacency_matrix, fixed_point_1, fixed_point_2):
                print("The two fixed points do not share an adjoining point.")
                print("freedom is 1")
                print("attempt: ", attempt)

                angles = np.linspace(0, 2*np.pi, 30)  # You can adjust this as needed
                for angle in angles:
                    draw_mechanism(coordinates, fixed_point_1, fixed_point_2, angle)

                frames = []
                for angle in angles:
                    frames.append(imageio.imread(f"frame_{angle}.png"))

                imageio.mimsave('mechanism.gif', frames, duration=0.1)  # Adjust duration as needed

                # Delete the temporary frames
                for angle in angles:
                    os.remove(f"frame_{angle}.png")

                # Use the function to visualize the mechanism


                break

if not share_adjoining_point(adjacency_matrix, fixed_point_1, fixed_point_2):
    angles = np.linspace(0, 2*np.pi, 30)  # You can adjust this as needed
    for angle in angles:
        draw_mechanism(coordinates, fixed_point_1, fixed_point_2, angle)

    frames = []
    for angle in angles:
        frames.append(imageio.imread(f"frame_{angle}.png"))

    imageio.mimsave('mechanism.gif', frames, duration=0.1)  # Adjust duration as needed

    # Delete the temporary frames
    for angle in angles:
        os.remove(f"frame_{angle}.png")
