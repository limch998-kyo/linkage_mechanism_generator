# from points_generator import generate_coordinates
from GNN_network import GNN
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

def share_adjoining_point(adj_matrix, fixed_point_1, fixed_point_2):
    for node in range(adj_matrix.shape[0]):
        if node != fixed_point_1 and node != fixed_point_2:
            if adj_matrix[fixed_point_1][node] == 1 and adj_matrix[fixed_point_2][node] == 1:
                return True
    return False

def visualize_linkage(coordinates, adjacency_matrix):
    # Plot the points
    for coord in coordinates:
        plt.scatter(coord[0], coord[1], s=50, c='red', marker='o')
    
    # Connect the points based on the adjacency matrix
    num_points = len(coordinates)
    for i in range(num_points):
        for j in range(num_points):
            if adjacency_matrix[i][j] == 1:
                plt.plot([coordinates[i][0], coordinates[j][0]],
                         [coordinates[i][1], coordinates[j][1]], 'b-')
    
    # Annotate the points
    for idx, coord in enumerate(coordinates):
        if idx == 0 or idx == 1:  # For P0 and P1
            plt.annotate(f'P{idx} (Fixed)', (coord[0], coord[1]), textcoords="offset points", xytext=(0,10), ha='center')
        else:
            plt.annotate(f'P{idx}', (coord[0], coord[1]), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


attempt = 0

input_dim = 4 * 2  # 4 2D coordinates
hidden_dim = 128
output_nodes = 4  # For instance, outputting 10 nodes

while True:
    attempt += 1
    gnn = GNN(input_dim, hidden_dim, output_nodes)
    inputs = torch.rand(1, input_dim)
    coordinates, adjacency_matrix = gnn(inputs)

    coordinates = coordinates.detach().numpy()[0]
    adjacency_matrix = adjacency_matrix[0]


    row_sums = torch.sum(adjacency_matrix, dim=1)
    # print(row_sums)
    if not (row_sums == 0).any().item() and not (row_sums == 1).any().item():
    # if True:
        sum_adj_matrix = torch.sum(adjacency_matrix)
        print(adjacency_matrix)
        if not share_adjoining_point(adjacency_matrix, 0, 1):
            print("The two fixed points do not share an adjoining point.")
            print("freedom is 1")
            print("attempt: ", attempt)
            visualize_linkage(coordinates, adjacency_matrix)
            break
