# from points_generator import generate_coordinates
from GNN_network import GNN
import torch
from torch_geometric.data import Data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from util import share_adjoining_point, visualize_linkage




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
