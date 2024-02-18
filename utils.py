from matplotlib import pyplot as plt
import math
import numpy as np
import torch
import random

def output_process(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, rotation_direction):
    coor_val = coor_val.detach().numpy()
    stage2_adjacency = stage2_adjacency.detach().numpy()
    all_coords = all_coords.detach().numpy()
    target_adjacency = target_adjacency.detach().numpy()
    target_coords = target_coords.detach().numpy()
    rotation_direction = rotation_direction.detach().numpy()

    return coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, rotation_direction

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



def calculate_angle(point1, point2):
    """Calculate angle between two points with respect to the positive x-axis."""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.atan2(dy, dx)

def seed_everything(seed=42):
    random.seed(seed)  # Seed Python's random module
    np.random.seed(seed)  # Seed Numpy (make sure to import numpy as np)
    torch.manual_seed(seed)  # Seed torch for CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Seed torch for CUDA operations
        torch.cuda.manual_seed_all(seed)  # Seed all GPUs if there are multiple
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False