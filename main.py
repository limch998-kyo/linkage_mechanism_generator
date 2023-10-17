# from points_generator import generate_coordinates
import torch
from util import share_adjoining_point, visualize_linkage, calculate_angle, euclidean_distance
import numpy as np
import pylinkage as pl
import matplotlib.pyplot as plt
from GNN_network import CombinedNetwork

def visualize_linkage_system(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, status_location):
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Define colors for each stage
    colors = ['r', 'g', 'b']

    # Stage 1
    if coor_val[0] == 1:  # Link exists for the first set
        ax.plot([crank_location[0], all_coords[0][0]], [crank_location[1], all_coords[0][1]], color=colors[0])
        ax.plot([all_coords[0][0], all_coords[1][0]], [all_coords[0][1], all_coords[1][1]], color=colors[0])
        ax.plot([all_coords[1][0], status_location[0]], [all_coords[1][1], status_location[1]], color=colors[0])
    
    if coor_val[2] == 1:  # Link exists for the second set
        ax.plot([crank_location[0], all_coords[2][0]], [crank_location[1], all_coords[2][1]], color=colors[0])
        ax.plot([all_coords[2][0], all_coords[3][0]], [all_coords[2][1], all_coords[3][1]], color=colors[0])
        ax.plot([all_coords[3][0], status_location[0]], [all_coords[3][1], status_location[1]], color=colors[0])

    # Stage 2
    for i in range(4, 8):
        if coor_val[i] == 1:
            joint_a, joint_b = stage2_adjacency[i-4]
            ax.plot([all_coords[joint_a][0], all_coords[i][0]], [all_coords[joint_a][1], all_coords[i][1]], color=colors[1])
            ax.plot([all_coords[joint_b][0], all_coords[i][0]], [all_coords[joint_b][1], all_coords[i][1]], color=colors[1])

    # Stage 3
    joint_a, joint_b = target_adjacency

    ax.plot([all_coords[joint_a][0], target_coords[0]], [all_coords[joint_a][1], target_coords[1]], color=colors[2])
    ax.plot([all_coords[joint_b][0], target_coords[0]], [all_coords[joint_b][1], target_coords[1]], color=colors[2])


    # Display the linkage system
    for idx, (x, y) in enumerate(all_coords):
        if coor_val[idx] == 1:
            ax.scatter(x, y, c='black')
    ax.scatter(crank_location[0], crank_location[1], c='orange', marker='o')  # Crank location
    ax.scatter(status_location[0], status_location[1], c='orange', marker='o')  # Status location
    ax.scatter(target_coords[0], target_coords[1], c='blue', marker='o')  # Target location
    plt.show()

input = []
target_location = [[5,5],[8,5],[5,4],[8,4]]
crank_location = [0,0]
status_location = [5,0]

input.append(crank_location)
input.append(status_location)
for i in range(len(target_location)):
    input.append(target_location[i])

input_tensor = torch.tensor([input], dtype=torch.float)

attempt = 0
while True:
    try:
        flag = False
        attempt += 1
        net = CombinedNetwork()
        coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = net(input_tensor)

        coor_val = coor_val.detach().numpy()
        stage2_adjacency = stage2_adjacency.detach().numpy()
        all_coords = all_coords.detach().numpy()
        target_adjacency = target_adjacency.detach().numpy()
        target_coords = target_coords.detach().numpy()
        coor_val = np.array([1, 1, 0, 0, 1, 1, 0, 0])

        if np.all(coor_val[:4] == 0):
            flag = True
        elif np.all(coor_val[-4:] == 0):
            flag = True



        for i in range(4, 8):
            joint_a, joint_b = stage2_adjacency[i-4]
            if coor_val[i]==0:
                continue
            elif coor_val[joint_a] == 0 or coor_val[joint_b] == 0:
                flag = True


        joint_a, joint_b = target_adjacency
        if coor_val[joint_a] == 0 or coor_val[joint_b] == 0:
            flag = True
        
        if flag:
            continue
        # print(np.sum(coor_val))
        # if np.sum(coor_val)>6:
        #     print(attempt)
        #     continue

        print(coor_val)
        print(stage2_adjacency)
        print(all_coords)
        print(target_adjacency)
        print(target_coords)

        # Call the visualization function



        visualize_linkage_system(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, status_location)
        break
    except Exception as e:  # Catch any exception
        print(f"An error occurred: {e}")  # Print the error for debugging
        continue  # Continue to the next iteration of the loop

# using joints connected to crankpoints
# make a gif visualization that simulates linkage mechanism when crank is rotating
# predict each joints location from joint0 to joint7 and use this information to locate target joint

# when predicting joints next frame location use intercept circle to calculate next location