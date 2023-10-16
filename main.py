# from points_generator import generate_coordinates
import torch
from util import share_adjoining_point, visualize_linkage, calculate_angle, euclidean_distance
import numpy as np
import pylinkage as pl

from GNN_network import GNN

nodes  = 4

attempt = 0
final_attempt = 1

input_dim = nodes * 2  # 4 2D coordinates
hidden_dim = 128
output_nodes = nodes  # For instance, outputting 10 nodes

while True:
    try:
        attempt += 1
        gnn = GNN(input_dim, hidden_dim, output_nodes)
        inputs = torch.rand(1, input_dim)
        coordinates, adjacency_matrix = gnn(inputs)

        coordinates = coordinates.detach().numpy()[0]
        adjacency_matrix = adjacency_matrix[0]

        # Ensure that the sum of the first row is 2
        if torch.sum(adjacency_matrix[0]) != 2:
            continue

        # Ensure that the second coordinate is connected to the first coordinate
        if adjacency_matrix[0][1] != 1:
            continue

        # Ensure that only one other coordinate (besides the second) is connected to the first coordinate
        if torch.sum(adjacency_matrix[0][2:]) != 1:
            continue

        # Your original constraints
        row_sums = torch.sum(adjacency_matrix, dim=1)
        if not (row_sums == 0).any().item() and not (row_sums == 1).any().item():
            sum_adj_matrix = torch.sum(adjacency_matrix)
            if not share_adjoining_point(adjacency_matrix, 0, 1):


                # Convert the 2D array into a list of tuple points
                points = [(coord[0], coord[1]) for coord in coordinates]
                # Find the index of a point that's connected to points[0] but not the second coordinate
                connected_points = np.where(adjacency_matrix[0] == 1)[0]  # Find all points connected to points[0]
                connected_point = None

                for point in connected_points:
                    if point != 1:  # Exclude the second coordinate
                        connected_point = point
                        break

                if connected_point is None:
                    raise ValueError("No suitable point found connected to the first coordinate!")

                # Calculate the initial angle
                initial_angle = calculate_angle(points[0], points[connected_point])

                # Calculate the distance between points[0] and the connected point
                crank_distance = euclidean_distance(points[0], points[connected_point])

                # Create the Crank object with the calculated distance and angle
                crank = pl.Crank(0, 1, joint0=points[0], distance=crank_distance, angle=initial_angle, name="B")


                # Find the index of a point that's connected to points[1] but not the first coordinate
                connected_to_second = np.where(adjacency_matrix[1] == 1)[0]  # All points connected to points[1]
                connected_to_second_point = None

                for point in connected_to_second:
                    if point != 0:  # Exclude the first coordinate
                        connected_to_second_point = point
                        break

                if connected_to_second_point is None:
                    raise ValueError("No suitable point found connected to the second coordinate!")

                # Calculate the distances for Pivot
                distance0 = euclidean_distance(points[0], points[connected_to_second_point])
                distance1 = euclidean_distance(points[1], points[connected_to_second_point])

                # Create the Pivot object with the calculated distances
                pin = pl.Pivot(connected_to_second_point, 1, joint0=crank, joint1=points[1], distance0=distance0, distance1=distance1)

                # Linkage definition
                my_linkage = pl.Linkage(joints=(crank, pin), order=(crank, pin))
                # Visualization
                pl.show_linkage(my_linkage)
                visualize_linkage(coordinates, adjacency_matrix)
                print("attempt: ", attempt)
                print("final stage attempt: ", final_attempt)
                break
    except Exception as e:  # Catch any exception
        final_attempt += 1
        # print(f"An error occurred: {e}")  # Print the error for debugging
        continue  # Continue to the next iteration of the loop


# import pylinkage as pl

# # Main motor
# crank = pl.Crank(0, 1, joint0=(0, 0), angle=.31, distance=1, name="B")
# # Close the loop
# pin = pl.Pivot(
#     3,2, joint0=crank, joint1=(3, 0), 
#     distance0=3, distance1=2, 
#     # name="C"
# )

# # Linkage definition
# my_linkage = pl.Linkage(
#     joints=(crank, pin),
#     order=(crank, pin),
#     # name="My four-bar linkage"
# )

# # Visualization
# pl.show_linkage(my_linkage)