from matplotlib import pyplot as plt
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