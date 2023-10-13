import numpy as np

def generate_coordinates(num_points=5, x_limit=[0, 1], y_limit=[0, 1]):
    x_coords = np.random.uniform(x_limit[0], x_limit[1], num_points)
    y_coords = np.random.uniform(y_limit[0], y_limit[1], num_points)
    
    return list(zip(x_coords, y_coords))