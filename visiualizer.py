import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def rotate_point_around_origin(px, py, angle):
    """Rotate a point around the origin (0, 0)."""
    qx = px * np.cos(angle) - py * np.sin(angle)
    qy = px * np.sin(angle) + py * np.cos(angle)
    return qx, qy

def visualize_with_rotation(coordinates, output, rotation_angle):
    """Visualize coordinates with a specific rotation around the first coordinate."""
    
    fig, ax = plt.subplots()
    fixed_point = coordinates[0]
    
    # Highlight the fixed axis point
    ax.scatter(*fixed_point, color='red', s=100, label='Fixed Point')
    
    # Rotate and plot the other points
    for px, py in coordinates[1:]:
        qx, qy = rotate_point_around_origin(px - fixed_point[0], py - fixed_point[1], rotation_angle)
        qx += fixed_point[0]
        qy += fixed_point[1]
        ax.scatter(qx, qy, color='blue', s=50)
    
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.set_xlim([-0.5, 2.0])  # You can adjust these limits based on your data
    ax.set_ylim([-0.5, 2.0])

    # Convert the figure to an image and return
    fig.canvas.draw()
    img_arr = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return img_arr

