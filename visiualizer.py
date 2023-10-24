
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_linkage_system(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, status_location, target_location_info, target_trace=[], Make_GIF=False, frame_num=0, marker_position=None, fail_num=3):


    # Create a new figure and axis
    fig, ax = plt.subplots()

    xmin = -10
    xmax = 10
    ymin = -10
    ymax = 10

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    # Define colors for each stage
    colors = ['r', 'g', 'b']
    if fail_num >0:
    # Stage 1
        if coor_val[0] == 1:  # Link exists for the first set
            ax.plot([crank_location[0], all_coords[0][0]], [crank_location[1], all_coords[0][1]], color=colors[0])
            ax.plot([all_coords[0][0], all_coords[1][0]], [all_coords[0][1], all_coords[1][1]], color=colors[0])
            ax.plot([all_coords[1][0], status_location[0]], [all_coords[1][1], status_location[1]], color=colors[0])
        
        if coor_val[2] == 1:  # Link exists for the second set
            ax.plot([crank_location[0], all_coords[2][0]], [crank_location[1], all_coords[2][1]], color=colors[0])
            ax.plot([all_coords[2][0], all_coords[3][0]], [all_coords[2][1], all_coords[3][1]], color=colors[0])
            ax.plot([all_coords[3][0], status_location[0]], [all_coords[3][1], status_location[1]], color=colors[0])

    if fail_num >1:
        # Stage 2
        for i in range(4, 8):
            if coor_val[i] == 1:
                joint_a, joint_b = stage2_adjacency[i-4]
                ax.plot([all_coords[joint_a][0], all_coords[i][0]], [all_coords[joint_a][1], all_coords[i][1]], color=colors[1])
                ax.plot([all_coords[joint_b][0], all_coords[i][0]], [all_coords[joint_b][1], all_coords[i][1]], color=colors[1])

    if fail_num >2:
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

    if marker_position:
        ax.scatter(marker_position[0], marker_position[1], c='purple', marker='o', s=50)  # Change color, marker, and size as per your needs

    # Draw the given target locations rectangle

    lower_left, width, height = target_location_info

    rect = Rectangle(lower_left, width, height, edgecolor='cyan', facecolor='none')  # Change color as per your needs
    ax.add_patch(rect)

    # Draw the trace of the target_coords
    for (x, y) in target_trace:
        ax.scatter(x, y, c='magenta', marker='.', s=10)  # Change color and size as per your needs

    if Make_GIF:
        plt.savefig(f"GIF_frames/frame_{frame_num}.png")
        plt.close()
        return
    else:
        plt.show()