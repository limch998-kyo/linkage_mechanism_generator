
import matplotlib.pyplot as plt

def visualize_linkage_system(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, status_location, Make_GIF=False, frame_num=0):


    # Create a new figure and axis
    fig, ax = plt.subplots()

    xmin = -3
    xmax = 3
    ymin = -3
    ymax = 3

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

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

    if Make_GIF:
        plt.savefig(f"GIF_frames/frame_{frame_num}.png")
        plt.close()
        return
    else:
        plt.show()