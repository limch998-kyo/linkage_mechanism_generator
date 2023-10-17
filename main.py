# from points_generator import generate_coordinates
import torch
import numpy as np
import imageio

from geometry import rotate_around_center, closest_intersection_point, euclidean_distance
from visiualizer import visualize_linkage_system
from GNN_network import CombinedNetwork
from 





input = []
target_location = [[5,5],[8,5],[5,4],[8,4]]
crank_location = [0,0]
status_location = [1,0]

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
        # coor_val = np.array([1, 1, 0, 0, 1, 1, 0, 0])

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



        # visualize_linkage_system(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, status_location)

        frame_num = 60
        angles = np.linspace(0, 4*np.pi, frame_num)  # You can adjust this as needed
        angles_delta = 2*np.pi/30

        linkage_valid = True

        crank_length = euclidean_distance(crank_location, all_coords[0])
        crank_length2 = euclidean_distance(crank_location, all_coords[2])

        link_fixed = euclidean_distance(all_coords[1], status_location)
        link_fixed2 = euclidean_distance(all_coords[3], status_location)

        #Second stage

        links_length = []
        for i in range(4, 8):
            link1_length = euclidean_distance(all_coords[i], all_coords[stage2_adjacency[i-4][0]])
            link2_length = euclidean_distance(all_coords[i], all_coords[stage2_adjacency[i-4][1]])
            links_length.append([link1_length, link2_length])

        #Third stage
        link1_length = euclidean_distance(target_coords, all_coords[target_adjacency[0]])
        link2_length = euclidean_distance(target_coords, all_coords[target_adjacency[1]])
        links_length.append([link1_length, link2_length])

        for frame in range(frame_num):
            # First stage
            if coor_val[0] == 1:
                crank_end = rotate_around_center(all_coords[0], angles_delta, crank_location)
                all_coords[0] = crank_end
                third_joint = closest_intersection_point(all_coords[1], all_coords[0], crank_length, status_location, link_fixed)
                all_coords[1] = third_joint
                if third_joint is None:
                    linkage_valid = False
                    break

            if coor_val[2] == 1:
                crank_end2 = rotate_around_center(all_coords[2], angles_delta, crank_location)
                all_coords[2] = crank_end2
                third_joint2 = closest_intersection_point(all_coords[3], all_coords[2], crank_length2, status_location, link_fixed2)
                all_coords[3] = third_joint2
                if third_joint2 is None:
                    linkage_valid = False
                    break

            # Second stage
            for i in range(4, 8):
                if coor_val[i] == 1:
                    joint_a, joint_b = stage2_adjacency[i-4]
                    moved_coord = closest_intersection_point(all_coords[i], all_coords[joint_a], links_length[i-4][0], all_coords[joint_b], links_length[i-4][1])
                    all_coords[i] = moved_coord
                    if moved_coord is None:
                        linkage_valid = False
                        break

            if not linkage_valid:
                break

            # Third stage
            joint_a, joint_b = target_adjacency
            moved_coord = closest_intersection_point(target_coords, all_coords[joint_a], links_length[-1][0], all_coords[joint_b], links_length[-1][1])
            target_coords = moved_coord

            visualize_linkage_system(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, status_location, Make_GIF=True, frame_num=frame)

        if not linkage_valid:
            continue
        frames = []
        for frame in range(frame_num):
            frames.append(imageio.imread(f"GIF_frames/frame_{frame}.png"))

        imageio.mimsave('mechanism.gif', frames, duration=0.1)  # Adjust duration as needed

        # Delete the temporary frames
        # for angle in angles:
        #     os.remove(f"GIF_frames/frame_{angle}.png")

        break
    except Exception as e:  # Catch any exception
        print(f"An error occurred: {e}")  # Print the error for debugging
        continue  # Continue to the next iteration of the loop

# using joints connected to crankpoints
# make a gif visualization that simulates linkage mechanism when crank is rotating
# predict each joints location from joint0 to joint7 and use this information to locate target joint

# when predicting joints next frame location use intercept circle to calculate next location
