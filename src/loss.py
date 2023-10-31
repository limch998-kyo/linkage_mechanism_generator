import torch
import numpy as np
import torch.nn as nn

from src.geometry_tensor import closest_intersection_point, rotate_around_center, euclidean_distance

def check_linkage_valid(coor_val, all_coords, stage2_adjacency, target_adjacency, angles_delta, crank_location, crank_to_revolutions, status_location, link_fixeds, links_length, target_coords):

    coor_val = torch.tensor(coor_val)
    all_coords = torch.tensor(all_coords)

    if torch.all(coor_val[:4] == 0):
        return False
    elif torch.all(coor_val[-4:] == 0):
        return False

    for i in range(4, 8):
        joint_a, joint_b = stage2_adjacency[i-4]
        if coor_val[i] == 0:
            continue
        elif coor_val[joint_a] == 0 or coor_val[joint_b] == 0:
            return False

    joint_a, joint_b = target_adjacency
    if coor_val[joint_a] == 0 or coor_val[joint_b] == 0:
        return False

    # First stage
    for i in range(0, 4, 2):
        if coor_val[i] == 1:
            crank_end = rotate_around_center(all_coords[i], angles_delta, crank_location)
            all_coords[i] = crank_end
            third_joint, reason = closest_intersection_point(all_coords[i+1], all_coords[i], crank_to_revolutions[i//2], status_location, link_fixeds[i//2])
            if third_joint is None:
                return False
            all_coords[i+1] = third_joint
    # Second stage
    for i in range(4, 8):
        if coor_val[i] == 1:
            joint_a, joint_b = stage2_adjacency[i-4]
            moved_coord, reason = closest_intersection_point(all_coords[i], all_coords[joint_a], links_length[i-4][0], all_coords[joint_b], links_length[i-4][1])
            if moved_coord is None:
                return False
            all_coords[i] = moved_coord
    # Third stage
    joint_a, joint_b = target_adjacency
    moved_coord, reason = closest_intersection_point(target_coords, all_coords[joint_a], links_length[-1][0], all_coords[joint_b], links_length[-1][1])
    if moved_coord is None:
        return False
    target_coords = moved_coord

    return True

def get_loss(coor_val, all_coords, target_coords, stage2_adjacency,target_adjacency,crank_location,status_location,target_location, frame_num=60):

    angles_delta = torch.tensor(2 * torch.pi / frame_num)

    first_target_coord = (target_location[0] + target_location[1]) / 2
    second_target_coord = (target_location[2] + target_location[3]) /2
    target_width = target_location[2][0] - target_location[0][0]

    moving_to_target1 = True
    initial_target_coords = target_coords
    initial_target_distance = euclidean_distance(initial_target_coords,first_target_coord)

    # Defining Link lengths
    # First stage
    crank_length = euclidean_distance(crank_location, all_coords[0])
    crank_length2 = euclidean_distance(crank_location, all_coords[2])
    crank_lengths = torch.stack([crank_length, crank_length2])

    crank_to_revolution = euclidean_distance(all_coords[0], all_coords[1])
    crank_to_revolution2 = euclidean_distance(all_coords[2], all_coords[3])
    crank_to_revolutions = torch.stack([crank_to_revolution, crank_to_revolution2])

    link_fixed = euclidean_distance(all_coords[1], status_location)
    link_fixed2 = euclidean_distance(all_coords[3], status_location)
    link_fixeds = torch.stack([link_fixed, link_fixed2])

    # Second stage
    # 5 rows (4 from the second stage + 1 from the third stage) and 2 columns (for link1 and link2 lengths)
    # Assuming all_coords, stage2_adjacency, target_coords, and target_adjacency are defined elsewhere
    links_length = torch.zeros(5, 2)

    # Second stage
    for i in range(4, 8):
        link1_length = euclidean_distance(all_coords[i], all_coords[stage2_adjacency[i-4][0]])
        link2_length = euclidean_distance(all_coords[i], all_coords[stage2_adjacency[i-4][1]])
        
        links_length[i-4, 0] = link1_length
        links_length[i-4, 1] = link2_length

    # Third stage
    output_link1_length = euclidean_distance(target_coords, all_coords[target_adjacency[0]])
    output_link2_length = euclidean_distance(target_coords, all_coords[target_adjacency[1]])

    links_length[4, 0] = output_link1_length
    links_length[4, 1] = output_link2_length

    # Flatten links_length directly without detaching
    links_length_tensor = links_length.flatten()

    # Concatenate all tensors
    all_lengths = torch.cat([crank_lengths, link_fixeds, links_length_tensor])

    # Compute the average
    overall_avg = torch.mean(all_lengths)

    loss = torch.tensor(0.0)


    for frame in range(frame_num):
        if not check_linkage_valid(coor_val.clone(), 
                            all_coords.clone(), 
                            stage2_adjacency.clone(), 
                            target_adjacency.clone(), 
                            angles_delta.clone(), 
                            crank_location.clone(), 
                            crank_to_revolutions.clone(), 
                            status_location.clone(), 
                            link_fixeds.clone(), 
                            links_length.clone(), 
                            target_coords.clone()
                            ):
            angles_delta = angles_delta * (-1.0)

            # print('angle changing')

# Assuming all variables are tensors.
        t = frame / frame_num # Ensure division is floating point.

        marker_offset = 0.5 * (1 - np.cos(2 * np.pi * t))

        # Assuming first_target_coord is a tensor of shape [2], 
        # you need to extract x and y components separately.
        marker_x_position = first_target_coord[0] + target_width * marker_offset
        marker_position = torch.tensor([marker_x_position, first_target_coord[1]], device=first_target_coord.device)


        # First stage
        for i in range(0,4,2):
            if coor_val[i] == 1:
                crank_end = rotate_around_center(all_coords[i], angles_delta, crank_location)
                all_coords[i] = crank_end
                third_joint, reason = closest_intersection_point(all_coords[i+1], all_coords[i], crank_to_revolutions[i//2], status_location, link_fixeds[i//2])
                if third_joint is None:
                    print('error1')
                    return
                else:
                    all_coords[i+1] = third_joint
                    

        # Second stage
        for i in range(4, 8):
            if coor_val[i] == 1:
                joint_a, joint_b = stage2_adjacency[i-4]
                moved_coord, reason = closest_intersection_point(all_coords[i], all_coords[joint_a], links_length[i-4][0], all_coords[joint_b], links_length[i-4][1])
                if moved_coord is None:
                    print(joint_a)
                    print(joint_b)
                    print(all_coords[i],all_coords[joint_a],link1_length, all_coords[joint_b],link2_length)
                    print('error2')
                    return
                else:
                    all_coords[i] = moved_coord
                        
        # Third stage
        joint_a, joint_b = target_adjacency


        moved_coord, reason = closest_intersection_point(target_coords, all_coords[joint_a], links_length[-1][0], all_coords[joint_b], links_length[-1][1])
        # print(moved_coord, reason, all_coords[joint_a], all_coords[joint_b])
        if moved_coord is None:
            print('error3')
            return
        else:
            # diff = moved_coord - target_coords
            # target_coords = target_coords + diff
            target_coords = moved_coord
                

        criterion = nn.MSELoss()
        loss1 = criterion(target_coords, marker_position)
        loss = loss + loss1


    if overall_avg.item() > 5:
        loss = loss + (overall_avg-5.0)*10.0

    return loss
