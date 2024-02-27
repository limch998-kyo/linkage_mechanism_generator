import torch
import numpy as np
import torch.nn as nn
import os
import imageio
import math

from src.geometry_tensor import closest_intersection_point, rotate_around_center, euclidean_distance
from visiualizer import visualize_linkage_system


def check_linkage_valid(coor_val, all_coords, stage2_adjacency, target_adjacency, rotation, crank_location, crank_to_revolutions, status_location, link_fixeds, links_length, target_coords, marker_position, device):

    if torch.all(coor_val[:4] == 0):
        return False, None
    elif torch.all(coor_val[-4:] == 0):
        return False, None

    for i in range(4, 8):
        joint_a, joint_b = stage2_adjacency[i-4]
        if coor_val[i] == 0:
            continue
        elif coor_val[joint_a] == 0 or coor_val[joint_b] == 0:
            return False, None

    joint_a, joint_b = target_adjacency
    if coor_val[joint_a] == 0 or coor_val[joint_b] == 0:
        return False, None

    # First stage
    for i in range(0, 4, 2):
        if coor_val[i] == 1:
            crank_end = rotate_around_center(all_coords[i], rotation, crank_location)
            all_coords[i] = crank_end
            third_joint = closest_intersection_point(all_coords[i+1], all_coords[i], crank_to_revolutions[i//2], status_location, link_fixeds[i//2])
            if third_joint is None:
                return False, None
            all_coords[i+1] = third_joint
    # Second stage
    for i in range(4, 8):
        if coor_val[i] == 1:
            joint_a, joint_b = stage2_adjacency[i-4]
            moved_coord = closest_intersection_point(all_coords[i], all_coords[joint_a], links_length[i-4][0], all_coords[joint_b], links_length[i-4][1])
            if moved_coord is None:
                return False, None
            all_coords[i] = moved_coord
    # Third stage
    joint_a, joint_b = target_adjacency
    moved_coord = closest_intersection_point(target_coords, all_coords[joint_a], links_length[-1][0], all_coords[joint_b], links_length[-1][1])
    if moved_coord is None:
        return False, None
    target_coords = moved_coord

    criterion = nn.MSELoss()
    loss1 = criterion(target_coords, marker_position)

    return True, loss1

def get_loss(coor_val, all_coords, target_coords, stage2_adjacency, target_adjacency, crank_location, status_location, target_location, epoch, device, frame_num=60, visualize=False, trajectory_type='linear', trajectory_data=None, marker_trace=None):


    # Determine the lower left corner, width, and height of target location

    angles_delta = torch.tensor(2 * torch.pi / frame_num)

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
    output_link1_length = euclidean_distance(target_coords.clone(), all_coords[target_adjacency[0]])
    output_link2_length = euclidean_distance(target_coords.clone(), all_coords[target_adjacency[1]])

    links_length[4, 0] = output_link1_length
    links_length[4, 1] = output_link2_length

    # Flatten links_length directly without detaching
    links_length_tensor = links_length.flatten()


    # Concatenate all tensors
    all_lengths = torch.cat([crank_lengths, link_fixeds, links_length_tensor])

    # Compute the average
    overall_avg = torch.mean(all_lengths)

    loss = torch.tensor(0.0)


    target_trace = []  # Store the trace of the target_coords

    flag = False
    
    if trajectory_type == 'circular' or trajectory_type == 'circular2':
        direction_list = [0, -0.1, -1]
    elif trajectory_type == 'sine':
        direction_list = [0, 0.1, 1]
    elif trajectory_type == 'linear' or trajectory_type == 'linear2':
        direction_list = [-1, -0.1, 0,0.1, 1]

    for frame in range(frame_num):
        # Use the provided marker_trace for the current frame
        marker_position = marker_trace[frame]

        temp_loss_list = []
        temp_direction_list = []
        for direction in direction_list:

            rotation = direction * angles_delta
            linkage_valid, temp_loss =  check_linkage_valid(coor_val.clone(), 
                                all_coords.clone(), 
                                stage2_adjacency.clone(), 
                                target_adjacency.clone(), 
                                rotation, 
                                crank_location.clone(), 
                                crank_to_revolutions.clone(), 
                                status_location.clone(), 
                                link_fixeds.clone(), 
                                links_length.clone(), 
                                target_coords.clone(),
                                marker_position,
                                device
                                )
            if linkage_valid:
                temp_loss_list.append(temp_loss.clone().detach())
                temp_direction_list.append(direction)
            else:
                continue
        
        if len(temp_loss_list) == 0:
            print('error occured')
            return
        if trajectory_type != 'linear' and trajectory_type != 'linear2':
            if len(temp_loss_list) >1:
                temp_direction_list = temp_direction_list[1:]
                temp_loss_list = temp_loss_list[1:]
        min_loss = min(temp_loss_list)
        min_index = temp_loss_list.index(min_loss)
        
        rotation = temp_direction_list[min_index] * angles_delta


        if not flag:
            # First stage
            for i in range(0,4,2):
                if coor_val[i] == 1:
                    crank_end = rotate_around_center(all_coords[i], rotation, crank_location)
                    all_coords[i] = crank_end
                    if torch.isnan(crank_to_revolutions[i//2]):
                        print('error0')
                        print(crank_to_revolutions)
                        print(all_coords)
                        return
                    third_joint = closest_intersection_point(all_coords[i+1], all_coords[i], crank_to_revolutions[i//2], status_location, link_fixeds[i//2])
                    if third_joint is None:
                        print('error1')
                        return
                    else:
                        all_coords[i+1] = third_joint
                        

            # Second stage
            for i in range(4, 8):
                if coor_val[i] == 1:
                    joint_a, joint_b = stage2_adjacency[i-4]
                    moved_coord = closest_intersection_point(all_coords[i], all_coords[joint_a], links_length[i-4][0], all_coords[joint_b], links_length[i-4][1])
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

            moved_coord = closest_intersection_point(target_coords, all_coords[joint_a], links_length[4][0], all_coords[joint_b], links_length[4][1])
            if moved_coord is None:
                print('error3')
                return
            else:
                target_coords = moved_coord
            
        flag = False

        if visualize:   
            target_trace.append(tuple(target_coords.clone().detach().cpu().numpy()))

            visualize_linkage_system(
                coor_val.clone().detach().cpu().numpy(),
                stage2_adjacency.clone().detach().cpu().numpy(),
                all_coords.clone().detach().cpu().numpy(),
                target_adjacency.clone().detach().cpu().numpy(),
                target_coords.clone().detach().cpu().numpy(),
                crank_location.clone().detach().cpu().numpy(),
                status_location.clone().detach().cpu().numpy(),
                target_trace, 
                temp_direction_list[min_index],
                Make_GIF=True, 
                frame_num=frame,
                marker_position=marker_position.cpu().numpy(),  # Also convert marker_position to CPU if it's on CUDA
                marker_trace=marker_trace
            )


        criterion = nn.MSELoss()
        loss1 = criterion(target_coords, marker_position)
        loss = loss + loss1


    # Proximity penalty initialization
    proximity_threshold = 0.05  # Set the threshold distance
    proximity_penalty = 0.0
    # Distance from origin penalty initialization
    origin = torch.tensor([0.0, 0.0], device=device)  # Define the origin point
    distance_from_origin_penalty = 0.0
    max_distance_threshold = 20.0  # Define maximum allowed distance from origin

    # Calculate the proximity penalty for all unique pairs of coordinates using a linear penalty
    num_coords = all_coords.shape[0]
    for i in range(num_coords):
        for j in range(i + 1, num_coords):
            distance = euclidean_distance(all_coords[i], all_coords[j])
            if distance < proximity_threshold:
                # Apply linear penalty if distance is less than threshold
                proximity_penalty += proximity_threshold - distance

    # Also include the crank and stationary locations in the proximity checks with a linear penalty
    for i in range(num_coords):
        distance_to_crank = euclidean_distance(all_coords[i], crank_location)
        distance_to_stationary = euclidean_distance(all_coords[i], status_location)
        if distance_to_crank < proximity_threshold:
            proximity_penalty += proximity_threshold - distance_to_crank
        if distance_to_stationary < proximity_threshold:
            proximity_penalty += proximity_threshold - distance_to_stationary

        # Incorporate the proximity penalty into the total loss
        loss += proximity_penalty

    # Calculate the distance from origin penalty for all coordinates
    for i in range(all_coords.shape[0]):
        distance = euclidean_distance(all_coords[i], origin)
        if distance > max_distance_threshold:
            # Apply penalty if distance is greater than threshold
            distance_from_origin_penalty += (distance - max_distance_threshold)**2

    # Apply the same for the crank and stationary locations if needed
    if euclidean_distance(crank_location, origin) > max_distance_threshold:
        distance_from_origin_penalty += (euclidean_distance(crank_location, origin) - max_distance_threshold)**2

    if euclidean_distance(status_location, origin) > max_distance_threshold:
        distance_from_origin_penalty += (euclidean_distance(status_location, origin) - max_distance_threshold)**2

    # Incorporate the distance from origin penalty into the total loss
    loss += distance_from_origin_penalty

    if overall_avg.item() > 5:
        loss = loss + (overall_avg-5.0)*10.0
    if visualize:
        frames = []
        current_directory = os.getcwd()
        # base_dir = os.path.dirname(current_directory)
        for frame in range(frame_num):
            frames.append(imageio.imread(f"{current_directory}/GIF_frames/frame_{frame}.png"))

        imageio.mimsave(f'{current_directory}/learn_process/mechanism_{epoch}.gif', frames, duration=0.1)  # Adjust duration as needed
        # print("mechanism gif saved")



    return loss
