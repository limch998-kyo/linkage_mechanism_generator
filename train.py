import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
def euclidean_distance(point1, point2):
    return torch.norm(point1 - point2)

def rotation_matrix(angle):
    cos_val = torch.cos(angle)
    sin_val = torch.sin(angle)
    return torch.tensor([[cos_val, -sin_val], 
                         [sin_val, cos_val]])

def rotate_around_center(coordinates, angle, center):

    # Step 1: Translate to origin
    translated_coordinates = coordinates - center
    
    # Step 2: Rotate around origin
    rotated_coordinates = torch.matmul(translated_coordinates, rotation_matrix(angle))
    
    # Step 3: Translate back to original position
    rotated_coordinates_back = rotated_coordinates + center
    
    return rotated_coordinates_back

def circle_intercept(P1, r1, P2, r2):
    x1, y1 = P1
    x2, y2 = P2
    
    # Distance between the centers
    d = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Check for no solution
    if d > r1 + r2:
        return None, "Circles are too far apart to intersect."
    if d < torch.abs(r1 - r2):
        return None, "One circle is inside the other, without touching."
    
    # Check for coincidence
    if d == 0 and r1 == r2:
        return torch.tensor([x1.item(), y1.item()]), "Circles are coincident."
    
    # Calculate the intersection point(s)
    a = (r1**2 - r2**2 + d**2) / (2*d)
    h = torch.sqrt(r1**2 - a**2)
    
    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d
    
    # Two intersection points
    x4_1 = x3 + h * (y2 - y1) / d
    y4_1 = y3 - h * (x2 - x1) / d
    
    x4_2 = x3 - h * (y2 - y1) / d
    y4_2 = y3 + h * (x2 - x1) / d
    
    return torch.tensor([x4_1.item(), y4_1.item(), x4_2.item(), y4_2.item()]), None

import torch

def closest_point(input_coord, potential_coords):
    input_coord_x, input_coord_y = input_coord[0], input_coord[1]
    potential_coords_x = potential_coords[::2]
    potential_coords_y = potential_coords[1::2]

    distances = torch.sqrt((potential_coords_x - input_coord_x)**2 + (potential_coords_y - input_coord_y)**2)
    index = torch.argmin(distances)
    
    # Extract the closest point's coordinates
    closest_x = potential_coords_x[index]
    closest_y = potential_coords_y[index]

    # Return the closest point as a PyTorch tensor
    return torch.tensor([closest_x.item(), closest_y.item()], requires_grad=True)


def closest_intersection_point(input_coord, P1, r1, P2, r2):
    intersections, reason = circle_intercept(P1, r1, P2, r2)
    if intersections is not None:
        return closest_point(input_coord, intersections), None
    else:
        return None, reason

def check_linkage_valid(coor_val, all_coords, stage2_adjacency, target_adjacency, frame_num, angles_delta, crank_location, crank_lengths, status_location, link_fixeds, links_length, target_coords):

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

    for _ in range(frame_num):
        # First stage
        for i in range(0, 4, 2):
            if coor_val[i] == 1:
                crank_end = rotate_around_center(all_coords[i], angles_delta, crank_location)
                all_coords[i] = crank_end
                third_joint = closest_intersection_point(all_coords[i+1], all_coords[i], crank_lengths[i//2], status_location, link_fixeds[i//2])
                all_coords[i+1] = third_joint
                if third_joint is None:
                    return False

        # Second stage
        for i in range(4, 8):
            if coor_val[i] == 1:
                joint_a, joint_b = stage2_adjacency[i-4]
                moved_coord = closest_intersection_point(all_coords[i], all_coords[joint_a], links_length[i-4][0], all_coords[joint_b], links_length[i-4][1])
                all_coords[i] = moved_coord
                if moved_coord is None:
                    return False

        # Third stage
        joint_a, joint_b = target_adjacency
        moved_coord = closest_intersection_point(target_coords, all_coords[joint_a], links_length[-1][0], all_coords[joint_b], links_length[-1][1])
        if moved_coord is None:
            return False
        target_coords = moved_coord

    return True

def get_loss(coor_val, all_coords, target_coords, stage2_adjacency,target_adjacency,crank_location,status_location,target_location, frame_num=60):


    coor_val = coor_val.clone()
    all_coords = all_coords.clone()
    target_coords = target_coords.clone()
    stage2_adjacency = stage2_adjacency.clone()
    target_adjacency = target_adjacency.clone()



    angles_delta = torch.tensor(2 * torch.pi / 60)

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

    link_fixed = euclidean_distance(all_coords[1], status_location)
    link_fixed2 = euclidean_distance(all_coords[3], status_location)
    link_fixeds = torch.stack([link_fixed, link_fixed2])

    # Second stage
    # 5 rows (4 from the second stage + 1 from the third stage) and 2 columns (for link1 and link2 lengths)
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

    # Flatten links_length and convert to a tensor
    links_length_tensor = torch.FloatTensor(links_length.detach().numpy().flatten())

    # Concatenate all tensors
    all_lengths = torch.cat([crank_lengths, link_fixeds, links_length_tensor])

    # Compute the average
    overall_avg = torch.mean(all_lengths)

    # print("Average of all lengths:", overall_avg.item())


    loss = torch.tensor(0.0)
    # print(all_coords)

    loss += overall_avg

    for frame in range(frame_num):

        if not check_linkage_valid(coor_val.clone(), 
                            all_coords.clone(), 
                            stage2_adjacency.clone(), 
                            target_adjacency.clone(), 
                            frame_num.clone(), 
                            angles_delta.clone(), 
                            crank_location.clone(), 
                            crank_lengths.clone(), 
                            status_location.clone(), 
                            link_fixeds.clone(), 
                            links_length.clone(), 
                            target_coords.clone()
                            ):
            angles_delta = angles_delta * (-1.0)

# Assuming all variables are tensors.
        t = frame / frame_num # Ensure division is floating point.

        marker_offset = 0.5 * (1 - np.cos(2 * np.pi * t))

        # Assuming first_target_coord is a tensor of shape [2], 
        # you need to extract x and y components separately.
        marker_x_position = first_target_coord[0] + target_width * marker_offset
        marker_position = torch.tensor([marker_x_position, first_target_coord[1]], device=first_target_coord.device, requires_grad=True)

        crank_lengths_copy = crank_lengths.clone()
        link_fixeds_copy = link_fixeds.clone()
        links_length_copy = links_length.clone()
        
        # First stage
        # print('loop1')
        for i in range(0,4,2):
            if coor_val[i] == 1:
                # print(all_coords[i])
                crank_end = rotate_around_center(all_coords[i], angles_delta, crank_location)
                # print('cranck_end',crank_end)
                all_coords[i] = crank_end
                original_point1 = all_coords[i+1].clone()
                crank_length = crank_lengths_copy[i//2]
                link_fixed = link_fixeds_copy[i//2]
                while True:
                    third_joint, reason = closest_intersection_point(all_coords[i+1], all_coords[i], crank_length, status_location, link_fixed)
                    if third_joint is None:
                        if reason == "Circles are too far apart to intersect.":
                            crank_length += 0.1
                            link_fixed += 0.1
                        elif reason == "One circle is inside the other, without touching.":
                            if crank_length > link_fixed:
                                link_fixed += 0.1
                            elif crank_length < link_fixed:
                                crank_length += 0.1
                            else:
                                print('error')
                        else:
                            print('error')
                    else:
                        loss1 = euclidean_distance(third_joint, all_coords[i+1])
                        loss = loss + loss1
                        all_coords[i+1] = third_joint
                        crank_lengths_copy[i//2] = crank_length
                        link_fixeds_copy[i//2] = link_fixed
                        

                        break
                # print('third_joint',third_joint)

        # Second stage
        # print('loop2')
        for i in range(4, 8):
            if coor_val[i] == 1:
                joint_a, joint_b = stage2_adjacency[i-4]
                link1_length = links_length_copy[i-4][0]
                link2_length = links_length_copy[i-4][1]
                while True:
                    moved_coord, reason = closest_intersection_point(all_coords[i], all_coords[joint_a], link1_length, all_coords[joint_b], link2_length)
                    # print(moved_coord, reason)
                    if moved_coord is None:
                        if reason == "Circles are too far apart to intersect.":
                            link1_length += 0.1
                            link2_length += 0.1
                        elif reason == "One circle is inside the other, without touching.":
                            if link1_length > link2_length:
                                link2_length += 0.1
                            elif link1_length < link2_length:
                                link1_length += 0.1
                            else:
                                print('error')
                        else:
                            print('error')
                    else:
                        loss2 = euclidean_distance(moved_coord, all_coords[i])
                        # loss = loss + loss2
                        all_coords[i] = moved_coord
                        links_length_copy[i-4][0] = link1_length
                        links_length_copy[i-4][1] = link2_length
                        break
                # print(all_coords)
        # print('loop3')
        # Third stage
        joint_a, joint_b = target_adjacency
        # print(joint_a,joint_b)
        # print(target_coords)
        # print(links_length)
        link1_length = links_length_copy[-1][0]
        link2_length = links_length_copy[-1][1]
        while True:

            moved_coord, reason = closest_intersection_point(target_coords, all_coords[joint_a], link1_length, all_coords[joint_b], link2_length)
            # print(moved_coord, reason, all_coords[joint_a], all_coords[joint_b])
            if moved_coord is None:
                if reason == "Circles are too far apart to intersect.":
                    link1_length += 0.1
                    link2_length += 0.1
                elif reason == "One circle is inside the other, without touching.":
                    if link1_length > link2_length:
                        link2_length += 0.1
                    elif link1_length < link2_length:
                        link1_length += 0.1
                    else:
                        print('error')
                else:
                    print('error')
            else:
                loss3 = euclidean_distance(moved_coord, target_coords)
                # loss = loss + loss3
                target_coords = moved_coord
                links_length_copy[-1][0] = link1_length
                links_length_copy[-1][1] = link2_length
                break

        # diff = moved_coord - target_coords
        # target_coords = target_coords + diff

        # print(moved_coord, target_coords, marker_position)

        # print('target_coords',target_coords)
        # loss = loss + euclidean_distance(target_coords, marker_position)
        # print(target_coords, marker_position)
        # loss = loss + F.mse_loss(target_coords, marker_position)
    # loss.backward()
    # print(loss)
    # print('target_coords',target_coords.grad)
        # print(frame)
        # print('end loop')
    # print('loss',loss)
    # if overall_avg.item() > 5:
    #     loss = loss + (overall_avg - 5.0)

    return loss
    # def evaluate_linkage(self):
    #     moving_to_target1 = True
    #     initial_target_coords = self.target_coords
    #     initial_target_distance = euclidean_distance(initial_target_coords, self.first_target_coord)



    #     total_score = 0

    #     for _ in range(self.frame_num):

    #         # First stage
    #         for i in range(0,4,2):
    #             if self.coor_val[i] == 1:
    #                 crank_end = rotate_around_center(self.all_coords[i], self.angles_delta, self.crank_location)
    #                 self.all_coords[i] = crank_end
    #                 third_joint = closest_intersection_point(self.all_coords[i+1], self.all_coords[i], self.crank_lengths[i//2], self.status_location, self.link_fixeds[i//2])
    #                 self.all_coords[i+1] = third_joint


    #         # Second stage
    #         for i in range(4, 8):
    #             if self.coor_val[i] == 1:
    #                 joint_a, joint_b = self.stage2_adjacency[i-4]
    #                 moved_coord = closest_intersection_point(self.all_coords[i], self.all_coords[joint_a], self.links_length[i-4][0], self.all_coords[joint_b], self.links_length[i-4][1])
    #                 self.all_coords[i] = moved_coord

    #         # Third stage
    #         joint_a, joint_b = self.target_adjacency
    #         moved_coord = closest_intersection_point(self.target_coords, self.all_coords[joint_a], self.links_length[-1][0], self.all_coords[joint_b], self.links_length[-1][1])

    #         # print(self.target_coords,self.all_coords[joint_a], self.links_length[-1][0], self.all_coords[joint_b], self.links_length[-1][1])
    #         self.target_coords = moved_coord
    #         # print('candi',self.target_coords)
    #         if moving_to_target1:
    #             current_target_distance = euclidean_distance(self.target_coords, self.first_target_coord)
    #             distance_reduced = initial_target_distance - euclidean_distance(self.target_coords, self.first_target_coord)
    #             total_score += distance_reduced
    #             initial_target_coords = self.target_coords
    #             initial_target_distance = current_target_distance
            
    #             if euclidean_distance(self.target_coords, self.first_target_coord) < 0.5:
    #                 initial_target_distance = euclidean_distance(self.target_coords, self.second_target_coord)
    #                 moving_to_target1 = False
    #                 continue

    #         if not moving_to_target1:
    #             current_target_distance = euclidean_distance(self.target_coords, self.second_target_coord)
    #             distance_reduced = initial_target_distance - euclidean_distance(self.target_coords, self.second_target_coord)
    #             total_score += distance_reduced
    #             initial_target_coords = self.target_coords
    #             initial_target_distance = current_target_distance
    #             if euclidean_distance(self.target_coords, self.second_target_coord) < 0.5:
    #                 initial_target_distance = euclidean_distance(self.target_coords, self.first_target_coord)
    #                 moving_to_target1 = True
    #                 continue
    #         # print(self.links_length)



    #     return total_score