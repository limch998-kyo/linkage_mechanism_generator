import numpy as np
import imageio
import os
from geometry import rotate_around_center, closest_intersection_point, euclidean_distance
from visiualizer import visualize_linkage_system



class Linkage_mechanism():
    def __init__(self, coor_val, all_coords,target_coords,stage2_adjacency, target_adjacency, crank_location, status_location, target_location,epoch = 0,frame_num=60, angles_delta=2*np.pi/60):
        # visualize_linkage_system(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, status_location)
        self.coor_val = coor_val
        self.all_coords = all_coords
        self.target_coords = target_coords
        self.stage2_adjacency = stage2_adjacency
        self.target_adjacency = target_adjacency
        self.crank_location = crank_location
        self.status_location = status_location
        self.target_location = target_location

        self.epoch = epoch
        self.frame_num = frame_num
        self.angles_delta = angles_delta

        self.sorted_target_locations = sorted(target_location, key=lambda k: (k[0], k[1]))

        self.first_target_coord = [(target_location[0][0]+target_location[1][0])/2, (target_location[0][1]+target_location[1][1])/2]
        self.second_target_coord = [(target_location[2][0]+target_location[3][0])/2, (target_location[2][1]+target_location[3][1])/2]

        # Determine the lower left corner, width, and height of target location
        self.target_lower_left = self.sorted_target_locations[0]
        self.target_width = self.sorted_target_locations[3][0] - self.sorted_target_locations[0][0]
        self.target_height = self.sorted_target_locations[3][1] - self.sorted_target_locations[0][1]
        self.target_location_info = [self.target_lower_left, self.target_width, self.target_height]

        #Defining Link lengths
        #First stage
        self.crank_length = euclidean_distance(self.crank_location, self.all_coords[0])
        self.crank_length2 = euclidean_distance(self.crank_location, self.all_coords[2])
        self.crank_lengths = [self.crank_length, self.crank_length2]

        self.link_fixed = euclidean_distance(self.all_coords[1], self.status_location)
        self.link_fixed2 = euclidean_distance(self.all_coords[3], self.status_location)
        self.link_fixeds = [self.link_fixed, self.link_fixed2]

        #Second stage

        self.links_length = []
        for i in range(4, 8):
            link1_length = euclidean_distance(self.all_coords[i], self.all_coords[self.stage2_adjacency[i-4][0]])
            link2_length = euclidean_distance(self.all_coords[i], self.all_coords[self.stage2_adjacency[i-4][1]])
            self.links_length.append([link1_length, link2_length])

        #Third stage
        self.output_link1_length = euclidean_distance(self.target_coords, self.all_coords[self.target_adjacency[0]])
        self.output_link2_length = euclidean_distance(self.target_coords, self.all_coords[self.target_adjacency[1]])
        self.links_length.append([self.output_link1_length, self.output_link2_length])

        # Flatten links_length
        self.links_length_array = np.array(self.links_length).flatten()
        
        # Concatenate all arrays
        self.all_lengths = np.concatenate([self.crank_lengths, self.link_fixeds, self.links_length_array])
        
        # Compute the average
        self.overall_avg = np.mean(self.all_lengths)

    def check_linkage_valid(self, coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, crank_lengths,link_fixeds, links_length,status_location, frame_num, angles_delta):

        if np.all(coor_val[:4] == 0):
            return False
        elif np.all(coor_val[-4:] == 0):
            return False

        for i in range(4, 8):
            joint_a, joint_b = stage2_adjacency[i-4]
            if self.coor_val[i]==0:
                continue
            elif coor_val[joint_a] == 0 or coor_val[joint_b] == 0:
                return False


        joint_a, joint_b = target_adjacency
        if coor_val[joint_a] == 0 or coor_val[joint_b] == 0:
            return False


        for _ in range(frame_num):
            # First stage
            for i in range(0,4,2):
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
    

    def visualize_linkage(self):
        target_trace = []  # Store the trace of the target_coords

        for frame in range(self.frame_num):
            print(self.stage2_adjacency)

            print
            if not self.check_linkage_valid(self.coor_val.copy(), 
                                    self.stage2_adjacency.copy(), 
                                    self.all_coords.copy(), 
                                    self.target_adjacency.copy(), 
                                    self.target_coords, 
                                    self.crank_location.copy(), 
                                    self.crank_lengths.copy(),
                                    self.link_fixeds.copy(), 
                                    self.links_length.copy(),
                                    self.status_location.copy(), 
                                    self.frame_num,
                                    self.angles_delta
                                    ):
                self.angles_delta = -self.angles_delta

            t = frame / self.frame_num  # t varies from 0 to 1
            marker_offset = 0.5 * (1 - np.cos(2 * np.pi * t))  # This will vary between 0 and 1 in a back-and-forth manner due to the cosine function

            marker_x_position = self.target_lower_left[0] + self.target_width * marker_offset
            marker_position = (marker_x_position, self.target_lower_left[1] + self.target_height/2)  

            # First stage
            for i in range(0,4,2):
                if self.coor_val[i] == 1:
                    crank_end = rotate_around_center(self.all_coords[i], self.angles_delta, self.crank_location)
                    self.all_coords[i] = crank_end
                    third_joint = closest_intersection_point(self.all_coords[i+1], self.all_coords[i], self.crank_lengths[i//2], self.status_location, self.link_fixeds[i//2])
                    self.all_coords[i+1] = third_joint


            # Second stage
            for i in range(4, 8):
                if self.coor_val[i] == 1:
                    joint_a, joint_b = self.stage2_adjacency[i-4]
                    moved_coord = closest_intersection_point(self.all_coords[i], self.all_coords[joint_a], self.links_length[i-4][0], self.all_coords[joint_b], self.links_length[i-4][1])
                    self.all_coords[i] = moved_coord

            # Third stage
            joint_a, joint_b = self.target_adjacency
            moved_coord = closest_intersection_point(self.target_coords, self.all_coords[joint_a], self.links_length[-1][0], self.all_coords[joint_b], self.links_length[-1][1])

            self.target_coords = moved_coord

            # Append the current target_coords to the trace list
            target_trace.append(tuple(self.target_coords))

            visualize_linkage_system(self.coor_val, 
                                    self.stage2_adjacency, 
                                    self.all_coords, 
                                    self.target_adjacency, 
                                    self.target_coords, 
                                    self.crank_location, 
                                    self.status_location,
                                    self.target_location_info, 
                                    target_trace, 
                                    Make_GIF=True, 
                                    frame_num=frame,
                                    marker_position=marker_position
                                    )

        frames = []
        current_directory = os.getcwd()
        # base_dir = os.path.dirname(current_directory)
        for frame in range(self.frame_num):
            frames.append(imageio.imread(f"{current_directory}/GIF_frames/frame_{frame}.png"))

        imageio.mimsave(f'{current_directory}/learn_process/mechanism_{self.epoch}.gif', frames, duration=0.1)  # Adjust duration as needed
        print("mechanism gif saved")



    def evaluate_linkage(self):
        # moving_to_target1 = True
        # initial_target_coords = self.target_coords
        # initial_target_distance = euclidean_distance(initial_target_coords, self.first_target_coord)


        # Calculating averages
        # print(self.overall_avg)

        loss = 0.0
        # print(self.all_coords)

        for frame in range(self.frame_num):


            t = frame / self.frame_num  # t varies from 0 to 1
            marker_offset = 0.5 * (1 - np.cos(2 * np.pi * t))  # This will vary between 0 and 1 in a back-and-forth manner due to the cosine function

            marker_x_position = self.target_lower_left[0] + self.target_width * marker_offset
            marker_position = (marker_x_position, self.target_lower_left[1] + self.target_height/2)  

            # First stage
            for i in range(0,4,2):
                if self.coor_val[i] == 1:
                    # print(self.all_coords[i])
                    crank_end = rotate_around_center(self.all_coords[i], self.angles_delta, self.crank_location)
                    # print('crank_end:',crank_end)
                    self.all_coords[i] = crank_end
                    third_joint = closest_intersection_point(self.all_coords[i+1], self.all_coords[i], self.crank_lengths[i//2], self.status_location, self.link_fixeds[i//2])
                    self.all_coords[i+1] = third_joint
                    # print('third_joint',third_joint)


            # Second stage
            for i in range(4, 8):
                if self.coor_val[i] == 1:
                    joint_a, joint_b = self.stage2_adjacency[i-4]
                    moved_coord = closest_intersection_point(self.all_coords[i], self.all_coords[joint_a], self.links_length[i-4][0], self.all_coords[joint_b], self.links_length[i-4][1])
                    self.all_coords[i] = moved_coord
                    # print(self.all_coords)
                    

            # Third stage
            joint_a, joint_b = self.target_adjacency
            # print(joint_a, joint_b)
            # print(self.target_coords)
            moved_coord = closest_intersection_point(self.target_coords, self.all_coords[joint_a], self.links_length[-1][0], self.all_coords[joint_b], self.links_length[-1][1])
            # print(self.links_length)
            # print(self.target_coords,self.all_coords[joint_a], self.links_length[-1][0], self.all_coords[joint_b], self.links_length[-1][1])
            self.target_coords = moved_coord

            loss = loss + euclidean_distance(self.target_coords, marker_position)
            # print(self.links_length)

        if self.overall_avg > 5:
            loss = np.sqrt(loss + (self.overall_avg - 5.0))
        else:
            loss = np.sqrt(loss)
        return loss