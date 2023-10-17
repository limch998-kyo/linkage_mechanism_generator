import numpy as np
import imageio
import os
from geometry import rotate_around_center, closest_intersection_point, euclidean_distance
from visiualizer import visualize_linkage_system



class Linkage_mechanism():
    def __init__(self, coor_val, all_coords,target_coords,stage2_adjacency, target_adjacency, crank_location, status_location, frame_num=30, angles_delta=2*np.pi/30):
        # visualize_linkage_system(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords, crank_location, status_location)
        self.coor_val = coor_val
        self.all_coords = all_coords
        self.target_coords = target_coords
        self.stage2_adjacency = stage2_adjacency
        self.target_adjacency = target_adjacency
        self.crank_location = crank_location
        self.status_location = status_location

        self.frame_num = frame_num
        self.angles_delta = angles_delta

        #Defining Link lengths
        #First stage
        self.crank_length = euclidean_distance(self.crank_location, self.all_coords[0])
        self.crank_length2 = euclidean_distance(self.crank_location, self.all_coords[2])

        self.link_fixed = euclidean_distance(self.all_coords[1], self.status_location)
        self.link_fixed2 = euclidean_distance(self.all_coords[3], self.status_location)

        #Second stage

        self.links_length = []
        for i in range(4, 8):
            link1_length = euclidean_distance(self.all_coords[i], self.all_coords[self.stage2_adjacency[i-4][0]])
            link2_length = euclidean_distance(self.all_coords[i], self.all_coords[self.stage2_adjacency[i-4][1]])
            self.links_length.append([link1_length, link2_length])

        #Third stage
        self.output_link1_length = euclidean_distance(self.target_coords, self.all_coords[self.target_adjacency[0]])
        self.output_link2_length = euclidean_distance(self.target_coords, self.all_coords[self.target_adjacency[1]])
        self.links_length.append([link1_length, link2_length])


    def check_linkage_valid(self):

        if np.all(self.coor_val[:4] == 0):
            return False
        elif np.all(self.coor_val[-4:] == 0):
            return False

        for i in range(4, 8):
            joint_a, joint_b = self.stage2_adjacency[i-4]
            if self.coor_val[i]==0:
                continue
            elif self.coor_val[joint_a] == 0 or self.coor_val[joint_b] == 0:
                return False


        joint_a, joint_b = self.target_adjacency
        if self.coor_val[joint_a] == 0 or self.coor_val[joint_b] == 0:
            return False

        for frame in range(self.frame_num):
            # First stage
            for i in range(0,4,2):
                if self.coor_val[i] == 1:
                    crank_end = rotate_around_center(self.all_coords[i], self.angles_delta, self.crank_location)
                    self.all_coords[i] = crank_end
                    third_joint = closest_intersection_point(self.all_coords[i+1], self.all_coords[i], self.crank_length, self.status_location, self.link_fixed)
                    self.all_coords[1] = third_joint
                    if third_joint is None:
                        return False


            # Second stage
            for i in range(4, 8):
                if self.coor_val[i] == 1:
                    joint_a, joint_b = self.stage2_adjacency[i-4]
                    moved_coord = closest_intersection_point(self.all_coords[i], self.all_coords[joint_a], self.links_length[i-4][0], self.all_coords[joint_b], self.links_length[i-4][1])
                    self.all_coords[i] = moved_coord
                    if moved_coord is None:
                        return False

            # Third stage
            joint_a, joint_b = self.target_adjacency
            moved_coord = closest_intersection_point(self.target_coords, self.all_coords[joint_a], self.links_length[-1][0], self.all_coords[joint_b], self.links_length[-1][1])
            if moved_coord is None:
                return False
            target_coords = moved_coord
        return True
    
    def visualize_linkage(self):


        frame_num = 30
        angles_delta = 2*np.pi/frame_num

        for frame in range(frame_num):
            # First stage
            for i in range(0,4,2):
                if self.coor_val[i] == 1:
                    crank_end = rotate_around_center(self.all_coords[i], angles_delta, self.crank_location)
                    self.all_coords[i] = crank_end
                    third_joint = closest_intersection_point(self.all_coords[i+1], self.all_coords[i], self.crank_length, self.status_location, self.link_fixed)
                    self.all_coords[1] = third_joint


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

            visualize_linkage_system(self.coor_val, self.stage2_adjacency, self.all_coords, self.target_adjacency, self.target_coords, self.crank_location, self.status_location, Make_GIF=True, frame_num=frame)

        frames = []
        current_directory = os.getcwd()
        # base_dir = os.path.dirname(current_directory)
        for frame in range(frame_num):
            frames.append(imageio.imread(f"{current_directory}/GIF_frames/frame_{frame}.png"))

        imageio.mimsave(f'{current_directory}/mechanism.gif', frames, duration=0.1)  # Adjust duration as needed
        print("mechanism gif saved")