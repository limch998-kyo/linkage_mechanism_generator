# from points_generator import generate_coordinates
import torch

from utils import output_process
from GNN_network import CombinedNetwork
from src.linkage_builder import Linkage_mechanism





input = []
target_location = [[-2,2.5],[-2,1.5],[2,2.5],[2,1.5]]
crank_location = [0,0]
status_location = [1,0]

input.append(crank_location)
input.append(status_location)

for i in range(len(target_location)):
    input.append(target_location[i])

input_tensor = torch.tensor([input], dtype=torch.float)

attempt = 0
while True:
    flag = False
    attempt += 1
    net = CombinedNetwork()
    coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = net(input_tensor)
    coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = output_process(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords)

    mechanism = Linkage_mechanism(coor_val,
                                  all_coords, 
                                  target_coords, 
                                  stage2_adjacency, 
                                  target_adjacency, 
                                  crank_location, 
                                  status_location,
                                  target_location
                                  )

    if mechanism.check_linkage_valid():
        print("valid linkage found at attempt: ", attempt)
        mechanism.visualize_linkage()
        break


# using joints connected to crankpoints
# make a gif visualization that simulates linkage mechanism when crank is rotating
# predict each joints location from joint0 to joint7 and use this information to locate target joint

# when predicting joints next frame location use intercept circle to calculate next location
