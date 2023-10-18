# from points_generator import generate_coordinates
import torch

from utils import output_process
from GNN_network import CombinedNetwork
from src.linkage_builder import Linkage_mechanism
from train import get_loss





input = []
target_location = [[-5,5.5],[-5,4.5],[5,5.5],[5,4.5]]
crank_location = [0,0]
status_location = [1,0]


# Convert each list into individual tensors
target_location_tensor = torch.tensor(target_location, dtype=torch.float)
crank_location_tensor = torch.tensor([crank_location], dtype=torch.float)
status_location_tensor = torch.tensor([status_location], dtype=torch.float)


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
    coor_val_copy = coor_val
    stage2_adjacency_copy = stage2_adjacency
    all_coords_copy = all_coords
    target_adjacency_copy = target_adjacency
    target_coords_copy = target_coords
    # print(all_coords)
    coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = output_process(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords)
    # print(all_coords)
    mechanism = Linkage_mechanism(coor_val.copy(),
                                  all_coords.copy(), 
                                  target_coords.copy(), 
                                  stage2_adjacency.copy(), 
                                  target_adjacency.copy(), 
                                  crank_location.copy(), 
                                  status_location.copy(),
                                  target_location.copy()
                                  )
    # print(all_coords)


    if mechanism.check_linkage_valid():

        print("valid linkage found at attempt: ", attempt)

        mechanism = Linkage_mechanism(coor_val.copy(),
                                    all_coords.copy(), 
                                    target_coords.copy(), 
                                    stage2_adjacency.copy(), 
                                    target_adjacency.copy(), 
                                    crank_location.copy(), 
                                    status_location.copy(),
                                    target_location.copy()
                                    )

        score = mechanism.evaluate_linkage()

        total_score = get_loss(coor_val_copy, 
                 all_coords_copy, 
                 target_coords_copy, 
                 stage2_adjacency_copy,
                 target_adjacency_copy,
                 crank_location_tensor[0],
                 status_location_tensor[0],
                 target_location_tensor)
        print("score: ", score)
        print("total_score: ", total_score)
        break



    


# using joints connected to crankpoints
# make a gif visualization that simulates linkage mechanism when crank is rotating
# predict each joints location from joint0 to joint7 and use this information to locate target joint

# when predicting joints next frame location use intercept circle to calculate next location
