# from points_generator import generate_coordinates
import torch

from visiualizer import visualize_linkage_system
from GNN_network import CombinedNetwork
from src.linkage_builder import Linkage_mechanism





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
    flag = False
    attempt += 1
    net = CombinedNetwork()
    coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = net(input_tensor)

    coor_val = coor_val.detach().numpy()
    stage2_adjacency = stage2_adjacency.detach().numpy()
    all_coords = all_coords.detach().numpy()
    target_adjacency = target_adjacency.detach().numpy()
    target_coords = target_coords.detach().numpy()

    mechanism = Linkage_mechanism(coor_val, all_coords, target_coords, stage2_adjacency, target_adjacency, crank_location, status_location)

    if mechanism.check_linkage_valid():
        print("valid linkage found")
        mechanism.visualize_linkage()
        break


# using joints connected to crankpoints
# make a gif visualization that simulates linkage mechanism when crank is rotating
# predict each joints location from joint0 to joint7 and use this information to locate target joint

# when predicting joints next frame location use intercept circle to calculate next location
