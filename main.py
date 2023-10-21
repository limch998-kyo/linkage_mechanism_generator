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


net = CombinedNetwork()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# env = LinkageEnvironment(input_tensor)
epochs = 10000

for epoch in range(epochs):

    coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = net(input_tensor)
    all_coords = all_coords/100.0
    target_coords = target_coords/200.0

    coor_val = torch.tensor([1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0])
    # print(target_adjacency)
    target_adjacency = torch.tensor([0,4])

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
    # if True:

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

        loss = get_loss(coor_val_copy, 
                 all_coords_copy, 
                 target_coords_copy, 
                 stage2_adjacency_copy,
                 target_adjacency_copy,
                 crank_location_tensor[0],
                 status_location_tensor[0],
                 target_location_tensor)

        # if overall_avg.item() > 10.0:
        #     if epoch % 10 == 0:
        #         print('epoch: ', epoch, 'mechanism invalid')
        #     for param in net.parameters():
        #         param.data = torch.randn_like(param)
        #     continue

        if epoch % 10 == 0:
            print('epoch: ', epoch, 'loss: ', loss.item(), 'score: ', score)
            if epoch % 100 == 0:
                mechanism = Linkage_mechanism(coor_val.copy(),
                                            all_coords.copy(), 
                                            target_coords.copy(), 
                                            stage2_adjacency.copy(), 
                                            target_adjacency.copy(), 
                                            crank_location.copy(), 
                                            status_location.copy(),
                                            target_location.copy(),
                                            epoch=epoch
                                            )
                mechanism.visualize_linkage()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        if epoch % 10 == 0:
            print('epoch: ', epoch, 'mechanism invalid')
        for param in net.parameters():
            param.data = torch.randn_like(param)
        continue



# net = CombinedNetwork()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# # env = LinkageEnvironment(input_tensor)
# epochs = 10000
# initilization_patience = 50
# fail_count = 0

# for epoch in range(epochs):
#     coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = net(input_tensor)

#     mechanism = reciprocate_movement(input_tensor)

#     loss = 1  # default small negative reward for invalid linkage

#     loss = mechanism.get_loss(
#         coor_val=coor_val,
#         stage2_adjacency=stage2_adjacency,
#         all_coords=all_coords,
#         target_adjacency=target_adjacency,
#         target_coords=target_coords,
#         crank_location=crank_location[0],
#         status_location=status_location[0],
#         target_location=target_location
#     )

#     loss = -loss



#     if loss == torch.tensor(1.0):
#         fail_count += 1
#         if fail_count > initilization_patience:

#         continue
#     else:
#         fail_count = 0



#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch % 100 == 0:
#         print(target_coords)
#         coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = output_process(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords)
#         print(target_coords)
#         crank_location_np = crank_location.detach().numpy()
#         status_location_np = status_location.detach().numpy()
#         target_location_np = target_location.detach().numpy()

#         mechanism = Linkage_mechanism(coor_val,
#                                       all_coords, 
#                                       target_coords, 
#                                       stage2_adjacency, 
#                                       target_adjacency, 
#                                       crank_location_np[0], 
#                                       status_location_np[0],
#                                       target_location_np
#                                       )
        
#         # if mechanism.check_linkage_valid():
#         print(loss)
#         print("valid linkage found at epoch: ", epoch)
#         # score = mechanism.evaluate_linkage()
#         # print('score: ', score)
#         mechanism.visualize_linkage()
    


# # using joints connected to crankpoints
# # make a gif visualization that simulates linkage mechanism when crank is rotating
# # predict each joints location from joint0 to joint7 and use this information to locate target joint

# # when predicting joints next frame location use intercept circle to calculate next location
