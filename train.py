# from points_generator import generate_coordinates
import torch

from utils import output_process
from src.linkage_builder import Linkage_mechanism
from src.loss import get_loss


from torch.optim.lr_scheduler import ExponentialLR


class Lingkage_mec_train():
    def __init__(self, net, crank_location, status_location, target_location, epochs=10000, lr=0.01, gamma=1.00, visualize_mec=False):
        self.net = net
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma

        self.crank_location = crank_location
        self.status_location = status_location
        self.target_location = target_location

        # Convert each list into individual tensors
        self.target_location_tensor = torch.tensor(target_location, dtype=torch.float)
        self.crank_location_tensor = torch.tensor([crank_location], dtype=torch.float)
        self.status_location_tensor = torch.tensor([status_location], dtype=torch.float)

        input = []
        input.append(crank_location)
        input.append(status_location)
        for i in range(len(target_location)):
            input.append(target_location[i])
        self.input_tensor = torch.tensor([input], dtype=torch.float)

        self.visualize_mec = visualize_mec

        self.optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        self.scheduler = ExponentialLR(self.optimizer, gamma=1.00)

    def train(self):
        for epoch in range(self.epochs):

            coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = self.net(self.input_tensor)
            all_coords = all_coords*5.0
            target_coords = target_coords*5.0
            stage2_adjacency = torch.tensor([[0,1],[2,3],[0,0],[0,0]])
            coor_val = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0])
            target_adjacency = torch.tensor([4,5])

            loss = get_loss(coor_val, 
                        all_coords, 
                        target_coords, 
                        stage2_adjacency,
                        target_adjacency,
                        self.crank_location_tensor[0],
                        self.status_location_tensor[0],
                        self.target_location_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 10 == 0:
                print('epoch: ', epoch, 'loss: ', loss.item())
                if epoch % 100 == 0 and self.visualize_mec:
                    coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = output_process(coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords)
                    mechanism = Linkage_mechanism(coor_val.copy(),
                                                all_coords.copy(), 
                                                target_coords.copy(), 
                                                stage2_adjacency.copy(), 
                                                target_adjacency.copy(), 
                                                self.crank_location.copy(), 
                                                self.status_location.copy(),
                                                self.target_location.copy(),
                                                epoch=epoch
                                                )

                    mechanism.visualize_linkage()