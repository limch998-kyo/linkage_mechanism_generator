# from points_generator import generate_coordinates
import torch

from utils import output_process
from src.linkage_builder import Linkage_mechanism
from src.loss import get_loss
from linkage_network import CombinedNetwork

from torch.optim.lr_scheduler import ExponentialLR


class Lingkage_mec_train():
    def __init__(self, crank_location, status_location, target_location, epochs=10000, lr=0.01, lr_min=0.0001, gamma=1.00,trajectory_type = 'linear', trajectory_data = [(-3, 0), (3, 0)], device = 'cpu',visualize_mec=False):
        self.net = CombinedNetwork()
        self.epochs = epochs
        self.lr = lr
        self.lr_min = lr_min
        self.gamma = gamma

        self.crank_location = crank_location
        self.status_location = status_location
        self.target_location = target_location

        # Convert each list into individual tensors
        if trajectory_type == 'linear':
            self.target_location_tensor = torch.tensor([trajectory_data[0],trajectory_data[1]], dtype=torch.float)
        else:
            self.target_location_tensor = torch.tensor(target_location, dtype=torch.float)
            # self.target_location_tensor = torch.tensor(target_location, dtype=torch.float)
        self.crank_location_tensor = torch.tensor([crank_location], dtype=torch.float)
        self.status_location_tensor = torch.tensor([status_location], dtype=torch.float)

        input = []
        input.append(crank_location)
        input.append(status_location)

        for i in range(len(target_location)):
            input.append(target_location[i])
        self.input_tensor = torch.tensor([input], dtype=torch.float)

        self.trajectory_type = trajectory_type
        self.trajectory_data = trajectory_data

        self.device = device

        self.visualize_mec = visualize_mec

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=1.00)

    def nan_to_num_hook(self, grad):
        if torch.isnan(grad).any():
            print("NaN gradient detected!")
            grad = torch.nan_to_num(grad)
        return grad

    def train(self):
        visualize = False
        is_lr_min = False  # Flag to check if current lr is lr_min
                # Register the gradient hook
        for param in self.net.parameters():
            param.register_hook(self.nan_to_num_hook)

        for epoch in range(self.epochs):
            coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = self.net(self.input_tensor)
            all_coords = all_coords*5.0
            target_coords = target_coords*5.0
            stage2_adjacency = torch.tensor([[0,1],[2,3],[0,0],[0,0]])
            coor_val = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0])
            target_adjacency = torch.tensor([4,5])

            if epoch % 10 == 0 and self.visualize_mec:
                visualize = True
            else:
                visualize = False

            loss = get_loss(coor_val, 
                        all_coords, 
                        target_coords, 
                        stage2_adjacency,
                        target_adjacency,
                        self.crank_location_tensor[0],
                        self.status_location_tensor[0],
                        self.target_location_tensor,
                        epoch,
                        device=self.device,
                        trajectory_type=self.trajectory_type,
                        trajectory_data=self.trajectory_data,
                        visualize=visualize)

            # Adjust learning rate based on loss value
            if not is_lr_min and loss.item() < 70:
                print(f"Reducing LR to {self.lr_min} as loss is below 70.")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_min
                is_lr_min = True
            elif is_lr_min and loss.item() > 70:
                print(f"Increasing LR to {self.lr} as loss is above 70.")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                is_lr_min = False

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scheduler.step()

            if epoch % 10 == 0:
                print('epoch: ', epoch, 'loss: ', loss.item())
