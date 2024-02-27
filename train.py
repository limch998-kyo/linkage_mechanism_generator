# from points_generator import generate_coordinates
import torch

from utils import output_process
from src.linkage_builder import Linkage_mechanism
from src.loss import get_loss
from linkage_network import CombinedNetwork

from torch.optim.lr_scheduler import ExponentialLR
from src.geometry_tensor import calculate_circular_position, calculate_sine_trajectory
import math
class Lingkage_mec_train():
    def __init__(self, crank_location, status_location, epochs=10000, lr=0.01, lr_min=0.0001, gamma=1.00,frame_num=60,trajectory_type = 'linear', trajectory_data = [(-3, 0), (3, 0)], device = 'cpu',visualize_mec=False):
        self.net = CombinedNetwork(input_size=frame_num+2)
        self.epochs = epochs
        self.lr = lr
        self.lr_min = lr_min
        self.loss_threshold = 50
        if trajectory_type == 'linear' or trajectory_type == 'linear2':
            self.loss_threshold = 10
        self.gamma = gamma

        self.crank_location = crank_location
        self.status_location = status_location

        # Convert each list into individual tensors
        if trajectory_type == 'linear' or trajectory_type == 'linear2':
            self.target_location_tensor = torch.tensor([trajectory_data[0],trajectory_data[1]], dtype=torch.float)
        else:
            self.target_location_tensor = None
            # self.target_location_tensor = torch.tensor(target_location, dtype=torch.float)
        self.crank_location_tensor = torch.tensor([crank_location], dtype=torch.float)
        self.status_location_tensor = torch.tensor([status_location], dtype=torch.float)

        # self.crank_location_tensor = self.crank_location_tensor.unsqueeze(0)
        # self.status_location_tensor = self.status_location_tensor.unsqueeze(0)

        # Calculate the total number of frames for one direction of the reciprocation
        half_frame_num = frame_num // 2
        marker_trace = []


        for frame in range(frame_num):

            # Determine the progress in the current half of the reciprocation cycle
            half_cycle_frame = frame % half_frame_num
            progress = half_cycle_frame / half_frame_num

            # Reverse the progress if we are in the second half of the reciprocation cycle
            if frame >= half_frame_num:
                progress = 1 - progress

            angle = 2 * math.pi * (frame / frame_num)
            if trajectory_type == 'linear' or trajectory_type == 'linear2':
                center_start, center_end = torch.tensor([trajectory_data[0],trajectory_data[1]], dtype=torch.float)   
                # Interpolate the marker's position between the start and end centers
                marker_position = center_start * (1 - progress) + center_end * progress
            elif trajectory_type == 'circular' or trajectory_type == 'circular2':
                center, radius = trajectory_data
                marker_position = calculate_circular_position(center, radius, angle)
            elif trajectory_type == 'sine':
                start_point, amplitude, wavelength, length = trajectory_data
                marker_position = calculate_sine_trajectory(start_point, amplitude, wavelength, length, frame, frame_num)
            else:
                raise ValueError("Unknown trajectory type")
            marker_trace.append(marker_position)

        # Step 2: Convert the list of positions into a tensor
        self.marker_positions_tensor = torch.stack(marker_trace)

        self.input_tensor = torch.cat((self.crank_location_tensor, self.status_location_tensor, self.marker_positions_tensor), dim=0)
        self.input_tensor = torch.transpose(self.input_tensor, 0, 1)

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
            all_coords, target_coords = self.net(self.input_tensor)
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
                            frame_num=60,  # or whatever your frame_num is
                            visualize=visualize,
                            trajectory_type=self.trajectory_type,
                            trajectory_data=self.trajectory_data,
                            marker_trace=self.marker_positions_tensor,
) 

            # Adjust learning rate based on loss value
            if not is_lr_min and loss.item() < self.loss_threshold:
                print(f"Reducing LR to {self.lr_min} as loss is below {self.loss_threshold}.")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_min
                is_lr_min = True
            elif is_lr_min and loss.item() > self.loss_threshold:
                print(f"Increasing LR to {self.lr} as loss is above {self.loss_threshold}.")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                is_lr_min = False

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 10 == 0:
                print('epoch: ', epoch, 'loss: ', loss.item())
