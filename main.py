from linkage_network import CombinedNetwork
from torch.optim.lr_scheduler import ExponentialLR
from train import Lingkage_mec_train
import math
# from utils import seed_everything
import torch
import random
import numpy as np


def seed_everything(seed=42):
    random.seed(seed)  # Seed Python's random module
    np.random.seed(seed)  # Seed Numpy (make sure to import numpy as np)
    torch.manual_seed(seed)  # Seed torch for CPU operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Seed torch for CUDA operations
        torch.cuda.manual_seed_all(seed)  # Seed all GPUs if there are multiple
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

crank_location = [-2,0]
status_location = [2,0]

# Define trajectory type and data
trajectory_type = 'linear'  # Options: 'linear', 'circular', 'elliptical', 'sine'
# Example trajectory data for circular: (center, radius)
# Adjust this based on the selected trajectory_type

if trajectory_type == 'linear2':
    crank_location = [-7.5,0]
    status_location = [-3.5,0]
    

trajectory_data = {'circular': [(0, 0), 4],
                   'circular2': [(0, 6), 3],
                    'sine': [(-5, 5), 2, 50, 10],  # start_point, amplitude, wavelength, length
                    'linear': [(-5, 5), (5, 5)],
                    'linear2': [(-2.5,5),(7.5,5)]}  # start and end points

selected_trajectory_data = trajectory_data[trajectory_type]

# net = CombinedNetwork()
# device = torch.device("cpu")
# net = net.to(device)

epochs = 3001
lr = 0.001
lr_min = 0.0001
gamma = 1.000
frame_num = 60


seed_everything(2024)

mechanism_train = Lingkage_mec_train(
                   crank_location, 
                   status_location, 
                   epochs=epochs, 
                   lr=lr, 
                   lr_min=lr_min,
                   gamma=gamma, 
                   frame_num=frame_num,
                   trajectory_type=trajectory_type,
                    trajectory_data=selected_trajectory_data,
                   visualize_mec=True
                   )

mechanism_train.train()
