from linkage_network import CombinedNetwork
from torch.optim.lr_scheduler import ExponentialLR
from train import Lingkage_mec_train
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

target_location = [[-5,5.5],[-5,4.5],[5,5.5],[5,4.5]]
crank_location = [-2,0]
status_location = [2,0]

# Define trajectory type and data
trajectory_type = 'linear'  # Options: 'linear', 'circular', 'elliptical', 'sine'
# Example trajectory data for circular: (center, radius)
# Adjust this based on the selected trajectory_type
trajectory_data = {'circular': [(0, 5), 3],
                    'elliptical': [(0, 0), (3, 2)],
                    'sine': [(-5, 5), 2, 50, 10],  # start_point, amplitude, wavelength, length
                    'linear': [(-5, 5), (5, 5)]}  # start and end points

selected_trajectory_data = trajectory_data[trajectory_type]

# net = CombinedNetwork()
# device = torch.device("cpu")
# net = net.to(device)

epochs = 10000
lr = 0.005
gamma = 1.000

seed_everything(42)

mechanism_train = Lingkage_mec_train(
                   crank_location, 
                   status_location, 
                   target_location, 
                   epochs=epochs, 
                   lr=lr, 
                   gamma=gamma, 
                   trajectory_type=trajectory_type,
                    trajectory_data=selected_trajectory_data,
                   visualize_mec=True
                   )

mechanism_train.train()
