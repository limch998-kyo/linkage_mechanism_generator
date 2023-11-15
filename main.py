import random
from GNN_network import CombinedNetwork
from train import Lingkage_mec_train
import torch
import multiprocessing as mp

# Random batch generation functions
def generate_random_coordinate(min_x, max_x, min_y, max_y):
    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)
    return [x, y]

def generate_random_batch(num_batches, target_bounds_1, target_bounds_2, crank_bounds, status_bounds):
    batches = []
    for _ in range(num_batches):
        target_location_1 = generate_random_coordinate(*target_bounds_1)
        target_location_2 = generate_random_coordinate(*target_bounds_2)
        crank_location = generate_random_coordinate(*crank_bounds)
        status_location = generate_random_coordinate(*status_bounds)
        batch = [[target_location_1, target_location_2], crank_location, status_location]
        batches.append(batch)
    return batches

# Random batch generation
num_batches = 100  # Example number of batches
num_batches_val = 1
target_bounds_1 = (-4, -2, 2, 4)  # First target coordinate boundaries
target_bounds_2 = (2, 4, 2, 4)    # Second target coordinate boundaries
crank_bounds = (-4, -1, -4, 0)       # Crank location boundaries
status_bounds = (1, 4, -4, 0)      # Status location boundaries

# Generate training and validation batches
input_batches = generate_random_batch(num_batches, target_bounds_1, target_bounds_2, crank_bounds, status_bounds)
validation_batches = generate_random_batch(num_batches_val, target_bounds_1, target_bounds_2, crank_bounds, status_bounds)

# Main training script
if __name__ == '__main__':
    # Set the start method for multiprocessing
    # mp.set_start_method('spawn')

    net = CombinedNetwork()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    sub_batch_size = 16
    steps_per_epoch = 16
    epochs = 1000
    lr = 0.005
    gamma = 1.000

    mechanism_train = Lingkage_mec_train(net,
                                         input_batches,
                                         validation_batches,
                                         sub_batch_size=sub_batch_size,
                                         steps_per_epoch=steps_per_epoch,
                                         device=device,
                                         epochs=epochs,
                                         lr=lr,
                                         gamma=gamma,
                                         visualize_mec=True
                                         )

    mechanism_train.train()
