from GNN_network import CombinedNetwork
from torch.optim.lr_scheduler import ExponentialLR
from train import Lingkage_mec_train
import torch




target_location = [[-5,5.5],[-5,4.5],[5,5.5],[5,4.5]]
crank_location = [-2,0]
status_location = [2,0]

batch = [target_location,crank_location,status_location]

input_batches = [batch,batch,batch,batch]

net = CombinedNetwork()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model to GPU
net = net.to(device)


epochs = 1000
lr = 0.005
gamma = 1.000



mechanism_train = Lingkage_mec_train(net, 
                                    input_batches,
                                    device,
                                    epochs=epochs, 
                                    lr=lr, 
                                    gamma=gamma, 
                                    visualize_mec=False
                   )

mechanism_train.train()
