from GNN_network import CombinedNetwork
from torch.optim.lr_scheduler import ExponentialLR
from train import Lingkage_mec_train




target_location = [[-5,5.5],[-5,4.5],[5,5.5],[5,4.5]]
crank_location = [-2,0]
status_location = [2,0]

net = CombinedNetwork()

epochs = 10000
lr = 0.01
gamma = 1.000

mechanism_train = Lingkage_mec_train(net, 
                   crank_location, 
                   status_location, 
                   target_location, 
                   epochs=epochs, 
                   lr=lr, 
                   gamma=gamma, 
                   visualize_mec=True
                   )

mechanism_train.train()
