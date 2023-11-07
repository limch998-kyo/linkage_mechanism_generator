# from points_generator import generate_coordinates
import torch
from torch.cuda.amp import autocast, GradScaler


from utils import output_process
from src.linkage_builder import Linkage_mechanism
from src.loss import get_loss


from torch.optim.lr_scheduler import ExponentialLR
from torch.profiler import profile, record_function, ProfilerActivity


class Lingkage_mec_train():
    def __init__(self, net, crank_location, status_location, target_location, device, epochs=10000, lr=0.01, gamma=1.00, visualize_mec=False):
        self.net = net
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma

        self.crank_location = crank_location
        self.status_location = status_location
        self.target_location = target_location
        self.device = device

        # Convert each list into individual tensors
        self.target_location_tensor = torch.tensor(target_location, dtype=torch.float, device=device)
        self.crank_location_tensor = torch.tensor([crank_location], dtype=torch.float, device=device)
        self.status_location_tensor = torch.tensor([status_location], dtype=torch.float, device=device)

        input = []
        input.append(crank_location)
        input.append(status_location)
        for i in range(len(target_location)):
            input.append(target_location[i])
        self.input_tensor = torch.tensor([input], dtype=torch.float)

        self.visualize_mec = visualize_mec

        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=1.00)
        self.scaler = GradScaler()

    def nan_to_num_hook(self, grad):
        if torch.isnan(grad).any():
            print("NaN gradient detected!")
            grad = torch.nan_to_num(grad)
        return grad

    def train(self):

        # Start the profiler
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_inference"):

        visualize = False
                # Register the gradient hook
        for param in self.net.parameters():
            param.register_hook(self.nan_to_num_hook)
        for epoch in range(self.epochs):

            # Define device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Model to GPU
            self.input_tensor = self.input_tensor.to(device)

            coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = self.net(self.input_tensor)
            all_coords = all_coords*5.0
            target_coords = target_coords*5.0


            stage2_adjacency = torch.tensor([[0,1],[2,3],[0,0],[0,0]],device=self.device)
            coor_val = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0],device=self.device)
            target_adjacency = torch.tensor([4,5],device=self.device)


            if epoch % 100 == 0 and self.visualize_mec:
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
                        visualize=visualize)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward() 
            self.scaler.step(self.optimizer)  
            self.scaler.update()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scheduler.step()

            if epoch % 10 == 0:
                print('epoch: ', epoch, 'loss: ', loss.item())


        # Print the profiler output
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))