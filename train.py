# from points_generator import generate_coordinates
import torch
import torch.multiprocessing as mp

from torch.cuda.amp import autocast, GradScaler
from GNN_network import CombinedNetwork

from utils import output_process
from src.linkage_builder import Linkage_mechanism
from src.loss import get_loss


from torch.optim.lr_scheduler import ExponentialLR
from torch.profiler import profile, record_function, ProfilerActivity
from torch.multiprocessing import Pool


class Lingkage_mec_train():
    def __init__(self, net, input_batches, device, epochs=10000, lr=0.01, gamma=1.00, visualize_mec=False):
        self.net = net
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.input_batches = input_batches  # Store the input batches
        self.device = device
        self.visualize_mec = visualize_mec

        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)
        self.scaler = GradScaler()

    def nan_to_num_hook(self, grad):
        if torch.isnan(grad).any():
            print("NaN gradient detected!")
            grad = torch.nan_to_num(grad)
        return grad



    def compute_gradients(self, state_dict, batch, device):
        print(self.state_dict)
        # print(state_dict)

        # Move the batch to the correct device
        target_location_tensor = torch.tensor(batch[0], dtype=torch.float, device=device)
        crank_location_tensor = torch.tensor([batch[1]], dtype=torch.float, device=device)
        status_location_tensor = torch.tensor([batch[2]], dtype=torch.float, device=device)

        # Move the batch to the correct device
        input = []
        input.append(batch[1])
        input.append(batch[2])
        for i in range(len(batch[0])):
            input.append(batch[0][i])
        input_tensor = torch.tensor([input], dtype=torch.float)
        
        # Forward pass
    # with autocast():
        # print(self.state_dict)
        net = CombinedNetwork()

        net.load_state_dict(state_dict)

        # for params in net.parameters():
        #     print(params)

        # Assuming that each input can be concatenated into a single tensor
        input_tensor = input_tensor.to(self.device)

        # print('input_tensor',input_tensor.shape)
        # print(input_tensor)


        coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = net(input_tensor)
        all_coords = all_coords*5.0
        target_coords = target_coords*5.0
        print(all_coords)

        stage2_adjacency = torch.tensor([[0,1],[2,3],[0,0],[0,0]],device=self.device)
        coor_val = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0],device=self.device)
        target_adjacency = torch.tensor([4,5],device=self.device)
        # print('start')
        # print(coor_val)
        # print(all_coords)
        # print(target_coords)
        # print(stage2_adjacency)
        # print(target_adjacency)
        # print(crank_location_tensor)
        # print(status_location_tensor)
        # print(target_location_tensor)
        # print('end')
        loss = get_loss(coor_val, 
                    all_coords, 
                    target_coords, 
                    stage2_adjacency,
                    target_adjacency,
                    crank_location_tensor[0],
                    status_location_tensor[0],
                    target_location_tensor,
                    self.epoch,
                    visualize=self.visualize_mec)

            # print(loss)
        # Backward pass to compute gradients
        net.zero_grad()  # Clear existing gradients
        loss.backward()

        # self.optimizer.zero_grad()
        # self.scaler.scale(loss).backward() 
        # self.scaler.step(self.optimizer)  
        # self.scaler.update()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        # self.scheduler.step()

        # Collect gradients, ensuring none are None
        gradients = [
            (param.grad.detach().clone().cpu() if param.grad is not None else torch.zeros_like(param).cpu())
            for param in net.parameters()
        ]
        return gradients

    def compute_gradients_wrapper(args):
        return compute_gradients(*args)

    def parallel_gradient_computation(self, net, input_batches, device):



        # Use the current network state_dict for all batches
        self.state_dict = net.state_dict()
        state_dict = self.state_dict

        
        # print(state_dict.device)
        # # Convert it to CPU for sharing with subprocesses
        # cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        # print(self.state_dict)

        # Create a pool of workers, the number of processes is usually set to the number of cores
        num_processes = mp.cpu_count()

        with Pool(processes=num_processes) as pool:
            # Map the compute_gradients function to the data for parallel computation
            # It automatically gathers the results in order

            all_gradients = pool.starmap(
                self.compute_gradients,
                [(state_dict, batch, device) for batch in input_batches]
            )


            # Average the gradients
        averaged_gradients = [torch.mean(torch.stack([g[i] for g in all_gradients]), dim=0) for i in range(len(all_gradients[0]))]
        return averaged_gradients

    def train(self):
        # self.net.share_memory()  # Allow network parameters to be shared between processes
        # Ensure CUDA operations are performed in the 'spawned' process


        # if mp.get_start_method(allow_none=True) is None:
        #     mp.set_start_method('spawn')
        for param in self.net.parameters():
            assert param.requires_grad, "Parameter does not require gradients."



        # Your training loop
        for epoch in range(self.epochs):
            self.epoch = epoch
            # Compute gradients in parallel
            averaged_gradients = self.parallel_gradient_computation(self.net, self.input_batches, self.device)
            # Before using the averaged gradients, move them to the correct device
            averaged_gradients = [g.to(device) for g in averaged_gradients]

            # Convert the total loss value into a tensor to backpropagate
            # total_loss = torch.tensor(total_loss_value, requires_grad=True, device=self.device)
            
            # Apply the averaged gradients
            for i, param in enumerate(self.net.parameters()):
                param.grad = averaged_gradients[i].to(self.device)

            # Step the optimizer
            self.optimizer.zero_grad()  # Clear any existing gradients
            self.scaler.step(self.optimizer)  # Update the model's parameters
            self.scaler.update()  # Update the scaler
            self.scheduler.step()  # Adjust the learning rate based on the scheduler

            # Optionally clip gradients
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            
            # if epoch % 10 == 0:
                # print('Epoch:', epoch, 'Total loss:', total_loss.item())



        # Print the profiler output
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))