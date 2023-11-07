# from points_generator import generate_coordinates
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler


from utils import output_process
from src.linkage_builder import Linkage_mechanism
from src.loss import get_loss


from torch.optim.lr_scheduler import ExponentialLR
from torch.profiler import profile, record_function, ProfilerActivity


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

    def compute_loss(self, net, batch, device, queue):
        # Move the batch to the correct device
        target_location_tensor = torch.tensor(batch[0], dtype=torch.float, device=device)
        crank_location_tensor = torch.tensor([batch[1]], dtype=torch.float, device=device)
        status_location_tensor = torch.tensor([batch[2]], dtype=torch.float, device=device)
        
        # Forward pass
        with autocast():
            # Assuming that each input can be concatenated into a single tensor
            input_tensor = torch.cat((crank_location_tensor, status_location_tensor, target_location_tensor), dim=1)
            input_tensor = input_tensor.to(self.device)


            coor_val, stage2_adjacency, all_coords, target_adjacency, target_coords = self.net(input_tensor)
            all_coords = all_coords*5.0
            target_coords = target_coords*5.0


            stage2_adjacency = torch.tensor([[0,1],[2,3],[0,0],[0,0]],device=self.device)
            coor_val = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0],device=self.device)
            target_adjacency = torch.tensor([4,5],device=self.device)

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

            queue.put(loss)

    def parallel_loss_computation(self, net, input_batches, device):
        net.share_memory()  # Allow network parameters to be shared between processes
        processes = []
        queue = mp.Queue()

        for batch in input_batches:
            p = mp.Process(target=self.compute_loss, args=(net, batch, device, queue))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Collect and sum up the losses from all processes
        total_loss = sum([queue.get() for _ in processes])
        return total_loss

    def train(self):
        # Ensure CUDA operations are performed in the 'spawned' process
        mp.set_start_method('spawn')

        # Your training loop
        for epoch in range(self.epochs):
            self.epoch = epoch
            # Compute losses in parallel
            total_loss_value = self.parallel_loss_computation(self.net, self.input_batches, self.device)

            # Convert the total loss value into a tensor to backpropagate
            total_loss = torch.tensor(total_loss_value, requires_grad=True, device=self.device)
            
            # Backpropagate
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward() 
            self.scaler.step(self.optimizer)  
            self.scaler.update()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
            self.scheduler.step()

            
            if epoch % 10 == 0:
                print('Epoch:', epoch, 'Total loss:', total_loss.item())



        # Print the profiler output
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))