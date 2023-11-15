import torch
import torch.multiprocessing as mp
import torch.utils.hooks as hooks
from GNN_network import CombinedNetwork
from src.loss import get_loss
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import warnings

# Suppress the specific warning about non-serializable backward hook
warnings.filterwarnings("ignore", message="backward hook <bound method")


class Lingkage_mec_train():
    def __init__(self, net, input_batches,validation_batches, sub_batch_size, steps_per_epoch,device, epochs=10000, lr=0.01, gamma=1.00, visualize_mec=False):

        # Register the nan_to_num_hook for each parameter
        for param in net.parameters():
            param.register_hook(self.nan_to_num_hook)

        self.sub_batch_size = sub_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.net = net
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.input_batches = input_batches  # Store the input batches
        self.validation_batches = validation_batches
        self.device = device
        self.visualize_mec = visualize_mec

        self.optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)

    def save_model(self, path):
        # Saving model state dictionary and other relevant information
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            # Add other states if necessary
        }, path)

    def load_model(self, path):
        # Load saved states
        checkpoint = torch.load(path)

        # Load state dictionaries
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load other states if necessary
        self.epoch = checkpoint['epoch']


    @hooks.unserializable_hook
    def nan_to_num_hook(self, grad):
        if torch.isnan(grad).any():
            print("NaN gradient detected!")
            grad = torch.nan_to_num(grad)
        return grad



    def compute_gradients(self, state_dict, batch, device, visualize=False):
        # print(self.state_dict)
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
        
        # print(input_tensor)

        # Ensure the input tensor is on CPU
        input_tensor = input_tensor.to("cpu")

        # Changes start here
        net = CombinedNetwork()
        net.load_state_dict(state_dict)

        # for params in net.parameters():
        #     print(params)

        # Assuming that each input can be concatenated into a single tensor
        input_tensor = input_tensor.to(self.device)

        all_coords, target_coords = net(input_tensor)
        all_coords = all_coords*5.0
        target_coords = target_coords*5.0
        # print(all_coords)

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
                    device,
                    visualize=visualize)

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
            param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            for param in net.parameters()
        ]
        # Detach the loss and clone it to ensure it's not part of the graph
        loss_value = loss.detach().clone().cpu()
        return gradients, loss_value


    def parallel_gradient_computation(self, net, input_batches, device):


        # Ensure state_dict is on CPU
        state_dict = {k: v.cpu() for k, v in net.state_dict().items()}


        # Create a pool of workers, the number of processes is usually set to the number of cores
        num_processes = mp.cpu_count()

        with mp.Pool(processes=num_processes) as pool:
            # Collect gradients and losses
            results = pool.starmap(
                self.compute_gradients,
                [(state_dict, batch, device) for batch in input_batches]
            )

        # Separate gradients and losses
        all_gradients = [result[0] for result in results]
        all_losses = [result[1] for result in results]

        # Average the gradients and losses
        averaged_gradients = [
            torch.mean(torch.stack([g[i] for g in all_gradients]), dim=0)
            for i in range(len(all_gradients[0]))
        ]
        averaged_loss = torch.mean(torch.stack(all_losses), dim=0)

        return averaged_gradients, averaged_loss

    def train(self):
        for epoch in range(self.epochs):
            self.epoch = epoch
            sub_batch = self.input_batches[epoch:epoch + self.sub_batch_size]
            training_loss_sum = 0

            with tqdm(range(self.steps_per_epoch), desc=f"Epoch {epoch+1}/{self.epochs}", unit="step") as t:
                for step in t:
                    # Compute gradients and losses for the sub-batch
                    averaged_gradients, averaged_loss = self.parallel_gradient_computation(self.net, sub_batch, "cpu")
                    
                    # Apply the averaged gradients
                    for j, param in enumerate(self.net.parameters()):
                        param.grad = averaged_gradients[j].to("cpu")

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

                    # Update the weights
                    self.optimizer.step()
                    self.scheduler.step()
                    training_loss_sum += averaged_loss.item()

                    # Update tqdm description with current loss
                    t.set_description(f"Epoch {epoch+1}/{self.epochs}, Step {step+1}/{self.steps_per_epoch}, Loss: {averaged_loss.item():.4f}")

            avg_training_loss = training_loss_sum / self.steps_per_epoch

            validation_loss_sum = 0
            # for val_step, validation_batch in enumerate(tqdm(self.validation_batches, desc="Validating", unit="batch")):
            #     # Process the validation batch and compute loss
            #     _, val_loss = self.compute_gradients(self.net.state_dict(), validation_batch, self.device, visualize=self.visualize_mec)
            #     validation_loss_sum += val_loss.item()

            for validation_batch in self.validation_batches:
                # Process the validation batch and compute loss
                _, val_loss = self.compute_gradients(self.net.state_dict(), validation_batch, self.device, visualize=self.visualize_mec)
                validation_loss_sum += val_loss.item()

            avg_validation_loss = validation_loss_sum / len(self.validation_batches)
            tqdm.write(f'Epoch {self.epoch+1} - Average Validation Loss: {avg_validation_loss:.4f}')
            
            # End-of-Epoch Summary
            # print(f'\nEpoch {epoch+1}/{self.epochs} Summary:')
            # print(f'  Average Training Loss: {avg_training_loss:.4f}')
            # print(f'  Average Validation Loss: {avg_validation_loss:.4f}')



        # Print the profiler output
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))