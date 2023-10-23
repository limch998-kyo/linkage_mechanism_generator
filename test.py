import torch
import torch.nn as nn

# Define target_coords and marker_position as tensors with requires_grad=True
target_coords = torch.tensor([-153.4103, 123.2133], dtype=torch.float32, requires_grad=True)
marker_position = torch.tensor([-4.9726, 5.0000], dtype=torch.float32, requires_grad=True)

# Calculate the mean squared error (MSE) loss
loss = nn.MSELoss()(target_coords, marker_position)

# Print the loss
print('loss:', loss)

# Perform backpropagation
loss.backward()

# Now you can access gradients
print('Gradient of target_coords:', target_coords.grad)
print('Gradient of marker_position:', marker_position.grad)