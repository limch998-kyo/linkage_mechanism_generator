import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple linear regression model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x)

# Define hyperparameters
input_dim = 3  # Let's assume we're using 3 features to predict the 2D coordinate
output_dim = 2  # 2D coordinate
learning_rate = 0.001
num_epochs = 100000

# Initialize model, criterion, and optimizer
model = SimpleModel(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Dummy dataset: 10 samples of 3 features each
x_train = torch.rand(10, input_dim)
# For simplicity, let's assume the target 2D coordinate for all samples is [5, 5]
y_train = torch.tensor([[5.0, 5.0] for _ in range(10)])
print(y_train.shape)
# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    y_pred = model(x_train)
    
    loss = criterion(y_pred, y_train)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training finished.")

# Test
with torch.no_grad():
    test_input = torch.rand(input_dim)
    predicted_coord = model(test_input)
    print(f"For test input {test_input}, predicted 2D coordinate is {predicted_coord}")
