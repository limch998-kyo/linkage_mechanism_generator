import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

# Let's use a simple model for demonstration
model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

# Start the profiler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
        model(inputs)

# Print the profiler output
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
