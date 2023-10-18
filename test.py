import torch

def euclidean_distance(point1, point2):
    return torch.norm(point1 - point2)

# Example usage:
p1 = torch.tensor([1.0, 2.0])
p2 = torch.tensor([4.0, 6.0])

distance = euclidean_distance(p1, p2)
print(distance)
