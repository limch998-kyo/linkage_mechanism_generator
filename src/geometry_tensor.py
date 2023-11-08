import torch

def euclidean_distance(point1, point2):
    return torch.norm(point1 - point2)

def rotation_matrix(angle, device=None):
    cos_val = torch.cos(angle)
    sin_val = torch.sin(angle)
    # If device is provided, use it, otherwise infer device from angle tensor
    device = device or angle.device
    return torch.tensor([[cos_val, -sin_val], 
                         [sin_val, cos_val]], device=device)


def rotate_around_center(coordinates, angle, center):
    # Infer device from coordinates tensor
    device = coordinates.device

    # Step 1: Translate to origin
    translated_coordinates = coordinates - center
    
    # Step 2: Rotate around origin
    # Pass the device to ensure the matrix is created on the correct device
    rotated_coordinates = torch.matmul(translated_coordinates, rotation_matrix(angle, device))
    
    # Step 3: Translate back to original position
    rotated_coordinates_back = rotated_coordinates + center
    
    return rotated_coordinates_back


def circle_intercept(P1, r1, P2, r2):
    x1, y1 = P1
    x2, y2 = P2
    
    epsilon = 1e-10  # small constant to prevent division by zero
    
    # Distance between the centers
    d = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Check if circles do not intersect
    # 1. One circle is contained within the other without touching
    # 2. The circles are too far apart to touch
    if d > r1 + r2 + epsilon or d < abs(r1 - r2) - epsilon:
        return None
    
    # Base calculations for intersection
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = torch.sqrt(torch.clamp(r1**2 - a**2, min=0))
    
    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d
    
    # Two intersection points
    x4_1 = x3 + h * (y2 - y1) / d
    y4_1 = y3 - h * (x2 - x1) / d
    
    x4_2 = x3 - h * (y2 - y1) / d
    y4_2 = y3 + h * (x2 - x1) / d
    
    return torch.stack([x4_1, y4_1, x4_2, y4_2])




def closest_point(input_coord, potential_coords):
    # Ensure the input tensors have gradient tracking enabled
    if not input_coord.requires_grad:
        input_coord.requires_grad_()
    if not potential_coords.requires_grad:
        potential_coords.requires_grad_()

    # Split the potential_coords into x and y coordinate arrays
    potential_coords_x = potential_coords[::2]
    potential_coords_y = potential_coords[1::2]

    # Calculate squared distances, no need to take the square root since we're just looking for the minimum distance
    distances = (potential_coords_x - input_coord[0])**2 + (potential_coords_y - input_coord[1])**2
    index = torch.argmin(distances).item()

    # Extract the closest point's coordinates
    closest_x = potential_coords_x[index]
    closest_y = potential_coords_y[index]

    # Combine them into a single tensor
    closest_coord = torch.stack([closest_x, closest_y])

    return closest_coord



def closest_intersection_point(input_coord, P1, r1, P2, r2):
    intersections = circle_intercept(P1, r1, P2, r2)
    if intersections is not None:
        return closest_point(input_coord, intersections)
    else:
        return None
