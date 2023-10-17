import numpy as np

def circle_intercept(P1, r1, P2, r2):
    x1, y1 = P1
    x2, y2 = P2
    
    # Distance between the centers
    d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Check for no solution
    if d > r1 + r2 or d < abs(r1 - r2):
        return None  # No intersection
    
    # Check for coincidence
    if d == 0 and r1 == r2:
        return [(x1, y1)]  # Infinite intersections, but return the center
    
    # Calculate the intersection point(s)
    a = (r1**2 - r2**2 + d**2) / (2*d)
    h = np.sqrt(r1**2 - a**2)
    
    x3 = x1 + a * (x2 - x1) / d
    y3 = y1 + a * (y2 - y1) / d
    
    # Two intersection points
    x4_1 = x3 + h * (y2 - y1) / d
    y4_1 = y3 - h * (x2 - x1) / d
    
    x4_2 = x3 - h * (y2 - y1) / d
    y4_2 = y3 + h * (x2 - x1) / d
    
    return [(x4_1, y4_1), (x4_2, y4_2)]

def closest_point(input_coord, potential_coords):
    distances = [np.sqrt((x - input_coord[0])**2 + (y - input_coord[1])**2) for x, y in potential_coords]
    return potential_coords[np.argmin(distances)]

def closest_intersection_point(input_coord, P1, r1, P2, r2):
    intersections = circle_intercept(P1, r1, P2, r2)
    if intersections:
        return closest_point(input_coord, intersections)
    else:
        return None  # No intersection
if __name__ == '__main__':
# Example usage:
    input_coord = (0, 2)
    P1 = (0, 0)
    r1 = 5
    P2 = (6, 0)
    r2 = 5

    print(closest_intersection_point(input_coord, P1, r1, P2, r2))
