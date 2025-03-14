#############################################
##        SIMPLE MOVEMENT CONTROLLER       ##
#############################################
import cv2
import numpy as np
from pyglet.gl import GL_LINES

arrow_graphics = []

def get_steering_and_speed(
    target_x: float,
    target_y: float,
    car_x: float,
    car_y: float,
    car_theta: float
) -> np.ndarray:
    """
    Simple movement controller that aims the car toward (target_x, target_y).

    Computes the desired heading and then returns:
       - steering: difference between desired heading and current heading (clipped to [-1, 1])
       - speed: proportional to (1 - |steering|) scaled and clipped.
    """
    desired_heading = np.arctan2(target_y - car_y, target_x - car_x)
    steering = desired_heading - car_theta
    steering = np.clip(steering, -1, 1)
    speed = 4.0 * (1 - np.abs(steering))
    speed = np.clip(speed, 0.0, 6.0)
    return np.array([[steering, speed]])


def detect_collison(fill_bitmap, car_x, car_y, neighborhood_check=3):
    """
    Detects if the car is about to collide with an obstacle.
    """
    h, w = fill_bitmap.shape
    for dy in range(-neighborhood_check, neighborhood_check+1):
        for dx in range(-neighborhood_check, neighborhood_check+1):

            # Skip the car's exact center pixel
            if -3<dx<3 and -3<dy<3:
                continue


            nx = car_x + dx
            ny = car_y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if fill_bitmap[ny, nx] == 0:
                    return True
    return False
    

def get_wall_normal(fill_bitmap, car_x, car_y, region=10):
    """
    Computes the wall normal vector from the environment bitmap.
    """
    edges = cv2.Canny(fill_bitmap, threshold1=50, threshold2=150)
    grad_x = cv2.Sobel(fill_bitmap, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(fill_bitmap, cv2.CV_32F, 0, 1, ksize=3)
    h, w = fill_bitmap.shape
    x0 = max(0, car_x - region - 2)
    x1 = min(w, car_x + region + 3)
    y0 = max(0, car_y - region - 2)
    y1 = min(h, car_y + region + 3)

    grad_vectors = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            if edges[y, x] == 0:
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                if not (abs(gx) < 1e-5 and abs(gy) < 1e-5):
                    grad_vectors.append([gx, gy])

    if len(grad_vectors) == 0:
        return np.array([0.0, 0.0])

    arr = np.array(grad_vectors, dtype=np.float32)
    mean_grad = np.mean(arr, axis=0)
    norm = np.linalg.norm(mean_grad) + 1e-8
    mean_grad /= norm
    normal = mean_grad
    return normal


def compute_collision_angle(wall_normal, car_direction_vec=np.array([0,1])):
    """
    Returns the angle (in degrees) between car direction and wall normal.
    """
    dot = np.dot(car_direction_vec, wall_normal)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    return angle

def collision_angle_penalty(fill_bitmap, car_x, car_y):
    """
    Computes an angle-based penalty if a collision is detected.
    """
    reward_delta = 0.0
    collided = detect_collison(fill_bitmap, car_x, car_y)
    if not collided:
        return 0.0

    wall_normal = get_wall_normal(fill_bitmap, car_x, car_y)
    angle_deg = 90 - compute_collision_angle(wall_normal)
    penalty = np.interp(abs(angle_deg), [0, 90], [0.1, 10000.0])
    reward_delta -= penalty
    return reward_delta

def distance_from_row_center(fill_bitmap, car_x, car_y):
    """
    Returns how far car_x is from the 'center' of the drivable area.
    """
    h, w = fill_bitmap.shape
    if not (0 <= car_y < h and 0 <= car_x < w):
        return None

    left_edge = car_x - 3
    while left_edge >= 0 and fill_bitmap[car_y, left_edge] == 255:
        left_edge -= 1
    left_edge += 1

    right_edge = car_x + 3
    while right_edge < w and fill_bitmap[car_y, right_edge] == 255:
        right_edge += 1
    right_edge -= 1

    if left_edge < 0 or right_edge >= w or left_edge >= right_edge:
        return None

    midpoint = (left_edge + right_edge) / 2.0
    halfwidth = ((right_edge - left_edge) / 2.0) - 2
    dist = abs(car_x - midpoint)
    norm_dist = dist / halfwidth
    return norm_dist

def centerline_reward(fill_bitmap, car_x, car_y, max_lane_halfwidth=50):
    """
    Computes a reward based on the car's proximity to the center of the lane.
    """
    dist = distance_from_row_center(fill_bitmap, car_x, car_y)
    if dist is None:
        return -1.0

    reward = max(0.0, 1.0 - dist)
    return reward


def render_arrow(env_renderer, flattened_path: np.ndarray):
    """
    Renders arrows along the planned path for visualization.
    """
    global arrow_graphics
    for arrow in arrow_graphics:
        arrow.delete()
    arrow_graphics = []
    
    points = flattened_path.reshape(-1, 2)
    scale = 50.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        arrow = env_renderer.batch.add(
            2, GL_LINES, None,
            ('v2f', (x0 * scale, y0 * scale, x1 * scale, y1 * scale)),
            ('c3B', (0, 255, 0, 0, 255, 0))
        )
        arrow_graphics.append(arrow)