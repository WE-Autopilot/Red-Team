import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import os
import gym
import cv2
import random
import bisect
import pickle
import math
from collections import deque
from typing import List, Tuple, Union
import time
import pyglet
from pyglet.gl import GL_LINES

##############################
##     GYM ENVIRONOMENT     ##
##############################
class SACF110Env(gym.Env):
    """
    This environment builds a new path only once the car has physically reached
    the previous path’s final point (i.e. within DIST_THRESHOLD).
    
    - The 32D action is interpreted as 16 local (x,y) increments.
    - Angles between increments are clamped (±10°) to ensure a smooth path.
    - A sub-index (0..15) tracks which waypoint is being pursued.
    - If a new action is provided before the path is finished, it is stored as pending.
    """
    DIST_THRESHOLD = 0.2  # [meters] threshold to consider a waypoint reached

    def __init__(self, f110_env: gym.Env):
        super().__init__()
        self.f110_env = f110_env
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(256,256), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
        
        self.car_length = 0.3
        self.vector_length = 0.5
        
        self.path_points = None    # List of 16 (x,y) points (global coordinates)
        self.sub_index = 16        # Forces a new path parse on first step
        self.pending_action = None # Latest agent action waiting to be used

        self.last_obs = None
        self.prev_x = None
        self.prev_y = None

    def reset(self):
        # Example starting pose: (0, 0) with 90° heading
        default_pose = np.array([[0.0, 0.0, np.pi/2]])
        obs, _, _, _ = self.f110_env.reset(default_pose)

        lidar_scan = obs['scans'][0]
        # Use FILL mode with a black background for the lidar bitmap
        bitmap = lidar_to_bitmap(lidar_scan, fov=2*np.pi, output_image_dims=(256,256),
                                 bg_color='black', draw_mode='FILL', channels=1)
        
        # Flip the bitmap vertically
        bitmap = np.flipud(bitmap).copy()

        self.last_obs = obs
        
        self.prev_x = obs['poses_x'][0]
        self.prev_y = obs['poses_y'][0]

        # Force new path
        self.path_points = None
        self.sub_index = 16
        self.pending_action = None

        return bitmap

    def step(self, raw_action: np.ndarray):
        """
        1) If the current path is finished and the car is near its final point,
           parse pending_action (or raw_action) to build a new path.
        2) If mid-path, store the latest action as pending without re-parsing.
        3) Compute a steering & speed command (via MPC) to drive toward the current waypoint.
        4) Advance the sub_index if the car is within DIST_THRESHOLD of the waypoint.
        """
        car_x = self.last_obs['poses_x'][0]
        car_y = self.last_obs['poses_y'][0]
        
        if self.path_points is None:
            self._parse_new_path(raw_action)
        else:
            if self.sub_index >= 16:
                final_x, final_y = self.path_points[-1]
                dist_to_final = np.hypot(final_x - car_x, final_y - car_y)
                if dist_to_final < self.DIST_THRESHOLD:
                    self._parse_new_path(raw_action)
                else:
                    self.pending_action = raw_action
            else:
                self.pending_action = raw_action

        target_x, target_y = self.path_points[self.sub_index]
        # Use the MPC controller (which computes steering and speed) for this step.
        action_out = MPC_controller(
            target_x, target_y,
            car_x, car_y,
            self.last_obs['poses_theta'][0]
        )

        obs, base_reward, done, info = self.f110_env.step(np.array([action_out]))

        lidar_scan = obs['scans'][0]
        bitmap = lidar_to_bitmap(lidar_scan, fov=2*np.pi ,output_image_dims=(256,256),
                                 bg_color='white', draw_mode='FILL', channels=1)
        # Flip the bitmap vertically
        bitmap = np.flipud(bitmap).copy()


        old_x, old_y = self.prev_x, self.prev_y
        new_x = obs['poses_x'][0]
        new_y = obs['poses_y'][0]
        dist_traveled = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2)
        self.prev_x, self.prev_y = new_x, new_y
        
        not_moving_penalty = -2.0 if dist_traveled < 0.001 else 0.0
        progress_reward = dist_traveled * 10.0
        
        lap_completion_bonus = 0.0
        if 'lap_time' in info and info['lap_time'] > 0:
            lap_t = info['lap_time']
            lap_completion_bonus = 500.0 - 10.0 * lap_t
            done = True

        # Calculate additional rewards and penalties
        center = np.array([bitmap.shape[1] // 2, bitmap.shape[0] // 2])
        car_x_centered = center[0]
        car_y_centered = center[1]-1
        angle_pen = collision_angle_penalty(bitmap, car_x_centered, car_y_centered) if done else 0.0
        center_r = centerline_reward(bitmap, car_x_centered, car_y_centered, max_lane_halfwidth=50)
        
        
        total_reward = (base_reward + progress_reward + lap_completion_bonus +
                        not_moving_penalty + angle_pen + center_r)
        if angle_pen != 0:
            print(f"angle_pen= {angle_pen}, center_r = {center_r}")
            done = True   # End episode on collision    

        self.last_obs = obs

        car_x2 = obs['poses_x'][0]
        car_y2 = obs['poses_y'][0]
        target_x2, target_y2 = self.path_points[self.sub_index]
        dist_to_waypoint = np.hypot(target_x2 - car_x2, target_y2 - car_y2)
        if dist_to_waypoint < self.DIST_THRESHOLD:
            self.sub_index += 1

        global current_planned_path
        flattened = []
        for px, py in self.path_points:
            flattened.extend([px, py])
        current_planned_path = np.array(flattened, dtype=np.float32)
        print(f"Total Reward: {total_reward}")

        return bitmap, total_reward, done, info

    def _parse_new_path(self, raw_action: np.ndarray):
        """
        Parse the provided (or pending) 32D action into 16 local increments,
        then compute a new global path based on the car's current pose.
        """
        if self.pending_action is not None:
            action_to_use = self.pending_action
            self.pending_action = None
        else:
            action_to_use = raw_action
        
        # Compute clamped vectors (each normalized to have unit length)
        increments = compute_vectors_with_angle_clamp(action_to_use)

        car_x = self.last_obs['poses_x'][0]
        car_y = self.last_obs['poses_y'][0]
        car_theta = self.last_obs['poses_theta'][0]

        front_x = car_x + self.car_length * np.cos(car_theta)
        front_y = car_y + self.car_length * np.sin(car_theta)

        new_points = [(front_x, front_y)]
        for i in range(16):
            dx, dy = increments[i]
            mag = np.sqrt(dx*dx + dy*dy) + 1e-8
            dx_norm, dy_norm = dx/mag, dy/mag
            dx_scaled = dx_norm * self.vector_length
            dy_scaled = dy_norm * self.vector_length
            
            # Rotate the increment from local to global frame
            global_dx = dx_scaled * np.cos(car_theta) - dy_scaled * np.sin(car_theta)
            global_dy = dx_scaled * np.sin(car_theta) + dy_scaled * np.cos(car_theta)
            
            px, py = new_points[-1]
            new_x = px + global_dx
            new_y = py + global_dy
            new_points.append((new_x, new_y))
        
        self.path_points = new_points[1:]
        self.sub_index = 0

###########################################
##   LIDAR TO BITMAP, COURTESY OF ALY    ##
###########################################
def _lidar_to_bitmap(
        scan: list[float],               
        winding_dir: str = 'CCW',          
        starting_angle: float = -np.pi/2,  
        max_scan_radius: float | None = None,
        scaling_factor: float | None = 10, 
        bg_color: str = 'white', 
        draw_center: bool = True,  
        output_image_dims: tuple[int] = (256, 256),
        target_beam_count: int = 600,
        fov: float = 2*np.pi,
        draw_mode: str = "FILL"
    ) -> np.ndarray:  
    """
    Creates a bitmap image from lidar scan data.
    Assumes rays are equally spaced over the field of view.
    
    Args:
        scan (list[float]): List of lidar measurements.
        winding_dir (str): Direction for the rays ('CW' or 'CCW').
        starting_angle (float): Offset from the positive x-axis.
        max_scan_radius (float | None): Maximum expected range; if provided, used for scaling.
        scaling_factor (float | None): Scaling factor if max_scan_radius is not given.
        bg_color (str): Background color, either 'white' or 'black'.
        draw_center (bool): Whether to draw a center marker.
        output_image_dims (tuple[int]): Dimensions (height, width) of the output image.
        target_beam_count (int): Number of beams (rays) to be used.
        fov (float): Field of view in radians.
        draw_mode (str): 'RAYS', 'POLYGON', or 'FILL' mode for drawing.
        
    Returns:
        np.ndarray: A single-channel (grayscale) bitmap image.
    """
    assert winding_dir in ['CW', 'CCW'], "winding_dir must be either CW or CCW"
    assert bg_color in ['black', 'white']
    assert draw_mode in ['RAYS', 'POLYGON', 'FILL']
    assert len(output_image_dims) == 2 and all(x > 0 for x in output_image_dims)
    assert 0 < target_beam_count < len(scan)
    assert 0 < fov <= 2*np.pi, "FOV must be between 0 and 2pi"

    if max_scan_radius is not None:
        scaling_factor = min(output_image_dims) / max_scan_radius
    elif scaling_factor is None:
        raise ValueError("Provide either max_scan_radius or scaling_factor")
    
    BG_COLOR, DRAW_COLOR = (0, 255) if bg_color == 'black' else (255, 0)
    image = np.ones(output_image_dims, dtype=np.uint8) * BG_COLOR
    direction = 1 if winding_dir == 'CCW' else -1

    indices = np.linspace(0, len(scan) - 1, target_beam_count, dtype=int)
    data = np.array(scan)[indices]
    angles = starting_angle + direction * fov * np.linspace(0, 1, target_beam_count)
    center = np.array([output_image_dims[0] // 2, output_image_dims[1] // 2])
    points = np.column_stack((
        np.rint(center[0] + scaling_factor * data * np.cos(angles)).astype(int),
        np.rint(center[1] + scaling_factor * data * np.sin(angles)).astype(int)
    ))

    if draw_mode == 'FILL':
        cv2.fillPoly(image, [points], DRAW_COLOR)
    elif draw_mode == 'POLYGON':
        cv2.polylines(image, [points], isClosed=True, color=DRAW_COLOR, thickness=1)
    elif draw_mode == 'RAYS':
        for p in points:
            cv2.line(image, tuple(center), tuple(p), color=DRAW_COLOR, thickness=1)
            cv2.rectangle(image, tuple(p - 2), tuple(p + 2), color=DRAW_COLOR, thickness=-1)

    if draw_center:
        cv2.rectangle(image, tuple(center - 2), tuple(center + 2),
                      color=BG_COLOR if draw_mode == "FILL" else DRAW_COLOR, thickness=-1)
    
    return image

def lidar_to_bitmap(
        scan: list[float],               
        winding_dir: str = 'CCW',          
        starting_angle: float = -np.pi/2,  
        max_scan_radius: float | None = None,
        scaling_factor: float | None = 10, 
        bg_color: str = 'white', 
        draw_center: bool = True,  
        output_image_dims: tuple[int] = (256, 256),
        target_beam_count: int = 600,
        fov: float = 2*np.pi,
        draw_mode: str = "POLYGON",
        channels: int = 1
    ) -> np.ndarray:  
    """
    Wraps _lidar_to_bitmap to optionally convert the grayscale image
    into a multi-channel image.
    
    Args:
        (see _lidar_to_bitmap for other parameters)
        channels (int): 1 (grayscale), 3 (RGB), or 4 (RGBA).
    
    Returns:
        np.ndarray: The lidar bitmap image.
    """
    assert channels in [1, 3, 4], "channels must be 1, 3, or 4"
    grayscale_img = _lidar_to_bitmap(scan, winding_dir, starting_angle,
                                     max_scan_radius, scaling_factor, bg_color,
                                     draw_center, output_image_dims,
                                     target_beam_count, fov, draw_mode)
    if channels == 1:
        return grayscale_img
    elif channels == 3:
        return np.stack([grayscale_img] * 3, axis=-1)
    elif channels == 4:
        alpha_channel = np.full_like(grayscale_img, 255)
        return np.stack([grayscale_img, grayscale_img, grayscale_img, alpha_channel], axis=-1)
    else:
        raise ValueError("Invalid number of channels. Supported: 1, 3, or 4.")

##############################
##        OPIUM MODEL       ##
##############################
class Actor(nn.Module):
    """
    The Actor outputs a 32D continuous action (in [-1,1]) representing 16 local (x,y) increments.
    Processes the 256x256 lidar bitmap through convolutional layers.
    """
    def __init__(self, action_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc_mean = nn.Linear(512, action_dim)
        self.fc_log_std = nn.Linear(512, action_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_std = torch.clamp(self.fc_log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = (dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)).sum(1, keepdim=True)
        return y_t, log_prob

class Critic(nn.Module):
    """
    The Critic estimates the Q-value for a given state (bitmap) and action (32D vector).
    """
    def __init__(self, action_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 28 * 28 + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

##############################
##      REPLAY BUFFER       ##
##############################
class ReplayBuffer:
    """
    Stores (state, action, reward, next_state, done) tuples for off-policy RL.
    """
    def __init__(self, capacity: int = 1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, d: bool):
        self.buffer.append((s, a, r, ns, d))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return s, a, r, ns, d
    
    def __len__(self) -> int:
        return len(self.buffer)

##############################
##        SAC AGENT         ##
##############################
class SACAgent:
    """
    Soft Actor-Critic agent for continuous control.
    
    Attributes:
        device: Torch device.
        actor: The policy network.
        critic1, critic2: The Q-value estimation networks.
        Target networks for critics (for soft updates).
    """
    def __init__(self, device: torch.device, action_dim: int = 32, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2, actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = Actor(action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic1 = Critic(action_dim).to(device)
        self.critic2 = Critic(action_dim).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        
        self.critic1_target = Critic(action_dim).to(device)
        self.critic2_target = Critic(action_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Select an action given the current state.
        
        Args:
            state (np.ndarray): Observation (expected shape: 256x256 or similar).
            evaluate (bool): If True, choose the mean (deterministic) action.
            
        Returns:
            np.ndarray: A 1D action vector (length 32).
        """
        st = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        if evaluate:
            with torch.no_grad():
                mean, _ = self.actor.forward(st)
                act = torch.tanh(mean)
                return act.cpu().numpy().flatten()
        else:
            with torch.no_grad():
                act, _ = self.actor.sample(st)
                return act.cpu().numpy().flatten()
    
    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 64) -> Tuple[float, float, float]:
        """
        Performs a SAC update (both actor and critics).
        
        Args:
            replay_buffer (ReplayBuffer): Buffer to sample transitions from.
            batch_size (int): Number of samples per update.
            
        Returns:
            Tuple containing (actor_loss, critic1_loss, critic2_loss).
        """
        if len(replay_buffer) < batch_size:
            return 0, 0, 0
        
        s, a, r, ns, d = replay_buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        if len(s.shape) == 3: s = s.unsqueeze(1)
        ns = torch.FloatTensor(ns).to(self.device)
        if len(ns.shape) == 3: ns = ns.unsqueeze(1)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(np.float32(d)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(ns)
            tq1 = self.critic1_target(ns, next_a)
            tq2 = self.critic2_target(ns, next_a)
            tq = torch.min(tq1, tq2) - self.alpha * next_logp
            tv = r + (1 - d) * self.gamma * tq
        
        cq1 = self.critic1(s, a)
        cq2 = self.critic2(s, a)
        c1_loss = F.mse_loss(cq1, tv)
        c2_loss = F.mse_loss(cq2, tv)
        
        self.critic1_optimizer.zero_grad()
        c1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        c2_loss.backward()
        self.critic2_optimizer.step()
        
        new_a, logp = self.actor.sample(s)
        q1n = self.critic1(s, new_a)
        q2n = self.critic2(s, new_a)
        qn = torch.min(q1n, q2n)
        a_loss = (self.alpha * logp - qn).mean()
        
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        for tp, p in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        
        return a_loss.item(), c1_loss.item(), c2_loss.item()

#######################################
## PATH CLAMP & MPC HELPER FUNCTIONS ##
#######################################
def clamp_vector_angle_diff(prev_angle: float, desired_angle: float, max_diff_deg: float = 10.0) -> float:
    """
    Ensures consecutive vectors differ by at most ±10° (or the given max_diff_deg).
    
    Args:
        prev_angle (float): Previous vector’s angle (radians).
        desired_angle (float): Desired current angle (radians).
        max_diff_deg (float): Maximum allowed deviation in degrees.
        
    Returns:
        float: The clamped angle (radians).
    """
    max_diff_rad = np.radians(max_diff_deg)
    angle_diff = (desired_angle - prev_angle + np.pi) % (2 * np.pi) - np.pi
    if angle_diff > max_diff_rad:
        return prev_angle + max_diff_rad
    elif angle_diff < -max_diff_rad:
        return prev_angle - max_diff_rad
    return desired_angle

def compute_vectors_with_angle_clamp(raw_action: np.ndarray, max_diff_deg: float = 10.0) -> np.ndarray:
    """
    Interprets a 32D raw action as 16 local (x,y) increments,
    forcing the first vector to be (1,0) and clamping subsequent angles.
    
    Args:
        raw_action (np.ndarray): 1D array of length 32.
        max_diff_deg (float): Maximum angle change between successive vectors.
        
    Returns:
        np.ndarray: (16, 2) array of clamped, normalized increments.
    """
    assert raw_action.shape == (32,), "Raw action must be a 32D vector (16 x 2D movements)."
    vectors = raw_action.reshape(16, 2)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    clamped_vectors = np.zeros_like(vectors)
    clamped_vectors[0] = [1, 0]
    prev_angle = np.arctan2(clamped_vectors[0][1], clamped_vectors[0][0])
    for i in range(1, 16):
        desired_angle = np.arctan2(vectors[i][1], vectors[i][0])
        clamped_angle = clamp_vector_angle_diff(prev_angle, desired_angle, max_diff_deg)
        clamped_vectors[i] = [np.cos(clamped_angle), np.sin(clamped_angle)]
        prev_angle = clamped_angle
    return clamped_vectors

def MPC_controller(target_x: float, target_y: float,
                   car_x: float, car_y: float,
                   car_theta: float) -> np.ndarray:
    """
    Computes steering and speed commands aiming the car from its current pose
    toward the target point.
    
    Args:
        target_x (float): Target X-coordinate.
        target_y (float): Target Y-coordinate.
        car_x (float): Current car X-coordinate.
        car_y (float): Current car Y-coordinate.
        car_theta (float): Current heading (radians).
    
    Returns:
        np.ndarray: 1D array [steering, speed] for the next simulator step.
    """
    desired_heading = np.arctan2(target_y - car_y, target_x - car_x)
    steering = desired_heading - car_theta
    steering = np.clip(steering, -1, 1)
    speed = 4.0 * (1 - np.abs(steering))
    speed = np.clip(speed, 0.0, 6.0)
    return np.array([steering, speed])

def detect_collison(fill_bitmap, car_x, car_y, neighborhood_check=3):
    """
    Detects if the car is about to collide with an obstacle.
    
    :param fill_bitmap: The filled bitmap image of the environment.
    :param car_x: Current car X position.
    :param car_y: Current car Y position.
    :param neighborhood_check: The number of pixels to check around the car.
    :return: True if a collision is imminent, False otherwise.
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
                # If a neighbor is white => off-track/collision
                if fill_bitmap[ny, nx] == 255:
                    return True
    return False
    

def get_wall_normal(fill_bitmap, car_x, car_y, region=10):
    """

    :param fill_bitmap: The filled bitmap image of the environment.
    :param car_x: Current car X position.
    :param car_y: Current car Y position.
    :param region: The maximum distance to search for a black pixel.
    :return: A 1D array representing the wall normal.
    """
    # 1. Canny Edge Detection
    edges = cv2.Canny(fill_bitmap, threshold1=50, threshold2=150)

    # 2. Sobel Gradients
    grad_x = cv2.Sobel(fill_bitmap, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(fill_bitmap, cv2.CV_32F, 0, 1, ksize=3)

    # 3. Gather gradient vectors at edges near (cx, cy)
    h, w = fill_bitmap.shape
    x0 = max(0, car_x - region - 2)
    x1 = min(w, car_x + region + 3)
    y0 = max(0, car_y - region - 2)
    y1 = min(h, car_y + region + 3)

    grad_vectors = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            if edges[y, x] == 255:  # It's an edge pixel
                gx = grad_x[y, x]
                gy = grad_y[y, x]
                if not (abs(gx) < 1e-5 and abs(gy) < 1e-5):
                    grad_vectors.append([gx, gy])

    if len(grad_vectors) == 0:
        return np.array([0.0, 0.0])

    # 4. Average the gradient vectors
    arr = np.array(grad_vectors, dtype=np.float32)
    mean_grad = np.mean(arr, axis=0)

    # 5. Normalize
    norm = np.linalg.norm(mean_grad) + 1e-8
    mean_grad /= norm

    # By default, the gradient points from darker to brighter.
    # If your "normal" should point inward or outward, you might flip or rotate:
    # For example, normal = mean_grad, or normal = -mean_grad, etc.
    normal = -mean_grad

    return normal


def compute_collision_angle(wall_normal, car_direction_vec=np.array([0,1])):
    """
    Returns the angle (in degrees) between direction_vec and wall_normal.

    :param car_direction_vec: The direction vector of the car.
    :param wall_normal: The normal vector of the wall.
    :return: The angle in degrees.
    """
    dot = np.dot(car_direction_vec, wall_normal)
    # Both are unit vectors => no need to divide by norms
    dot = np.clip(dot, -1.0, 1.0)  # numerical safety
    angle = np.degrees(np.arccos(dot))
    return angle

def collision_angle_penalty(fill_bitmap, car_x, car_y):
    """
    Check collision. If collision is detected, compute angle-based penalty.

    :param fill_bitmap: The filled bitmap image of the environment.
    :param car_x: Current X position.
    :param car_y: Current Y position.
    :return: The penalty value.
    """
    reward_delta = 0.0
    collided = detect_collison(fill_bitmap, car_x, car_y)
    if not collided:
        return 0.0  # No collision => no penalty

    wall_normal = get_wall_normal(fill_bitmap, car_x, car_y)
    angle_deg = 90 - compute_collision_angle(wall_normal)
    print(f"Collision angle: {angle_deg} degrees")
    # Map angle to penalty
    penalty = np.interp(abs(angle_deg), [0, 90], [0.1, 10000.0])
    reward_delta -= penalty
    return reward_delta

def distance_from_row_center(fill_bitmap, car_x, car_y):
    """
    Returns how far car_x is from the 'center' of the drivable area
    on the row car_y in the fill_bitmap.

    :param fill_bitmap: The filled bitmap image of the environment.
    :param car_x: Current car X position.
    :param car_y: Current car Y position.
    :return: The distance from the center
    """
    h, w = fill_bitmap.shape

    # Safety check
    if not (0 <= car_y < h and 0 <= car_x < w):
        return None  # Car is out of bounds

    # 1. Find left boundary
    left_edge = car_x - 3
    while left_edge >= 0 and fill_bitmap[car_y, left_edge] == 0:
        left_edge -= 1
    # Move one pixel into white area
    left_edge += 1

    # 2. Find right boundary
    right_edge = car_x + 3
    while right_edge < w and fill_bitmap[car_y, right_edge] == 0:
        right_edge += 1
    # Move one pixel into white area
    right_edge -= 1

    # If we found valid edges
    if left_edge < 0 or right_edge >= w or left_edge >= right_edge:
        # Possibly means car is off track or no white area in that row
        return None

    # 3. Midpoint
    midpoint = (left_edge + right_edge) / 2.0
    half_width = ((right_edge - left_edge) / 2.0) - 2;

    # 4. Distance from center
    dist = abs(car_x - midpoint)
    norm_dist = dist / half_width
    # print(f"Normal Distance from center: {norm_dist:.2f} pixels")
    # 5. Return distance
    return norm_dist

def centerline_reward(fill_bitmap, car_x, car_y, max_lane_halfwidth=50):
    """
    If the car is near the 'center' of the lane (in that row),
    give higher reward. If far, give lower reward.
    """
    dist = distance_from_row_center(fill_bitmap, car_x, car_y)
    if dist is None:
        # Car might be off track => big penalty or zero reward
        return -1.0

    # Normalize distance by half-lane width
    norm_dist = dist  # e.g., 0 = center, 1 = near boundary
    # Reward could be: R = 1 - norm_dist (bounded to [0, 1] if dist <= max_lane_halfwidth)
    reward = max(0.0, 1.0 - norm_dist)
    return reward

##############################
##  DISPLAYING EVERYTHING   ##
##############################
arrow_graphics = []
current_planned_path = None

def render_arrow(env_renderer, flattened_path: np.ndarray):
    """
    Renders arrows along the planned path for visualization.
    
    Args:
        env_renderer: The environment renderer (expects a pyglet batch).
        flattened_path (np.ndarray): Flattened array of path points.
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

def render_callback(env_renderer):
    """
    Callback for the simulator renderer to display the planned path.
    """
    global current_planned_path
    if current_planned_path is not None:
        render_arrow(env_renderer, current_planned_path)

##############################
##      MAIN TRAINING LOOP  ##
##############################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f110_env = gym.make('f110_gym:f110-v0', map='example_map', map_ext='.png',
                        num_agents=1, timestep=0.015)
    f110_env.add_render_callback(render_callback)
    
    env = SACF110Env(f110_env)
    agent = SACAgent(device, action_dim=32)
    replay_buffer = ReplayBuffer()
    
    max_episodes = 1000
    max_steps = 2000
    batch_size = 64
    update_after = 1000
    update_every = 50
    
    total_steps = 0
    for ep in range(max_episodes):
        obs = env.reset()
        ep_reward = 0
        for st in range(max_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            print(f"Episode {ep} Reward={ep_reward:.2f}")
            
            f110_env.render("human")
            cv2.imshow("LiDAR Bitmap", obs)
            cv2.waitKey(1)
            
            if total_steps > update_after and total_steps % update_every == 0:
                a_loss, c1_loss, c2_loss = agent.update(replay_buffer, batch_size)
                print(f"Step {total_steps}: Actor={a_loss:.4f}, Critic1={c1_loss:.4f}, Critic2={c2_loss:.4f}")
            
            if done:
                break
        print(f"Episode {ep} Reward={ep_reward:.2f}")
        
    
    torch.save(agent.actor.state_dict(), "sac_actor.pth")
    cv2.destroyAllWindows()
    print("Training complete, model saved as sac_actor.pth")

    # reward = 0.0

    # # 1. Collision angle penalty
    # angle_pen = collision_angle_penalty(fill_bitmap, car_x, car_y)
    # reward += angle_pen  # This is negative if collision

    # # 2. Centerline reward
    # center_r = centerline_reward(fill_bitmap, car_x, car_y, max_lane_halfwidth=50)
    # reward += center_r  # Higher if near center, 0 or negative if off track

# Run the main function.
if __name__ == "__main__":
    main()