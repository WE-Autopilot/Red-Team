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
import cvxpy as cp
from scipy.interpolate import CubicSpline
from collections import deque
from typing import List, Tuple, Union
import time
import pyglet
from pyglet.gl import GL_LINES

# Global variables for rendering callbacks
arrow_graphics = []
current_planned_path = None


##############################
##     GYM ENVIRONOMENT     ##
##############################

class SACF110Env(gym.Env):
    """
    Custom F1Tenth environment with SAC integration and MPC path following
    Handles high-level path planning and low-level MPC control
    """
    
    DIST_THRESHOLD = 0.2  # Waypoint reaching threshold
    MPC_PARAMS = {
        'desired_velocity': 2.0,    # m/s
        'timestep': 0.1,            # seconds
        'total_steps': 10,          # planning steps
        'horizon_length': 5,        # MPC horizon
        'state_cost': np.diag([1.0, 1.0, 0.1, 0.1]),
        'input_cost': np.diag([0.1, 0.1]),
        'terminal_cost': np.diag([10.0, 10.0, 1.0, 1.0])
    }

    def __init__(self, f110_env: gym.Env):
        super().__init__()
        self.f110_env = f110_env
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                              shape=(256,256), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1, high=1, 
                                         shape=(32,), dtype=np.float32)
        
        # Path planning parameters
        self.car_length = 0.3
        self.vector_length = 0.5
        self.path_points = None
        self.sub_index = 16
        self.pending_action = None

        # State tracking
        self.last_obs = None
        self.prev_position = None
        self.current_planned_path = None
        self.map_scale = 10.0  # pixels per meter
        self.map_origin = (128, 128)  

    def reset(self):
        """Reset environment with default pose and clear path history"""
        default_pose = np.array([[0.0, 0.0, np.pi/2]])  # x, y, theta
        obs, _, _, _ = self.f110_env.reset(default_pose)
        
        # Process initial observation
        lidar_scan = obs['scans'][0]
        # Use FILL mode with a black background for the lidar bitmap with full FOV and one channel
        bitmap = lidar_to_bitmap(lidar_scan, fov=2*np.pi, output_image_dims=(256,256),
                                 bg_color='black', draw_mode='FILL', channels=1)
        # Flip the bitmap vertically
        bitmap = np.flipud(bitmap).copy()
        # Store the computed lidar bitmap in the observation
        obs['lidar_bitmap'] = bitmap

        self.last_obs = obs
        self.prev_position = np.array([obs['poses_x'][0], obs['poses_y'][0]])

        # Reset path tracking
        self.path_points = None
        self.sub_index = 16
        self.pending_action = None
        self.current_planned_path = None

        return bitmap

    def step(self, raw_action: np.ndarray):
        """
        Execute one timestep with SAC action and MPC control
        Returns:
            bitmap: Processed LIDAR observation
            total_reward: Calculated reward for this step
            done: Episode completion flag
            info: Additional information
        """
        # Get current state
        car_state = {
            'x': self.last_obs['poses_x'][0],
            'y': self.last_obs['poses_y'][0],
            'theta': self.last_obs['poses_theta'][0]
        }

        # Path management
        if self.path_points is None or self.sub_index >= 16:
            self._handle_path_update(raw_action, car_state)

        # MPC control calculation
        mpc_action = self._calculate_mpc_control(car_state)

        # Step simulation
        obs, base_reward, done, info = self.f110_env.step(mpc_action)
        
        # Process new observation
        lidar_scan = obs['scans'][0]
        # Use full FOV, black background, FILL mode, 1 channel and flip vertically
        bitmap = lidar_to_bitmap(lidar_scan, fov=2*np.pi, output_image_dims=(256,256),
                                 bg_color='black', draw_mode='FILL', channels=1)
        bitmap = np.flipud(bitmap).copy()
        # Add the lidar bitmap into the new observation
        obs['lidar_bitmap'] = bitmap

        # Calculate rewards using the previous observation's lidar bitmap
        reward_components = self._calculate_rewards(obs, done)
        total_reward = sum(reward_components.values())

        # Update state
        self._update_path_index(obs)
        self.last_obs = obs
        self.prev_position = np.array([obs['poses_x'][0], obs['poses_y'][0]])

        # Update visualization
        self._update_path_visualization()

        return bitmap, total_reward, done, info

    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
       px = int(self.map_origin[0] + x * self.map_scale)
       py = int(self.map_origin[1] + y * self.map_scale)
       return np.clip(px, 0, 255), np.clip(py, 0, 255)

    def _handle_path_update(self, raw_action: np.ndarray, car_state: dict):
        """Manage path creation and updates"""
        if self.pending_action is not None:
            action_to_use = self.pending_action
            self.pending_action = None
        else:
            action_to_use = raw_action

        # Convert SAC action to path vectors
        increments = compute_vectors_with_angle_clamp(action_to_use)
        self.path_points = self._calculate_global_path(increments, car_state)
        self.sub_index = 0

    def _calculate_global_path(self, increments: np.ndarray, car_state: dict) -> list:
        """Convert local vectors to global path coordinates"""
        path = []
        x, y = car_state['x'], car_state['y']
        theta = car_state['theta']
        
        # Start from front of car
        front_x = x + self.car_length * np.cos(theta)
        front_y = y + self.car_length * np.sin(theta)
        path.append((front_x, front_y))

        # Convert local increments to global coordinates
        for dx, dy in increments:
            dx_scaled = dx * self.vector_length
            dy_scaled = dy * self.vector_length
            
            # Rotate to global frame
            global_dx = dx_scaled * np.cos(theta) - dy_scaled * np.sin(theta)
            global_dy = dx_scaled * np.sin(theta) + dy_scaled * np.cos(theta)
            
            new_x = path[-1][0] + global_dx
            new_y = path[-1][1] + global_dy
            path.append((new_x, new_y))

        return path[1:]  # Skip initial point

    def _calculate_mpc_control(self, car_state: dict) -> np.ndarray:
        """Calculate low-level control using MPC"""
        path_array = np.array(self.path_points)
        mpc_params = self.MPC_PARAMS

        # Get MPC control inputs, now passing current velocity from the last observation
        control_seq = MPC_controller(
            path=path_array,
            desiredVelocity=mpc_params['desired_velocity'],
            timeStep=mpc_params['timestep'],
            totalSteps=mpc_params['total_steps'],
            horizonLength=mpc_params['horizon_length'],
            stateCost=mpc_params['state_cost'],
            inputCost=mpc_params['input_cost'],
            terminalCost=mpc_params['terminal_cost'],
            current_vel_x=self.last_obs['linear_vels_x'][0],
            current_vel_y=self.last_obs['linear_vels_y'][0]
        )

        # Convert MPC output to simulator action
        current_speed = np.hypot(self.last_obs['linear_vels_x'][0],
                                 self.last_obs['linear_vels_y'][0])
        steering, throttle = MPC_converter(
            x_accel=control_seq[0][0],
            y_accel=control_seq[0][1],
            current_speed=current_speed,
            current_steer=self.last_obs.get('steering', [0.0])[0],
            max_steer=0.4189,  # ~24 degrees
            max_accel=3.0,
            max_velo=8.0,
            min_velo=-4.0
        )

        # Return a 2D array (num_agents x 2) to satisfy the environment indexing
        return np.array([[steering, throttle]])

    def _calculate_rewards(self, obs: dict, done: bool) -> dict:
        """Calculate reward components"""
        rewards = {
            'base': 0.0,
            'progress': 0.0,
            'collision': 0.0,
            'centering': 0.0
        }

        # Collision detection
        px, py = self._world_to_pixel(obs['poses_x'][0], obs['poses_y'][0])
        collision = detect_collison(self.last_obs['lidar_bitmap'], px, py)
        rewards['collision'] = -100.0 if collision else 0.0

        # Progress reward
        new_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        dist = np.linalg.norm(new_pos - self.prev_position)
        rewards['progress'] = dist * 10.0

        # Centering bonus
        centering = centerline_reward(
            fill_bitmap=self.last_obs['lidar_bitmap'],
            car_x=int(obs['poses_x'][0]),
            car_y=int(obs['poses_y'][0])
        )
        rewards['centering'] = centering * 2.0

        # Lap completion
        if 'lap_time' in obs and obs['lap_time'] > 0:
            rewards['lap'] = 500.0 - 10.0 * obs['lap_time']

        return rewards

    def _update_path_index(self, obs: dict):
        """Update waypoint index based on current position"""
        current_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        target_pos = np.array(self.path_points[self.sub_index])
        dist = np.linalg.norm(current_pos - target_pos)
        
        if dist < self.DIST_THRESHOLD:
            self.sub_index += 1

    def _update_path_visualization(self):
        """Update visualization of planned path"""
        if self.path_points is not None:
            flattened = []
            for px, py in self.path_points:
                flattened.extend([px, py])
            self.current_planned_path = np.array(flattened, dtype=np.float32)
            global current_planned_path
            current_planned_path = self.current_planned_path

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
        st = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
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
def compute_vectors_with_angle_clamp(raw_action: np.ndarray, 
                                   max_diff_deg: float = 10.0) -> np.ndarray:
    """Convert raw action to path vectors with angle constraints"""
    vectors = raw_action.reshape(16, 2)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    
    clamped = np.zeros_like(vectors)
    clamped[0] = [1, 0]
    prev_angle = 0.0
    
    for i in range(1, 16):
        desired_angle = np.arctan2(vectors[i,1], vectors[i,0])
        clamped_angle = clamp_vector_angle_diff(prev_angle, desired_angle, max_diff_deg)
        clamped[i] = [np.cos(clamped_angle), np.sin(clamped_angle)]
        prev_angle = clamped_angle
        
    return clamped

def clamp_vector_angle_diff(prev_angle: float, desired_angle: float,
                          max_diff_deg: float) -> float:
    """Clamp angle difference between consecutive path segments"""
    max_diff_rad = np.deg2rad(max_diff_deg)
    angle_diff = (desired_angle - prev_angle + np.pi) % (2*np.pi) - np.pi
    return prev_angle + np.clip(angle_diff, -max_diff_rad, max_diff_rad)


############################
##     MPC CONTROLLER     ##
############################

def MPC_controller(path: np.ndarray, desiredVelocity: float, timeStep: float, totalSteps: int, horizonLength: int, stateCost: np.ndarray, inputCost: np.ndarray, terminalCost: np.ndarray, current_vel_x: float = 0.0, current_vel_y: float = 0.0) -> np.ndarray:
    """
    Computes control input(x and y acceleration) at each timeStep along path
    
    :param path: Array of vectors of projected path car should follow. Should start at current x, y coordinates of car
    :param desiredVelocity: Desired constant speed along the track
    :param timeStep: Time step (seconds), how often our simulation will update
    :param totalSteps: Total simulation steps, how long the simulation will run
    :param horizonLength: MPC horizon (number of steps), how far ahead the controller plans
    :param stateCost: MPC cost weights, penalizes deviations from the reference trajectory (the vectorized path)
    :param inputCost: MPC cost weights, penalizes deviations from the reference trajectory (the vectorized path)
    :param terminalCost: MPC cost weights, penalizes deviations from the reference trajectory (the vectorized path)
    :param current_vel_x: Current x-velocity of the car
    :param current_vel_y: Current y-velocity of the car

    :return: An array of [x_acceleration, y_acceleration] for the converter
    """
    # Calculates the distance between each pair of points along path, and adds them all to one cumulative arc length
    dists = [0]
    for i in range(1, len(path)):
        dists.append(dists[-1] + np.linalg.norm(path[i] - path[i-1]))
    dists = np.array(dists)

    '''
    Cublic Splines are cubic functions used to interpolate between points, maintaining smoothness
    between the points. This is useful for creating the paths for this project.
    '''
    # Uses the cubic spline function to interpolate between the points on the track
    cs_x = CubicSpline(dists, path[:, 0])
    cs_y = CubicSpline(dists, path[:, 1])

    # 2D double-integrator model:
    # State: [x, y, vx, vy]; Control: [ax, ay]
    A = np.array([[1, 0, timeStep, 0],
                [0, 1, 0, timeStep],
                [0, 0, 1,  0],
                [0, 0, 0,  1]])
    B = np.array([[0.5*timeStep**2, 0],
                [0, 0.5*timeStep**2],
                [timeStep, 0],
                [0, timeStep]])

    # Precompute the reference trajectory along the drawn track.
    # For each simulation time (plus horizon), compute the reference state.
    # We use s = v_des * t (i.e., the distance along the track increases at constant speed).
    ref_traj = np.zeros((totalSteps + horizonLength + 1, 4)) # 4x4 Array to store the reference trajectory at each time step (+ the horizon)
    for i in range(totalSteps + horizonLength + 1):
        t = i * timeStep # Current time
        s = desiredVelocity * t  # arc-length traveled along the track

        # If s exceeds the maximum distance of the drawn track, hold the last point.
        if s > dists[-1]:
            s = dists[-1]
        
        # Compute the reference position from the spline.
        x_ref = cs_x(s)
        y_ref = cs_y(s)
        
        # Compute the derivative (velocity components) from the spline derivatives.
        vx_ref = cs_x.derivative()(s)
        vy_ref = cs_y.derivative()(s)
        
        # Optionally normalize the velocity to the desired speed.
        speed = np.hypot(vx_ref, vy_ref) # Calculates magnitue of the velocity vector
        if speed > 1e-3:
            vx_ref = desiredVelocity * vx_ref / speed
            vy_ref = desiredVelocity * vy_ref / speed
        else:
            vx_ref = 0
            vy_ref = 0
        
        ref_traj[i, :] = np.array([x_ref, y_ref, vx_ref, vy_ref])

    u_history = [] # Record of control inputs at each timeStep
    state_history = [] # Record of car state at each timeStep

    # Set the initial state.
    # Here we start at the first point of the drawn track, with current velocity.
    x_current = np.array([path[0, 0], path[0, 1], current_vel_x, current_vel_y])
    state_history.append(x_current)

    # Iterates through the simulation steps
    for t in range(totalSteps):
        # Define cvxpy variables for the state and control over the horizon.
        x = cp.Variable((4, horizonLength+1)) # Array to store the state at each time step
        u = cp.Variable((2, horizonLength)) # Array to store the control input at each time step
        
        cost = 0
        constraints = []
        
        # Initial condition for the horizon. ensuring the first state in the horizon = the current state
        constraints += [x[:, 0] == x_current]
        
        # Build the cost function and dynamics constraints over the horizon.
        for k in range(horizonLength):
            ref_state = ref_traj[t + k] # The reference state at the current step in the horizon
            cost += cp.quad_form(x[:, k] - ref_state, stateCost) + cp.quad_form(u[:, k], inputCost)
            constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
            constraints += [u[:, k] <= np.array([1.0, 1.0]),
                            u[:, k] >= np.array([-1.0, -1.0])]
        
        # Terminal cost for the final state in the horizon.
        ref_state_terminal = ref_traj[t + horizonLength]
        cost += cp.quad_form(x[:, horizonLength] - ref_state_terminal, terminalCost)
        
        # Solve the MPC optimization problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        # Extract the first control input from the optimal sequence.
        u_apply = u[:, 0].value
        if u_apply is None:
            u_apply = np.zeros(2)
        u_history.append(u_apply)

        # Update the current state using the system dynamics.
        x_current = A @ x_current + B @ u_apply

        state_history.append(x_current)
    
    u_history = np.array(u_history)
    state_history = np.array(state_history)

    return np.array(u_history)

def MPC_converter(x_accel: float, y_accel: float, current_speed: float, current_steer: float, max_steer: float, max_accel: float, max_velo: float, min_velo: float) -> np.ndarray:
    """
    Takes MPC Controller control inputs(x and y accelration) and convertes them into a 1D Array of [steering, thrust]
    
    :param x_accel: MPC calculated x-acceleration of car
    :param y_accel: MPC calculated y-acceleration of car
    :param current_speed: Current speed of car
    :param current_steer: Current steering angle of car
    :param max_steer: Maximum possible steering angle of car
    :param max_accel: Maximum possible acceleration of car
    :param max_velo: Maximum possible velocity of car (forwards)
    :param min_velo: Minimum possible velocity of car (backwards)

    :return: A 1D array [steering, thrust] for the simulator step.
    """
    target_angle = np.arctan2(y_accel, x_accel)
    angle_diff = (target_angle - current_steer + np.pi) % (2*np.pi) - np.pi
    steering = np.clip(angle_diff, -max_steer, max_steer)
    
    # Calculate acceleration in direction of current heading
    forward_accel = x_accel * np.cos(current_steer) + y_accel * np.sin(current_steer)
    throttle = np.clip(forward_accel, -1.0, 1.0)
    
    return np.array([steering, throttle])


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
            # Skip the car's exact center pixel (if within a small box)
            if -3 < dx < 3 and -3 < dy < 3:
                continue

            nx = car_x + dx
            ny = car_y + dy
            if 0 <= nx < w and 0 <= ny < h:
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
            if edges[y, x] == 255:
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
    dot = np.clip(dot, -1.0, 1.0)
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
        return 0.0

    wall_normal = get_wall_normal(fill_bitmap, car_x, car_y)
    angle_deg = 90 - compute_collision_angle(wall_normal)
    print(f"Collision angle: {angle_deg} degrees")
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
    if not (0 <= car_y < h and 0 <= car_x < w):
        return None

    left_edge = car_x - 3
    while left_edge >= 0 and fill_bitmap[car_y, left_edge] == 0:
        left_edge -= 1
    left_edge += 1

    right_edge = car_x + 3
    while right_edge < w and fill_bitmap[car_y, right_edge] == 0:
        right_edge += 1
    right_edge -= 1

    if left_edge < 0 or right_edge >= w or left_edge >= right_edge:
        return None

    midpoint = (left_edge + right_edge) / 2.0
    half_width = ((right_edge - left_edge) / 2.0) - 2
    dist = abs(car_x - midpoint)
    norm_dist = dist / half_width
    return norm_dist

def centerline_reward(fill_bitmap, car_x, car_y, max_lane_halfwidth=50):
    """
    If the car is near the 'center' of the lane (in that row),
    give higher reward. If far, give lower reward.
    """
    dist = distance_from_row_center(fill_bitmap, car_x, car_y)
    if dist is None:
        return -1.0

    norm_dist = dist
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

if __name__ == "__main__":
    main()
