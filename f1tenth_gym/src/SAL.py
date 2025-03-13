import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import os
import gym
import cv2
import random
from pyglet.gl import GL_LINES
from collections import deque
from typing import List, Tuple, Union
from weap_util.lidar import lidar_to_bitmap

# Global variables for rendering callbacks
arrow_graphics = []
current_planned_path = None


##############################
##     GYM ENVIRONMENT      ##
##############################

class SACF110Env(gym.Env):
    """
    Custom F1Tenth environment with SAC integration and simple low-level control.
    Handles high-level path planning and uses a basic steering controller for movement.
    """
    
    DIST_THRESHOLD = 0.2  # Waypoint reaching threshold

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
        default_pose = np.array([[0.0, 0.0, 1.57]])  # x, y, theta
        obs, _, _, _ = self.f110_env.reset(default_pose)
        
        # Process initial observation
        lidar_scan = obs['scans'][0]
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(256,256),
                                bg_color='black', draw_mode="FILL", winding_dir='CW', starting_angle=np.pi/2)
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
        Execute one timestep with SAC action and a simple steering controller.
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
        if self.path_points is None or self.sub_index >= len(self.path_points):
            self._handle_path_update(raw_action, car_state)

        # Use the next waypoint from the planned path and compute control using the simple controller.
        target_x, target_y = self.path_points[self.sub_index]
        action_out = get_steering_and_speed(target_x, target_y,
                                            car_state['x'], car_state['y'],
                                            car_state['theta'])
        
        # Check if the computed speed is 0 (or very close to 0)
        if np.isclose(action_out[0, 1], 0.0, atol=1e-6):
            # Simulate a crash: assign a crash penalty and force an episode restart
            crash_penalty = -100.0
            info = {"crash": True, "reason": "velocity_zero"}
            obs = self.reset()
            return obs, crash_penalty, True, info

        # Step simulation with the simple control action
        obs, base_reward, done, info = self.f110_env.step(action_out)
        
        # Process new observation
        lidar_scan = obs['scans'][0]
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(256,256),
                                bg_color='black', draw_mode="FILL", winding_dir='CW', starting_angle=np.pi/2)
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

    def _calculate_rewards(self, obs: dict, done: bool) -> dict:
        """Calculate reward components with adjustments for efficient learning."""
        rewards = {}

        # Time penalty: encourage faster completion.
        rewards['time_penalty'] = -0.1

        # Progress reward: reward distance traveled with a higher multiplier.
        new_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        dist = np.linalg.norm(new_pos - self.prev_position)
        rewards['progress'] = dist * 15.0  # increased multiplier from 10.0 to 15.0

        # Collision detection and penalty: use a heavy penalty plus an angle-based adjustment.
        px, py = self._world_to_pixel(obs['poses_x'][0], obs['poses_y'][0])
        collision = detect_collison(self.last_obs['lidar_bitmap'], px, py)
        if collision:
            # collision_angle_penalty returns a small negative value (more penalty for shallow angles)
            angle_penalty = collision_angle_penalty(self.last_obs['lidar_bitmap'],
                                                    int(obs['poses_x'][0]),
                                                    int(obs['poses_y'][0]))
            rewards['collision'] = -150.0 + angle_penalty  # base heavy penalty adjusted by angle
        else:
            rewards['collision'] = 0.0

        # Centering bonus: reward staying near the center of the drivable area.
        centering = centerline_reward(
            fill_bitmap=self.last_obs['lidar_bitmap'],
            car_x=int(obs['poses_x'][0]),
            car_y=int(obs['poses_y'][0])
        )
        rewards['centering'] = centering * 3.0  # increased multiplier from 2.0 to 3.0

        # Lap completion bonus: encourage fast lap completion.
        if 'lap_time' in obs and obs['lap_time'] > 0:
            rewards['lap'] = 600.0 - 15.0 * obs['lap_time']  # increased base reward and penalty rate
        else:
            rewards['lap'] = 0.0

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

##############################
##        OPIUM MODEL       ##
##############################
class Actor(nn.Module):
    """
    The Actor outputs a 32D continuous action (in [-1,1]) representing 16 local (x,y) increments.
    Processes the 256x256 lidar bitmap through convolutional layers.
    """
    def __init__(self, action_dim: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 28 * 28, 512)
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
    def __init__(self, action_dim: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 28 * 28 + action_dim, 512)
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
    """
    def __init__(self, device: torch.device, action_dim: int = 16, gamma: float = 0.99,
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
        c1_loss = nn.MSELoss()(cq1, tv)
        c2_loss = nn.MSELoss()(cq2, tv)
        
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
## PATH CLAMP & HELPER FUNCTIONS ##
#######################################
def compute_vectors_with_angle_clamp(raw_action: np.ndarray, 
                                   max_diff_deg: float = 10.0) -> np.ndarray:
    """Convert raw action to path vectors with angle constraints"""
    vectors = raw_action.reshape(8, 2)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    
    clamped = np.zeros_like(vectors)
    clamped[0] = [1, 0]
    prev_angle = 0.0
    
    for i in range(1, 8):
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


#############################################
##        SIMPLE MOVEMENT CONTROLLER       ##
#############################################
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


def detect_collison(fill_bitmap, car_x, car_y, neighborhood_check=1):
    """
    Detects if the car is about to collide with an obstacle.
    """
    h, w = fill_bitmap.shape
    for dy in range(-neighborhood_check, neighborhood_check+1):
        for dx in range(-neighborhood_check, neighborhood_check+1):
            if dx == 0 and dy == 0:
                continue
            nx = car_x + dx
            ny = car_y + dy
            if 0 <= nx < w and 0 <= ny < h:
                if fill_bitmap[ny, nx] == 255:
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
    x0 = max(0, car_x - region)
    x1 = min(w, car_x + region + 1)
    y0 = max(0, car_y - region)
    y1 = min(h, car_y + region + 1)

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
    angle_deg = compute_collision_angle(wall_normal)
    penalty = np.interp(abs(angle_deg), [0, 90], [0.1, 1.0])
    reward_delta -= penalty
    return reward_delta

def distance_from_row_center(fill_bitmap, car_x, car_y):
    """
    Returns how far car_x is from the 'center' of the drivable area.
    """
    h, w = fill_bitmap.shape
    if not (0 <= car_y < h and 0 <= car_x < w):
        return None

    left_edge = car_x
    while left_edge >= 0 and fill_bitmap[car_y, left_edge] == 255:
        left_edge -= 1
    left_edge += 1

    right_edge = car_x
    while right_edge < w and fill_bitmap[car_y, right_edge] == 255:
        right_edge += 1
    right_edge -= 1

    if left_edge < 0 or right_edge >= w or left_edge >= right_edge:
        return None

    midpoint = (left_edge + right_edge) / 2.0
    dist = abs(car_x - midpoint)
    return dist

def centerline_reward(fill_bitmap, car_x, car_y, max_lane_halfwidth=50):
    """
    Computes a reward based on the car's proximity to the center of the lane.
    """
    dist = distance_from_row_center(fill_bitmap, car_x, car_y)
    if dist is None:
        return -1.0

    norm_dist = dist / max_lane_halfwidth
    reward = max(0.0, 1.0 - norm_dist)
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

def render_callback(env_renderer):
    """
    Callback for the simulator renderer to display the planned path.
    """
    global current_planned_path
    if current_planned_path is not None:
        render_arrow(env_renderer, current_planned_path)


####################################################
##      MAIN TRAINING LOOP  AND MODEL SAVING      ##
####################################################

def load_latest_checkpoint(agent, checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found. Starting from scratch.")
        return
    # Look for files that match our naming scheme, e.g., sac_actor_v*.pth
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("sac_actor_v") and f.endswith(".pth")]
    if checkpoint_files:
        # Sort by version number extracted from filename (e.g., v1, v2, etc.)
        checkpoint_files.sort(key=lambda x: int(x.split("v")[1].split(".")[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Loading latest checkpoint: {checkpoint_path}")
        agent.actor.load_state_dict(torch.load(checkpoint_path))
    else:
        print("No checkpoint files found. Starting from scratch.")

# In your main training loop, before starting training:
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f110_env = gym.make('f110_gym:f110-v0', map='example_map', map_ext='.png',
                        num_agents=1, timestep=0.015)
    f110_env.add_render_callback(render_callback)
    
    env = SACF110Env(f110_env)
    agent = SACAgent(device, action_dim=16)
    
    # Try to resume from the latest checkpoint
    load_latest_checkpoint(agent, checkpoint_dir="checkpoints")
    
    replay_buffer = ReplayBuffer()
    
    max_episodes = 1000
    max_steps = 2000
    batch_size = 64
    update_after = 1000
    update_every = 50

    # Ensure checkpoint directory exists
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
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
            
            f110_env.render("human")
            cv2.imshow("LiDAR Bitmap", obs)
            cv2.waitKey(1)
            
            if total_steps > update_after and total_steps % update_every == 0:
                a_loss, c1_loss, c2_loss = agent.update(replay_buffer, batch_size)
                print(f"Step {total_steps}: Actor={a_loss:.4f}, Critic1={c1_loss:.4f}, Critic2={c2_loss:.4f}")
            
            if done:
                break
        print(f"Episode {ep} Reward={ep_reward:.2f}")
        
        # Save checkpoint every 50 episodes
        if (ep + 1) % 50 == 0:
            version = (ep + 1) // 50
            checkpoint_path = os.path.join(checkpoint_dir, f"sac_actor_v{version}.pth")
            torch.save(agent.actor.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    torch.save(agent.actor.state_dict(), os.path.join(checkpoint_dir, "sac_actor_final.pth"))
    cv2.destroyAllWindows()
    print("Training complete, model saved.")

if __name__ == "__main__":
    main()
