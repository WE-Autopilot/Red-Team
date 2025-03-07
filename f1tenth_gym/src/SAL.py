import numpy as np
import cv2
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import time
from typing import List, Tuple, Union
import pyglet
from pyglet.gl import GL_LINES

###########################################
##   LIDAR TO BITMAP, COURTESY OF ALY    ##
###########################################

def _lidar_to_bitmap(
        scan: list[float],               
        winding_dir: str='CCW',          
        starting_angle: float=-np.pi/2,  
        max_scan_radius: float | None = None,
        scaling_factor: float | None = 10, 
        bg_color: str = 'white', 
        draw_center: bool=True,  
        output_image_dims: tuple[int]=(256, 256),
        target_beam_count: int=600,
        fov: float=2*np.pi,
        draw_mode: str="FILL"
    ) -> np.ndarray:  
    """
    Creates a bitmap image based on lidar input.
    Assumes rays are equally spaced within the FOV.
    Internal only DO NOT USE. Use lidar_to_bitmap instead.

    Args:
        scan (list[float]): A list of lidar measurements.

        winding_dir (str): The direction that the rays wind. Must either be CW or CCW in a right handed coord system.
        
        starting_angle (float): The offset from the pos-x axis that points "up" or "forward.
        
        max_scan_radius (float | None): The maximum range expected from the scans. Used to scale the value into the image if given.
        
        scaling_factor (float | None): Scaling factor for the ranges from the scan.
        
        bg_color (str): Either \'white\' or \'black\'. The accent color is always the opposite.
        
        draw_center (bool): Should this function draw a square in the center of the bitmap?
        
        output_image_dims (tuple[int]): The dimensions of the output image. Should be square but not enforced.
        
        target_beam_count (int): The target number of beams (rays) cast into the environment.
        
        fov (float): The field of view of the car measured in radians. Note: the output will look pinched if this is setup incorrectly.

        draw_mode (str): How should the final image be drawn. Can be \'RAYS\' (view the ray casts - keep beam count low), \'POLYGON\' (draws the outline of the rays), or \'FILL\' (filled in driveable and nondriveable boundary). 

    Returns:
        np.ndarray: A single-channel, grayscale image with a birds-eye-view of the lidar scan.
    """
    assert winding_dir in ['CW', 'CCW'], "winding_dir must be either clockwise or counterclockwise"
    assert bg_color in ['black', 'white']
    assert draw_mode in ['RAYS', 'POLYGON', 'FILL']
    assert len(output_image_dims) == 2
    assert all([x > 0 for x in output_image_dims]), "output_image_dims must be at least 1x1"
    assert 0 < target_beam_count < len(scan)
    assert 0 < fov <= 2*np.pi, "FOV must be between 0 and 2pi"

    if max_scan_radius is not None:
        scaling_factor = min(output_image_dims) / max_scan_radius
    elif scaling_factor is None:
        raise ValueError("Must provide either max_scan_radius or scaling_factor")

    BG_COLOR, DRAW_COLOR = (0, 255) if bg_color == 'black' else (255, 0)

    # Initialize a blank grayscale image for the output
    image = np.ones((output_image_dims[0], output_image_dims[1]), dtype=np.uint8) * BG_COLOR

    # Direction factor
    dir = 1 if winding_dir == 'CCW' else -1

    # Select target beam count using linspace for accurate downsampling
    indices = np.linspace(0, len(scan) - 1, target_beam_count, dtype=int)
    data = np.array(scan)[indices]

    # Precompute angles
    angles = starting_angle + dir * fov * np.linspace(0, 1, target_beam_count)

    # Compute (x, y) positions
    center = np.array([output_image_dims[0] // 2, output_image_dims[1] // 2])
    points = np.column_stack((
        np.rint(center[0] + scaling_factor * data * np.cos(angles)).astype(int),
        np.rint(center[1] + scaling_factor * data * np.sin(angles)).astype(int)
    ))

    # draw according to the correct mode
    if draw_mode == 'FILL':
        cv2.fillPoly(image, [points], DRAW_COLOR)
    elif draw_mode == 'POLYGON':
        cv2.polylines(image, [points], isClosed=True, color=DRAW_COLOR, thickness=1)
    elif draw_mode == 'RAYS':
        for p in points:
            cv2.line(image, tuple(center), tuple(p), color=DRAW_COLOR, thickness=1)
            cv2.rectangle(image, tuple(p - 2), tuple(p + 2), color=DRAW_COLOR, thickness=-1)

    # Draw center point if needed
    if draw_center:
        cv2.rectangle(image, tuple(center - 2), tuple(center + 2), color=BG_COLOR if draw_mode == "FILL" else DRAW_COLOR, thickness=-1)
    
    return image

def lidar_to_bitmap(
        scan: list[float],               
        winding_dir: str='CCW',          
        starting_angle: float=-np.pi/2,  
        max_scan_radius: float | None = None,
        scaling_factor: float | None = 10, 
        bg_color: str = 'white', 
        draw_center: bool=True,  
        output_image_dims: tuple[int]=(256, 256),
        target_beam_count: int=600,
        fov: float=2*np.pi,
        draw_mode: str="POLYGON",
        channels: int=1
    ) -> np.ndarray:  
    """
    Creates a bitmap image based on lidar input.
    Assumes rays are equally spaced within the FOV.

    Args:
        scan (list[float]): A list of lidar measurements.

        winding_dir (str): The direction that the rays wind. Must either be CW or CCW in a right handed coord system.
        
        starting_angle (float): The offset from the pos-x axis that points "up" or "forward.
        
        max_scan_radius (float | None): The maximum range expected from the scans. Used to scale the value into the image if given.
        
        scaling_factor (float | None): Scaling factor for the ranges from the scan.
        
        bg_color (str): Either \'white\' or \'black\'. The accent color is always the opposite.
        
        draw_center (bool): Should this function draw a square in the center of the bitmap?
        
        output_image_dims (tuple[int]): The dimensions of the output image. Should be square but not enforced.
        
        target_beam_count (int): The target number of beams (rays) cast into the environment.

        fov (float): The field of view of the car measured in radians. Note: the output will look pinched if this is setup incorrectly.

        draw_mode (str): How should the final image be drawn. Can be \'RAYS\' (view the ray casts - keep beam count low), \'POLYGON\' (draws the outline of the rays), or \'FILL\' (filled in driveable and nondriveable boundary). 

        channels (int): The number of channels in the output. Must be 1 (grayscale), 3 (RGB), or 4 (RGBA). Default is 1.
    Returns:
        np.ndarray: An image with a birds-eye-view of the lidar scan.
    """
    assert channels in [1, 3, 4], "channels must 1, 3, or 4"

    grayscale_img = _lidar_to_bitmap(scan, winding_dir, starting_angle, max_scan_radius, scaling_factor, bg_color, draw_center, output_image_dims, target_beam_count, fov, draw_mode)
    if channels == 1:
        return grayscale_img  # Shape: (256, 256)
    elif channels == 3:
        return np.stack([grayscale_img] * 3, axis=-1)  # Shape: (256, 256, 3)
    elif channels == 4:
        alpha_channel = np.full_like(grayscale_img, 255)  # Alpha is fully opaque (255)
        return np.stack([grayscale_img, grayscale_img, grayscale_img, alpha_channel], axis=-1)  # Shape: (256, 256, 4)
    else:
        raise ValueError("Invalid number of channels. Supported: 1 (grayscale), 3 (RGB), 4 (RGBA)")
##############################
##        OPIUM MODEL       ##
##############################

class Actor(nn.Module):
    def __init__(self, action_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*28*28, 512)
        self.fc_mean = nn.Linear(512, action_dim)
        self.fc_log_std = nn.Linear(512, action_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return y_t, log_prob

class Critic(nn.Module):
    def __init__(self, action_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*28*28 + action_dim, 512)
        self.fc2 = nn.Linear(512, 1)
    
    def forward(self, x, action):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s,a,r,ns,d):
        self.buffer.append((s,a,r,ns,d))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,ns,d = map(np.stack, zip(*batch))
        return s,a,r,ns,d
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, device, action_dim=32, gamma=0.99, tau=0.005, alpha=0.2,
                 actor_lr=3e-4, critic_lr=3e-4):
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
    
    def select_action(self, state, evaluate=False):
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
    
    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return 0, 0, 0
        s,a,r,ns,d = replay_buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        if len(s.shape)==3: s = s.unsqueeze(1)
        ns = torch.FloatTensor(ns).to(self.device)
        if len(ns.shape)==3: ns = ns.unsqueeze(1)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        d = torch.FloatTensor(np.float32(d)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_a, next_logp = self.actor.sample(ns)
            tq1 = self.critic1_target(ns, next_a)
            tq2 = self.critic2_target(ns, next_a)
            tq = torch.min(tq1, tq2) - self.alpha * next_logp
            tv = r + (1-d)*self.gamma*tq
        
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
        
        for tp, p in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        for tp, p in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
        
        return a_loss.item(), c1_loss.item(), c2_loss.item()

#######################################
## PATH CLAMP & MPC HELPER FUNCTIONS ##
#######################################

def clamp_angle_diff(prev_angle: float, desired_angle: float, max_diff_deg: float=10.0) -> float:
    """Ensures desired_angle doesn't deviate from prev_angle by more than max_diff_deg (in degrees)."""
    delta = desired_angle - prev_angle
    delta = (delta + np.pi) % (2*np.pi) - np.pi
    max_diff_rad = np.deg2rad(max_diff_deg)
    if delta > max_diff_rad:
        delta = max_diff_rad
    elif delta < -max_diff_rad:
        delta = -max_diff_rad
    return prev_angle + delta

def compute_vectors_with_angle_clamp(raw_action: np.ndarray) -> np.ndarray:
    """
    Interpret raw_action (shape=(32,)) as 16 increments in [-1,1]^2,
    but clamp each vector's angle so it cannot deviate more than ±10° from the previous.
    The first vector is forced to be (1,0) in local heading space (straight forward).
    Returns an array of shape (16,2) in [-1,1], but with angle constraints.
    """
    increments = raw_action.reshape(16, 2)
    # Force the first vector to be purely forward => angle=0, magnitude=1
    increments[0] = np.array([1.0, 0.0], dtype=np.float32)

    clamped = np.zeros_like(increments)
    clamped[0] = increments[0]

    prev_angle = 0.0  # first vector angle

    for i in range(1, 16):
        dx, dy = increments[i]
        mag = np.sqrt(dx*dx + dy*dy) + 1e-8
        angle = np.arctan2(dy, dx)
        # Clamp angle relative to the previous angle
        angle = clamp_angle_diff(prev_angle, angle, max_diff_deg=10.0)
        # Keep the same magnitude
        new_dx = mag * np.cos(angle)
        new_dy = mag * np.sin(angle)
        clamped[i] = np.array([new_dx, new_dy], dtype=np.float32)
        prev_angle = angle

    return clamped

def rotate_local_to_global(dx: float, dy: float, heading: float) -> Tuple[float, float]:
    """Rotate the local vector (dx, dy) by 'heading' radians into the global frame."""
    global_dx = dx * np.cos(heading) - dy * np.sin(heading)
    global_dy = dx * np.sin(heading) + dy * np.cos(heading)
    return global_dx, global_dy

def get_steering_and_speed(
    target_x: float,
    target_y: float,
    car_x: float,
    car_y: float,
    car_theta: float
) -> np.ndarray:
    """Simple 'MPC' that aims the car at the given (target_x, target_y)."""
    desired_heading = np.arctan2(target_y - car_y, target_x - car_x)
    steering = desired_heading - car_theta
    steering = np.clip(steering, -1, 1)
    
    speed = 4.0 * (1 - np.abs(steering))
    speed = np.clip(speed, 0.0, 6.0)
    return np.array([steering, speed])

##############################
##     GYM ENVIRONOMENT     ##
##############################

class SACF110Env(gym.Env):
    """
    This environment only builds a new path once the car physically
    finishes the old path (i.e., is within DIST_THRESHOLD of the final point).
    
    - We parse the 32D action => 16 local increments => clamp angles => build 16 global points
      from the car's front.
    - We have a 'sub_index' from 0..15 to track which point we're heading to.
    - If the agent tries to provide a new action while sub_index < 16, we store it in 'pending_action'
      but do not parse it yet.
    - Once sub_index=16 and the car is physically near that final point, we parse the pending_action
      (if any) to build a new path => sub_index=0 again.
    """
    DIST_THRESHOLD = 0.2  # distance threshold (in meters) for "reaching" a point

    def __init__(self, f110_env: gym.Env):
        super().__init__()
        self.f110_env = f110_env
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(256,256), dtype=np.uint8)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
        
        self.car_length = 0.3
        self.vector_length = 0.5
        
        self.path_points = None    # list of 16 (x,y)
        self.sub_index = 16        # force parse new path on first step
        self.pending_action = None # store the agent's latest action if we haven't used it yet

        self.last_obs = None
        self.prev_x = None
        self.prev_y = None

    def reset(self):
        # Example: start at (0,0) with 90° heading
        default_pose = np.array([[0.0, 0.0, 1.57]])
        obs, _, _, _ = self.f110_env.reset(default_pose)

        lidar_scan = obs['scans'][0]
        # Use FILL mode with a black background to fill the shape
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(256,256), bg_color='black', draw_mode='FILL')
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
        1) If sub_index>=16 and we've physically reached the final old path point,
           parse pending_action or raw_action => build new path => sub_index=0
        2) Else if sub_index<16, ignore raw_action but store it as pending_action
        3) Move the car one step toward path_points[sub_index]
        4) If the car is within DIST_THRESHOLD of path_points[sub_index], sub_index++
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
        action_out = get_steering_and_speed(
            target_x, target_y,
            car_x, car_y,
            self.last_obs['poses_theta'][0]
        )

        obs, base_reward, done, info = self.f110_env.step(np.array([action_out]))

        lidar_scan = obs['scans'][0]
        # Again, use FILL mode with black background in the updated observation.
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(256,256), bg_color='black', draw_mode='FILL')
        
        collision_penalty = -100.0 if done else 0.0
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
        
        total_reward = (base_reward
                        + progress_reward
                        + lap_completion_bonus
                        + not_moving_penalty
                        - collision_penalty)

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

        return bitmap, total_reward, done, info

    def _parse_new_path(self, raw_action: np.ndarray):
        if self.pending_action is not None:
            action_to_use = self.pending_action
            self.pending_action = None
        else:
            action_to_use = raw_action
        
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
            
            global_dx = dx_scaled * np.cos(car_theta) - dy_scaled * np.sin(car_theta)
            global_dy = dx_scaled * np.sin(car_theta) + dy_scaled * np.cos(car_theta)
            
            px, py = new_points[-1]
            new_x = px + global_dx
            new_y = py + global_dy
            new_points.append((new_x, new_y))
        
        self.path_points = new_points[1:]
        self.sub_index = 0

###############################
##  DISPLAYING EVERYTHING    ##
###############################

arrow_graphics = []
current_planned_path = None

def render_arrow(env_renderer, flattened_path):
    global arrow_graphics
    for arrow in arrow_graphics:
        arrow.delete()
    arrow_graphics = []
    
    points = flattened_path.reshape(-1, 2)
    scale = 50.0
    for i in range(len(points)-1):
        x0, y0 = points[i]
        x1, y1 = points[i+1]
        arrow = env_renderer.batch.add(
            2, GL_LINES, None,
            ('v2f', (x0*scale, y0*scale, x1*scale, y1*scale)),
            ('c3B', (0, 255, 0, 0, 255, 0))
        )
        arrow_graphics.append(arrow)

def render_callback(env_renderer):
    global current_planned_path
    if current_planned_path is not None:
        render_arrow(env_renderer, current_planned_path)

###############################################
##              MAIN TRAINING LOOP           ##
###############################################

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
    cv2.destroyAllWindows()  # Close the bitmap window when done
    print("Training complete, model saved as sac_actor.pth")

if __name__=="__main__":
    main()
