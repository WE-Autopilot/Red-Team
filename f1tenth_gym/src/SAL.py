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


##############################
##     GYM ENVIRONOMENT     ##
##############################

class SACF110Env(gym.Env):
    """
    A custom environment that:
      - Wraps the f1tenth_gym environment (self.f110_env).
      - On each step, interprets the 32D action as 16 local increments.
      - Clamps angles, builds a path, picks the first vector, uses MPC to get (steering, speed).
      - Steps the underlying environment and returns the new 256x256 LiDAR-based observation.
    """

    def __init__(self, f110_env: gym.Env):
        """
        :param f110_env: The underlying f1tenth environment instance.
        """
        super().__init__()
        self.f110_env = f110_env

        # Define observation space as a 256x256 grayscale image
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(256, 256), dtype=np.uint8
        )

        # Define action space as 32D => 16 (x,y) increments
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(32,), dtype=np.float32
        )

        # Additional placeholders for state tracking if needed
        self.last_obs = None

    def reset(self):
        """
        Resets the underlying f1tenth environment and returns a 256x256 LiDAR-based bitmap.
        """
        # Example: you might have a default pose
        default_pose = np.array([[0.0, 0.0, 0.0]])
        obs, reward, done, info = self.f110_env.reset(default_pose)

        # Convert LiDAR to 256x256 bitmap
        lidar_scan = obs['scans'][0]  # f110_gym typically has 'scans'
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(256, 256))

        self.last_obs = obs
        return bitmap

    def step(self, raw_action: np.ndarray):
        """
        Interprets the 32D action as 16 local increments, clamps angles, builds a path,
        picks the first vector, uses MPC to compute [steering, speed], steps the simulator,
        returns (bitmap, reward, done, info).
        """

        # 1) Convert raw_action => shape (16,2), clamp angles
        increments = compute_vectors_with_angle_clamp(raw_action)  # shape (16,2)

        # 2) Build a path in global coords. 
        #    For simplicity, let's just take the FIRST vector as the target, ignoring the rest.
        #    If you want to hold the entire path, you'd do more logic here.
        car_x = self.last_obs['poses_x'][0]
        car_y = self.last_obs['poses_y'][0]
        car_theta = self.last_obs['poses_theta'][0]

        # The first local increment is increments[0]. Suppose we rotate it by car_theta to get global coords:
        dx_local, dy_local = increments[0]
        # If you want to rotate to global:
        global_dx = dx_local * np.cos(car_theta) - dy_local * np.sin(car_theta)
        global_dy = dx_local * np.sin(car_theta) + dy_local * np.cos(car_theta)

        target_x = car_x + global_dx
        target_y = car_y + global_dy

        # 3) Use the MPC_controller to get [steering, speed]
        control = MPC_controller(target_x, target_y, car_x, car_y, car_theta)
        # control is e.g. np.array([steering, speed])

        # 4) Step the underlying environment
        obs, base_reward, done, info = self.f110_env.step(np.array([control]))

        # 5) Convert new LiDAR => new 256x256 observation
        lidar_scan = obs['scans'][0]
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(256, 256))

        # 6) Compute a custom reward. 
        #    For now, let's do base_reward minus collision penalty:
        collision_penalty = -100.0 if done else 0.0
        total_reward = base_reward + collision_penalty

        self.last_obs = obs
        return bitmap, total_reward, done, info


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
        draw_mode: str="POLYGON"
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
        
        beam_dropout (float): How much of the scan to dropout. I.e., 0 means all beams are drawn, 0.3 means 30% of beams are dropped.
        
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

    BG_COLOR, DRAW_COLOR = (0, 180) if bg_color == 'black' else (255, 20)

    # Initialize a blank grayscale image for the output
    image = np.ones((output_image_dims[0], output_image_dims[1]), dtype=np.uint8) * BG_COLOR

    # Direction factor
    dir = 1 if winding_dir == 'CCW' else -1

    # Select target beam count using linspace for accurate downsampling
    indices = np.linspace(0, len(scan) - 1, target_beam_count, dtype=int)
    data = np.array(scan)[indices]

    # Precompute angles
    angles = starting_angle + dir * fov * np.linspace(0, 1, target_beam_count)

    # Compute (x, y) positions in one step
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

    # Draw center point
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
        
        beam_dropout (float): How much of the scan to dropout. I.e., 0 means all beams are drawn, 0.3 means 30% of beams are dropped.
        
        fov (float): The field of view of the car measured in radians. Note: the output will look pinched if this is setup incorrectly.

        draw_mode (str): How should the final image be drawn. Can be \'RAYS\' (view the ray casts - keep beam count low), \'POLYGON\' (draws the outline of the rays), or \'FILL\' (filled in driveable and nondriveable boundary). 

    Returns:
        np.ndarray: A single-channel, grayscale image with a birds-eye-view of the lidar scan.
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
    """
    Purpose: The Actor outputs a 32D continuous action (in [-1,1]) representing 16 local 2D increments.
    It uses convolutional layers to process the 256×256 bitmap and outputs a value based on model performance.
    """
    def __init__(self, action_dim: int = 32):
        """
        Initializes the Actor network.
        
        :param action_dim: The dimensionality of the action vector (e.g. 32).
        """
        # Initialize the data set.
        super(Actor, self).__init__()

        # Define convolutional layers to process the 256x256 bitmap.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=3) # Output: (batch_size, 32, 64, 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: (batch_size, 64, 32, 32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Output: (batch_size, 128, 16, 16)
        
        self.flatten = nn.Flatten() # Flatten the output into a 1D vector.

        # Define a fully connected layer to output probability.
        self.fc1 = nn.Linear(128 * 16 * 16, 512) # (batch_size, 512)
        self.fc_mean = nn.Linear(512, action_dim) # (batch_size, action_dim)
        self.fc_log_std = nn.Linear(512, action_dim) # (batch_size, action_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor network.
        
        :param x: A (batch, 1, 256, 256) input tensor (the bitmap observation).
        :return: (mean, log_std) for the Gaussian distribution over actions.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)

        return mean, log_std

    
    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action using the reparameterization trick.
        
        :param x: A (batch, 1, 256, 256) input tensor.
        :return: (action, log_prob), where 'action' is in [-1,1]^action_dim,
                 and 'log_prob' is the log probability of that action.
        """
        mean, log_std = self.forward(x) # Retrieve an action from the network.
        std = torch.exp(0.5 * log_std) # Get the standard deviation.
        eps = torch.randn_like(mean) # Random noise.
        action = mean + std * eps # Take a sample from the distribution.
        
        # Calculate the log probability.
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1) # Sum over all dimensions of the action.

        return action, log_prob
    
class Critic(nn.Module):
    """
    Purpose: The Critic estimates the Q-value of a given state (the bitmap) and action (the 32D vector). 
    It also uses convolutional layers for the state, then concatenates the action for a final Q-value estimate 
        (Q-Values or Action-Values : These represent the expected rewards for taking an action in a specific state).
    """
    
    def __init__(self, name,beta, checkPoint_dir="sac", action_dim: int = 32):
        """
        Initializes the Critic network.
        
        :param action_dim: Dimensionality of the action vector (e.g. 32).
        :param name: name for model checkpointing
        """
        # the guy has:
        # 1. the learning rate
        # 2. dimensions of the environment (itd be 256 x 256 but we dont need this since its known)
        # 3. dimensions of the fully connected layers also not needed we can just do within
        # 4. name for model checkpointing which im going to add
        # 5. checkpoint directory which im going to add 
        super(Critic,self).__init__()
        self.action_dim = action_dim
        self.name = name
        self.beta = beta
        self.checkPoint_dir = checkPoint_dir
        self.checkPoint_file = os.path.join(self.checkPoint_dir,name)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=3) # Output: (batch_size, 32, 64, 64)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: (batch_size, 64, 32, 32)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Output: (batch_size, 128, 16, 16)
        
        self.flatten = nn.Flatten() # Flatten the output into a 1D vector.

        # Define a fully connected layer to output probability.
        self.fc1 = nn.Linear(128 * 16 * 16+action_dim, 256) #Evaluates value of state and action pair 
        self.fc2 = nn.Linear(256,256)
        self.q = nn.Linear(256,1)

        self.optimizer = optim.Adam(self.parameters(),lr=beta)
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network, estimating Q-value.
        
        :param x: A (batch, 1, 256, 256) input tensor (the bitmap observation).
        :param action: A (batch, action_dim) tensor of actions.
        :return: A (batch, 1) tensor representing Q-values for state-action pairs.
        """
        state = F.relu(self.conv1(x))
        state = F.relu(self.conv2(state))
        state = F.relu(self.conv3(state))
        state = self.flatten(state)


        action_value = self.fc1(torch.cat([state,action],dim = 1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q


class Sample:
    """
    Wraps a transition for prioritized replay.
    """
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.weight = 1.0
        self.cumulative_weight = 1.0

    def is_interesting(self):
        return self.done or self.reward != 0

    def __lt__(self, other):
        return self.cumulative_weight < other.cumulative_weight
    
class ReplayBuffer:
    """
    Purpose: The ReplayBuffer stores (state, action, reward, next_state, done) tuples for off-policy RL. 
    It supports pushing new transitions and sampling random batches for training.
    """
    def __init__(self, capacity: int = 1000000, prioritized_replay: bool = False, base_output_dir: str = "."):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.prioritized_replay = prioritized_replay

        # For prioritized replay
        self.num_interesting_samples = 0
        self.batches_drawn = 0

        # Optionally set up saving directory
        self.save_buffer_dir = os.path.join(base_output_dir, "models")
        if not os.path.isdir(self.save_buffer_dir):
            os.makedirs(self.save_buffer_dir)
        self.file = "replay_buffer.dat"
        """
        Constructs a replay buffer for storing transitions.
        
        :param capacity: Maximum number of transitions to store.
        """
    
    def push(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, d: bool):
        if self.prioritized_replay:
            sample = Sample(s, a, r, ns, d)
        else:
            sample = (s, a, r, ns, d)

        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.position] = sample
        
        if self.prioritized_replay:
            self._update_weights()
        self.position = (self.position + 1) % self.capacity
        """
        Adds a transition to the replay buffer.
        
        :param s: State (observation) array.
        :param a: Action array.
        :param r: Reward (float).
        :param ns: Next state (observation) array.
        :param d: Done flag (boolean).
        """
    
    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            raise IndexError(f"Not enough samples ({len(self.buffer)}) to draw a batch of {batch_size}")

        if self.prioritized_replay:
            self.batches_drawn += 1
            return self._draw_prioritized_batch(batch_size)
        else:
            sample_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in sample_indices]
            states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
            return states, actions, rewards, next_states, dones
        """
        Samples a batch of transitions from the buffer.
        
        :param batch_size: Number of transitions to sample.
        :return: (states, actions, rewards, next_states, dones) as stacked arrays.
        """
    
    def __len__(self) -> int:
        return len(self.buffer)
        """
        :return: Current number of transitions in the buffer.
        """
    def save(self):
        with open(os.path.join(self.save_buffer_dir, self.file), "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, file):
        with open(file, "rb") as f:
            self.buffer = pickle.load(f)

    def _truncate_list_if_necessary(self):
        # Truncate the buffer if it exceeds 105% of capacity.
        if len(self.buffer) > self.capacity * 1.05:
            if self.prioritized_replay:
                truncated_weight = 0
                for i in range(self.capacity, len(self.buffer)):
                    truncated_weight += self.buffer[i].weight
                    if self.buffer[i].is_interesting():
                        self.num_interesting_samples -= 1
            self.buffer = self.buffer[-self.capacity:]
            if self.prioritized_replay:
                for sample in self.buffer:
                    sample.cumulative_weight -= truncated_weight

    def _draw_prioritized_batch(self, batch_size: int):
        # Assumes self.buffer is sorted by cumulative_weight
        batch = []
        probe = Sample(None, 0, 0, None, False)
        while len(batch) < batch_size:
            # Choose a random number between 0 and the last sample's cumulative weight
            probe.cumulative_weight = random.uniform(0, self.buffer[-1].cumulative_weight)
            index = bisect.bisect_right(self.buffer, probe)
            sample = self.buffer[index]
            # Decay the sample's weight slightly
            sample.weight = max(1.0, 0.8 * sample.weight)
            if sample not in batch:
                batch.append(sample)
        if self.batches_drawn % 100 == 0:
            cumulative = 0
            for sample in self.buffer:
                cumulative += sample.weight
                sample.cumulative_weight = cumulative
        # Convert Sample objects into tuples for consistency with training code
        batch_tuples = [(s.state, s.action, s.reward, s.next_state, s.done) for s in batch]
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch_tuples))
        return states, actions, rewards, next_states, dones


    def _update_weights(self):
        if len(self.buffer) > 1:
            last_sample = self.buffer[-1]
            last_sample.cumulative_weight = last_sample.weight + self.buffer[-2].cumulative_weight

        if self.buffer[-1].is_interesting():
            self.num_interesting_samples += 1
            # Boost neighboring samples; number depends on frequency of "interesting" samples
            uninteresting_range = max(1, len(self.buffer) / max(1, self.num_interesting_samples))
            uninteresting_range = int(uninteresting_range)
            for i in range(uninteresting_range, 0, -1):
                index = len(self.buffer) - i
                if index < 1:
                    break
                boost = 1.0 + 3.0 / math.exp(i / (uninteresting_range / 6.0))
                self.buffer[index].weight *= boost
                self.buffer[index].cumulative_weight = self.buffer[index].weight + self.buffer[index - 1].cumulative_weight

class SACAgent:
    def __init__(self, device: torch.device, action_dim: int = 32, gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2, actor_lr: float = 3e-4, critic_lr: float = 3e-4):
        """
        Initializes the Soft Actor-Critic agent.
        
        :param device: Torch device (CPU or CUDA).
        :param action_dim: Dimensionality of the action vector (e.g. 32).
        :param gamma: Discount factor.
        :param tau: Soft update coefficient for target critics.
        :param alpha: Entropy temperature (entropy regularization).
        :param actor_lr: Learning rate for the actor.
        :param critic_lr: Learning rate for the critics.
        """

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """
        Selects an action from the current policy.
        
        :param state: A 2D (256,256) or 3D (1,256,256) array representing the observation.
        :param evaluate: If True, use the mean action (deterministic); else sample stochastically.
        :return: A 1D array of shape (action_dim,) in [-1,1].
        """

    def update(self, replay_buffer: 'ReplayBuffer', batch_size: int = 64) -> Tuple[float, float, float]:
        """
        Performs one SAC update step (actor + critics).
        
        :param replay_buffer: The ReplayBuffer containing transitions.
        :param batch_size: Number of transitions to sample for the update.
        :return: (actor_loss, critic1_loss, critic2_loss) as floats.
        """


def clamp_vector_angle_diff(prev_angle: float, desired_angle: float, max_diff_deg: float = 10.0) -> float:
    """
    Purpose: Ensures consecutive vectors differ by at most ±10° (or another chosen angle). Helps keep the path smooth.
    
    :param prev_angle: The angle of the previous vector (radians).
    :param desired_angle: The angle of the current vector (radians).
    :param max_diff_deg: Maximum allowed deviation in degrees.
    :return: The clamped angle in radians.
    """

    max_diff_rad = np.radians(max_diff_deg) # Converts degrees to radians
    angle_diff = desired_angle - prev_angle # Gets the difference between the desired and the previous angles

    #Ensures angle differences stay within [-π, π] to prevent large jumps when crossing ±180°.
    angle_diff = (desired_angle - prev_angle + np.pi) % (2 * np.pi) - np.pi

    # If the angle difference is greater than EX: 10 degrees, then clamp
    if angle_diff > max_diff_rad:
        return prev_angle + max_diff_rad
    elif angle_diff < - max_diff_rad:
        return prev_angle - max_diff_rad
    
    return desired_angle # if it is within the limit (ex:10 degree) that keep it as it is (ex: 5 degrees)

def compute_vectors_with_angle_clamp(raw_action: np.ndarray, max_diff_deg: float = 10.0) -> np.ndarray:
    """
    Interprets 'raw_action' (shape=(32,)) as 16 increments in [-1,1]^2,
    forcing the first vector to be (1,0) and clamping subsequent angles ±10°.
    
    :param raw_action: A 1D array of length 32 (16 x 2).
    :return: A (16,2) array of clamped increments in [-1,1].
    """
    assert raw_action.shape == (32,), "Raw action must be a 32D vector (16 x 2D movements)."

    # Reshape the action vector into 16 movement vectors of (x, y)
    vectors = raw_action.reshape(16, 2)

    # Normalize each (x, y) vector to have unit length (ensuring direction is preserved)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # It creates an output array for clamped movement vectors
    clamped_vectors = np.zeros_like(vectors)

    # First movement vector is fixed to (1,0) for consistent direction
    clamped_vectors[0] = [1, 0]  
    prev_angle = np.arctan2(clamped_vectors[0][1], clamped_vectors[0][0])  # Get initial angle

    for i in range(1, 16):
        desired_angle = np.arctan2(vectors[i][1], vectors[i][0])  # It creates the desired angle
        clamped_angle = clamp_vector_angle_diff(prev_angle, desired_angle, max_diff_deg)  # This is the Clamp angle

        # Converts the clamped angle back to (x, y)
        clamped_vectors[i] = [np.cos(clamped_angle), np.sin(clamped_angle)]
        prev_angle = clamped_angle  # This update the previous angle

    return clamped_vectors

############################
##     MPC CONTROLLER     ##
############################

def MPC_controller(target_x: float, target_y: float, car_x: float, car_y: float, car_theta: float) -> np.ndarray:
    """
    Computes steering and speed commands aiming from (car_x, car_y, car_theta) to (target_x, target_y).
    
    :param target_x: X-coordinate of the target point in global space.
    :param target_y: Y-coordinate of the target point in global space.
    :param car_x: Current car X position.
    :param car_y: Current car Y position.
    :param car_theta: Current car heading in radians.
    :return: A 1D array [steering, speed] for the simulator step.
    """
    desired_heading = np.arctan2(target_y - car_y, target_x - car_x)
    steering = desired_heading - car_theta
    steering = np.clip(steering, -1, 1)
    speed = 4.0 * (1 - abs(steering))
    speed = np.clip(speed, 0, 5.0)
    return np.array([steering, speed])


##################
##     MAIN     ##
##################
def main():
    # Make a random 256x256 bitmap.
    rand_bitmap = torch.rand(1, 1, 256, 256) # Batch size of 1, 1 channel, 256x256 size.
    rand_bitmap = torch.round(rand_bitmap) # Make it binary (0 or 1).
    print(f"Random Bitmap: \n {rand_bitmap} \n") # Print random bitmap.

    # Create the Actor instance and send it to the device (CPU or GPU).
    actor = Actor(32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = actor.to(device)
    rand_bitmap = rand_bitmap.to(device)

    # Return a sample of an action from the Actor network.
    data = actor.sample(rand_bitmap)
    print(f"Action: \n {data[0]} \n")
    print(f"Log Prob: \n {data[1]} \n")

# Run the main function.
if __name__ == "__main__":
    main()
