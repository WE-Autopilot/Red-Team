import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import cv2
import numpy as np
import cvxpy as cp
from scipy.interpolate import CubicSpline

##############################
##     GYM ENVIRONOMENT     ##
##############################

class SACF110Env(gym.Env):
    print("Ben will do this")


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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the actor network.
        
        :param x: A (batch, 1, 256, 256) input tensor (the bitmap observation).
        :return: (mean, log_std) for the Gaussian distribution over actions.
        """
    
    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action using the reparameterization trick.
        
        :param x: A (batch, 1, 256, 256) input tensor.
        :return: (action, log_prob), where 'action' is in [-1,1]^action_dim,
                 and 'log_prob' is the log-probability of that action.
        """

class Critic(nn.Module):
    """
    Purpose: The Critic estimates the Q-value of a given state (the bitmap) and action (the 32D vector). 
    It also uses convolutional layers for the state, then concatenates the action for a final Q-value estimate 
        (Q-Values or Action-Values : These represent the expected rewards for taking an action in a specific state).
    """
    
    def __init__(self, action_dim: int = 32):
        """
        Initializes the Critic network.
        
        :param action_dim: Dimensionality of the action vector (e.g. 32).
        """
    
    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the critic network, estimating Q-value.
        
        :param x: A (batch, 1, 256, 256) input tensor (the bitmap observation).
        :param action: A (batch, action_dim) tensor of actions.
        :return: A (batch, 1) tensor representing Q-values for state-action pairs.
        """


class ReplayBuffer:
    """
    Purpose: The ReplayBuffer stores (state, action, reward, next_state, done) tuples for off-policy RL. 
    It supports pushing new transitions and sampling random batches for training.
    """
    def __init__(self, capacity: int = 1000000):
        self.capacity = capacity
        # creating empty list to store experiences and setting current position for inserting new experiences to index 0
        self.buffer = []
        self.position = 0
        """
        Constructs a replay buffer for storing transitions.
        
        :param capacity: Maximum number of transitions to store.
        """
    
    def push(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, d: bool):
        if len(self.buffer) < self.capacity: #if the buffer isn't at capacity, expand it
            self.buffer.append(None)
        self.buffer[self.position] = (s, a, r, ns, d) #inserting pushed experience

        self.position = (self.position + 1) % self.capacity #incrementing position index
        """
        Adds a transition to the replay buffer.
        
        :param s: State (observation) array.
        :param a: Action array.
        :param r: Reward (float).
        :param ns: Next state (observation) array.
        :param d: Done flag (boolean).
        """
    
    def sample(self, batch_size: int):
        sample_indices = np.random.choice(len(self.buffer), batch_size, replace=False) #gets random indices without any being repeated
        batch = [self.buffer[i] for i in sample_indices]

        states, actions, rewards, next_states, done = map(np.stack, zip(*batch))
        
        return states, actions, rewards, next_states, done
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

def compute_vectors_with_angle_clamp(raw_action: np.ndarray) -> np.ndarray:
    """
    Interprets 'raw_action' (shape=(32,)) as 16 increments in [-1,1]^2,
    forcing the first vector to be (1,0) and clamping subsequent angles ±10°.
    
    :param raw_action: A 1D array of length 32 (16 x 2).
    :return: A (16,2) array of clamped increments in [-1,1].
    """

############################
##     MPC CONTROLLER     ##
############################

def MPC_controller(path: np.ndarray, desiredVelocity: float, timeStep: float, totalSteps: float, horizonLength: float, stateCost: np.ndarray, inputCost: np.ndarray, terminalCost: np.ndarray) -> np.ndarray:
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
    # Here we start at the first point of the drawn track, with zero velocity.
    x_current = np.array([path[0, 0], path[0, 1], 0, 0])
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
            cost += cp.quad_form(x[:, k] - ref_state, stateCost) + cp.quad_form(u[:, k], inputCost) # Adds a penalty to any deviation from the reference state and control input
            constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]] # Constraints on the state dynamics
            constraints += [u[:, k] <= np.array([1.0, 1.0]),
                            u[:, k] >= np.array([-1.0, -1.0])] # Constraints on the control inputs (between -1 & 1)
        
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

        state_history.append(x_current, u_apply)
    
    # u_history and state_history can be combined into state_history. Optional
    u_history = np.array(u_history)
    state_history = np.array(state_history)

    return u.history

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
    # Calculate total acceleration
    total_accel = np.sqrt(x_accel**2 + y_accel**2)
    
    # Normalize the acceleration to within given limits
    thrust = min(total_accel, max_accel)
    
    # Calculate the desired angle from acceleration (direction of desired velocity)
    desired_steer = np.arctan2(y_accel, x_accel)  # Angle of the desired velocity vector
    
    # Calculate the steering angle difference
    steer_diff = desired_steer - current_steer
    
    # Normalize steering
    if np.fabs(steer_diff) > 1e-4:
        final_steer = (steer_diff / np.fabs(steer_diff)) * max_steer
    else:
        final_steer = 0.0
    
    return np.array([final_steer, thrust])

##################
##     MAIN     ##
##################

def main():
    print("Ben will also do this")
    

if __name__=="__main__":
    main()

