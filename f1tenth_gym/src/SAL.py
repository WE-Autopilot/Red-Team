import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np

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
        """
        Constructs a replay buffer for storing transitions.
        
        :param capacity: Maximum number of transitions to store.
        """
    
    def push(self, s: np.ndarray, a: np.ndarray, r: float, ns: np.ndarray, d: bool):
        """
        Adds a transition to the replay buffer.
        
        :param s: State (observation) array.
        :param a: Action array.
        :param r: Reward (float).
        :param ns: Next state (observation) array.
        :param d: Done flag (boolean).
        """
    
    def sample(self, batch_size: int):
        """
        Samples a batch of transitions from the buffer.
        
        :param batch_size: Number of transitions to sample.
        :return: (states, actions, rewards, next_states, dones) as stacked arrays.
        """
    
    def __len__(self) -> int:
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



##################
##     MAIN     ##
##################

def main():
    print("not working on it yet")
    

if __name__=="__main__":
    main()

