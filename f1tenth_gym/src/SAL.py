import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
import os


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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    
    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
