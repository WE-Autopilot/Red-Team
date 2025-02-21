import torch
import torch.nn as nn
import torch.nn.functional as F

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