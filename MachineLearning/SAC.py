import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque

# ================================
# 🔹 1. REPLAY BUFFER
# ================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float),
            torch.tensor(action, dtype=torch.float),
            torch.tensor(reward, dtype=torch.float).unsqueeze(1),
            torch.tensor(next_state, dtype=torch.float),
            torch.tensor(done, dtype=torch.float).unsqueeze(1),
        )

    def size(self):
        return len(self.buffer)

# ================================
# 🔹 2. NETWORKS (Actor & Critic)
# ================================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)  # Prevents large variances
        std = log_std.exp()
        return mu, std

    def sample_action(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.max_action
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 Network
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q1 = nn.Linear(256, 1)

        # Q2 Network
        self.fc3 = nn.Linear(state_dim + action_dim, 256)
        self.fc4 = nn.Linear(256, 256)
        self.q2 = nn.Linear(256, 1)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.fc1(xu))
        q1 = F.relu(self.fc2(q1))
        q1 = self.q1(q1)

        q2 = F.relu(self.fc3(xu))
        q2 = F.relu(self.fc4(q2))
        q2 = self.q2(q2)

        return q1, q2  # Two Q-values for stability

# ================================
# 🔹 3. SAC AGENT
# ================================
class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(1000000)
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Entropy coefficient

        # Copy weights to target critic
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return

        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        # Update Critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample_action(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * min_q_next

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        action_new, log_prob_new = self.actor.sample_action(state)
        q1_new, q2_new = self.critic(state, action_new)
        actor_loss = (self.alpha * log_prob_new - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target network
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

# ================================
# 🔹 4. TRAIN SAC
# ================================
env = gym.make("f110-v0", map="example_map", num_agents=1)
sac_agent = SAC(state_dim=env.observation_space.shape[0], 
                action_dim=env.action_space.shape[0], 
                max_action=env.action_space.high[0])

num_episodes = 500
batch_size = 64

for episode in range(num_episodes):
    state = env.reset()
    state = state[0] if isinstance(state, tuple) else state  # Gym API change
    total_reward = 0

    for step in range(200):  # Max steps per episode
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action, _ = sac_agent.actor.sample_action(state_tensor)
        action = action.detach().numpy()[0]

        next_state, reward, done, _, _ = env.step(action)
        sac_agent.store_transition(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        sac_agent.update(batch_size)

        if done:
            break

    print(f"Episode {episode+1}: Reward = {total_reward:.2f}")

env.close()
