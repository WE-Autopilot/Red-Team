import os
import cv2
import gym
import torch

from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from sacf110env import SACF110Env, render_callback

# Config
DO_RENDER = True
RENDER_SPEED = 'human_fast' # either human or human_fast
MAP_PATH = '../assets/example_map'
CHECKPOINT_DIR = '../output/checkpoints'

# training hyperparams
BATCH_SIZE = 64
UPDATE_EVERY = 50
UPDATE_AFTER = 1000

def load_latest_checkpoint(agent, checkpoint_dir):
    # if there is no checkpoint dir, make it
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found. Creating it..")
        return
    
    # Look for files that match our naming scheme, e.g., sac_actor_v*.pth
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("sac_actor_v") and f.endswith(".pth")]

    # if there are checkpoint files, load the latest one
    if checkpoint_files:
        # Sort by version number extracted from filename (e.g., v1, v2, etc.)
        checkpoint_files.sort(key=lambda x: int(x.split("v")[1].split(".")[0]))

        # get the latest one
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        # load it
        print(f"Loading latest checkpoint: {checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint files found. Starting from scratch.")

# In your main training loop, before starting training:
def main(do_render: bool, render_speed="human_fast"):
    # find torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create gym
    f110_env = gym.make('f110_gym:f110-v0', map=MAP_PATH, map_ext='.png', num_agents=1, timestep=0.015)
    
    # if we're rendering, add it
    if do_render:
        f110_env.add_render_callback(render_callback)
    
    # initialize the environment and SAC Agent
    env = SACF110Env(f110_env)
    agent = SACAgent(device, action_dim=16)
    
    # Try to resume from the latest checkpoint
    load_latest_checkpoint(agent, CHECKPOINT_DIR)
    
    # initialize replay buffer
    replay_buffer = ReplayBuffer()
    
    # Ensure checkpoint directory exists
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    # Infinite training loop
    ep = 0
    total_steps = 0
    while True:  
        ep += 1
        obs = env.reset()
        ep_reward = 0

        # idk what this does
        while True:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            
            if do_render:
                f110_env.render(render_speed)
                cv2.imshow("LiDAR Bitmap", obs)
                cv2.waitKey(1)
            
            if total_steps > UPDATE_AFTER and total_steps % UPDATE_EVERY == 0:
                a_loss, c1_loss, c2_loss = agent.update(replay_buffer, BATCH_SIZE)
                print(f"Step {total_steps}: Actor={a_loss:.4f}, Critic1={c1_loss:.4f}, Critic2={c2_loss:.4f}")
            
            if done:
                break
        print(f"Episode {ep} Reward={ep_reward:.2f}")
        
        # Save a checkpoint every 25 episodes
        if ep % 25 == 0:
            version = ep // 25
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"sac_actor_v{version}.pth")
            torch.save(agent.actor.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main(DO_RENDER, RENDER_SPEED)