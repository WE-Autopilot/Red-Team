import os
import cv2
import gym
import torch
import random
import numpy as np

from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from sacf110env import SACF110Env, render_callback

# ----------------------------------------------------------------
# Config
DO_RENDER = True
RENDER_SPEED = 'human_fast'  # either 'human' or 'human_fast'
MAP_PATH = '../assets/example_map'
CHECKPOINT_DIR = '../out/checkpoints'
CHECKPOINT_INTERVAL = 1000  # after how many episodes do we save a checkpoint?

# Training hyperparams
BATCH_SIZE = 128
UPDATE_EVERY = 50     # Update every environment step
UPDATE_AFTER = 1000  # Start updating after 1000 transitions in replay
# ----------------------------------------------------------------


def changeMap(old_f110_env):
    """
    Closes the old env (optional) and returns a new f110_env, 
    along with the new custom environment and the chosen orientation angle.
    """
    if old_f110_env is not None:
        # Optionally close the old environment to free resources
        old_f110_env.close()
    
    # Randomly pick a map and orientation
    listOfMaps = {
        "maps/BrandsHatch_map": np.pi/5,
        "maps/Budapest_map": np.pi*5/6,
        "maps/IMS_map": np.pi/2,
        "maps/Spielberg_map": np.pi/6,
        "../assets/example_map": 1.57
    }
    key, value = random.choice(list(listOfMaps.items()))
    
    # Create the new low-level environment
    new_f110_env = gym.make('f110_gym:f110-v0', 
                            map=key, map_ext='.png', 
                            num_agents=1, timestep=0.015)
    
    # Optionally add rendering callback
    # (Only do this if you want the new env to render as well)
    # new_f110_env.add_render_callback(render_callback)
    
    # Wrap it in the custom environment
    new_env = SACF110Env(new_f110_env)
    
    return new_f110_env, new_env, value


def load_latest_checkpoint(agent, checkpoint_dir):
    """Loads the latest checkpoint if found."""
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found. Creating it..")
        os.makedirs(checkpoint_dir)
        return
    
    # Look for files matching "sac_actor_v*.pth"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                        if f.startswith("sac_actor_v") and f.endswith(".pth")]
    
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("v")[1].split(".")[0]))
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        
        print(f"Loading latest checkpoint: {checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint files found. Starting from scratch.")


def main(do_render: bool, render_speed="human_fast"):
    # Find torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create initial gym env
    f110_env = gym.make('f110_gym:f110-v0', 
                        map=MAP_PATH, 
                        map_ext='.png', 
                        num_agents=1, 
                        timestep=0.015)

    # If we're rendering, add the callback
    if do_render:
        f110_env.add_render_callback(render_callback)
    
    # Wrap in our custom environment
    env = SACF110Env(f110_env)
    
    # Create the SAC Agent (with a smaller LR)
    agent = SACAgent(
        device=device, 
        action_dim=16, 
        actor_lr=3e-4, 
        critic_lr=3e-4
    )
    
    # Try to resume from the latest checkpoint
    load_latest_checkpoint(agent, CHECKPOINT_DIR)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer()
    
    # Training loop
    ep = 0
    total_steps = 0
    # Arbitrary orientation for the first run
    orientation = 1.57
    
    while True:
        ep += 1
        # Reset the environment
        obs = env.reset(orientation)
        ep_reward = 0.0
        
        while True:
            # Select action
            action = agent.select_action(obs)
            
            # Step the environment
            next_obs, reward, done, info = env.step(action)
            
            # Push transition to replay
            replay_buffer.push(obs, action, reward, next_obs, done)
            
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            
            # Render if desired
            if do_render:
                f110_env.render(render_speed)
                cv2.imshow("LiDAR Bitmap", obs)
                cv2.waitKey(1)
            
            # Update if we have enough data
            if total_steps > UPDATE_AFTER and total_steps % UPDATE_EVERY == 0:
                a_loss, c1_loss, c2_loss = agent.update(replay_buffer, BATCH_SIZE)
                print(f"Step {total_steps}: "
                      f"Actor={a_loss:.4f}, Critic1={c1_loss:.4f}, Critic2={c2_loss:.4f}")
            
            if done:
                break
        
        print(f"Episode {ep} Reward={ep_reward:.2f}")
        
        # Save a checkpoint every CHECKPOINT_INTERVAL episodes
        if ep % CHECKPOINT_INTERVAL == 0:
            version = ep // CHECKPOINT_INTERVAL
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, 
                f"sac_actor_v{version}.pth"
            )
            torch.save(agent.actor.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            
            # Change the map (optional)
            f110_env, env, orientation = changeMap(f110_env)


if __name__ == "__main__":
    main(DO_RENDER, RENDER_SPEED)
