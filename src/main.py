import os

import cv2
import gym
import torch
import random
import gc

from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from sacf110env import SACF110Env, render_callback


def changeMap(f110_env):

    print("hi")
    value = random.randrange(0,5)
    listOfMaps = ["maps/BrandsHatch_map","maps/Budapest_map","maps/IMS_map","maps/Spielberg_map","../assets/example_map"]

    print(listOfMaps[value])
    f110_env.update_map(map_path = "./assets/example_map",map_ext = ".png")

def load_latest_checkpoint(agent, checkpoint_dir="../output/checkpoints"):
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint files found. Starting from scratch.")

# In your main training loop, before starting training:
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f110_env = gym.make('f110_gym:f110-v0', map='../assets/example_map', map_ext='.png',
                        num_agents=1, timestep=0.015)
    
    
    from pyglet.gl import GL_LINES
    f110_env.add_render_callback(render_callback)
    
    env = SACF110Env(f110_env)
    agent = SACAgent(device, action_dim=16)
    
    # Try to resume from the latest checkpoint
    load_latest_checkpoint(agent, checkpoint_dir="../out/checkpoints")
    
    replay_buffer = ReplayBuffer()
    
    batch_size = 64
    update_after = 1000
    update_every = 50

    # Ensure checkpoint directory exists
    checkpoint_dir = "../out/checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    total_steps = 0
    ep = 0
    while True:  # Infinite training loop
        ep += 1
        obs = env.reset()
        ep_reward = 0
        while True:
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
        
        # Save a checkpoint every 25 episodes
        if ep % 2 == 0:
            version = ep // 25
            checkpoint_path = os.path.join(checkpoint_dir, f"sac_actor_v{version}.pth")
            torch.save(agent.actor.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            changeMap(f110_env=f110_env)

if __name__ == "__main__":
    main()

