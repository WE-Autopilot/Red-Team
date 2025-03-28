import os
import cv2
import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from sac_agent import SACAgent
from replay_buffer import ReplayBuffer
from sacf110env import SACF110Env, render_callback
import state_machine.py

# ----------------------------------------------------------------
# Config
DO_RENDER = True  # We'll render the LiDAR bitmap, sim view, and planned path arrows.
RENDER_SPEED = 'human_fast'  # Use this speed for rendering.
MAP_PATH = '../assets/example_map'
CHECKPOINT_DIR = '../out/checkpoints'
CHECKPOINT_INTERVAL = 1000  # after how many episodes do we save a checkpoint?

# Training hyperparams
BATCH_SIZE = 128
UPDATE_EVERY = 50     # Update every environment step
UPDATE_AFTER = 1000   # Start updating after 1000 transitions in replay
GRAPH_CHECKPOINT = 5 # output to graph every 250 episodes
# ----------------------------------------------------------------


def changeMap(old_f110_env):
    """
    Closes the old env (optional) and returns a new f110_env, 
    along with the new custom environment and the chosen orientation angle.
    """
    if old_f110_env is not None:
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
    
    new_f110_env = gym.make('f110_gym:f110-v0', 
                            map=key, map_ext='.png', 
                            num_agents=1, timestep=0.015)
    
    # Turn on sim rendering by adding the render callback.
    new_f110_env.add_render_callback(render_callback)
    
    new_env = SACF110Env(new_f110_env)
    
    return new_f110_env, new_env, value


def load_latest_checkpoint(agent, checkpoint_dir):
    """Loads the latest checkpoint if found."""
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found. Creating it..")
        os.makedirs(checkpoint_dir)
        return
    
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    f110_env = gym.make('f110_gym:f110-v0', 
                        map=MAP_PATH, 
                        map_ext='.png', 
                        num_agents=1, 
                        timestep=0.015)
    
    # Turn on full sim rendering by adding the render callback.
    if do_render:
        f110_env.add_render_callback(render_callback)
    
    env = SACF110Env(f110_env)
    
    agent = SACAgent(
        device=device, 
        action_dim=16, 
        actor_lr=3e-3, 
        critic_lr=3e-3
    )
    
    load_latest_checkpoint(agent, CHECKPOINT_DIR)
    
    replay_buffer = ReplayBuffer()
    
    episode_rewards = []
    block_avg_rewards = []
    
    total_steps = 0
    orientation = 1.57  # starting orientation
    
    # Setup real-time learning graph (updates every GRAPH_CHECKPOINT episodes)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    reward_line, = ax.plot([], [], label='Avg Episode Reward per {} Episodes'.format(GRAPH_CHECKPOINT), marker='o')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('Real-Time Learning: Average Reward (per {} episodes)'.format(GRAPH_CHECKPOINT))
    ax.legend()
    
    ep = 0
    while True:
        ep += 1
        obs = env.reset(orientation)
        ep_reward = 0.0
        
        done = False
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, done, info = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            total_steps += 1
            
            if do_render:
                # Render the sim view (if any) and the LiDAR bitmap.
                f110_env.render(render_speed)
                cv2.imshow("LiDAR Bitmap", obs)
                cv2.waitKey(1)
            
            if total_steps > UPDATE_AFTER and total_steps % UPDATE_EVERY == 0:
                a_loss, c1_loss, c2_loss = agent.update(replay_buffer, BATCH_SIZE)
                print(f"Step {total_steps}: Actor Loss={a_loss:.4f}, Critic1 Loss={c1_loss:.4f}, Critic2 Loss={c2_loss:.4f}")
        
        episode_rewards.append(ep_reward)
        if ep_reward != -150:
            print(f"Episode {ep} Reward={ep_reward:.2f}")
        
        if ep % GRAPH_CHECKPOINT == 0:
            block_avg = np.mean(episode_rewards[-GRAPH_CHECKPOINT:])
            block_avg_rewards.append(block_avg)
            x_vals = np.arange(GRAPH_CHECKPOINT, GRAPH_CHECKPOINT*(len(block_avg_rewards)+1), GRAPH_CHECKPOINT)
            reward_line.set_data(x_vals, block_avg_rewards)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
        
        if ep % CHECKPOINT_INTERVAL == 0:
            version = ep // CHECKPOINT_INTERVAL
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"sac_actor_v{version}.pth")
            torch.save(agent.actor.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
            f110_env, env, orientation = changeMap(f110_env)
    
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main(DO_RENDER, RENDER_SPEED)
