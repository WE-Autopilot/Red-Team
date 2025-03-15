import os
import cv2
import gym
import torch
import numpy as np

from sac_agent import SACAgent
from sacf110env import SACF110Env, render_callback

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the F1Tenth environment using the hardcoded map path
    f110_env = gym.make('f110_gym:f110-v0',
                        map='../src/maps/Spielberg_map',
                        map_ext='.png',
                        num_agents=1,
                        timestep=0.015)
    
    # Wrap the base environment in the custom SAC environment wrapper
    env = SACF110Env(f110_env)
    
    # Initialize the SAC Agent; only the actor is used for evaluation.
    agent = SACAgent(
        device=device,
        action_dim=16,
        actor_lr=3e-4,
        critic_lr=3e-4
    )
    
    # Hardcode the checkpoint path and load it
    checkpoint_path = os.path.join("..", "out", "checkpoints", "sac_actor_v149.pth")
    print(f"Loading checkpoint from: {checkpoint_path}")
    agent.actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    num_episodes = 10
    for ep in range(1, num_episodes+1):
        obs = env.reset(np.pi/6)  
        done = False
        ep_reward = 0.0
        
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            
            # Display the LiDAR bitmap
            cv2.imshow("LiDAR Bitmap", obs)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"Episode {ep} Reward: {ep_reward:.2f}")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
