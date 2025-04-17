import time
import numpy as np
import gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from f110_gym.envs.f110_env import F110Env

class F110LidarEnv(gym.Env):
    """
    F110 environment with 36 LIDAR sensors spaced 10 degrees each, and no max_range clipping.
    """
    def __init__(self, map_path, num_agents=1, timestep=0.01, max_steering=0.4, 
                 max_throttle=1.0, max_episode_steps=2000):
        super().__init__()
        
        self.f110 = F110Env(
            map=map_path,
            map_ext=".png",
            num_agents=num_agents,
            timestep=timestep
        )
        
        # Observation: 36 LIDAR rays + normalized speed. Assuming F110's max lidar range is 20.0
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=20.0,
            shape=(36 + 1,),
            dtype=np.float32
        )

        # Action space: [steering, throttle] with minimum throttle set at 0.4.
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steering, 0.4]),
            high=np.array([max_steering, max_throttle]),
            dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        init_pose = np.array([[0.0, 0.0, np.pi/2]])  # Starting position
        obs, _, _, _ = self.f110.reset(init_pose)
        return self._get_observation(obs)

    def step(self, action):
        self.num_steps += 1
        
        # Ensure minimum throttle is maintained.
        action[1] = np.clip(action[1], 0.4, 1.0)
        
        # Step the underlying environment.
        obs, _, done, info = self.f110.step(np.array([action]))
        
        observation = self._get_observation(obs)
        reward = self._calculate_reward(obs, action)
        
        if self.num_steps >= self.max_episode_steps:
            done = True
            
        return observation, reward, done, info

    def _get_observation(self, obs_dict):
        """Sample 36 LIDAR rays (every 10 degrees) and add normalized speed."""
        scan = obs_dict['scans'][0]  # Original 1080-element scan
        # Sample every 30th element to get 36 rays (1080/30 = 36)
        sampled_scan = scan[::30]
        
        # Normalized speed (0-1 scale, assuming max ~4 m/s)
        speed = obs_dict['linear_vels_x'][0] / 4.0
        normalized_speed = np.clip(speed, 0.0, 1.0)
        
        return np.concatenate([sampled_scan, [normalized_speed]]).astype(np.float32)

    def _calculate_reward(self, obs_dict, action):
        """Reward function using the 36 sampled LIDAR rays."""
        speed = obs_dict['linear_vels_x'][0]
        speed_reward = speed * 0.2
        
        # Process sampled LIDAR for safety
        scan = obs_dict['scans'][0]
        sampled_scan = scan[::30]  # Same 36-ray sampling as observation
        
        # Penalize closeness to obstacles (under 2.0 meters)
        safety_penalty = np.sum([max(0, 1.0 - (v / 2.0)) for v in sampled_scan])
        
        steering_penalty = abs(action[0]) * 0.1
        collision_penalty = 10.0 if self.f110.sim.agents[0].in_collision else 0.0
        
        return speed_reward - safety_penalty - steering_penalty - collision_penalty

    def render(self, mode='human'):
        return self.f110.render(mode)

def train_model():
    MAP_PATH = "../assets/example_map"  # Update with your map path
    
    env = F110LidarEnv(
        map_path=MAP_PATH,
        max_steering=0.4,
        max_throttle=1.0,
        max_episode_steps=1000
    )
    
    check_env(env)  # Verify that the environment adheres to Gym's interface

    # Define the network architecture for SAC.
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], qf=[64, 64])
    )
    
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        gamma=0.99,
        tensorboard_log="./f110_lidar_logs"
    )
    
    try:
        # Infinite training loop. Will only exit when an exception (crash or manual interrupt) occurs.
        while True:
            model.learn(total_timesteps=100000)
            # Optionally, include the timestep number in the save filename:
            model.save("f110_lidar_sac")
            print("Checkpoint saved.")
    except KeyboardInterrupt:
        print("Training interrupted! Saving the model before exit.")
        model.save("f110_lidar_sac")
    except Exception as e:
        print(f"An error occurred: {e}\nSaving the model before exit.")
        model.save("f110_lidar_sac")
    
    return model, env

def demo_rendering(model, env):
    """
    Runs an episode with continuous rendering,
    forcing the throttle to always be at maximum.
    """
    obs = env.reset()
    done = False
    while not done:
        env.render("human_fast")
        
        action, _ = model.predict(obs)
        action[1] = 2  # Force high throttle
        
        obs, reward, done, info = env.step(action)
    
    env.close()

if __name__ == "__main__":
    model, env = train_model()
    demo_rendering(model, env)