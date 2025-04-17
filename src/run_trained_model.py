import time
import numpy as np
import gym
from stable_baselines3 import SAC
from f110_gym.envs.f110_env import F110Env
import pyglet

class F110LidarEnv(gym.Env):
    """
    Simplified F110 environment using full 1080 LIDAR rays for obstacle detection.
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
        
        self.max_range = 10.0  # Maximum sensor range in meters
        
        # Observation includes all 1080 lidar readings plus a normalized speed
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=self.max_range,
            shape=(1080 + 1,),
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

    def reset(self, angle=np.pi/2):
        self.num_steps = 0
        init_pose = np.array([[0.0, 0.0, angle]])  # Starting position
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
        """Process full LIDAR scan (1080 rays) and add normalized speed."""
        scan = obs_dict['scans'][0]  # Assumes a 1080-element array.
        clipped_scan = np.clip(scan, 0, self.max_range)
        
        # Normalize speed assuming a maximum of ~4 m/s.
        speed = obs_dict['linear_vels_x'][0] / 4.0
        normalized_speed = np.clip(speed, 0.0, 1.0)
        
        return np.concatenate([clipped_scan, [normalized_speed]]).astype(np.float32)

    def _calculate_reward(self, obs_dict, action):
        """Simple reward function encouraging speed and safety."""
        speed = obs_dict['linear_vels_x'][0]
        speed_reward = speed * 0.2
        
        # Use full lidar scan for safety calculations.
        scan = obs_dict['scans'][0]
        clipped_scan = np.clip(scan, 0, self.max_range)
        safety_penalty = np.sum([max(0, 1.0 - (v / 2.0)) for v in clipped_scan])
        
        steering_penalty = abs(action[0]) * 0.1
        collision_penalty = 10.0 if self.f110.sim.agents[0].in_collision else 0.0
        
        return speed_reward - safety_penalty - steering_penalty - collision_penalty

    def render(self, mode='human'):
        return self.f110.render(mode)


def demo_rendering(model, env):
    """
    Runs an episode with continuous rendering until a collision is detected.
    """
    obs = env.reset()
    done = False
    steps = 0

    while not done:
        steps += 1
        env.render("human_fast")
        
        action, _ = model.predict(obs)
        action[1] = 2  # Force high throttle
        
        obs, reward, done, info = env.step(action)
        
        if env.f110.sim.agents[0].in_collision:
            print("Collision detected! Ending run.")
            break

    env.close()


if __name__ == "__main__":
    MAP_PATH = "../assets/example_map"  # Update with your map path

    env = F110LidarEnv(
        map_path=MAP_PATH,
        max_steering=0.4,
        max_throttle=2,
        max_episode_steps=1000000
    )
    
    # Load your pre-trained model. Use the updated model file (e.g. "f110_lidar_sac.zip")
    model = SAC.load("f110_lidar_sac.zip")
    
    demo_rendering(model, env)
