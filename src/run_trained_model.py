import time
import numpy as np
import gym
from stable_baselines3 import SAC
from f110_gym.envs.f110_env import F110Env

class F110LineSensorEnv(gym.Env):
    """
    Simplified F110 environment with 5 line sensors for basic obstacle detection.
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
        
        # Sensor configuration (angles in degrees relative to car heading)
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.max_range = 10.0  # Maximum sensor range in meters
        
        # Observation space: 5 sensor readings + current speed
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=self.max_range,
            shape=(len(self.sensor_angles) + 1,),
            dtype=np.float32
        )

        # Action space: [steering, throttle]
        self.action_space = gym.spaces.Box(
            low=np.array([-max_steering, 0.4]),  # Minimum throttle to maintain speed
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
        
        # Ensure minimum throttle is maintained
        action[1] = np.clip(action[1], 0.4, 1.0)
        
        # Step the underlying environment
        obs, _, done, info = self.f110.step(np.array([action]))
        
        observation = self._get_observation(obs)
        reward = self._calculate_reward(obs, action)
        
        if self.num_steps >= self.max_episode_steps:
            done = True
            
        return observation, reward, done, info

    def _get_observation(self, obs_dict):
        """Process LIDAR scan into 5 sensor readings and add speed."""
        scan = obs_dict['scans'][0]
        sensor_readings = []
        
        # Convert angles to LIDAR indices
        for angle in self.sensor_angles:
            idx = int((angle + 135) / 0.25)
            idx = np.clip(idx, 0, len(scan)-1)
            distance = scan[idx] if scan[idx] < self.max_range else self.max_range
            sensor_readings.append(distance)
        
        # Add normalized speed (0-1 scale)
        speed = obs_dict['linear_vels_x'][0] / 4.0  # Assuming max speed ~4 m/s
        sensor_readings.append(np.clip(speed, 0.0, 1.0))
        
        return np.array(sensor_readings, dtype=np.float32)

    def _calculate_reward(self, obs_dict, action):
        """Simple reward function encouraging speed and safety."""
        speed = obs_dict['linear_vels_x'][0]
        speed_reward = speed * 0.2
        
        sensor_values = self._get_observation(obs_dict)[:-1]
        safety_penalty = sum([max(0, 1.0 - (v/2.0)) for v in sensor_values])

        # New steering alignment reward
        # Quadratic penalty for steering away from 0, making it exponentially more costly
        steering_penalty = (action[0] ** 2) * 0.5
        # Optional: Add a small bonus for being close to 0 steering
        steering_alignment_bonus = np.exp(-abs(action[0]) * 5)
        collision_penalty = 10.0 if self.f110.sim.agents[0].in_collision else 0.0
        
        return speed_reward - safety_penalty - steering_penalty + steering_alignment_bonus - collision_penalty

    def render(self, mode='human'):
        return self.f110.render(mode)

def demo_rendering(model, env):
    """
    Runs an episode with continuous rendering until a collision is detected.
    """
    obs = env.reset()
    done = False
    while not done:
        env.render("human_fast")
        
        action, _ = model.predict(obs)
        action[1] = 2  # Force high throttle
        
        obs, reward, done, info = env.step(action)
        
        # Check for collision using the simulation's flag.
        if env.f110.sim.agents[0].in_collision:
            print("Collision detected! Ending run.")
            break
    
    env.close()

if __name__ == "__main__":
    MAP_PATH = "../assets/example_map"  # Update with your map path

    env = F110LineSensorEnv(
        map_path=MAP_PATH,
        max_steering=0.4,
        max_throttle=2,
        max_episode_steps=10000000
    )
    
    # Load your pre-trained model (ensure the file name/extension is correct)
    model = SAC.load("f110_line_sensor_sac.zip")
    
    demo_rendering(model, env)
