import time
import numpy as np
import gym
import pyglet
import random
import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from f110_gym.envs.f110_env import F110Env
import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

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
        self.sensor_angles = np.arange(-134.645,134.645,9.97370976287)
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

    def reset(self, angle = np.pi/2):
        self.num_steps = 0
        init_pose = np.array([[0.0, 0.0, angle]])  # Starting position
        obs, _, _, _ = self.f110.reset(init_pose)
        return self._get_observation(obs)
    
    def map_reset(self):
     
         maps = ["BrandsHatch","Budapest","example","IMS","Spielberg"]
         index = random.randrange(0,5)

         current_directory = os.getcwd()
         map_path = os.path.abspath(os.path.join(current_directory, "..", "assets", maps[index] + "_map.yaml"))
         self.f110.update_map(map_path, ".png")
 
         unit_Circle = np.array([0,np.pi/6,np.pi/4,np.pi/3,np.pi/2,2*np.pi/3,3*np.pi/4,5*np.pi/6,np.pi,7*np.pi/6,5*np.pi/4,4*np.pi/3,3*np.pi/2,5*np.pi/3,7*np.pi/4,11*np.pi/6])
         distance = 0
         best_angle = 0
         for angle in unit_Circle:
             observation = self.reset(angle=angle)
             value_straight_ahead = observation[13]
 
             if(value_straight_ahead>distance):
                 distance = value_straight_ahead
                 best_angle = angle
 
             if(distance>=9.99):
                 best_angle = angle
                 break
         
 
 
         init_pose = np.array([[0.0, 0.0, best_angle]])  # Starting position
         obs, _, _, _ = self.f110.reset(init_pose)
         observation = self._get_observation(obs)
 
         if self.f110.renderer is not None:
            self.f110.renderer.poses = None
            self.f110.renderer.batch = pyglet.graphics.Batch()
            self.f110.renderer.update_obs(obs)
            self.f110.renderer.update_map("../assets/"+maps[index]+"_map",".png")
 
         return observation


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
        scan = obs_dict['scans'][0]
        n = len(scan)
        fov = 2 * np.deg2rad(134.645)
        scan_angles = np.linspace(-fov/2, fov/2, n)
        desired_angles = np.linspace(-fov/2, fov/2, len(self.sensor_angles))

        sensor_vals = np.interp(desired_angles, scan_angles, scan)
        sensor_vals = np.clip(sensor_vals, 0.0, self.max_range)

        speed = obs_dict['linear_vels_x'][0] / 4.0
        return np.concatenate([sensor_vals, [np.clip(speed, 0.0, 1.0)]]).astype(np.float32)


    def _calculate_reward(self, obs_dict, action):
        speed = obs_dict['linear_vels_x'][0]
        speed_reward = 0.2 * speed

        obs = self._get_observation(obs_dict)
        sensor_vals = obs[:-1] / self.max_range       # normalize to [0,1]
        # average penalty, so it stays O(1) regardless of ray count
        safety_penalty = np.mean(np.maximum(0, 1.0 - sensor_vals))
        steering_penalty = 0.1 * abs(action[0])
        collision_penalty = 10.0 if self.f110.sim.agents[0].in_collision else 0.0

        return speed_reward - safety_penalty - steering_penalty - collision_penalty

    def render(self, mode='human'):
        return self.f110.render(mode)
    
class CustomCallback(BaseCallback):
     def __init__(self, verbose = 0):
         super().__init__(verbose)
 
     def _on_step(self):
         if(self.num_timesteps % 10000 == 0):
             self.training_env.envs[0].map_reset()
        
         if(self.num_timesteps % 100000 == 0):
             self.model.save("f110_line_sensor_sac")
         return True

def train_model():
    print("""
    Enter 1 or 2.
    1: To continue training last saved model
    2: To train a new model
    """)
    choice = input()

    MAP_PATH = "../assets/example_map"  # Update with your map path

    env = F110LineSensorEnv(
        map_path=MAP_PATH,
        max_steering=0.4,
        max_throttle=1.0,
        max_episode_steps=1000
    )
    callBack = CustomCallback()
    check_env(env)  # Verify that your environment adheres to Gym's interface

    # Define the network architecture for SAC.
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], qf=[64, 64])
    )

    if choice == "2" :
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
            tensorboard_log="./f110_line_sensor_logs",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else :
        model = SAC.load("f110_line_sensor_sac.zip", env=env, device="cuda")

    try:
        print("Model is on device:", model.device)
        # Train indefinitely in chunks of 100,000 timesteps.
        model.learn(total_timesteps=5000000,callback=callBack)
        model.save("f110_line_sensor_sac")
    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        model.save("f110_line_sensor_sac")
    
    return model, env

def demo_rendering(model, env):
    """
    Runs an episode with continuous rendering,
    forcing the throttle to always be at maximum.
    """
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        steps=steps+1
        env.render("human_fast")
        
        action, _ = model.predict(obs)
        action[1] = 1.0
        
        obs, reward, done, info = env.step(action)

        if steps%1000 == 0:
              obs = env.map_reset()
    
    env.close()

if __name__ == "__main__":
    model, env = train_model()
    demo_rendering(model, env)
