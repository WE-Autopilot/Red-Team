import time
import numpy as np
import gym
import pyglet
import random
import os
from stable_baselines3 import SAC
from f110_gym.envs.f110_env import F110Env
import pyglet
import random
from statistics import mean

logs = {"speed":[],"safety":[], "steering":[],"collision":[], "center-bonus":[],"timeStep_Reward":[],"direction":[]}
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
        self.prev_penalty = 0.0
        self.prev_dist = 0.0
        self.angle_number = None
        self.prev_pose = np.array([0.0,0.0])

    def reset(self, angle=np.pi/2):
        self.angle_number = None
        self.prev_penalty = 0.0
        self.prev_dist = 0.0
        self.prev_pose = np.array([0.0,0.0])
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
             value_straight_ahead = observation[14]
 
             if(value_straight_ahead>distance):
                 distance = value_straight_ahead
                 best_angle = angle
 
             if(distance>=9.99):
                 best_angle = angle
                 break
         
 
 
         init_pose = np.array([[0.0, 0.0, best_angle]])  # Starting position
         obs, _, _, _ = self.f110.reset(init_pose)
         observation = self._get_observation(obs)
 
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
        
        # if self.num_steps >= self.max_episode_steps:
        #     done = True

        collided = self.f110.sim.agents[0].in_collision

        if(collided):    
            done = True
            
        return observation, reward, done, info

    def _get_observation(self, obs_dict):
            scan = obs_dict['scans'][0]
            n = len(scan) #Length or number of inputs from the scan, shoudl be 1080
            fov = 4.7
            scan_angles = np.linspace(-fov/2, fov/2, n) #returns an array of 1080 evenly spaced angles between -134.645 and 134.645
            desired_angles = np.linspace(-fov/2, fov/2, len(self.sensor_angles)) #returns an array of 27 evenly spaced angles between -134.645 and 134.645

            sensor_vals = np.interp(desired_angles, scan_angles, scan) #scan is what is returned for each of the scan angles. This returns an array an array of the distance values at all our desired scan angles.
            sensor_vals = np.clip(sensor_vals, 0.0, self.max_range) #All sensor values are clipped at max range, so if they are greater than 10 they are converted to 10

            speed = obs_dict['linear_vels_x'][0] / 4.0
            return np.concatenate([sensor_vals, [np.clip(speed, 0.0, 1.0)]]).astype(np.float32)

    def _calculate_reward(self, obs_dict, action):
        obs = self._get_observation(obs_dict)
        All_Sensors = obs[:-1] / self.max_range
        speed = obs_dict['linear_vels_x'][0]

        # Centering bonus (same as yours)
        left = obs[:7]
        right = obs[20:27][::-1]
        center = obs[7:20]/self.max_range
        center_bonuses = []
        for i in range(len(left)):
            total = left[i] + right[i]
            epsilon = 1e-6
            ratio = left[i] / (total + epsilon)
            if 0.3 <= ratio <= 0.5:
                bonus = (ratio - 0.3) / 0.2
            elif 0.5 < ratio <= 0.7:
                bonus = (0.7 - ratio) / 0.2
            else:
                bonus = 0.0
            center_bonuses.append(bonus)

        center_bonus = mean(center_bonuses)

        # Safety penalty (corrected weighting)
        center_penalties = np.maximum(0, 0.4 - center)
        left_penalties = np.maximum(0,0.08 - (left/self.max_range))
        right_penalties = np.maximum(0,0.08 - (right/self.max_range))
        penalties = np.concatenate([left_penalties, center_penalties, right_penalties])
        weights = np.cos(self.sensor_angles) + 1.0
        weights /= weights.sum()
        safety_penalty = float((weights * penalties).sum())

        # Speed reward (scaled by safety improvement)
        safety_delta = max(0.0, self.prev_penalty - safety_penalty)
        speed_reward = np.clip(speed / 4.0, 0.0, 1.0) * (1 + safety_delta)

        if safety_penalty - self.prev_penalty > 0.001:
            speed_reward = 0

        # Collision penalty
        collision_penalty = 10.0 if (np.min(All_Sensors) < 0.05 or self.f110.sim.agents[0].in_collision) else 0.0

        # Time step reward
        time_step_reward = 0.2 * (speed_reward + center_bonus)

        # Steering penalty
        steering_penalty = 0.1 * abs(action[0])

        

         # 1) compute preferred bearing each step
        scan = obs[:-1]            # 27 distances
        thetas = np.deg2rad(self.sensor_angles)  # convert to radians

        # weighted sum of unit vectors
        x = np.dot(scan, np.cos(thetas))
        y = np.dot(scan, np.sin(thetas))
        preferred = np.arctan2(y, x)  # in [−π,π]

        # 2) project forward speed onto that bearing
        v_forward = np.clip(speed/4.0, 0.0, 1.0)  # your normalized speed
        direction_reward = v_forward * np.cos(preferred)


        # Logging
        logs["speed"].append(speed_reward)
        logs["safety"].append(safety_penalty)
        logs["collision"].append(collision_penalty)
        logs["steering"].append(steering_penalty)
        logs["center-bonus"].append(center_bonus)
        logs["timeStep_Reward"].append(time_step_reward)
        logs["direction"].append(direction_reward)
        
        self.prev_penalty = safety_penalty

        return (
            speed_reward +
            time_step_reward +
            center_bonus -
            safety_penalty -
            collision_penalty -
            steering_penalty + direction_reward
        )

    def render(self, mode='human'):
        return self.f110.render(mode)

def demo_rendering(model, env):
    """
    Runs an episode with continuous rendering until a collision is detected.
    """
    obs = env.reset()

    sensor_vals = obs[:-1] / env.max_range       # normalize to [0,1]
        # average penalty, so it stays O(1) regardless of ray count
    safety_penalty = np.mean(np.maximum(0, 1.0 - sensor_vals))
    print(safety_penalty)

    done = False
    steps = 0
    while not done:
        steps=steps+1
        env.render("human_fast")
        
        action, _ = model.predict(obs)
        action[1] = 2  # Force high throttle
        
        obs, reward, done, info = env.step(action)
        
        # Check for collision using the simulation's flag.
        if env.f110.sim.agents[0].in_collision:
            print("Collision detected! Ending run.")
            break

        if steps%100000 == 0:
             obs = env.map_reset()
    
    env.close()

if __name__ == "__main__":
    MAP_PATH = "../assets/comp_studd/map3"  # Update with your map path

    env = F110LineSensorEnv(
        map_path=MAP_PATH,
        max_steering=0.4,
        max_throttle=2,
        max_episode_steps=1000000
    )
    
    # Load your pre-trained model (ensure the file name/extension is correct)
    model = SAC.load("f110_line_sensor_sac.zip")
    
    demo_rendering(model, env)