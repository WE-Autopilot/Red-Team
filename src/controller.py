import os
import numpy as np
from weap_util.abstract_controller import AbstractModel
import gym
gym.__version__ = "0.21.0"   # lie to SB3 about the gym version
gym.spaces.Box.shape = property(lambda self: self._shape)
from stable_baselines3 import SAC

class Controller(AbstractModel):
    def __init__(self, model_path="f110_line_sensor_sac.zip"):
        super().__init__()
        self.model = None
        self.model_path = model_path

        self.sensor_angles = np.arange(-134.645, 134.645, 9.97370976287)
        self.max_range = 10.0
        self.fov = 4.7
        print(self.model_path)
        self.first_run = True

    def startup(self):
        print("LOL")
        pass

    def init(self):
        pass

    def eval(self, obs, timestamp = 0):
        # pull out the 1080-beam scan
        if(self.first_run):
            self.model = SAC.load(self.model_path)
            self.first_run = False
        scan = obs["scans"][0]
        n = len(scan)
        # interpolate down to your 27 angles
        scan_angles = np.linspace(-self.fov/2, self.fov/2, n)
        desired = np.linspace(-self.fov/2, self.fov/2, len(self.sensor_angles))
        vals = np.interp(desired, scan_angles, scan)
        vals = np.clip(vals, 0.0, self.max_range)
        speed = obs["linear_vels_x"][0] / 4.0
        speed = float(np.clip(speed, 0.0, 1.0))
        obs_vec = np.concatenate([vals, [speed]]).astype(np.float32)

        action, _states = self.model.predict(obs_vec, deterministic=True)
        throttle = action[1]
        steer = action[0]


        throttle = float(np.clip(throttle, 0.4, 10.0))

        return throttle, float(steer)

    def shutdown(self):
        pass
