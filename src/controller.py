import os
import numpy as np
from weap_util.abstract_controller import AbstractModel
from stable_baselines3 import SAC

class Controller(AbstractModel):
    def __init__(self, model_path="f110_line_sensor_sac.zip"):
        super().__init__()
        self.model = SAC.load(model_path)

        self.sensor_angles = np.arange(-134.645, 134.645, 9.97370976287)
        self.max_range = 10.0
        self.fov = 4.7

    def init():
        pass

    def startup(self, waypoints):
        self.waypoints = waypoints

    def eval(self, obs, timestamp = 0):
        # 1) pull out the 1080-beam scan
        scan = obs["scans"][0]
        n = len(scan)
        # 2) interpolate down to your 27 angles
        scan_angles = np.linspace(-self.fov/2, self.fov/2, n)
        desired = np.linspace(-self.fov/2, self.fov/2, len(self.sensor_angles))
        vals = np.interp(desired, scan_angles, scan)
        vals = np.clip(vals, 0.0, self.max_range)
        # 3) normalized speed
        speed = obs["linear_vels_x"][0] / 4.0
        speed = float(np.clip(speed, 0.0, 1.0))
        # 4) build obs vector
        obs_vec = np.concatenate([vals, [speed]]).astype(np.float32)

        # 5) get action from SAC
        steer, throttle = self.model.predict(obs_vec, deterministic=True)
        # enforce throttle ≥ minimum if you like:
        throttle = float(np.clip(throttle, 0.4, 2.0))

        return throttle, float(steer)

    def shutdown(self):
        pass
