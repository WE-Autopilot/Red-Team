##############################
##     GYM ENVIRONMENT      ##
##############################
from typing import Tuple

import gym
import numpy as np
from weap_util.lidar import lidar_to_bitmap
import random

from path_clamp import compute_vectors_with_angle_clamp, clamp_vector_angle_diff
from movement_controller import render_arrow, get_steering_and_speed, detect_collison, centerline_reward

current_planned_path = None
theta = 1.57

def render_callback(env_renderer):
    """
    Callback for the simulator renderer to display the planned path.
    """
    global current_planned_path
    if current_planned_path is not None:
        render_arrow(env_renderer, current_planned_path)

class SACF110Env(gym.Env):
    """
    Custom F1Tenth environment with SAC integration and simple low-level control.
    Handles high-level path planning and uses a basic steering controller for movement.
    """
    
    DIST_THRESHOLD = 0.2  # Waypoint reaching threshold

    def __init__(self, f110_env: gym.Env):
        super().__init__()
        self.f110_env = f110_env
        self.observation_space = gym.spaces.Box(low=0, high=255, 
                                              shape=(128,128), dtype=np.uint8)
        
        self.action_space = gym.spaces.Box(low=-1, high=1, 
                                           shape=(16,), dtype=np.float32)
        
        # Path planning parameters
        self.car_length = 0.3
        self.vector_length = 0.5
        self.path_points = None
        self.sub_index = 16
        self.pending_action = None

        # State tracking
        self.last_obs = None
        self.prev_position = None
        self.current_planned_path = None
        self.map_scale = 2.5  # pixels per meter
        self.map_origin = (32, 32)  
    
    
    def reset(self,theta = 1.57):
        """Reset environment with default pose and clear path history"""
        self.theta = theta
        default_pose = np.array([[0.0, 0.0, theta]])  # x, y, theta
        obs, _, _, _ = self.f110_env.reset(default_pose)
        
        # Process initial observation
        lidar_scan = obs['scans'][0]
        bitmap = lidar_to_bitmap(
            lidar_scan, 
            output_image_dims=(128, 128),
            bg_color='black', 
            draw_mode="FILL", 
            winding_dir='CW', 
            starting_angle=np.pi/2
        )
        # Store the computed lidar bitmap in the observation
        obs['lidar_bitmap'] = bitmap
        self.last_obs = obs
        self.prev_position = np.array([obs['poses_x'][0], obs['poses_y'][0]])

        # Reset path tracking
        self.path_points = None
        self.sub_index = 16
        self.pending_action = None
        self.current_planned_path = None

        return bitmap

        
    def step(self, raw_action: np.ndarray):
        """
        Execute one timestep using the SAC action and a simple steering controller.
        This version uses a simplified reward signal that:
        - Heavily rewards forward progress.
        - Treats very low forward velocity as a crash.
        - Applies a small time penalty.
        """
        # Get current car state
        car_state = {
            'x': self.last_obs['poses_x'][0],
            'y': self.last_obs['poses_y'][0],
            'theta': self.last_obs['poses_theta'][0]
        }

        # Initialize path if needed
        if self.path_points is None:
            self._handle_path_update(raw_action, car_state)

        # Use the first waypoint in the sliding window to compute the low-level action
        target_x, target_y = self.path_points[0]
        action_out = get_steering_and_speed(target_x, target_y,
                                            car_state['x'], car_state['y'],
                                            car_state['theta'])
        
        # Here, if the computed speed is below 0.1, we assume the car has stopped.
        if action_out[0, 1] < 0.1:
            stop_penalty = -300.0
            info = {"stop": True, "reason": "low_velocity"}
            obs = self.reset(self.theta)
            return obs, stop_penalty, True, info

        # Execute the low-level action in the underlying environment.
        # We ignore the base reward and use our own.
        obs, _, done, info = self.f110_env.step(action_out)

        # Process the LiDAR scan to generate a bitmap observation.
        lidar_scan = obs['scans'][0]
        bitmap = lidar_to_bitmap(lidar_scan, output_image_dims=(128, 128),
                                bg_color='black', draw_mode="FILL",
                                winding_dir='CW', starting_angle=np.pi/2)
        obs['lidar_bitmap'] = bitmap

        # Compute our simplified reward:
        reward_components = self._calculate_rewards(obs, done)
        # If a severe crash penalty is applied, mark the episode done.
        if reward_components.get('collision', 0.0) < -150:
            done = True
        total_reward = sum(reward_components.values())

        # Update the path if the car has reached the current target waypoint.
        self._update_path_index(obs, raw_action)

        # Update state tracking.
        self.last_obs = obs
        self.prev_position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        self._update_path_visualization()

        return bitmap, total_reward, done, info


    def _world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
       px = int(self.map_origin[0] + x * self.map_scale)
       py = int(self.map_origin[1] + y * self.map_scale)
       return np.clip(px, 0, 255), np.clip(py, 0, 255)

    def _handle_path_update(self, raw_action: np.ndarray, car_state: dict):
        """
        Initial full path generation using the SAC action.
        """
        if self.pending_action is not None:
            action_to_use = self.pending_action
            self.pending_action = None
        else:
            action_to_use = raw_action

        increments = compute_vectors_with_angle_clamp(action_to_use)
        self.path_points = self._calculate_global_path(increments, car_state)
        # Ensure the sliding window length (e.g. 8 waypoints)
        self.path_points = self.path_points[:8]

    def _calculate_global_path(self, increments: np.ndarray, car_state: dict) -> list:
        """Convert local vectors to global path coordinates"""
        path = []
        x, y = car_state['x'], car_state['y']
        theta = car_state['theta']
        
        # Start from front of car
        front_x = x + self.car_length * np.cos(theta)
        front_y = y + self.car_length * np.sin(theta)
        path.append((front_x, front_y))

        # Convert local increments to global coordinates
        for dx, dy in increments:
            dx_scaled = dx * self.vector_length
            dy_scaled = dy * self.vector_length
            
            # Rotate to global frame
            global_dx = dx_scaled * np.cos(theta) - dy_scaled * np.sin(theta)
            global_dy = dx_scaled * np.sin(theta) + dy_scaled * np.cos(theta)
            
            new_x = path[-1][0] + global_dx
            new_y = path[-1][1] + global_dy
            path.append((new_x, new_y))

        return path[1:]  # Skip initial point

    def _calculate_rewards(self, obs: dict, done: bool) -> dict:
        rewards = {}

        # Get the car's center in the LiDAR bitmap.
        car_x = self.last_obs['lidar_bitmap'].shape[1] // 2
        car_y = self.last_obs['lidar_bitmap'].shape[0] // 2
        current_bitmap = obs['lidar_bitmap']

        # Heavily reward forward progress.
        new_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        progress = np.linalg.norm(new_pos - self.prev_position)
        rewards['progress'] = progress * 50.0  # Increased multiplier for strong incentive

        # Reward for speed
        dt = 0.015  # matches the timestep
        speed = progress / dt  # forward speed (m/s)
        rewards['speed'] = speed * 0.005  # Scale factor to reward high speeds


        return rewards



    def _update_path_index(self, obs: dict, raw_action: np.ndarray):
        """
        Check if the car has reached the first waypoint in the window.
        If so, remove it and generate a new waypoint using raw_action.
        """
        current_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        target_pos = np.array(self.path_points[0])
        dist = np.linalg.norm(current_pos - target_pos)
        
        if dist < self.DIST_THRESHOLD:
            self._shift_path(raw_action)

    def _shift_path(self, raw_action: np.ndarray):
        """
        Remove the first waypoint and generate a new one appended at the end.
        The new vector is computed based on the last segment’s direction and the new raw_action.
        """
        # Remove the reached waypoint
        self.path_points.pop(0)
        
        # Get the last waypoint and determine its direction
        last_point = np.array(self.path_points[-1])
        if len(self.path_points) >= 2:
            second_last = np.array(self.path_points[-2])
            last_angle = np.arctan2(last_point[1] - second_last[1],
                                    last_point[0] - second_last[0])
        else:
            # Fallback: use the current car heading if not enough points
            last_angle = self.last_obs['poses_theta'][0]
        
        # Use the first two elements of raw_action as a hint for the new direction
        raw_vector = raw_action[:2]
        norm = np.linalg.norm(raw_vector) + 1e-8
        raw_vector = raw_vector / norm
        desired_angle = np.arctan2(raw_vector[1], raw_vector[0])
        
        # Clamp the change in angle relative to the previous segment
        clamped_angle = clamp_vector_angle_diff(last_angle, desired_angle, 10.0)
        new_vector = np.array([np.cos(clamped_angle), np.sin(clamped_angle)])
        
        # Compute the new waypoint from the last point
        new_waypoint = (last_point[0] + new_vector[0] * self.vector_length,
                        last_point[1] + new_vector[1] * self.vector_length)
        self.path_points.append(new_waypoint)

    def _update_path_visualization(self):
        """Update visualization of planned path"""
        if self.path_points is not None:
            flattened = []
            for px, py in self.path_points:
                flattened.extend([px, py])
            self.current_planned_path = np.array(flattened, dtype=np.float32)
            global current_planned_path
            current_planned_path = self.current_planned_path
