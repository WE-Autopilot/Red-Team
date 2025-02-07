import time
import yaml
import gym
import numpy as np
import pyglet
import datetime
from argparse import Namespace
import os

from pyglet.gl import GL_POINTS
from f110_gym.envs.base_classes import Integrator

drawn_lidar_points = []
global_obs = None

def render_lidar_points(env_renderer, obs):
    global drawn_lidar_points, random_arrow

    if obs is None or 'scans' not in obs:
        return

    lidar_scan = obs['scans'][0]
    pose_x = obs['poses_x'][0]
    pose_y = obs['poses_y'][0]
    pose_theta = obs['poses_theta'][0]

    n_beams = len(lidar_scan)
    angles = np.linspace(-135, 135, n_beams) * np.pi / 180.0

    xs_local = lidar_scan * np.cos(angles)
    ys_local = lidar_scan * np.sin(angles)

    cos_t = np.cos(pose_theta)
    sin_t = np.sin(pose_theta)
    xs_global = pose_x + cos_t * xs_local - sin_t * ys_local
    ys_global = pose_y + sin_t * xs_local + cos_t * ys_local

    scale = 50.0
    xs_scaled = xs_global * scale
    ys_scaled = ys_global * scale

    num_points = len(xs_scaled)
    if len(drawn_lidar_points) < num_points:
        for i in range(num_points):
            b = env_renderer.batch.add(
                1, GL_POINTS, None,
                ('v3f/stream', [xs_scaled[i], ys_scaled[i], 0.0]),
                ('c3B/stream', [255, 0, 0])
            )
            drawn_lidar_points.append(b)
    else:
        for i in range(num_points):
            drawn_lidar_points[i].vertices = [
                xs_scaled[i], ys_scaled[i], 0.0
            ]

def render_callback(env_renderer):
    def render_arrow(env_renderer, arrow_vec): # method to render the vector arrow
        if random_arrow is None:
            return
        
        x, y, theta = arrow_vec
        scale = 50.0
        x_scaled = x * scale
        y_scaled = y * scale
        arrow_length = 64 # arrow length in pixels

        # getting coordinates of the arrowhead
        x_head = x_scaled + arrow_length * np.cos(theta)
        y_head = y_scaled + arrow_length * np.sin(theta)
        arrowhead_size = 10
        left_x = x_head - arrowhead_size * np.cos(theta + np.pi / 6)
        left_y = y_head - arrowhead_size * np.sin(theta - np.pi / 6)
        right_x = x_head - arrowhead_size * np.cos(theta + np.pi / 6)
        right_y = y_head - arrowhead_size * np.sin(theta + np.pi / 6)

        #drawing the arrow line
        env_renderer.batch.add(
            2, pyglet.gl.GL_LINES, None,
            ('v3f', (x_scaled, y_scaled, 0.0, x_head, y_head, 0.0)), # vertex positions
            ('c3B', (0, 255, 0, 0, 255, 0)) # arrow colour (green)
        )

        #drawing the arrowhead
        env_renderer.batch.add(
            3, pyglet.gl.GL_TRIANGLES, None,
            ('v3f', (x_head, y_head, 0.0, left_x, left_y, 0.0, right_x, right_y, 0.0)), #vertex positions
            ('c3B', (0, 255, 0, 0, 255, 0, 0, 255, 0)) #arrow colour (green)
        )

    global global_obs

    e = env_renderer
    # Modified window resizing logic
    if not hasattr(e, 'window_resized'):
        # Get windows as list from WeakSet
        windows = list(pyglet.app.windows)
        if windows:
            window = windows[0]
            window.set_size(256, 256)
            e.window_resized = True
        else:
            print("Warning: No Pyglet window found for resizing")

    # Rest of camera positioning code
    x = e.cars[0].vertices[::2]
    y = e.cars[0].vertices[1::2]
    top, bottom, left, right = max(y), min(y), min(x), max(x)
    e.score_label.x = left
    e.score_label.y = top - 700
    e.left = left - 800
    e.right = right + 800
    e.top = top + 800
    e.bottom = bottom - 800

    render_lidar_points(env_renderer, global_obs)
    render_arrow(env_renderer, random_arrow)

def main():
    dataset = []
    episode_count = 0
    save_interval = 5
    save_path = "lidar_datasets"
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)  # Add this line
    while True:
        global global_obs, random_arrow

        with open('config_example_map.yaml') as file:
            conf_dict = yaml.safe_load(file)
        conf = Namespace(**conf_dict)

        env = gym.make(
            'f110_gym:f110-v0',
            map=conf.map_path,
            map_ext=conf.map_ext,
            num_agents=1,
            timestep=0.01,
            integrator=Integrator.RK4,
            render_options={'window_size': (256, 256)}
        )
        env.add_render_callback(render_callback)

        # Random spawn parameters
        random_x = np.random.uniform(-2.0, 2.0)
        random_y = np.random.uniform(-2.0, 2.0)
        random_theta = np.random.uniform(-np.pi, np.pi)
        print(f"Episode {episode_count} - Spawn: x={random_x:.2f}, y={random_y:.2f}, theta={random_theta:.2f}")
        random_arrow = np.array([random_x, random_y, random_theta])

        init_poses = np.array([[random_x, random_y, random_theta]])
        obs, _, done, _ = env.reset(init_poses)
        global_obs = obs
        env.render(mode='human')

        episode_data = []
        
        for i in range(10):
            if done:
                break
            # Random actions
            random_steer = np.random.uniform(-0.5, 0.5)
            random_speed = np.random.uniform(0.0, 3.0)
            action = np.array([[random_steer, random_speed]])

            obs, reward, done, info = env.step(action)
            global_obs = obs
            env.render(mode='human')
            time.sleep(0.1)

            # Process LiDAR data into tensor
            lidar_scan = obs['scans'][0]
            max_range = 30.0
            angles = np.linspace(-135, 135, len(lidar_scan)) * np.pi / 180.0

            grid_size = 256
            x_min, x_max = -10.0, 10.0
            y_min, y_max = -10.0, 10.0

            tensor = np.zeros((grid_size, grid_size), dtype=np.uint8)

            for beam_idx in range(len(lidar_scan)):
                range_ = lidar_scan[beam_idx]
                if range_ >= max_range:
                    continue
                angle = angles[beam_idx]
                x = range_ * np.cos(angle)
                y = range_ * np.sin(angle)

                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    continue

                i_row = int(((x - x_min) / (x_max - x_min)) * (grid_size - 1))
                i_col = int(((y - y_min) / (y_max - y_min)) * (grid_size - 1))

                i_row = np.clip(i_row, 0, grid_size - 1)
                i_col = np.clip(i_col, 0, grid_size - 1)

                tensor[i_row, i_col] = 1

            episode_data.append(tensor)

        dataset.extend(episode_data)
        episode_count += 1

        # Periodic saving
        if episode_count % save_interval == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_path, f"lidar_dataset_{timestamp}_ep{episode_count}.npz")
            np.savez_compressed(filename, data=np.array(dataset))
            print(f"Saved {len(dataset)} samples to {filename}")
            dataset = []

if __name__ == "__main__":
    main()