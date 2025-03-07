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
vector_array = np.empty((64, 2)) #global vector array -- storing (x,y) pairs for beginning coordinates of each vector arrow
global_obs = None

#globals for arrow vector generation
arrow_graphics = [] # array to store arrow graphics so they can be removed later
car_length = 0.3
scale = 50

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

def make_init_arrow(arrow_vec): #function to generate coordinates needed to draw first vector
    if arrow_vec is None:
        return
    
    x, y, theta = arrow_vec

    #computing front of the car using its orientation
    front_x = x + car_length * np.cos(theta)
    front_y = y + car_length * np.sin(theta)

    # Add the initial arrow coordinates to the vector array
    vector_array[0] = (front_x, front_y) #putting x and y coordinates of arrow beginning in the vector array

    x_scaled = front_x * scale 
    y_scaled = front_y * scale 
    arrow_length = (6 * car_length)/ 64 * scale # arrow length in pixels

    # getting coordinates of the arrowhead
    x_head = x_scaled + arrow_length * np.cos(theta)
    y_head = y_scaled + arrow_length * np.sin(theta)

    return x_scaled, y_scaled, x_head, y_head, theta

def make_vector_path(env_renderer, init_arrow): #function to generate the rest of the vector arrows in the path
    #initializing the starting x and y to the head of the initial arrow, and storing theta in next_trajec
    next_x_start = init_arrow[2]
    next_y_start = init_arrow[3]
    next_trajec = init_arrow[4]
    arrow_length = (6 * car_length)/64 * scale

    for c in range (63): #generating the remaining 63 vector arrows in the path
        vector_array[c+1] = (next_x_start, next_y_start) #putting x and y coordinates of arrow beginning in the vector array

        next_x_head = next_x_start + arrow_length * np.cos(next_trajec)
        next_y_head = next_y_start + arrow_length * np.sin(next_trajec)
        
        next_line = env_renderer.batch.add( #rendering the current vector arrow
                2, pyglet.gl.GL_LINES, None,
                ('v3f', (next_x_start, next_y_start, 0.0, next_x_head, next_y_head, 0.0)), # vertex positions
                ('c3B', (0, 255, 0, 0, 255, 0)) # arrow colour (green)
            )
        arrow_graphics.append(next_line) #adding the arrow to the arrow_graphics array so it can be cleared later
        
        # updating starting x and y to head of the previous arrow before next arrow is generated
        next_x_start = next_x_head
        next_y_start = next_y_head

def render_arrow(env_renderer, arrow_vec): # method to render the vector arrow
        global arrow_graphics

        # section below clears the arrow that was previously generated
        for arrow in arrow_graphics: 
            arrow.delete()
        arrow_graphics = []

        this_arrow = make_init_arrow(arrow_vec) #generating coords for the initial vector arrow
        #drawing the arrow line
        arrow_line = env_renderer.batch.add(
            2, pyglet.gl.GL_LINES, None,
            ('v3f', (this_arrow[0], this_arrow[1], 0.0, this_arrow[2], this_arrow[3], 0.0)), # vertex positions
            ('c3B', (0, 255, 0, 0, 255, 0)) # arrow colour (green)
        )
        arrow_graphics.append(arrow_line) #adding the arrow line to the arrow_graphics array so it can be cleared later
        
        make_vector_path(env_renderer, this_arrow) #calling make_vector_path on the initial vector arrow

def render_callback(env_renderer):
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
        random_arrow = None #setting current random_arrow to none so new one can be generated
        arrow_graphics = [] # clearing any stored arrow graphics
        vector_array = np.empty((64, 2)) #clearing vector path array so coordinates from next random generation can overwrite

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
