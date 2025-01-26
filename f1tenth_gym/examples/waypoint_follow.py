import time
import yaml
import gym
import numpy as np
from argparse import Namespace

from pyglet.gl import GL_POINTS

# For convenience, we store the latest observation in a global so we can draw it.
# In a more advanced setup, you'd have a class structure to avoid globals.
LATEST_OBS = None

def render_lidar_points(env_renderer):
    """
    Render LiDAR points from LATEST_OBS onto the simulator screen.
    """
    global LATEST_OBS
    
    # If no observations yet, do nothing.
    if LATEST_OBS is None:
        return

    # Extract car pose and LiDAR scan from observation (assuming single agent).
    # obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0] are the car pose
    # obs['scans'][0] is an array of LiDAR distances (range readings).
    car_x = LATEST_OBS['poses_x'][0]
    car_y = LATEST_OBS['poses_y'][0]
    car_theta = LATEST_OBS['poses_theta'][0]
    lidar_ranges = LATEST_OBS['scans'][0]

    # Typical F1TENTH LiDAR spans ~270 degrees; 
    # adjust angles if your simulator is configured differently.
    num_beams = len(lidar_ranges)
    angle_min = -135.0 * np.pi / 180.0
    angle_max =  135.0 * np.pi / 180.0
    lidar_angles = np.linspace(angle_min, angle_max, num_beams)

    # We'll create a fresh batch of points each frame. 
    # (Alternatively, store them in a list and update them to be more efficient.)
    lidar_batch = env_renderer.batch.add(
        num_beams,              # how many points
        GL_POINTS,              # draw mode
        None,                   # group
        ('v3f/stream', np.zeros(num_beams * 3)),   # vertex positions (x, y, z=0)
        ('c3B/stream', np.zeros(num_beams * 3, dtype=np.uint8)) # colors
    )

    # Each beam -> transform from polar (range, angle) to global (x, y)
    # Then scale for rendering (f110_gym often scales real meters by 50).
    coords = []
    colors = []
    for i in range(num_beams):
        r = lidar_ranges[i]
        angle = lidar_angles[i]

        # Ignore invalid or too-distant readings (if any).
        if not np.isfinite(r) or r > 20.0:
            # Just push an off-screen coordinate or keep it near the car at zero range
            gx = car_x
            gy = car_y
        else:
            # Convert local (r,angle) → global (x,y)
            gx = car_x + r * np.cos(car_theta + angle)
            gy = car_y + r * np.sin(car_theta + angle)

        # Scale up for rendering (sim coords often multiplied ~50).
        sx = 50.0 * gx
        sy = 50.0 * gy

        # Collect final coords. z=0
        coords.extend([sx, sy, 0.0])
        # Example color: bright green
        colors.extend([0, 255, 0])

    # Update the batch data in one shot
    lidar_batch.vertices = coords
    lidar_batch.colors = colors


def render_callback(env_renderer):
    global LATEST_OBS

    # Update camera to follow the car (assuming single agent).
    # This is similar logic to the original script, but stripped down.
    if LATEST_OBS is not None:
        x = LATEST_OBS['poses_x'][0]
        y = LATEST_OBS['poses_y'][0]

        # Just a naive bounding box around the car for the camera
        e = env_renderer
        e.left   = 50.0 * (x - 10.0)
        e.right  = 50.0 * (x + 10.0)
        e.bottom = 50.0 * (y - 10.0)
        e.top    = 50.0 * (y + 10.0)

    # Draw the LiDAR points
    render_lidar_points(env_renderer)


def main():
    # Load a config to pick a map, spawn location, etc. 
    # Minimal example: we only care that 'map_path', 'map_ext', etc. exist.
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Create environment (single agent). 
    # integrator can be RK4 or EULER if the environment supports it.
    env = gym.make(
        'f110_gym:f110-v0',
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01
    )

    # Add our custom render callback to draw LiDAR
    env.add_render_callback(render_callback)

    # Reset environment with initial pose
    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]])
    )
    env.render()

    global LATEST_OBS
    LATEST_OBS = obs  # store the initial observation so we can render

    laptime = 0.0
    start = time.time()

    # Run until done. We'll keep the car stationary (steer=0, speed=0).
    while not done:
        # Car won't move, but LiDAR will still be drawn each step.
        action = np.array([[0.0, 0.0]])

        obs, step_reward, done, info = env.step(action)
        laptime += step_reward

        # Update global obs, then render
        LATEST_OBS = obs
        env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


if __name__ == '__main__':
    main()
