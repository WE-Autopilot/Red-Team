import time
import yaml
import gym
import numpy as np
import pyglet
import datetime
import cvxpy as cp
import matplotlib.pyplot as plt
from argparse import Namespace
import os

from pyglet.gl import GL_POINTS
from f110_gym.envs.base_classes import Integrator
from scipy.interpolate import CubicSpline

drawn_lidar_points = []
vector_array = np.empty((64, 2)) #global vector array -- storing (x,y) pairs for beginning coordinates of each vector arrow
global_obs = None

state_history = None #MPC state history

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

def MPC():
    global vector_array
    global state_history

    # Calculates the distance between each pair of vector points on the track, and adds them all to one cumulative arc length
    dists = [0]
    for i in range(1, len(vector_array)):
        dists.append(dists[-1] + np.linalg.norm(vector_array[i] - vector_array[i-1]))

    dists = np.array(dists)

    # Remove duplicates (if any) and sort both the distances and vector_array accordingly
    unique_indices = np.unique(dists, return_index=True)[1]
    dists = dists[unique_indices]
    vector_array = vector_array[unique_indices]

    # Ensure dists are strictly increasing by checking
    if np.any(np.diff(dists) <= 0):  # If any value is non-increasing
        print("Warning: Dists is not strictly increasing. Sorting...")
        sorted_indices = np.argsort(dists)  # Get the sorted indices
        dists = dists[sorted_indices]  # Sort distances
        vector_array = vector_array[sorted_indices]  # Reorder vector_array to match sorted dists



    # Uses the cubic spline function to interpolate between the points on the track
    cs_x = CubicSpline(dists, vector_array[:, 0])
    cs_y = CubicSpline(dists, vector_array[:, 1])

        # ---------------- MPC Controller ----------------

    # Simulation parameters
    timeStep = 0.1  # time step (seconds), how often our simulation will update
    totalSteps = 1000  # total simulation steps, how long the simulation will run
    horizonLength = 10  # MPC horizon (number of steps), how far ahead the controller plans

    # 2D double-integrator model:
    # State: [x, y, vx, vy]; Control: [ax, ay]
    A = np.array([[1, 0, timeStep, 0],
                [0, 1, 0, timeStep],
                [0, 0, 1,  0],
                [0, 0, 0,  1]])
    B = np.array([[0.5 * timeStep**2, 0],
                [0, 0.5 * timeStep**2],
                [timeStep, 0],
                [0, timeStep]])

    # MPC cost weights, penalizes deviations from the reference trajectory (the vectorized path)
    stateCost = np.diag([1, 1, 0.6, 0.6])
    inputCost = np.diag([0.01, 0.01])
    terminalCost = stateCost

    # Define a desired constant speed along the track.
    desiredVelocity = 2.5

    # Precompute the reference trajectory along the drawn track.
    # For each simulation time (plus horizon), compute the reference state.
    # We use s = v_des * t (i.e., the distance along the track increases at constant speed).
    ref_traj = np.zeros((totalSteps + horizonLength + 1, 4))  # 4x4 Array to store the reference trajectory at each time step (+ the horizon)
    for i in range(totalSteps + horizonLength + 1):
        t = i * timeStep  # Current time
        s = desiredVelocity * t  # arc-length traveled along the track

        # If s exceeds the maximum distance of the drawn track, hold the last point.
        if s > dists[-1]:
            s = dists[-1]
        
        # Compute the reference position from the spline.
        x_ref = cs_x(s)
        y_ref = cs_y(s)
        
        # Compute the derivative (velocity components) from the spline derivatives.
        vx_ref = cs_x.derivative()(s)
        vy_ref = cs_y.derivative()(s)
        
        # Optionally normalize the velocity to the desired speed.
        speed = np.hypot(vx_ref, vy_ref)  # Calculates magnitue of the velocity vector
        if speed > 1e-3:
            vx_ref = desiredVelocity * vx_ref / speed
            vy_ref = desiredVelocity * vy_ref / speed
        else:
            vx_ref = 0
            vy_ref = 0
        
        ref_traj[i, :] = np.array([x_ref, y_ref, vx_ref, vy_ref])

    # Set the initial state.
    # Here we start at the first point of the drawn track, with zero velocity.
    x_current = np.array([vector_array[0, 0], vector_array[0, 1], 0, 0])

    # ---------------- MPC Simulation ----------------

    state_history = []

    # Iterates through the simulation steps
    for t in range(totalSteps):
        # Define cvxpy variables for the state and control over the horizon.
        x = cp.Variable((4, horizonLength + 1))  # Array to store the state at each time step
        u = cp.Variable((2, horizonLength))  # Array to store the control input at each time step
        
        cost = 0
        constraints = []
        
        # Initial condition for the horizon. ensuring the first state in the horizon = the current state
        constraints += [x[:, 0] == x_current]
        
        # Build the cost function and dynamics constraints over the horizon.
        for k in range(horizonLength):
            ref_state = ref_traj[t + k]  # The reference state at the current step in the horizon
            cost += cp.quad_form(x[:, k] - ref_state, stateCost) + cp.quad_form(u[:, k], inputCost)  # Adds a penalty to any deviation from the reference state and control input
            constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k]]  # Constraints on the state dynamics
            constraints += [u[:, k] <= np.array([1.0, 1.0]),
                            u[:, k] >= np.array([-1.0, -1.0])]  # Constraints on the control inputs (between -1 & 1)
        
        # Terminal cost for the final state in the horizon.
        ref_state_terminal = ref_traj[t + horizonLength]
        cost += cp.quad_form(x[:, horizonLength] - ref_state_terminal, terminalCost)
        
        # Solve the MPC optimization problem.
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        
        # Extract the first control input from the optimal sequence.
        u_apply = u[:, 0].value
        if u_apply is None:
            u_apply = np.zeros(2)
        
        # Update the current state using the system dynamics.
        x_current = A @ x_current + B @ u_apply
        state_history.append(x_current)

    # The state history will now contain the full trajectory the car would take
    state_history = np.array(state_history)


def render_MPC(env_renderer, state_history):
    # section below clears the arrow that was previously generated
    

    if state_history is not None and len(state_history) > 1:
        for i in range(len(state_history) - 1):
            x1, y1 = state_history[i][0], state_history[i][1]  # Extract x, y positions from the state
            x2, y2 = state_history[i + 1][0], state_history[i + 1][1]  # Next state

            # Add a line to represent the path from one point to the next in the MPC trajectory
            trajectory = env_renderer.batch.add(
                2, pyglet.gl.GL_LINES, None,
                ('v2f', (x1, y1, x2, y2)),  # Line between two consecutive points
                ('c3B', (255, 255, 255) * 2)  # White color for the MPC path
            )



def render_callback(env_renderer):
    global global_obs

    global state_history

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
    render_MPC(env_renderer, state_history)

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

        MPC()

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
