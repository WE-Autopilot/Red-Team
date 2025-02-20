# ===============================================================
# Zain Syed, February 2025
# MPC Controller for a Car Following a given path, for WEAP
# I HIGHLY RECOMMEND READING THEORY.md BEFORE READING THIS CODE
# ===============================================================

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline


# ---------------- Path Drawing ----------------

def draw_track():
    """
    This function prompts the user to draw out points
    These points will be used to create a path for the car to follow along
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Draw your racetrack.\nClick to add points; press Enter when finished.")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    
    track_points = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)
    return np.array(track_points)

# We are using points as opposed to a free drawing situation to reduce computational complexity
# It also allows us to more accurately simulate the car's pathfinding algorithm which will be guided by a series of 2D Vectors
track = draw_track()

# Calculates the distance between each pair of points on the track, and adds them all to one cumulative arc length
dists = [0]
for i in range(1, len(track)):
    dists.append(dists[-1] + np.linalg.norm(track[i] - track[i-1]))
dists = np.array(dists)

'''
Cublic Splines are cubic functions used to interpolate between points, maintaining smoothness
between the points. This is useful for creating the paths for this project.
'''
# Uses the cubic spline function to interpolate between the points on the track
cs_x = CubicSpline(dists, track[:, 0])
cs_y = CubicSpline(dists, track[:, 1])
# We also can define a circle radius value if we wish, however our paths are straight lines, so it isn't necessary




# ---------------- MPC Controller ----------------

# Simulation parameters
timeStep = 0.1 # time step (seconds), how often our simulation will update
totalSteps = 30 # total simulation steps, how long the simulation will run
horizonLength = 10 # MPC horizon (number of steps), how far ahead the controller plans


# 2D double-integrator model:
# State: [x, y, vx, vy]; Control: [ax, ay]
A = np.array([[1, 0, timeStep, 0],
              [0, 1, 0, timeStep],
              [0, 0, 1,  0],
              [0, 0, 0,  1]])
B = np.array([[0.5*timeStep**2, 0],
              [0, 0.5*timeStep**2],
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
ref_traj = np.zeros((totalSteps + horizonLength + 1, 4)) # 4x4 Array to store the reference trajectory at each time step (+ the horizon)
for i in range(totalSteps + horizonLength + 1):
    t = i * timeStep # Current time
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
    speed = np.hypot(vx_ref, vy_ref) # Calculates magnitue of the velocity vector
    if speed > 1e-3:
        vx_ref = desiredVelocity * vx_ref / speed
        vy_ref = desiredVelocity * vy_ref / speed
    else:
        vx_ref = 0
        vy_ref = 0
    
    ref_traj[i, :] = np.array([x_ref, y_ref, vx_ref, vy_ref])


state_history = []

# Set the initial state.
# Here we start at the first point of the drawn track, with zero velocity.
x_current = np.array([track[0, 0], track[0, 1], 0, 0])
state_history.append(x_current)

# Iterates through the simulation steps
for t in range(totalSteps):
    # Define cvxpy variables for the state and control over the horizon.
    x = cp.Variable((4, horizonLength+1)) # Array to store the state at each time step
    u = cp.Variable((2, horizonLength)) # Array to store the control input at each time step
    
    cost = 0
    constraints = []
    
    # Initial condition for the horizon. ensuring the first state in the horizon = the current state
    constraints += [x[:, 0] == x_current]
    
    # Build the cost function and dynamics constraints over the horizon.
    for k in range(horizonLength):
        ref_state = ref_traj[t + k] # The reference state at the current step in the horizon
        cost += cp.quad_form(x[:, k] - ref_state, stateCost) + cp.quad_form(u[:, k], inputCost) # Adds a penalty to any deviation from the reference state and control input
        constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]] # Constraints on the state dynamics
        constraints += [u[:, k] <= np.array([1.0, 1.0]),
                        u[:, k] >= np.array([-1.0, -1.0])] # Constraints on the control inputs (between -1 & 1)
    
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

state_history = np.array(state_history)

fig, ax = plt.subplots(figsize=(8, 8))

# Adjust plot limits based on the drawn track.
ax.set_xlim(np.min(track[:, 0]) - 5, np.max(track[:, 0]) + 5)
ax.set_ylim(np.min(track[:, 1]) - 5, np.max(track[:, 1]) + 5)
ax.set_aspect('equal')
ax.set_title("MPC Following Your Drawn Racetrack")

# Plot the drawn track (reference) for context and the state history along with it
ax.plot(track[:, 0], track[:, 1], 'r--', label="Drawn Track")
ax.plot(state_history[:, 0], state_history[:, 1], 'g-', label="Car Trajectory")
ax.legend()

# Car drawing parameters (the car is represented as a rectangle).
car_length = 1.0
car_width = 0.5
car_patch = Rectangle((0, 0), car_length, car_width, fc='blue', ec='black')
ax.add_patch(car_patch)

sim_steps = state_history.shape[0]

def init():
    car_patch.set_visible(True)
    return car_patch,

def animate(i):
    # Retrieve the current state.
    x = state_history[i, 0]
    y = state_history[i, 1]
    vx = state_history[i, 2]
    vy = state_history[i, 3]
    
    # Estimate orientation from the velocity vector (if moving).
    theta = np.arctan2(vy, vx) if np.hypot(vx, vy) > 1e-3 else 0
    
    # Create an affine transform: first rotate about (0,0), then translate to (x,y).
    trans = (plt.matplotlib.transforms.Affine2D()
             .rotate_around(0, 0, theta)
             .translate(x, y)
             + ax.transData)
    car_patch.set_transform(trans)
    
    # Adjust the rectangle so that its center is at (0,0) before transformation.
    car_patch.set_xy((-car_length/2, -car_width/2))
    car_patch.set_width(car_length)
    car_patch.set_height(car_width)
    
    return car_patch,

ani = animation.FuncAnimation(fig, animate, frames=sim_steps,
                              init_func=init, blit=True, interval=100, repeat=True)
plt.show()
