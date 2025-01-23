# This is another trial run with gpt, trying to visualize the MPC controller
# This one doesn't crash, but the optimization function is so bad that our "car" is just ridiculously stupid
# I am going to try to use elements from these 2 codes, as well as some online resources to make something that actually works

import pygame
import numpy as np
import scipy.optimize

# PyGame Setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MPC Controller Visualization")
clock = pygame.time.Clock()

# Define colors
WHITE = (255, 255, 255)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# MPC Parameters
N = 10  # Prediction Horizon
dt = 0.1  # Time step

# **Adjusted Path: More Spaced-Out Sine Wave Waypoints**
target_path = [(x, HEIGHT // 2 + 120 * np.sin(x * 0.02)) for x in range(50, WIDTH, 30)]

# Initial State: [x, y, theta, v]
x, y, theta, v = 100, HEIGHT // 2, 0, 2
history = [(x, y)]  # Store the trajectory

def dynamics(state, control, dt):
    """ Bicycle model dynamics """
    x, y, theta, v = state
    a, delta = control  # Acceleration, Steering angle

    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += v * np.tan(delta) * dt / 20  # Wheelbase L = 20
    v += a * dt
    v = max(0.1, v)  # Prevent stopping completely

    return np.array([x, y, theta, v])

def mpc_controller(state, target):
    """ Optimized MPC Controller with Directional Penalization """
    
    def cost_function(u):
        """ Cost function that penalizes deviation and stopping """
        pred_state = np.copy(state)
        cost = 0
        for i in range(N):
            pred_state = dynamics(pred_state, u[2 * i: 2 * i + 2], dt)
            
            # Distance cost
            cost += (pred_state[0] - target[i][0]) ** 2 + (pred_state[1] - target[i][1]) ** 2
            
            # Penalize abrupt changes in control
            if i > 0:
                cost += 0.05 * (u[2 * i] - u[2 * (i - 1)]) ** 2
                cost += 0.05 * (u[2 * i + 1] - u[2 * (i - 1) + 1]) ** 2
            
            # Encourage forward movement (minimize deceleration)
            cost -= 5 * pred_state[3]  # Reward moving forward

        return cost

    u0 = np.zeros(2 * N)  # Initial guess: zero acceleration & steering
    bounds = [(-1, 1), (-0.5, 0.5)] * N  # Acceleration [-1,1], Steering [-0.5,0.5]

    result = scipy.optimize.minimize(cost_function, u0, bounds=bounds, method='SLSQP')
    return result.x[:2] if result.success else [0.5, 0]  # Apply slight acceleration if stuck

running = True
while running:
    screen.fill(WHITE)
    
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get closest target points
    target = target_path[:N]

    # Apply MPC Control
    control = mpc_controller([x, y, theta, v], target)
    x, y, theta, v = dynamics([x, y, theta, v], control, dt)
    history.append((x, y))

    # Draw Target Path (Green)
    for point in target_path:
        pygame.draw.circle(screen, GREEN, (int(point[0]), int(point[1])), 4)

    # Draw Robot (Red)
    pygame.draw.circle(screen, RED, (int(x), int(y)), 8)

    # Draw Trajectory (Blue Dots for history)
    for hx, hy in history[-100:]:
        pygame.draw.circle(screen, BLUE, (int(hx), int(hy)), 3)

    pygame.display.update()
    clock.tick(30)  # Limit FPS

pygame.quit()