# Red-MPC-controller

# Model Predictive Control (MPC) for F1TENTH Simulator

## Overview
This repository contains an **MPC (Model Predictive Control) controller** designed for the **F1TENTH simulator**. Initially, the controller will be used for **pathfinding towards the cursor** in an **open plane environment**. The goal is to leverage MPC for dynamically computing optimal control inputs based on a predictive model of the vehicle's motion. As the project progresses, the controller will be refined, until it will be the basis behind our autonomous driving software by the mid-year competition.

## Features
- **Pathfinding to Cursor**: The controller computes an optimal trajectory to navigate towards the cursor position.
- **MPC-Based Control**: Uses a predictive model to optimize control inputs over a short horizon.
- **F1TENTH Simulator Compatibility**: Designed to integrate with the F1TENTH environment.
- **Dynamic Replanning**: Continuously updates the trajectory to adapt to new cursor positions.

## Installation
Ensure you have the **F1TENTH simulator** and dependencies installed: detailed instructions can be found [here](https://github.com/WE-Autopilot/f1tenth_gym)

## Roadmap
- [ ] Implement basic MPC for cursor tracking in open space
- [ ] Add collision avoidance
- [ ] Optimize control input constraints
- [ ] Implement lane-following behavior

## Contributing
Pull requests are welcome. Please ensure all changes are tested in the simulator before submitting.
