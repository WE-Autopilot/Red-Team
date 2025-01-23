# Machine Learning Outline

## Model Creation

## Overview
This repository implements a neural network model for **autonomous vehicle path planning** in the F1TENTH simulator. The model processes **top-down 2D maps** (256x256 grids) and real-time dynamic parameters (e.g., speed, turning radius) to generate a **sequence of path vectors** for an MPC controller. The goal is to enable smooth, dynamically feasible trajectories in open environments, with future extensions for obstacle avoidance and on-track racing.  

## Model Architecture  
1. **Input Layer**:  
   - `(4, 256, 256)` tensor (4-channel map: obstacles, track boundaries, etc.).  
2. **CNN Backbone**:  
   - Convolutional layers: `8 → 16 → 32 → 64` channels, downsampling to `16x16` spatial resolution.  
   - Final `16x16` convolution collapses features to `512x1x1`.  
3. **Dynamic Fusion**:  
   - Concatenates CNN output (`512-dim`) with dynamic parameters (`1-dim` speed) → `513-dim` vector.  
4. **Linear Projection**:  
   - Sequential FC layers: `513 → 512 → 512 → 256 → 128`.  
   - Output: `128-dim` path vector (10 waypoints + metadata).  

---

## Loss Function  
- **Positional MSE**: Penalizes deviations in predicted vs. ground-truth `(x, y, heading)`.  
- **Curvature Regularization**:  
  ```python  
  loss += λ * torch.sum(torch.clamp(|pred_curvature| - max_turn, min=0) ** 2)  

## Installation
Ensure you have the **F1TENTH simulator** and dependencies installed: detailed instructions can be found [here](https://github.com/WE-Autopilot/f1tenth_gym)

## Roadmap
- [ ] Implement Convolutional Neural Network (CNN) Backbone
- [ ] Integrate Dynamic Parameters into Linear Layers
- [ ] Define Path Vector Output Structure
- [ ] Implement Hybrid Loss Function (Maybe MSE)
- [ ] Set Up Training Pipeline
- [ ] Tune Layer Dimensions and Forward Pass, unit tests and the likes
- [ ] Set Up Training Pipeline
