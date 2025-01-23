# Data Team: Data Gathering & Preprocessing Tasks  

## Objective  
Collect, annotate, and preprocess **2D maps**, **vehicle dynamics**, and **expert trajectories** to train the autonomous driving model.  

---

### Phase 1: Data Collection  
1. **Define Data Specifications**:  
   - **Inputs**:  
     - 4-channel `256x256` maps (obstacles, track boundaries, start/goal positions, dynamic objects).  
     - Vehicle state: Speed (mph or m/s), steering angle, yaw rate, acceleration.  
   - **Outputs**:  
     - Expert path vectors (10 waypoints with `x, y, heading, curvature, speed`).  

2. **Simulator Data Recording**:  
   - **Expert Demonstrations**:  
     - Record 10,000+ laps in the F1TENTH simulator with varying tracks (open plane, circuits).  
     - Capture edge cases: sudden stops, sharp U-turns, recovery from drift.  
   - **Synchronization**:  
     - Log maps, vehicle state, and trajectories at **10Hz** with <5ms timestamp misalignment.  

3. **Real-World Data (Optional)**:  
   - Deploy lidar-equipped F1TENTH cars to collect occupancy grids + telemetry.  
   - Use motion capture systems (e.g., OptiTrack) for ground-truth trajectory annotation.  

---

### Phase 2: Preprocessing Workflow  
1. **Raw Data Cleaning**:  
   - Filter out corrupted data (e.g., simulator crashes, sensor dropouts).  
   - Align map frames with vehicle state using timestamps.  

2. **Map Standardization**:  
   - Convert raw simulator/CAD maps to 4-channel tensors:  
     - Apply Gaussian blur to obstacle channels for noise reduction.  
     - Encode start/goal positions as 2D heatmaps.  
   - Resize all maps to `256x256` resolution.  

3. **Trajectory Post-Processing**:  
   - Smooth expert paths using **B-spline interpolation** to remove jitter.  
   - Compute curvature for each waypoint:  
     ```python  
     curvature = np.abs(d_heading / ds)  # ds = distance between waypoints  
     ```  

4. **Data Augmentation**:  
   - **Map Augmentation**:  
     - Random rotations (±20°), translations (±10px), and flips.  
     - Overlay synthetic obstacles (e.g., random rectangles/circles).  
   - **Dynamic Parameter Augmentation**:  
     - Scale speed values (±30%) to simulate varying momentum.  
     - Add Gaussian noise to steering angles (σ = 0.5°).  

5. **Dataset Organization**:  
   - Split data into:  
     - **Training (70%)**: General driving scenarios.  
     - **Validation (15%)**: Edge cases (e.g., tight corners).  
     - **Test (15%)**: Unseen tracks/environments.  
   - Store as compressed PyTorch tensors (`.pt` files).  

6. **Normalization**:  
   - Scale map pixel values to `[0, 1]`.  
   - Normalize vehicle speed: `(speed - μ_speed) / σ_speed`.  
   - Clip curvature values to `[-max_turn, max_turn]`.  

---

### Phase 3: Quality Assurance  
1. **Validation Checks**:  
   - **Feasibility**: Ensure 100% of expert trajectories are kinematically feasible.  
   - **Synchronization**: Reject samples with >10ms misalignment between map/state/trajectory.  

2. **Visualization Tools**:  
   - Build a Python tool to overlay maps, vehicle states, and trajectories (use Matplotlib/OpenCV).  
   - Flag outliers (e.g., discontinuous paths, impossible curvature jumps).  

3. **Dataset Versioning**:  
   - Use **DVC (Data Version Control)** to track changes.  
   - Tag datasets with metadata (e.g., `v1.2-simulator-20k-samples`).  

---

### Deliverables  
- **Processed Dataset**:  
  - Train/val/test splits in `.pt` format.  
  - Metadata: Track layouts, augmentation logs, normalization stats.  
- **Preprocessing Scripts**:  
  - Data cleaning, augmentation, and normalization pipelines.  
  - QA and visualization tools.  
- **Documentation**:  
  - Data schema, collection protocols, and dataset statistics.  

---

**Tools & Infrastructure**:  
- **Data Logging**: ROS2 (for simulator), Bag files (for real-world).  
- **Processing**: Python, NumPy, OpenCV, PyTorch.  
- **Storage**: AWS S3 (raw data), local NAS (processed datasets).  

**Key Requirements**:  
- **Volume**: ~2TB storage for 100,000+ training samples.  
- **Latency**: Preprocessing pipeline completes in <2 hours per 10k samples.  
- **Compliance**: Anonymize real-world data (if used) to remove identifiable info.  