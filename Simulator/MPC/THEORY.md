# Theory & Explanation Behind MPC Code

*If you are too lazy to read it all, you can skip to the [**Summary**](#summary) to go over the basics of each concept*

## Basic MPC Parameters:
For the sake of this project, I am currently using the following parameters:<br>
- A Time Step of 0.1 seconds | How often our simulation will update, this can be altered to be longer, or even work with distance as opposed to time (this will represent dt in the upcoming steps)
- 100 Total Steps, this is how many 0.1 second bursts our program will fun for, this value is not at all rigid and will vary based on track length
- A Horizon Length of 10, this essentially means we are looking and calculation 10 steps ahead, meaning we will make decisions one second prior to being in a particular state

---
<br>

## 2D Double-Integrator Model:

### Primitive Math:
In a continuous time model, we can consider a point of mass, in our case, a car, and it's position/velocity in a 2x2 plane (this of course applies to F1TENTH but also the real world because a car will not move in the 3rd dimension for the sake of this project). We can represent these relevant values in a 4D vector, 2 position values, and 2 velocity values: 

$$M(t) =\begin{bmatrix}x(t)\\\ y(t)\\\ v_x(t)\\\ v_y(t) \end{bmatrix}$$

We can take a look at control inputs as well, in our case a 2D vector to store acceleration values:

$$C(t) =\begin{bmatrix}a_x(t) \\\a_y(t)\end{bmatrix}$$

We know that Acceleration is simply the derivative of Velocity, and that Velocity is simply the derivative of Position (with respect to time):

$$\frac{d}{dt}x(t) = v_x(t)$$  $$\frac{d}{dt}y(t) = v_y(t)$$
$$\frac{d}{dt}v_x(t) = a_x(t)$$  $$\frac{d}{dt}v_y(t) = a_y(t)$$ 

Now we know that M(t) consists of the position vector and the velocity vector, meaning we can say:
$$\frac{d}{dt}M(t) = F\cdot M(t) + G\cdot C(t)$$
Where:

$$F = \begin{bmatrix} 0&0&1&0\\\0&0&0&1\\\0&0&0&0\\\0&0&0&0 \end{bmatrix}$$  $$G=\begin{bmatrix} 0&0\\\0&0\\\1&0\\\0&1 \end{bmatrix}$$

This works because the matrix F clearly extracts the $v_x$ & $v_y$ values and expresses it as the derivatives of $x$ & $y$ which is exactly what it is. Similarly, matrix G extracts the $a_x$ & $a_y$ values and expresses them as the derivatives of $v_x$ & $v_y$.

Now of course, we are looking at a moving car, not a static one, so for the sake of our example, we have to find a way to implement change in position and velocity. We can define time steps to do so.

<br>

### Definition of A & B matricies:

**Note: keep in mind dt represents one time-step**

We know that we can find the position by simply integrating the velocity at a given time. Of course the purpose of an MPC controller is to look ahead, and we can do that!

$$x(t+dt) = x(t) + v_x(t)dt + \frac{1}{2}a_x(t)dt^2$$
$$y(t+dt) = y(t) + v_y(t)dt + \frac{1}{2}a_y(t)dt^2$$

Of course we update position at every step, so we should do the same with velocity:

$$v_x(t+dt) = v_x(t) + a_x(t)dt$$
$$v_y(t+dt) = v_y(t) + a_y(t)dt$$

**Now, lets say we want to define the state of the car at time $k$, we can define our own equation:**

$$M[k+1] = AM[k] + BC[k]$$

Where the A Matrix is in charge of capturing change in state when there is no input from the control (in other words when $C[k] = 0$):

We can simply define the following equations from the predictive position and velocity equations defined above, the only difference this time being acceleration=0 (due to no control input)

$$x(k+1) = x(k) + v_x(k)dt$$
$$y(k+1) = y(k) + v_y(k)dt$$
$$v_x(k+1) = v_x(k)$$
$$v_y(k+1) = v_y(k)$$

We can represent this as:

$$ A = \begin{bmatrix} 1&0&dt&0\\\0&1&0&dt\\\0&0&1&0\\\ 0&0&0&1 \end{bmatrix}$$

This works super systematically:<br>
The first row reprents $x(t) + v_x(t)dt$, stating that the new position, is just the old position plus the distance travelled in $dt$<br>
The second row represents the same thing, but with the position and velocity in regards to the y-direction
The third & forth row represent the fact that velocity does not change over time in the absence of acceleration

Now for the B Matrix, it describes how the control inputs (accelerations) can affect the state during a time step. It will handle the acceleration side of things for us. Recall how we removed the acceleration components in the A Matr$$ix, in this iteration we will be removing the position and velocity components to balance out the equation.

We know acceleration at k+1 is the following for x and y:

$$\frac{1}{2}a_x[k]dt^2$$ $$\frac{1}{2}a_y[k]dt^2$$

We also know that this is how acceleration relates to velocity:

$$v_xdt$$ $$v_ydt$$

With this knowledge, we can construct our B matrix to complete the equation:

$$B = \begin{bmatrix} 0.5dt^2&0\\\0&0.5dt^2\\\dt&0\\\0&dt \end{bmatrix}$$

Again, we can justify this, the first and second row outline the change in position with respect to acceleration, while the third and forth rows outline the change in velocity with respect to acceleration. It may seem complex, but it is simply just a representation of the basic Physics equations we defined above, just remember that acceleration is the derivative of velocity, which is the derivative of position.

Now we finally have our final equation for the Discrete-Time Model, this will allow us to calculate ahead of time, and figure out potential future position, velocity, and acceleration values based on information gathered from out horizon.

$$M[k+1] = A = \begin{bmatrix} 1&0&dt&0\\\0&1&0&dt\\\0&0&1&0\\\ 0&0&0&1 \end{bmatrix}M[k] + \begin{bmatrix} 0.5dt^2&0\\\0&0.5dt^2\\\dt&0\\\0&dt \end{bmatrix}C[k]$$

<br>

***To maintain code clarity, we will be using the timeStep variable to represent dt, because thats all dt is, a single step of time***

---
<br>


## MPC Cost Function Weights:
The three things we have defined are State Cost, Input/Control Cost, & Terminal Cost.<br><br>
State Cost refers to mistakes made in our position and speed. If you see the code, we have defined stateError as a diagonal matrix [1, 1, 0.25, 0.25]. The first 2 values penalie mistakes in position, the last 2 penalize mistakes in velocity (notice how we care more about where we re as opposed to how fast we are), these penalty weights can be changed differently, and they will effect how are car reacts to mistakes moving forward.<br><br>
Input Cost refers to mistakes made in acceleration, which are not weighed nearly as much as position or velocity, as of now, our diagonal matrix consists of [0.01, 0.01], emphasizing that they aren't penalized as much
<br><br>
Terminal Cost is a cost applied to the end of the horizon. It is a little more complicated and not relevant as of now, so we will set Terminal Cost to be equal to State Cost for the time being, as it does not involve the input cost whatsoever!

## Reference Trajectory:

The **reference trajectory** is a series of desired states that the controller aims to track. In our implementation, the reference trajectory is generated from the user-drawn racetrack using cubic spline interpolation. By parameterizing the track using its cumulative arc-length, we obtain a smooth function for both the position and its derivatives (which approximate the velocity).

In english, our reference trajectory is just the track (Or in our case the 64-long vectorized path)

We define the arc-length as  
$$
s = v_{des} \cdot t,
$$  
where:
- \( v_{des} \) is the desired constant speed,
- \( t \) is the elapsed time.

The interpolated reference state at time \( t \) is then given by:
$$
M_{ref}(t) = \begin{bmatrix} x_{ref}(t) \\ y_{ref}(t) \\ v_{x,ref}(t) \\ v_{y,ref}(t) \end{bmatrix},
$$
with:
- $\left( x_{ref}(t) = cs_x(s) \right)$,
- $\left( y_{ref}(t) = cs_y(s) \right)$,
- $\left( v_{x,ref}(t) \right)$ and $\left( v_{y,ref}(t) \right)$ obtained from the derivatives $\left( cs_x'(s) \right)$ and $\left( cs_y'(s) \right)$, respectively, then normalized so that the speed equals $\left( v_{des} \right)$.

This approach ensures that the reference trajectory continuously provides the desired position and velocity information along the track.

---
<br>

## System Dynamics and Constraints

The optimization is subject to the discrete-time dynamics of our 2D double-integrator model:
$$M[k+1] = A\, M[k] + B\, C[k]$$
with the matrices defined as:

$$A = \begin{bmatrix} 1 & 0 & dt & 0 \\ 0 & 1 & 0 & dt \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad B = \begin{bmatrix} 0.5\,dt^2 & 0 \\ 0 & 0.5\,dt^2 \\ dt & 0 \\ 0 & dt \end{bmatrix}$$
Here, $ dt $ is the time step between consecutive states.

Additional constraints are imposed on the control inputs to ensure they remain within allowable bounds:
$$C_{min} \leq C[k] \leq C_{max}$$

These constraints, along with the system dynamics, define the feasible set of control sequences over the prediction horizon.

---
<br>

## Receding Horizon Strategy

A key feature of MPC is its **receding horizon** (or moving horizon) strategy:

- **Optimization at Each Time Step:**  
  At time step $k$, the controller solves the optimization problem over the next $N$ steps, generating an optimal sequence of control inputs:
  $$\{ C[k], C[k+1], \dots, C[k+N-1] \}$$

- **Implementation of the First Control Action:**  
  Only the first control input $C[k]$ is applied to the system. The state is updated, and the optimization is resolved at the next time step $k+1$ using the new state information.

- **Adaptive Behavior:**  
  This approach allows the controller to adapt to disturbances and modeling errors in real time because it continuously recalculates the optimal control strategy based on the latest state and reference trajectory.

---
<br>

## Summary

In summary:

- **Basic MPC Parameters:**
  - **Time Step (dt):** The simulation updates every 0.1 seconds. This parameter defines how frequently the controller updates and can be adjusted as needed.
  - **Total Steps:** The simulation runs for a fixed number of steps (e.g., 100 or 1000), which may be adapted based on the track length or desired simulation duration.
  - **Horizon Length:** The controller plans ahead for 10 time steps, meaning it optimizes its control actions over a 1-second future window at each iteration.

- **2D Double-Integrator Model:**
  - **State Representation:**  
    The car's state is defined as: $M(t) = \begin{bmatrix} x(t) \\ y(t) \\ v_x(t) \\ v_y(t) \end{bmatrix}$ 
    
    where \(x(t)\) and \(y(t)\) denote the position, and \(v_x(t)\) and \(v_y(t)\) denote the velocity components.

  - **Control Input:**  
    The control (acceleration) is represented by:
    $C(t) = \begin{bmatrix} a_x(t) \\ a_y(t) \end{bmatrix}$
  - **Continuous-Time Dynamics:**  
    The evolution of the state is modeled by:
    $\frac{d}{dt}M(t) = F\, M(t) + G\, C(t)$
    with
    $$F = \begin{bmatrix} 0 & 0 & 1 & 0 \\[4mm] 0 & 0 & 0 & 1 \\[4mm] 0 & 0 & 0 & 0 \\[4mm] 0 & 0 & 0 & 0 \end{bmatrix}, \quad
    G = \begin{bmatrix} 0 & 0 \\[4mm] 0 & 0 \\[4mm] 1 & 0 \\[4mm] 0 & 1 \end{bmatrix}$$
  - **Discrete-Time Model:**  
    By discretizing with a time step $dt$, we obtain:

    $$M[k+1] = A\, M[k] + B\, C[k],$$
    where
    $$A = \begin{bmatrix} 1 & 0 & dt & 0 \\[4mm] 0 & 1 & 0 & dt \\[4mm] 0 & 0 & 1 & 0 \\[4mm] 0 & 0 & 0 & 1 \end{bmatrix}, \quad
    B = \begin{bmatrix} 0.5\,dt^2 & 0 \\[4mm] 0 & 0.5\,dt^2 \\[4mm] dt & 0 \\[4mm] 0 & dt \end{bmatrix}.$$

- **MPC Cost Function and Receding Horizon Strategy:**
  - The controller minimizes a cost function that penalizes the deviation of the state from a reference trajectory (weighted by \( Q \) and \( Q_f \)) as well as the magnitude of the control inputs (weighted by \( R \)).
  - Only the first control input from the optimized sequence is applied at each time step. Then the optimization problem is re-solved at the next time step with the updated state, allowing the controller to adapt to any disturbances or changes in the system.


- The **Reference Trajectory** is generated from a user-defined track via cubic spline interpolation. It provides the desired position and velocity along the path, parameterized by arc-length $s = v_{des} \cdot t$

- The **MPC Problem** minimizes a cost function that balances state tracking (via $Q$ and $Q_f$) and control effort (via $R$) over a finite horizon, while respecting the system dynamics:
  $M[k+1] = A\, M[k] + B\, C[k]$

- The **Receding Horizon** strategy ensures that only immediate control actions are implemented, allowing the controller to continuously update its decisions based on new information.

This framework enables robust, real-time trajectory tracking for the car, making the MPC controller well-suited for applications like autonomous vehicle path following.