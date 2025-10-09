# 🚀 Quick Start

<div class="doc-summary">
  <strong>Get up and running with SoRoMoX in minutes!</strong> This hands-on guide walks you through your first soft robot simulation with step-by-step examples.
</div>

---

## 🎯 Your First Simulation

Let's dive right in with a classic example - simulating a double pendulum to understand the basics using the new Pendulum class API:

=== "🔧 Setup"

    ```python
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from soromox.systems import pendulum

    # Define parameters for a double pendulum
    num_links = 2
    params = {
        "L": jnp.array([0.5, 0.3]),      # Link lengths [m]
        "Lc": jnp.array([0.25, 0.15]),   # Center of mass positions [m]
        "m": jnp.array([1.0, 0.5]),      # Masses [kg]
        "I": jnp.array([0.1, 0.05]),     # Moments of inertia [kg⋅m²]
        "g": jnp.array([0.0, -9.81]),    # Planar gravity vector [m/s²]
    }
    ```

=== "🏗️ Initialize"

    ```python
    # Create the system using the class API
    robot = pendulum.Pendulum(params)

    # Set initial conditions
    q0 = jnp.array([jnp.pi/4, jnp.pi/6])   # Initial angles [rad]
    qd0 = jnp.array([0.0, 0.0])            # Initial velocities [rad/s]
    state0 = jnp.concatenate([q0, qd0])    # State = [q, qd]
    ```

=== "⚡ Simulate"

    ```python
    # Configure simulation
    t_span = (0.0, 5.0)  # 5 seconds
    dt = 0.01           # 10ms timestep

    # Zero control torques
    u = jnp.zeros((num_links,))

    # Integrate using the class helper (Diffrax under the hood)
    ts, qs, qds = robot.resolve_upon_time(
        q0=q0,
        qd0=qd0,
        u=u,
        t0=t_span[0],
        t1=t_span[1],
        dt=dt,
        save_every_n_steps=1,
    )
    ```

=== "📊 Analyze"

    ```python
    # Extract results
    t = ts
    q = qs
    qd = qds

    # Compute end-effector (last tip) trajectory — returns [theta, px, py]
    end_effector_positions = jnp.array([
        robot.forward_kinematics_tips(q_i)[-1, 1:] for q_i in q  # Get [px, py]
    ])

    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Joint angles over time
    ax1.plot(t, q[:, 0], label='Joint 1', linewidth=2, color='#1f77b4')
    ax1.plot(t, q[:, 1], label='Joint 2', linewidth=2, color='#ff7f0e')
    ax1.set_ylabel('Joint Angles [rad]')
    ax1.set_title('🔄 Joint Angles vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Joint velocities over time
    ax2.plot(t, qd[:, 0], label='Joint 1', linewidth=2, color='#1f77b4')
    ax2.plot(t, qd[:, 1], label='Joint 2', linewidth=2, color='#ff7f0e')
    ax2.set_ylabel('Joint Velocities [rad/s]')
    ax2.set_title('💨 Joint Velocities vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # End-effector trajectory
    ax3.plot(end_effector_positions[:, 0], end_effector_positions[:, 1], 
             linewidth=3, color='#2ca02c', alpha=0.8)
    ax3.scatter(end_effector_positions[0, 0], end_effector_positions[0, 1], 
                s=100, color='green', marker='o', label='Start', zorder=5)
    ax3.scatter(end_effector_positions[-1, 0], end_effector_positions[-1, 1], 
                s=100, color='red', marker='s', label='End', zorder=5)
    ax3.set_xlabel('X Position [m]')
    ax3.set_ylabel('Y Position [m]')
    ax3.set_title('🎯 End-Effector Trajectory')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')

    plt.tight_layout()
    plt.show()
    ```

!!! success "🎉 Congratulations!"
    You've just simulated your first robot with SoRoMoX! The pendulum exhibits chaotic motion due to its nonlinear dynamics.

---

## 🧠 Core Concepts

### 🏭 Factory Pattern

SoRoMoX uses an elegant factory pattern to create system functions:

```python title="System Creation"
ode_fn, forward_kinematics, jacobian_fn, _ = system.factory(parameters)
```

**Returns:**

- `ode_fn` → Differential equations for numerical integration
- `forward_kinematics` → Position and orientation computation  
- `jacobian_fn` → Velocity relationships and sensitivities
- `_` → Additional system-specific functions

!!! tip "Why Factory Pattern?"
    This approach allows JAX to optimize the entire computation graph at compile time, resulting in much faster execution.

### 📋 Parameter Dictionary

Each robot system expects a structured parameter dictionary:

```python title="Parameter Structure"
params = {
    "L": link_lengths,        # Physical dimensions
    "m": masses,             # Inertial properties  
    "I": inertias,          # Rotational inertia
    "g": gravity,           # Environmental forces
    # ... system-specific parameters
}
```

!!! note "Units Matter"
    Always use consistent SI units (meters, kilograms, seconds) for reliable results.

### 🔄 State Representation

Robot states follow a consistent `[positions, velocities]` format:

```python title="State Vector"
# For an n-DOF system:
positions = q      # Shape: (n,)
velocities = qd # Shape: (n,)
state = jnp.concatenate([q, qd])  # Shape: (2n,)
```

---

## 🤖 Soft Robot Example

Ready for something more advanced? Let's simulate a soft continuum robot:

=== "🌊 Continuum Robot"

    ```python
    from soromox.systems.planar_pcs import PlanarPCS
    from soromox.parameters import Params

    # Create a 3-segment soft robot
    num_segments = 3
    params = Params.default_planar_pcs(num_segments)

    # Initialize the PCS robot
    robot = PlanarPCS(
        num_segments=num_segments,
        params=params,
    )

    # Define configuration (strains)
    q = jnp.array([0.1, 0.0, 0.0,  # Segment 1: [curvature, shear_x, shear_y]
                   0.2, 0.0, 0.0,  # Segment 2
                   0.3, 0.0, 0.0]) # Segment 3

    # Compute forward kinematics along the robot
    s_values = jnp.linspace(0, robot.L.sum(), 100)
    backbone_shape = jnp.array([
        robot.forward_kinematics(q, s) for s in s_values
    ])

    # Extract positions for plotting
    x_positions = backbone_shape[:, 1]  # X coordinates
    y_positions = backbone_shape[:, 2]  # Y coordinates

    # Visualize the robot shape
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, y_positions, 'b-', linewidth=4, label='Robot Backbone')
    plt.scatter(x_positions[0], y_positions[0], s=150, color='green', 
                marker='o', label='Base', zorder=5)
    plt.scatter(x_positions[-1], y_positions[-1], s=150, color='red', 
                marker='s', label='Tip', zorder=5)
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title('🌊 Soft Continuum Robot Shape')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    ```

=== "📐 Jacobian Analysis"

    ```python
    # Compute Jacobian at the end-effector
    s_tip = robot.L.sum()  # End of the robot
    J = robot.jacobian(q, s_tip)

    print(f"🔍 Jacobian shape: {J.shape}")
    print(f"📏 Jacobian matrix:\n{J}")

    # Analyze manipulability
    manipulability = jnp.sqrt(jnp.linalg.det(J @ J.T))
    print(f"💪 Manipulability index: {manipulability:.4f}")

    # Compute workspace boundary
    theta_range = jnp.linspace(0, 2*jnp.pi, 100)
    workspace_boundary = []

    for theta in theta_range:
        # Unit direction in task space
        direction = jnp.array([jnp.cos(theta), jnp.sin(theta), 0.0])
        
        # Compute maximum reach in this direction
        # (This is a simplified analysis - real workspace computation is more complex)
        max_reach = jnp.linalg.norm(J @ direction)
        workspace_boundary.append(max_reach * direction[:2])

    workspace_boundary = jnp.array(workspace_boundary)

    # Plot workspace
    plt.figure(figsize=(8, 8))
    plt.plot(workspace_boundary[:, 0], workspace_boundary[:, 1], 
             'r--', linewidth=2, label='Approximate Workspace')
    plt.scatter(x_positions[-1], y_positions[-1], s=150, color='blue', 
                marker='o', label='Current Tip Position')
    plt.xlabel('X Position [m]')
    plt.ylabel('Y Position [m]')
    plt.title('🎯 Robot Workspace Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()
    ```

---

## 🎯 Next Steps

<div class="feature-grid">
  <div class="feature-card">
    <h3><span class="icon">📚</span> [Explore Examples](examples.md)</h3>
    <p>Dive deeper with comprehensive examples and tutorials</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">📖</span> [API Reference](../api/systems.md)</h3>
    <p>Complete documentation of all classes and functions</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🤝</span> [Contributing](../development/contributing.md)</h3>
    <p>Learn how to contribute to the SoRoMoX project</p>
  </div>
  
  <div class="feature-card">
    <h3><span class="icon">🔬</span> Advanced Topics</h3>
    <p>Control theory, optimization, and machine learning applications</p>
  </div>
</div>

---

## 💡 Pro Tips

!!! tip "Performance Optimization"
    
    === "🚀 JIT Compilation"
        ```python
        import jax
        
        # Compile functions for faster execution
        fast_kinematics = jax.jit(robot.forward_kinematics)
        fast_jacobian = jax.jit(robot.jacobian)
        ```
    
    === "📊 Vectorization"
        ```python
        # Process multiple configurations at once
        q_batch = jnp.array([[0.1, 0.0, 0.0],
                            [0.2, 0.0, 0.0],
                            [0.3, 0.0, 0.0]])
        
        # Vectorized computation
        batch_kinematics = jax.vmap(
            lambda q: robot.forward_kinematics(q, s_tip)
        )(q_batch)
        ```
    
    === "🎯 Gradient Computation"
        ```python
        # Automatic differentiation for optimization
        def objective(q):
            pos = robot.forward_kinematics(q, s_tip)
            target = jnp.array([1.0, 0.5, 0.0])  # Target pose
            return jnp.sum((pos - target)**2)
        
        # Compute gradients
        grad_fn = jax.grad(objective)
        gradient = grad_fn(q)
        ```

!!! warning "Common Pitfalls"
    
    - **Singular Configurations**: Check manipulability before inverse operations
    - **Physical Limits**: Ensure parameters respect material constraints  
    - **Numerical Precision**: Use appropriate tolerances for convergence
    - **Memory Usage**: Be mindful of array sizes in batch operations
