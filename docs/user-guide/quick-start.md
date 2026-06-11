# 🚀 Quick Start

**Get up and running with SoRoMoX in minutes!** This hands-on guide walks you through your first soft robot simulation with step-by-step examples.

---

## 🎯 Your First Simulation

Let's dive right in with a classic example - simulating a double pendulum to understand the basics using the Pendulum class API:

!!! tip "Tabbed Interface"
    This example is broken down into four tabs. Click through each tab to see the complete workflow: **Setup** → **Initialize** → **Simulate** → **Analyze**. All code blocks build on previous tabs.

=== "🔧 Setup"

    ```python
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from soromox.systems import Pendulum, PendulumParams, SystemState
    from soromox.rendering import ViserRenderer

    # Define parameters for a double pendulum
    num_links = 2
    params = PendulumParams(
        length=jnp.array([0.5, 0.3]),
        center_of_mass_length=jnp.array([0.25, 0.15]),
        mass=jnp.array([1.0, 0.5]),
        moment_inertia=jnp.array([0.1, 0.05]),
        gravity=jnp.array([0.0, -9.81]),
        joint_stiffness=jnp.zeros((num_links, num_links)),
        joint_damping=jnp.zeros((num_links, num_links)),
        joint_rest_configuration=jnp.zeros((num_links,)),
        radius=jnp.array([0.025, 0.015]),
    )
    ```

=== "🏗️ Initialize"

    ```python
    # Create the system using the class API
    robot = Pendulum(params=params)

    # Set initial conditions
    q0 = jnp.array([jnp.pi/4, jnp.pi/6])   # Initial angles [rad]
    qd0 = jnp.array([0.0, 0.0])            # Initial velocities [rad/s]
    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))
    ```

=== "⚡ Simulate"

    ```python
    # Configure simulation
    t_span = (0.0, 5.0)  # 5 seconds
    solver_dt = 0.01    # 10ms timestep

    # Zero control torques
    u = jnp.zeros((num_links,))

    # Integrate using the class helper (Diffrax under the hood)
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=t_span[1],
        solver_dt=solver_dt,
        save_dt=solver_dt,
    )
    ts = trajectory.t
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    u_ts = trajectory.u
    ```

=== "📊 Analyze"

    ```python
    # Extract results from trajectory

    # Compute end-effector (last tip) trajectory over time (using vmap for batched time sample processing)
    # forward_kinematics_tips returns [theta, px, py] for each link tip
    xee_ts = jax.vmap(robot.forward_kinematics_tips)(q_ts)[:,-1,1:]
    ```

    ```python
    # Create visualization with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Joint angles over time
    ax1.plot(t, q_ts[:, 0], label='Joint 1', linewidth=2, color='#1f77b4')
    ax1.plot(t, q_ts[:, 1], label='Joint 2', linewidth=2, color='#ff7f0e')
    ax1.set_ylabel('Joint Angles [rad]')
    ax1.set_title('🔄 Joint Angles vs Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Joint velocities over time
    ax2.plot(t, qd_ts[:, 0], label='Joint 1', linewidth=2, color='#1f77b4')
    ax2.plot(t, qd_ts[:, 1], label='Joint 2', linewidth=2, color='#ff7f0e')
    ax2.set_ylabel('Joint Velocities [rad/s]')
    ax2.set_title('💨 Joint Velocities vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: End-effector trajectory in workspace
    ax3.plot(xee_ts[:, 0], xee_ts[:, 1],  # x and y coordinates of the end-effector
             linewidth=3, color='#2ca02c', alpha=0.8)
    ax3.scatter(xee_ts[0, 0], xee_ts[0, 1], 
                s=100, color='green', marker='o', label='Start', zorder=5)
    ax3.scatter(xee_ts[-1, 0], xee_ts[-1, 1], 
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

    ```python
    # Interactive 3D visualization with ViserRenderer
    # Opens a web-based interface in your browser for interactive exploration
    viser_renderer = ViserRenderer(robot, num_points=50, backbone_style="discrete")
    viser_renderer.render_sequence(
        ts,
        q_ts,
        playback_speed=1.0,
        loop=True,
        autoplay=True,
        plot_configurations=True,
        robot_name="Pendulum",
    )
    # The visualization will open in your browser automatically
    # You can interact with the 3D scene, play/pause animation, and view plots
    ```

!!! success "🎉 Congratulations!"
    You've just simulated your first robot with SoRoMoX! The pendulum exhibits chaotic motion due to its nonlinear dynamics. 
    
    **What you learned:**
    - How to define system parameters
    - How to initialize a robot system
    - How to run a simulation using `rollout_to()`
    - How to extract and visualize results
    
    **Next:** Explore the [Core Concepts](#core-concepts) below to understand the architecture, or jump to [Soft Robot Examples](#soft-robot-example) for more advanced simulations.

---

## 🧠 Core Concepts

### 🏗️ Object-Oriented Architecture

SoRoMoX uses an object-oriented design based on Equinox dataclasses. Systems are instantiated directly as classes:

```python title="System Creation"
from soromox.systems import (
    ArticulatedSoftRobot,
    ArticulatedSoftRobotParams,
    PCS,
    PCSParams,
)
# For spatial articulated soft robots
robot = ArticulatedSoftRobot(params=ArticulatedSoftRobotParams(...))

# For soft continuum robots (PCS, GVS, HSA)
robot = PCS(params=PCSParams(...))
```

**Key Benefits:**

- **Extensibility**: Easy to subclass and modify methods (e.g., custom actuation mappings)
- **JAX Compatibility**: Full support for JIT compilation, automatic differentiation, and vectorization
- **Type Safety**: Clear interfaces with static type checking
- **Performance**: Optimized computation graphs compiled at runtime

!!! note "Migration from Factory Pattern"
    SoRoMoX previously used a factory pattern, but has migrated to object-oriented classes for better extensibility and maintainability. All systems now inherit from `DynamicalSystem` or `SoftRobot` base classes. See the [API Reference](../api/overview.md) for details.

### 📋 Typed Parameters

Each robot system expects a typed Equinox PyTree params object. Numeric fields are
JAX arrays, so same-shape updates can flow through `jit`, `grad`, and `vmap`
without changing the compiled structure.

```python title="Parameter Structure"
params = PCSParams(
    length=link_lengths,
    radius=radii,
    density=densities,
    reference_strain=reference_strain,
    gravity=gravity,
    young_modulus=young_modulus,
    shear_modulus=shear_modulus,
    damping_matrix=damping_matrix,
    base_pose=base_pose,
)
robot = PCS(params=params)
robot = robot.update_params(length=new_lengths)
```

!!! note "Units Matter"
    Always use consistent SI units (meters, kilograms, seconds) for reliable results.

### 🔄 State Representation

Robot states are represented using the `SystemState` dataclass, which encapsulates time, state vector, and optional actuation/control state:

```python title="SystemState"
from soromox.systems import SystemState

# For an n-DOF system:
positions = q      # Shape: (n,)
velocities = qd    # Shape: (n,)
y = jnp.concatenate([q, qd])  # State vector, shape: (2n,)

# Create a SystemState instance
state = SystemState(
    t=0.0,                    # Current time
    y=y,                      # State vector [q, qd]
    u=None,                   # Optional: actuation input
    control_state=None        # Optional: controller state (e.g., integrator terms)
)

# Access state components
time = state.t
state_vector = state.y
q, qd = jnp.split(state.y, 2)  # Split into positions and velocities
```

---

## 🤖 Soft Robot Example

Ready for something more advanced? Let's simulate a soft continuum robot:

=== "🌊 Continuum Robot"

    ```python
    from soromox.systems import PlanarPCS, PlanarPCSParams

    # Create a 3-segment soft robot
    num_segments = 3
    segment_lengths = 0.1 * jnp.ones((num_segments,))
    # Damping matrix (optional but recommended for stability)
    # Structure: diagonal matrix with damping coefficients for each strain component
    # [bending, shear_x, shear_y] per segment, scaled by segment length
    damping_matrix = 1e-3 * jnp.diag(
        jnp.repeat(jnp.array([[1e0, 1e3, 1e3]]), num_segments, axis=0).flatten()
        * segment_lengths[:, None].flatten()
    )
    params = PlanarPCSParams(
        length=segment_lengths,
        radius=0.02 * jnp.ones((num_segments,)),
        density=1070.0 * jnp.ones((num_segments,)),
        reference_strain=jnp.tile(jnp.array([0.0, 1.0, 0.0]), num_segments),
        gravity=jnp.array([0.0, 9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        base_angle=jnp.array(jnp.pi / 2),
    )
    # Note: Damping helps stabilize simulations and represents material dissipation.
    # For static analysis, you can omit this or set to zero.

    # Initialize the PCS robot
    robot = PlanarPCS(params=params)

    # Define configuration (strains)
    # Each segment has 3 strain components: [curvature, shear_x, shear_y]
    q = jnp.array([0.1, 0.0, 0.0,  # Segment 1: [curvature, shear_x, shear_y]
                   0.2, 0.0, 0.0,  # Segment 2
                   0.3, 0.0, 0.0]) # Segment 3

    # Compute forward kinematics along the robot
    s_values = jnp.linspace(0, robot.L.sum(), 100)
    backbone_shape = robot.forward_kinematics_batched(q, s_values)

    # Extract positions for plotting
    # forward_kinematics returns [theta, px, py] for planar systems
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

- **📚 [Explore Examples](examples.md)**: Dive deeper with examples and tutorials covering all robot types and use cases.
- **📖 [API Reference](../api/overview.md)**: Complete documentation of all classes, methods, and functions.
- **🤝 [Contributing](../development/contributing.md)**: Learn how to contribute to SoRoMoX.
- **🔬 Advanced Topics**: Control theory, optimization, and machine learning applications with SoRoMoX.

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
    
    - **Numerical Precision**: Enable double precision with `jax.config.update("jax_enable_x64", True)`
    - **Unstable Simulations**: Use a smaller timestep and a higher order solver to prevent numerical instability and improve accuracy of the simulation.
    - **Slow Batched Simulations**: If batched simulations are slow, try using a smaller batch size and execute on a desktop-size GPU.
    - **Automatic Differentiation**: Please use automatic differentiation with care as it can be very slow and memory-intensive - in particular, when differentiating through the simulation integration.
