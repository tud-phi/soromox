"""
Operational-Space Impedance Control Example for PCS (Piecewise Constant Strain) Soft Robot.

This example demonstrates how to use the OperationalSpaceImpedanceControlTracker to control a
two-segment PCS soft robot in operational space, where the task is defined as
tracking the end-effector position.

The operational-space impedance controller uses partial feedback linearization
to cancel the original task dynamics and replace them with a desired impedance
behavior characterized by stiffness K_x and damping D_x gains.

References:
    Della Santina, C., Katzschmann, R. K., Bicchi, A., & Rus, D. (2020).
    Model-based dynamic feedback control of a planar soft robot: trajectory
    tracking and interaction with the environment. The International Journal
    of Robotics Research, 39(4-5), 490-513.

    Stölzle, M. (2025). Safe yet Precise Soft Robots: Incorporating Physics
    into Learned Models for Control. Dissertation, Delft University of Technology.
    https://doi.org/10.4233/uuid:24c1f667-8fd6-431a-bb78-11d22f8cb3da
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # Double precision

from soromox.control import OperationalSpaceImpedanceControlTracker, ReferenceTrajectory
from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.rendering import Open3DRenderer
from soromox.systems import PCS, PCSParams, SystemState


def main():
    # =========================================================================
    # Robot Configuration: Two-Segment PCS
    # =========================================================================
    num_segments = 2
    rho = 1070 * jnp.ones((num_segments,))  # Volumetric density [kg/m^3]

    # Define the initial scalar-first quaternion base pose.
    p0 = jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0])

    segment_lengths = 1e-1 * jnp.ones((num_segments,))

    # Damping matrix
    damping_matrix = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * segment_lengths[:, None]
        ).flatten()
    )
    params = PCSParams(
        base_pose=p0,
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    # Initialize robot
    robot = PCS(params=params)
    num_dofs = robot.num_active_strains
    total_length = float(jnp.sum(segment_lengths))

    print(f"Number of DOFs: {num_dofs}")
    print(f"Number of actuators: {robot.num_actuators}")
    print(f"Total robot length: {total_length:.3f} m")

    # =========================================================================
    # Operational Space Definition
    # =========================================================================
    # Define the operational space as the end-effector position (x, y, z)
    # We place the task point at the tip of the robot (s = L_total)
    s_ps = jnp.array([total_length])

    # For PCS robots, the Jacobian has shape (6, num_dofs) for each point:
    # [omega_x, omega_y, omega_z, v_x, v_y, v_z]
    # We select only the linear velocity components (position tracking)
    # task_selector selects [v_x, v_y, v_z] which are indices 3, 4, 5
    task_selector = jnp.array([False, False, False, True, True, True])

    # Create operational space dynamics
    osd = OperationalSpaceDynamics(
        robot=robot,
        s_ps=s_ps,
        task_selector=task_selector,
    )

    print(f"Operational space dimension (velocity): {osd.n_operational_space}")
    print(f"Operational space dimension (pose): {osd.n_pose_operational_space}")
    print(f"Full pose dimension: {osd.n_points * osd.n_pose_dim}")

    # =========================================================================
    # Reference Trajectory Definition (End-Effector Position)
    # =========================================================================
    t0, t1 = 0.0, 10.0
    ts = jnp.linspace(t0, t1, 1000)

    # Get the initial FULL end-effector pose for reference
    # IMPORTANT: The reference trajectory must provide FULL poses (all components),
    # even though only some components (e.g., position) are part of the task.
    # The controller will ignore components not in the task.
    q0 = jnp.zeros((num_dofs,))
    x0_full = osd.operational_space_poses(q0)  # shape: (n_points * n_pose_dim,)
    print(f"Initial end-effector pose (full): {x0_full}")

    # Define a trajectory that moves the end-effector in a simple pattern
    # Starting from the initial position and making small oscillations
    # For 3D PCS with ROTATION_VECTOR: x0_full = [rot_x, rot_y, rot_z, p_x, p_y, p_z]
    # We modify position components (indices 3, 4, 5) while keeping rotation constant
    def x_des_fn(t):
        """Desired FULL end-effector pose as a function of time."""
        # Start from rest position and move in y-z plane
        # The robot initially points along the x-axis (after the base rotation)
        period = 4.0
        amplitude_y = 0.02  # 2 cm amplitude in y
        amplitude_z = 0.01  # 1 cm amplitude in z

        # Smooth trajectory with sinusoidal motion
        # Ramp up the motion smoothly in the first second
        ramp = jnp.minimum(t / 1.0, 1.0)

        # x0_full is [rot_x, rot_y, rot_z, p_x, p_y, p_z] for 3D PCS
        # Modify position components (indices 3, 4, 5)
        x_des = x0_full.at[4].add(ramp * amplitude_y * jnp.sin(2 * jnp.pi * t / period))
        x_des = x_des.at[5].add(ramp * amplitude_z * jnp.sin(4 * jnp.pi * t / period))

        return x_des

    # Create reference trajectory with rotation representation for proper velocity derivation
    # When rotation_representation is provided, ReferenceTrajectory automatically derives
    # velocities correctly (handling the geometric relationship between orientation and
    # angular velocity)
    reference_trajectory = ReferenceTrajectory(
        ts=ts,
        x_des_fn=x_des_fn,
        rotation_representation=osd.rotation_representation,
        n_points=osd.n_points,
        is_planar=osd.is_planar,
    )

    # =========================================================================
    # Impedance Controller Setup
    # =========================================================================
    # Define operational space gains
    # K_x: Stiffness gain (makes the closed-loop stiffer -> faster convergence)
    # D_x: Damping gain (suppresses oscillations)

    # For a second-order system with critical damping: D = 2 * sqrt(K)
    K_x = 1.0 * jnp.ones((osd.n_operational_space,))  # Stiffness [N/m]
    D_x = 0.2 * jnp.ones((osd.n_operational_space,))  # Damping [N·s/m]

    # Create the impedance controller
    controller = OperationalSpaceImpedanceControlTracker(
        operational_space_dynamics=osd,
        reference_trajectory=reference_trajectory,
        K_x=K_x,
        D_x=D_x,
    )

    print(f"Operational space stiffness K_x: {K_x}")
    print(f"Operational space damping D_x: {D_x}")

    # =========================================================================
    # Simulation Setup
    # =========================================================================
    # Initial conditions
    qd0 = jnp.zeros((num_dofs,))  # Zero initial velocity
    y0 = jnp.concatenate([q0, qd0])

    # Create initial system state
    initial_state = SystemState(
        t=jnp.array(t0),
        y=y0,
        u=jnp.zeros((robot.num_actuators,)),
        control_state=None,  # Impedance controller is stateless
    )

    # Simulation parameters
    solver_dt = 1e-4
    save_dt = 0.01

    # =========================================================================
    # Run Closed-Loop Simulation
    # =========================================================================
    print("\nRunning closed-loop simulation with impedance control...")

    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
    )

    # Extract results
    t_traj = trajectory.t
    q_traj, qd_traj = jnp.split(trajectory.y, 2, axis=1)
    assert trajectory.u is not None
    u_traj = trajectory.u

    # Compute FULL operational space coordinates along the trajectory (for visualization)
    x_traj_full = jax.vmap(osd.operational_space_poses)(q_traj)

    # Compute desired FULL trajectory at saved times
    x_des_traj_full = jax.vmap(x_des_fn)(t_traj)

    # Extract position components for plotting (indices 3, 4, 5 for 3D PCS)
    # For 3D robots with ROTATION_VECTOR: full pose = [rot_x, rot_y, rot_z, p_x, p_y, p_z]
    pos_indices = jnp.array([3, 4, 5])  # Position components
    x_traj_pos = x_traj_full[:, pos_indices]
    x_des_traj_pos = x_des_traj_full[:, pos_indices]

    print(f"Simulation completed. {len(t_traj)} time steps saved.")

    # =========================================================================
    # Plotting Results
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Define colors
    colors = plt.cm.tab10.colors

    # Plot 1: End-effector position tracking
    ax1 = axes[0]
    axis_labels = ["x", "y", "z"]
    for i in range(3):
        color = colors[i % len(colors)]
        ax1.plot(
            t_traj,
            x_traj_pos[:, i],
            color=color,
            linewidth=2,
            label=f"$p_{{{axis_labels[i]}}}$",
        )
        ax1.plot(
            t_traj,
            x_des_traj_pos[:, i],
            "--",
            color=color,
            linewidth=1.5,
            alpha=0.7,
            label=f"$p_{{{axis_labels[i]},des}}$",
        )
    ax1.set_ylabel("Position [m]")
    ax1.set_title("End-Effector Position Tracking (Operational Space)")
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tracking error in operational space
    ax2 = axes[1]
    tracking_error = x_des_traj_pos - x_traj_pos
    for i in range(3):
        color = colors[i % len(colors)]
        ax2.plot(
            t_traj,
            tracking_error[:, i] * 1000,  # Convert to mm
            color=color,
            label=f"$e_{{{axis_labels[i]}}}$",
            linewidth=2,
        )
    ax2.set_ylabel("Tracking Error [mm]")
    ax2.set_title("Operational Space Tracking Error")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control input
    ax3 = axes[2]
    num_actuators_to_plot = min(robot.num_actuators, 6)
    for i in range(num_actuators_to_plot):
        color = colors[i % len(colors)]
        ax3.plot(t_traj, u_traj[:, i], color=color, label=f"$u_{i}$", linewidth=1.5)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Control Input")
    ax3.set_title("Actuator Inputs")
    ax3.legend(loc="upper right", ncol=2)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("control_pcs_with_impedance_results.pdf", dpi=200)
    plt.show()

    # Plot configuration space evolution
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Plot configuration
    ax1 = axes2[0]
    num_dofs_to_plot = min(num_dofs, 6)
    for i in range(num_dofs_to_plot):
        color = colors[i % len(colors)]
        ax1.plot(t_traj, q_traj[:, i], color=color, label=f"$q_{i}$", linewidth=2)
    ax1.set_ylabel("Configuration")
    ax1.set_title("Configuration Space Evolution")
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot configuration velocities
    ax2 = axes2[1]
    for i in range(num_dofs_to_plot):
        color = colors[i % len(colors)]
        ax2.plot(
            t_traj, qd_traj[:, i], color=color, label=f"$\\dot{{q}}_{i}$", linewidth=2
        )
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Velocity")
    ax2.set_title("Configuration Space Velocities")
    ax2.legend(loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("control_pcs_with_impedance_config.pdf", dpi=200)
    plt.show()

    # Compute and print summary statistics
    rmse = jnp.sqrt(jnp.mean(tracking_error**2, axis=0))
    print(f"\nRMSE tracking error: {rmse * 1000} mm")
    print(f"Mean RMSE: {jnp.mean(rmse) * 1000:.3f} mm")

    print("\nPlots saved to:")
    print("  - control_pcs_with_impedance_results.pdf")
    print("  - control_pcs_with_impedance_config.pdf")

    if Open3DRenderer is None:
        print("\nOpen3DRenderer unavailable. Install open3d to view the animation.")
    else:
        target_radius = float(jnp.mean(params["r"])) * 0.5
        target_positions = jnp.asarray(x_des_traj_pos)[None, :, :]
        renderer = Open3DRenderer(robot, num_points=50)
        renderer.render_sequence(
            ts=t_traj,
            q_ts=q_traj,
            playback_speed=1.0,
            dynamic_spheres_positions=target_positions,
            dynamic_spheres_radii=jnp.array([target_radius]),
            dynamics_spheres_colors=jnp.array([[0.1, 0.6, 0.9]]),
            window_name="Operational-Space Tracking (Open3D)",
        )


if __name__ == "__main__":
    main()
