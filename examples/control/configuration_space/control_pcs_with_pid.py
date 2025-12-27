"""
PID Control Example for PCS (Piecewise Constant Strain) Soft Robot.

This example demonstrates how to use the PIDController to control a PCS soft robot
in configuration space using continuous-time closed-loop simulation.

The controller tracks a sequence of step setpoints in the bending strain (kappa_y)
of the first segment, which better illustrates the step response characteristics
and motivates the potential benefits of model-based control terms.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # Double precision

from soromox.control import PIDControl, PIDControllerState, ReferenceTrajectory
from soromox.control.configuration_space import PIDController
from soromox.rendering import Open3DRenderer
from soromox.systems import PCS, SystemState


def main():
    # =========================================================================
    # Robot Configuration
    # =========================================================================
    num_segments = 1
    rho = 1070 * jnp.ones((num_segments,))  # Volumetric density [kg/m^3]

    params = {
        "p0": jnp.array([jnp.pi / 2, jnp.pi / 2, 0.0, 0.0, 0.0, 0.0]),
        "L": 1e-1 * jnp.ones((num_segments,)),  # Segment length [m]
        "r": 2e-2 * jnp.ones((num_segments,)),  # Segment radius [m]
        "rho": rho,
        "g": jnp.array([0.0, 0.0, 9.81]),  # Gravity vector [m/s^2]
        "E": 2e3 * jnp.ones((num_segments,)),  # Elastic modulus [Pa]
        "G": 1e3 * jnp.ones((num_segments,)),  # Shear modulus [Pa]
    }
    # Damping matrix
    params["D"] = 1e-3 * jnp.diag(
        (
            jnp.repeat(
                jnp.array([[1e0, 1e0, 1e0, 1e3, 1e3, 1e3]]), num_segments, axis=0
            )
            * params["L"][:, None]
        ).flatten()
    )

    # Initialize robot
    robot = PCS(num_segments=num_segments, params=params)
    num_dofs = robot.num_active_strains

    print(f"Number of DOFs: {num_dofs}")
    print(f"Number of actuators: {robot.num_actuators}")

    # =========================================================================
    # Reference Trajectory Definition (Step Setpoints)
    # =========================================================================
    # Define a sequence of step setpoints for bending (kappa_y of first segment)
    # This demonstrates the step response of the controller
    t0, t1 = 0.0, 10.0
    ts = jnp.linspace(t0, t1, 1000)

    # Define step setpoint times and values for kappa_y (index 1)
    step_times = jnp.array([0.0, 2.0, 4.0, 6.0, 8.0])
    step_values = jnp.array([0.0, 1.0, -0.5, 0.8, 0.0])

    def x_des_fn(t):
        """Desired configuration as a function of time (step setpoints)."""
        q_des = jnp.zeros((num_dofs,))
        # Find which step we're in
        step_idx = jnp.searchsorted(step_times, t, side="right") - 1
        step_idx = jnp.clip(step_idx, 0, len(step_values) - 1)
        # Set kappa_y (index 1) to the current setpoint
        q_des = q_des.at[1].set(step_values[step_idx])
        return q_des

    reference_trajectory = ReferenceTrajectory(ts=ts, x_des_fn=x_des_fn)

    # =========================================================================
    # PID Controller Setup
    # =========================================================================
    # Define PID gains
    # Note: The gains must be tuned to the specific robot dynamics.
    # For PCS, strains are [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z]
    # per segment, where kappa are rotational (curvatures) and sigma are linear.
    # These have different units and magnitudes, so they need different gains.

    # Gains for rotational strains (curvatures) - units: [rad/m]
    Kp_rot = 1e-2
    Ki_rot = 1e-3
    Kd_rot = 1e-4

    # Gains for linear strains (elongation/shear) - dimensionless
    Kp_lin = 1e-1
    Ki_lin = 1e-3
    Kd_lin = 1e-2

    # Build gain vectors: [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z] per segment
    Kp_segment = jnp.array([Kp_rot, Kp_rot, Kp_rot, Kp_lin, Kp_lin, Kp_lin])
    Ki_segment = jnp.array([Ki_rot, Ki_rot, Ki_rot, Ki_lin, Ki_lin, Ki_lin])
    Kd_segment = jnp.array([Kd_rot, Kd_rot, Kd_rot, Kd_lin, Kd_lin, Kd_lin])

    # Tile for all segments
    Kp = jnp.tile(Kp_segment, num_segments)
    Ki = jnp.tile(Ki_segment, num_segments)
    Kd = jnp.tile(Kd_segment, num_segments)

    # Create PID control with tanh saturation for anti-windup
    pid_control = PIDControl(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        saturation_fn="tanh",  # Use tanh saturation
        gamma=1.0,  # Saturation scaling
    )

    # Create the PID controller
    controller = PIDController(
        robot=robot,
        reference_trajectory=reference_trajectory,
        pid_control=pid_control,
    )

    # =========================================================================
    # Simulation Setup
    # =========================================================================
    # Initial conditions
    q0 = jnp.zeros((num_dofs,))  # Start at rest configuration
    qd0 = jnp.zeros((num_dofs,))  # Zero initial velocity
    y0 = jnp.concatenate([q0, qd0])

    # Initial control state (using PIDControllerState dataclass)
    control_state_0 = PIDControllerState.zero(int(num_dofs))

    # Create initial system state
    initial_state = SystemState(
        t=jnp.array(t0),
        y=y0,
        u=jnp.zeros((robot.num_actuators,)),
        control_state=control_state_0,  # Track controller state
    )

    # Simulation parameters
    solver_dt = 1e-4
    save_dt = 0.01

    # =========================================================================
    # Run Closed-Loop Simulation
    # =========================================================================
    print("Running closed-loop simulation with PID control...")

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
    assert trajectory.control_state is not None
    control_state_traj: PIDControllerState = trajectory.control_state
    integral_error_traj = control_state_traj.integral_error

    # Compute reference trajectory at saved times
    q_des_traj = jax.vmap(x_des_fn)(t_traj)

    print(f"Simulation completed. {len(t_traj)} time steps saved.")

    # =========================================================================
    # Plotting Results
    # =========================================================================
    # Define colors for each DOF (consistent between actual and desired)
    colors = plt.cm.tab10.colors  # Use tab10 colormap for distinct colors

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Configuration tracking
    ax1 = axes[0]
    num_dofs_to_plot = min(int(num_dofs), 3)
    for i in range(num_dofs_to_plot):
        color = colors[i % len(colors)]
        ax1.plot(
            t_traj,
            q_traj[:, i],
            color=color,
            linewidth=2,
            label=f"$q_{i}$",
        )
        ax1.plot(
            t_traj,
            q_des_traj[:, i],
            "--",
            color=color,
            linewidth=1.5,
            alpha=0.7,
            label=f"$q_{{{i},des}}$",
        )
    ax1.set_ylabel("Configuration [rad/m]")
    ax1.set_title("Configuration Tracking (Step Response)")
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Tracking error
    ax2 = axes[1]
    tracking_error = q_des_traj - q_traj
    for i in range(num_dofs_to_plot):
        color = colors[i % len(colors)]
        ax2.plot(
            t_traj, tracking_error[:, i], color=color, label=f"$e_{i}$", linewidth=2
        )
    ax2.set_ylabel("Tracking Error [rad/m]")
    ax2.set_title("Tracking Error")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Control input
    ax3 = axes[2]
    for i in range(min(robot.num_actuators, 3)):
        color = colors[i % len(colors)]
        ax3.plot(t_traj, u_traj[:, i], color=color, label=f"$u_{i}$", linewidth=2)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Control Input")
    ax3.set_title("Control Input")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("control_pcs_with_pid_results.pdf", dpi=200)
    plt.show()

    # Plot integral error evolution
    fig2, ax = plt.subplots(figsize=(10, 4))
    for i in range(num_dofs_to_plot):
        color = colors[i % len(colors)]
        ax.plot(
            t_traj,
            integral_error_traj[:, i],
            color=color,
            label=f"$\\int e_{i}$",
            linewidth=2,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Integral Error")
    ax.set_title("Integral Error Evolution")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("control_pcs_with_pid_integral_error.pdf", dpi=200)
    plt.show()

    print(
        "Plots saved to control_pcs_with_pid_results.png and control_pcs_with_pid_integral_error.png"
    )

    if Open3DRenderer is None:
        print("Open3DRenderer unavailable. Install open3d to view the animation.")
    else:
        renderer = Open3DRenderer(robot, num_points=50)
        renderer.render_sequence(
            ts=t_traj,
            q_ts=q_traj,
            playback_speed=1.0,
            window_name="Configuration-Space PID Control (Open3D)",
        )


if __name__ == "__main__":
    main()
