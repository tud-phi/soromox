"""
Setpoint Regulation Comparison for Actuation-Space Controllers.

This example compares the performance of different actuation-space regulators
for controlling a Tendon-Actuated PCS soft robot to statically-feasible setpoints.
The controllers compared are:
    - PIDController (model-free baseline)
    - PotentialCompensationRegulator (feedforward: G(q_des) + tau_el(q_des))
    - PotentialCancellationRegulator (feedforward: G(q) + tau_el(q))
    - GravityCancellationRegulator (feedforward: G(q) + tau_el(q_des))

All controllers use the same PID gains to isolate the effect of the model-based
feedforward term.

The robot is a one-segment Tendon-Actuated PCS with 3 tendons that are:
    - Parallel to the backbone (zero slope)
    - Evenly spaced 120 degrees apart around the circumference

To ensure the setpoint is statically feasible under actuation, we first simulate
the open-loop system with constant tendon tensions until steady-state, then use
the final configuration as the setpoint for the closed-loop regulation.
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # Double precision

from soromox.control import PIDControl, PIDControllerState, ReferenceTrajectory
from soromox.control.actuation_space import (
    GravityCancellationRegulator,
    PIDController,
    PotentialCancellationRegulator,
    PotentialCompensationRegulator,
)
from soromox.coordinate_transformations.actuation_space import ActuationSpaceDynamics
from soromox.rendering import Open3DRenderer
from soromox.systems import (
    LinearTendonRoutingParams,
    PCSParams,
    SystemState,
    TendonActuatedPCS,
    TendonActuatedPCSParams,
)


def create_robot() -> tuple[TendonActuatedPCS, int]:
    """
    Create a one-segment Tendon-Actuated PCS robot with 3 tendons.

    The tendons are:
        - Parallel to the backbone (zero slope: my=0, mz=0)
        - Evenly spaced 120 degrees apart around the circumference
        - All attached to the end of the single segment

    Returns:
        robot: The TendonActuatedPCS instance.
        num_dofs: Number of degrees of freedom.
    """
    num_segments = 1
    radius = 2e-2  # Segment radius [m]
    rho = 1070 * jnp.ones((num_segments,))  # Volumetric density [kg/m^3]

    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    material_damping_coefficient = 362.0
    body_params = PCSParams(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        length=segment_lengths,
        radius=radius * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        material_damping_coefficient=material_damping_coefficient,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    # Tendon routing: 3 tendons at 120 degrees apart, parallel to backbone
    # Tendon positions at angles: 0°, 120°, 240° from the +z axis
    # d_y = r * sin(theta), d_z = r * cos(theta)
    tendon_offset_radius = 0.8 * radius  # Slightly inside the robot radius
    angles_deg = jnp.array([0.0, 120.0, 240.0])
    angles_rad = jnp.deg2rad(angles_deg)

    active_tendon_routing = LinearTendonRoutingParams(
        y_intercept=tendon_offset_radius * jnp.sin(angles_rad),
        z_intercept=tendon_offset_radius * jnp.cos(angles_rad),
        y_slope=jnp.zeros(3),
        z_slope=jnp.zeros(3),
        attachment_segment_index=jnp.array([0, 0, 0]),
    )

    robot = TendonActuatedPCS(
        params=TendonActuatedPCSParams(
            body=body_params,
            active_tendon_routing=active_tendon_routing,
        ),
    )

    num_dofs = robot.num_active_strains

    return robot, int(num_dofs)


def find_steady_state_configuration(
    robot: TendonActuatedPCS,
    u_constant: jnp.ndarray,
    sim_duration: float = 10.0,
    solver_dt: float = 1e-4,
) -> jnp.ndarray:
    """
    Find the steady-state configuration under constant actuation.

    Simulates the robot with constant tendon tensions until the velocities
    become negligible, indicating steady-state has been reached.

    Args:
        robot: The TendonActuatedPCS robot.
        u_constant: Constant tendon tensions to apply.
        sim_duration: Duration of the simulation.
        solver_dt: Time step for the solver.

    Returns:
        q_ss: Steady-state configuration.
    """
    num_dofs = robot.num_active_strains

    # Start from rest configuration
    q0 = jnp.zeros((num_dofs,))
    qd0 = jnp.zeros((num_dofs,))
    y0 = jnp.concatenate([q0, qd0])

    initial_state = SystemState(t=jnp.array(0.0), y=y0)

    # Simulate with constant actuation
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u_constant,
        t1=sim_duration,
        solver_dt=solver_dt,
        save_dt=0.1,  # Save less frequently for efficiency
    )

    # Extract final state
    q_final, qd_final = jnp.split(trajectory.y[-1], 2)

    # Check that velocities are small (steady-state reached)
    velocity_norm = jnp.linalg.norm(qd_final)
    print(f"  Final velocity norm: {velocity_norm:.2e}")
    if velocity_norm > 1e-3:
        print(
            "  Warning: Steady-state may not have been reached. "
            "Consider increasing sim_duration."
        )

    return q_final


def create_pid_control(num_actuators: int) -> PIDControl:
    """
    Create PID control gains for actuation-space control.

    The gains are sized for the actuated coordinates (tendon lengths).

    Args:
        num_actuators: Number of actuators (tendons).

    Returns:
        PIDControl instance with tuned gains.
    """
    # Gains for tendon length tracking
    # Units: tendon lengths are in meters
    Kp = 5e1 * jnp.ones((num_actuators,))
    Ki = 5e0 * jnp.ones((num_actuators,))
    Kd = 1e0 * jnp.ones((num_actuators,))

    return PIDControl(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        saturation_fn="tanh",
        gamma=10.0,
    )


def create_setpoint_trajectory(
    q_des: jnp.ndarray,
    t0: float,
    t1: float,
) -> ReferenceTrajectory:
    """
    Create a constant setpoint trajectory for regulation.

    Args:
        q_des: Desired configuration (constant setpoint).
        t0: Start time.
        t1: End time.

    Returns:
        ReferenceTrajectory with constant setpoint.
    """
    ts = jnp.linspace(t0, t1, 100)

    def x_des_fn(t):
        """Desired configuration as a function of time (constant setpoint)."""
        return q_des

    reference_trajectory = ReferenceTrajectory(ts=ts, x_des_fn=x_des_fn)

    return reference_trajectory


def run_simulation(
    robot: TendonActuatedPCS,
    controller,
    num_actuators: int,
    t0: float,
    t1: float,
    solver_dt: float = 1e-4,
    save_dt: float = 0.01,
) -> dict:
    """Run closed-loop simulation with the given controller."""
    num_dofs = robot.num_active_strains

    # Initial conditions: start from rest configuration
    q0 = jnp.zeros((num_dofs,))
    qd0 = jnp.zeros((num_dofs,))
    y0 = jnp.concatenate([q0, qd0])

    # Initial control state (sized for actuated coordinates)
    control_state_0 = PIDControllerState.zero(num_actuators)

    # Create initial system state
    initial_state = SystemState(
        t=jnp.array(t0),
        y=y0,
        u=jnp.zeros((robot.num_actuators,)),
        control_state=control_state_0,
    )

    # Run simulation
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

    return {
        "t": t_traj,
        "q": q_traj,
        "qd": qd_traj,
        "u": trajectory.u,
    }


def compute_metrics(results: dict, q_des: jnp.ndarray, strain_indices: list) -> dict:
    """Compute performance metrics for the simulation results."""
    t = results["t"]
    q = results["q"]

    # Compute tracking error
    tracking_error = q_des[None, :] - q

    # Root mean square error for each strain
    rmse = jnp.sqrt(jnp.mean(tracking_error[:, strain_indices] ** 2, axis=0))

    # Maximum absolute error
    max_error = jnp.max(jnp.abs(tracking_error[:, strain_indices]), axis=0)

    # Steady-state error (last 10% of simulation)
    n_ss = max(1, len(t) // 10)
    ss_error = jnp.mean(jnp.abs(tracking_error[-n_ss:, strain_indices]), axis=0)

    return {
        "rmse": rmse,
        "max_error": max_error,
        "ss_error": ss_error,
        "tracking_error": tracking_error,
    }


def main():
    # =========================================================================
    # Setup
    # =========================================================================
    robot, num_dofs = create_robot()
    num_actuators = robot.num_actuators

    print(f"Number of DOFs: {num_dofs}")
    print(f"Number of actuators: {num_actuators}")

    # Create actuation space dynamics transformer
    asd = ActuationSpaceDynamics(robot)

    # =========================================================================
    # Find Statically-Feasible Setpoint
    # =========================================================================
    print("\nFinding steady-state configuration under constant actuation...")

    # Apply constant tendon tensions (negative = pulling)
    # Use asymmetric tensions to create an interesting configuration
    u_constant = jnp.array([-0.05, -0.02, -0.01])
    print(f"  Applied tendon tensions: {u_constant}")

    q_des = find_steady_state_configuration(robot, u_constant, sim_duration=10.0)
    print(f"  Steady-state configuration (q_des): {q_des}")

    # Compute the corresponding tendon lengths at steady-state
    y_des = asd.actuated_unactuated_coordinates(q_des)
    print(f"  Actuation-space coordinates (y_des): {y_des}")

    # Time parameters for closed-loop control
    t0, t1 = 0.0, 5.0

    # Create shared PID control (same gains for all controllers)
    pid_control = create_pid_control(num_actuators)

    # Create reference trajectory (in configuration space)
    reference_trajectory = create_setpoint_trajectory(q_des, t0, t1)

    # =========================================================================
    # Define Controllers
    # =========================================================================
    controllers = {
        "PID (model-free)": PIDController,
        "Potential Compensation": PotentialCompensationRegulator,
        "Potential Cancellation": PotentialCancellationRegulator,
        "Gravity Cancellation": GravityCancellationRegulator,
    }

    # Store results
    all_results = {}

    # =========================================================================
    # Run Simulations
    # =========================================================================
    for name, ControllerClass in controllers.items():
        print(f"\nRunning simulation with {name}...")

        # Suppress warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            controller = ControllerClass(
                actuation_space_dynamics=asd,
                reference_trajectory=reference_trajectory,
                pid_control=pid_control,
                reference_in_configuration_space=True,
            )

        results = run_simulation(robot, controller, num_actuators, t0, t1)
        all_results[name] = results
        print(f"  Completed {len(results['t'])} time steps.")

    # =========================================================================
    # Compute Metrics
    # =========================================================================
    # Analyze all strains (for a 1-segment robot: 6 strains)
    strain_indices = list(range(num_dofs))
    strain_names = [
        r"$\kappa_x$",
        r"$\kappa_y$",
        r"$\kappa_z$",
        r"$\sigma_x$",
        r"$\sigma_y$",
        r"$\sigma_z$",
    ][:num_dofs]

    print("\n" + "=" * 70)
    print("Performance Metrics (RMSE)")
    print("=" * 70)
    header = f"{'Controller':<25}"
    for name in strain_names:
        header += f" {name:>10}"
    print(header)
    print("-" * 70)

    for name, _results in all_results.items():
        metrics = compute_metrics(results, q_des, strain_indices)
        all_results[name]["metrics"] = metrics
        rmse = metrics["rmse"]
        row = f"{name:<25}"
        for val in rmse:
            row += f" {val:>10.4f}"
        print(row)

    print("\n" + "=" * 70)
    print("Performance Metrics (Steady-State Error)")
    print("=" * 70)
    header = f"{'Controller':<25}"
    for name in strain_names:
        header += f" {name:>10}"
    print(header)
    print("-" * 70)

    for name in all_results:
        ss_error = all_results[name]["metrics"]["ss_error"]
        row = f"{name:<25}"
        for val in ss_error:
            row += f" {val:>10.4f}"
        print(row)

    # =========================================================================
    # Plotting
    # =========================================================================
    # Define colors for each controller
    colors = {
        "PID (model-free)": "#E24A33",  # Red-orange
        "Potential Compensation": "#348ABD",  # Blue
        "Potential Cancellation": "#988ED5",  # Purple
        "Gravity Cancellation": "#8EBA42",  # Green
    }

    # Select key strains to plot (curvatures and axial strain)
    plot_strain_indices = [1, 2, 3]  # kappa_y, kappa_z, sigma_x
    plot_strain_names = [
        r"$\kappa_y$ [rad/m]",
        r"$\kappa_z$ [rad/m]",
        r"$\sigma_x$ [-]",
    ]

    # Create figure for strain tracking comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for idx, (strain_idx, strain_name) in enumerate(
        zip(plot_strain_indices, plot_strain_names)
    ):
        ax = axes[idx]

        # Plot desired setpoint
        all_results["PID (model-free)"]["t"]
        ax.axhline(
            y=q_des[strain_idx],
            color="k",
            linestyle="--",
            linewidth=2,
            label="Desired",
            alpha=0.7,
        )

        # Plot actual trajectories for each controller
        for name, results in all_results.items():
            ax.plot(
                results["t"],
                results["q"][:, strain_idx],
                color=colors[name],
                linewidth=1.5,
                label=name,
            )

        ax.set_ylabel(f"{strain_name}")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Actuation-Space Setpoint Regulation: Strain Tracking Comparison")

    plt.tight_layout()
    plt.savefig("actuation_space_regulation_tracking.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    # Create figure for tracking errors
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for idx, (strain_idx, strain_name) in enumerate(
        zip(plot_strain_indices, plot_strain_names)
    ):
        ax = axes[idx]

        for name, results in all_results.items():
            tracking_error = results["metrics"]["tracking_error"]
            ax.plot(
                results["t"],
                tracking_error[:, strain_idx],
                color=colors[name],
                linewidth=1.5,
                label=name,
            )

        ax.set_ylabel(f"Error {strain_name}")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Actuation-Space Setpoint Regulation: Tracking Error Comparison")

    plt.tight_layout()
    plt.savefig("actuation_space_regulation_errors.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    # Create figure for control inputs (tendon tensions)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    tendon_names = ["Tendon 1 (0°)", "Tendon 2 (120°)", "Tendon 3 (240°)"]

    for idx in range(num_actuators):
        ax = axes[idx]

        # Plot reference steady-state actuation
        ax.axhline(
            y=u_constant[idx],
            color="k",
            linestyle="--",
            linewidth=2,
            label="Reference (steady-state)" if idx == 0 else None,
            alpha=0.7,
        )

        for name, results in all_results.items():
            ax.plot(
                results["t"],
                results["u"][:, idx],
                color=colors[name],
                linewidth=1.5,
                label=name,
            )

        ax.set_ylabel(f"{tendon_names[idx]} [N]")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(
        "Actuation-Space Setpoint Regulation: Control Inputs (Tendon Tensions)"
    )

    plt.tight_layout()
    plt.savefig("actuation_space_regulation_inputs.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    # Create bar chart for RMSE comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    controller_names = list(all_results.keys())
    x = jnp.arange(len(controller_names))
    width = 0.25

    for i, (strain_idx, strain_name) in enumerate(
        zip(plot_strain_indices, plot_strain_names)
    ):
        rmse_values = [
            all_results[name]["metrics"]["rmse"][strain_idx]
            for name in controller_names
        ]
        ax.bar(x + i * width, rmse_values, width, label=strain_name)

    ax.set_xlabel("Controller")
    ax.set_ylabel("RMSE")
    ax.set_title("Actuation-Space Setpoint Regulation: RMSE Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(controller_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("actuation_space_regulation_rmse.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    print("\nPlots saved to:")
    print("  - actuation_space_regulation_tracking.pdf")
    print("  - actuation_space_regulation_errors.pdf")
    print("  - actuation_space_regulation_inputs.pdf")
    print("  - actuation_space_regulation_rmse.pdf")

    if Open3DRenderer is None:
        print("Open3DRenderer unavailable. Install open3d to view the animation.")
    else:
        render_name = "PID (model-free)"
        if render_name not in all_results:
            render_name = next(iter(all_results.keys()))
        render_results = all_results[render_name]
        renderer = Open3DRenderer(robot, num_points=50)
        renderer.render_sequence(
            ts=render_results["t"],
            q_ts=render_results["q"],
            playback_speed=1.0,
            window_name=f"Actuation-Space Regulation ({render_name})",
        )


if __name__ == "__main__":
    main()
