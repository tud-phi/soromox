"""
Setpoint Regulation Comparison for PCS (Piecewise Constant Strain) Soft Robot.

This example compares the performance of different configuration-space regulators
for controlling a PCS soft robot to a sequence of setpoints. The controllers compared are:
    - PIDController (model-free baseline)
    - PotentialCompensationRegulator (feedforward: G(q_des) + tau_el(q_des))
    - PotentialCancellationRegulator (feedforward: G(q) + tau_el(q))
    - GravityCancellationRegulator (feedforward: G(q) + tau_el(q_des))

All controllers use the same PID gains to isolate the effect of the model-based
feedforward term.

The setpoints are defined on the two bending strains (kappa_y, kappa_z) and
the axial stretch strain (sigma_x).
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # Double precision

from soromox.control import PIDControl, PIDControllerState, ReferenceTrajectory
from soromox.control.configuration_space import (
    GravityCancellationRegulator,
    PIDController,
    PotentialCancellationRegulator,
    PotentialCompensationRegulator,
)
from soromox.rendering import Open3DRenderer
from soromox.systems import PCS, PCSParams, SystemState


def create_robot() -> tuple[PCS, int]:
    """Create the PCS robot and return it along with the number of DOFs."""
    num_segments = 1
    rho = 1070 * jnp.ones((num_segments,))  # Volumetric density [kg/m^3]

    segment_lengths = 1e-1 * jnp.ones((num_segments,))
    material_damping_coefficient = 362.0
    params = PCSParams(
        base_pose=jnp.array([0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 0.0]),
        length=segment_lengths,
        radius=2e-2 * jnp.ones((num_segments,)),
        density=rho,
        gravity=jnp.array([0.0, 0.0, 9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        material_damping_coefficient=material_damping_coefficient,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    robot = PCS(params=params)
    num_dofs = robot.num_active_strains

    return robot, int(num_dofs)


def create_pid_control(num_dofs: int, num_segments: int) -> PIDControl:
    """
    Create PID control gains that are used by all controllers.

    The gains are structured for PCS strains:
    [kappa_x, kappa_y, kappa_z, sigma_x, sigma_y, sigma_z] per segment.
    """
    # Gains for rotational strains (curvatures) - units: [rad/m]
    Kp_rot = 1e-2
    Ki_rot = 1e-3
    Kd_rot = 1e-4

    # Gains for linear strains (elongation/shear) - dimensionless
    Kp_lin = 1e-1
    Ki_lin = 1e-3
    Kd_lin = 1e-2

    # Build gain vectors per segment
    Kp_segment = jnp.array([Kp_rot, Kp_rot, Kp_rot, Kp_lin, Kp_lin, Kp_lin])
    Ki_segment = jnp.array([Ki_rot, Ki_rot, Ki_rot, Ki_lin, Ki_lin, Ki_lin])
    Kd_segment = jnp.array([Kd_rot, Kd_rot, Kd_rot, Kd_lin, Kd_lin, Kd_lin])

    # Tile for all segments
    Kp = jnp.tile(Kp_segment, num_segments)
    Ki = jnp.tile(Ki_segment, num_segments)
    Kd = jnp.tile(Kd_segment, num_segments)

    return PIDControl(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        saturation_fn="tanh",
        gamma=1.0,
    )


def create_setpoint_trajectory(
    num_dofs: int, t0: float, t1: float
) -> tuple[ReferenceTrajectory, jnp.ndarray]:
    """
    Create a sequence of step setpoints for regulation.

    The setpoints are defined on:
    - kappa_y (index 1): bending around y-axis
    - kappa_z (index 2): bending around z-axis
    - sigma_x (index 3): axial stretch
    """
    ts = jnp.linspace(t0, t1, 1000)

    # Define step setpoint times
    step_times = jnp.array([0.0, 3.0, 6.0, 9.0, 12.0])

    # Setpoint values for each strain component
    # kappa_y: bending in y [rad/m]
    kappa_y_values = jnp.array([0.0, 2.0, -1.0, 1.5, 0.0])
    # kappa_z: bending in z [rad/m]
    kappa_z_values = jnp.array([0.0, 0.5, -0.8, 0.3, 0.0])
    # sigma_x: axial stretch (small variations around 0, which means no stretch from reference)
    sigma_x_values = jnp.array([0.0, 0.02, -0.01, 0.01, 0.0])

    def x_des_fn(t):
        """Desired configuration as a function of time (step setpoints)."""
        q_des = jnp.zeros((num_dofs,))
        # Find which step we're in
        step_idx = jnp.searchsorted(step_times, t, side="right") - 1
        step_idx = jnp.clip(step_idx, 0, len(kappa_y_values) - 1)

        # Set kappa_y (index 1)
        q_des = q_des.at[1].set(kappa_y_values[step_idx])
        # Set kappa_z (index 2)
        q_des = q_des.at[2].set(kappa_z_values[step_idx])
        # Set sigma_x (index 3)
        q_des = q_des.at[3].set(sigma_x_values[step_idx])

        return q_des

    reference_trajectory = ReferenceTrajectory(ts=ts, x_des_fn=x_des_fn)

    return reference_trajectory, step_times


def run_simulation(
    robot: PCS,
    controller,
    num_dofs: int,
    t0: float,
    t1: float,
    solver_dt: float = 1e-4,
    save_dt: float = 0.01,
) -> dict:
    """Run closed-loop simulation with the given controller."""
    # Initial conditions
    q0 = jnp.zeros((num_dofs,))
    qd0 = jnp.zeros((num_dofs,))
    y0 = jnp.concatenate([q0, qd0])

    # Initial control state
    control_state_0 = PIDControllerState.zero(num_dofs)

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


def compute_metrics(results: dict, q_des_fn, strain_indices: list) -> dict:
    """Compute performance metrics for the simulation results."""
    t = results["t"]
    q = results["q"]

    # Compute desired trajectory at saved times
    q_des = jax.vmap(q_des_fn)(t)

    # Compute tracking error for selected strains
    tracking_error = q_des - q

    # Root mean square error for each strain
    rmse = jnp.sqrt(jnp.mean(tracking_error[:, strain_indices] ** 2, axis=0))

    # Maximum absolute error
    max_error = jnp.max(jnp.abs(tracking_error[:, strain_indices]), axis=0)

    # Integral of squared error (ISE)
    dt = t[1] - t[0]
    ise = jnp.sum(tracking_error[:, strain_indices] ** 2, axis=0) * dt

    return {
        "rmse": rmse,
        "max_error": max_error,
        "ise": ise,
        "tracking_error": tracking_error,
        "q_des": q_des,
    }


def main():
    # =========================================================================
    # Setup
    # =========================================================================
    robot, num_dofs = create_robot()
    num_segments = 1

    print(f"Number of DOFs: {num_dofs}")
    print(f"Number of actuators: {robot.num_actuators}")

    # Time parameters
    t0, t1 = 0.0, 15.0

    # Create shared PID control (same gains for all controllers)
    pid_control = create_pid_control(num_dofs, num_segments)

    # Create reference trajectory
    reference_trajectory, step_times = create_setpoint_trajectory(num_dofs, t0, t1)

    # Define controllers to compare
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
    import warnings

    for name, ControllerClass in controllers.items():
        print(f"\nRunning simulation with {name}...")

        # Suppress actuation warnings for cleaner output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            controller = ControllerClass(
                robot=robot,
                reference_trajectory=reference_trajectory,
                pid_control=pid_control,
            )

        results = run_simulation(robot, controller, num_dofs, t0, t1)
        all_results[name] = results
        print(f"  Completed {len(results['t'])} time steps.")

    # =========================================================================
    # Compute Metrics
    # =========================================================================
    # Strain indices to analyze: kappa_y (1), kappa_z (2), sigma_x (3)
    strain_indices = [1, 2, 3]
    strain_names = [r"$\kappa_y$", r"$\kappa_z$", r"$\sigma_x$"]

    print("\n" + "=" * 70)
    print("Performance Metrics (RMSE)")
    print("=" * 70)
    print(
        f"{'Controller':<25} {strain_names[0]:>12} {strain_names[1]:>12} {strain_names[2]:>12}"
    )
    print("-" * 70)

    for name, results in all_results.items():
        metrics = compute_metrics(
            results, reference_trajectory.x_des_fn, strain_indices
        )
        all_results[name]["metrics"] = metrics
        rmse = metrics["rmse"]
        print(f"{name:<25} {rmse[0]:>12.4f} {rmse[1]:>12.4f} {rmse[2]:>12.4f}")

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

    # Create figure for strain tracking comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for idx, (strain_idx, strain_name) in enumerate(zip(strain_indices, strain_names)):
        ax = axes[idx]

        # Plot desired trajectory (same for all controllers)
        t = all_results["PID (model-free)"]["t"]
        q_des = all_results["PID (model-free)"]["metrics"]["q_des"]
        ax.plot(t, q_des[:, strain_idx], "k--", linewidth=2, label="Desired", alpha=0.7)

        # Plot actual trajectories for each controller
        for name, results in all_results.items():
            ax.plot(
                results["t"],
                results["q"][:, strain_idx],
                color=colors[name],
                linewidth=1.5,
                label=name,
            )

        # Add vertical lines at step times
        for st in step_times[1:]:
            ax.axvline(x=st, color="gray", linestyle=":", alpha=0.5)

        ax.set_ylabel(f"{strain_name}")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Setpoint Regulation: Strain Tracking Comparison")

    plt.tight_layout()
    plt.savefig("setpoint_regulation_tracking.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    # Create figure for tracking errors
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for idx, (strain_idx, strain_name) in enumerate(zip(strain_indices, strain_names)):
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

        # Add vertical lines at step times
        for st in step_times[1:]:
            ax.axvline(x=st, color="gray", linestyle=":", alpha=0.5)

        ax.set_ylabel(f"Error {strain_name}")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title("Setpoint Regulation: Tracking Error Comparison")

    plt.tight_layout()
    plt.savefig("setpoint_regulation_errors.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    # Create bar chart for RMSE comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    controller_names = list(all_results.keys())
    x = jnp.arange(len(controller_names))
    width = 0.25

    for i, (_strain_idx, strain_name) in enumerate(zip(strain_indices, strain_names)):
        rmse_values = [
            all_results[name]["metrics"]["rmse"][i] for name in controller_names
        ]
        ax.bar(x + i * width, rmse_values, width, label=strain_name)

    ax.set_xlabel("Controller")
    ax.set_ylabel("RMSE")
    ax.set_title("Setpoint Regulation: RMSE Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(controller_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("setpoint_regulation_rmse.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    print("\nPlots saved to:")
    print("  - setpoint_regulation_tracking.pdf")
    print("  - setpoint_regulation_errors.pdf")
    print("  - setpoint_regulation_rmse.pdf")

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
            window_name=f"Setpoint Regulation ({render_name})",
        )


if __name__ == "__main__":
    main()
