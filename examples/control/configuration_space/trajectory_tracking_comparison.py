"""
Trajectory Tracking Comparison for PCS (Piecewise Constant Strain) Soft Robot.

This example compares the performance of different configuration-space controllers
for tracking smooth trajectories on a PCS soft robot. The comparison includes:

Tracking Controllers (designed for dynamic trajectory tracking):
    - ComputedTorqueTracker (full inverse dynamics at current state)
    - FeedforwardCompensationTracker (inverse dynamics at desired trajectory)
    - MixedStateFeedbackTracker (mixed evaluation strategy)

Regulators (designed for setpoint regulation, tested on slow trajectory only):
    - PIDController (model-free baseline)
    - PotentialCompensationRegulator (feedforward: G(q_des) + tau_el(q_des))
    - PotentialCancellationRegulator (feedforward: G(q) + tau_el(q))
    - GravityCancellationRegulator (feedforward: G(q) + tau_el(q_des))

All controllers use the same PID gains to isolate the effect of the model-based
feedforward term.

Two trajectory speeds are tested:
    - Slow trajectory: Tests quasi-static tracking (regulators may perform well)
    - Fast trajectory: Tests dynamic tracking (requires proper trajectory trackers)

The trajectories are defined on the two bending strains (kappa_y, kappa_z) and
the axial stretch strain (sigma_x).
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)  # Double precision

from soromox.control import PIDControl, PIDControllerState, ReferenceTrajectory
from soromox.control.configuration_space import (
    ComputedTorqueTracker,
    FeedforwardCompensationTracker,
    GravityCancellationRegulator,
    MixedStateFeedbackTracker,
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


def create_smooth_trajectory(
    num_dofs: int,
    t0: float,
    t1: float,
    frequency_scale: float = 1.0,
) -> ReferenceTrajectory:
    """
    Create a smooth sinusoidal trajectory for tracking.

    The trajectory is defined on:
    - kappa_y (index 1): bending around y-axis
    - kappa_z (index 2): bending around z-axis
    - sigma_x (index 3): axial stretch

    Args:
        num_dofs: Number of degrees of freedom.
        t0: Start time.
        t1: End time.
        frequency_scale: Scale factor for trajectory frequency.
            1.0 = slow trajectory, 3.0 = fast trajectory.
    """
    ts = jnp.linspace(t0, t1, 1000)

    # Base frequencies (scaled by frequency_scale)
    omega_y = 0.3 * frequency_scale  # rad/s for kappa_y
    omega_z = 0.2 * frequency_scale  # rad/s for kappa_z
    omega_x = 0.25 * frequency_scale  # rad/s for sigma_x

    # Amplitudes
    amp_kappa_y = 2.0  # [rad/m]
    amp_kappa_z = 1.0  # [rad/m]
    amp_sigma_x = 0.02  # dimensionless (small axial stretch)

    def x_des_fn(t):
        """Desired configuration as a function of time."""
        q_des = jnp.zeros((num_dofs,))

        # Smooth start/stop using raised cosine ramp
        # Ramp up over first 10% of trajectory, ramp down over last 10%
        duration = t1 - t0
        ramp_duration = 0.1 * duration

        # Ramp factor: 0 at t0, 1 during middle, 0 at t1
        ramp_up = 0.5 * (1 - jnp.cos(jnp.pi * jnp.clip((t - t0) / ramp_duration, 0, 1)))
        ramp_down = 0.5 * (
            1
            + jnp.cos(
                jnp.pi * jnp.clip((t - (t1 - ramp_duration)) / ramp_duration, 0, 1)
            )
        )
        ramp = ramp_up * ramp_down

        # Sinusoidal trajectories with phase offsets for interesting motion
        q_des = q_des.at[1].set(ramp * amp_kappa_y * jnp.sin(omega_y * t))
        q_des = q_des.at[2].set(ramp * amp_kappa_z * jnp.sin(omega_z * t + jnp.pi / 4))
        q_des = q_des.at[3].set(ramp * amp_sigma_x * jnp.sin(omega_x * t + jnp.pi / 2))

        return q_des

    return ReferenceTrajectory(ts=ts, x_des_fn=x_des_fn)


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


def plot_trajectory_comparison(
    all_results: dict,
    title: str,
    filename_prefix: str,
    strain_indices: list,
    strain_names: list,
    colors: dict,
):
    """Plot tracking comparison for a set of controllers."""
    # Create figure for strain tracking comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    first_controller = list(all_results.keys())[0]

    for idx, (strain_idx, strain_name) in enumerate(zip(strain_indices, strain_names)):
        ax = axes[idx]

        # Plot desired trajectory (same for all controllers)
        t = all_results[first_controller]["t"]
        q_des = all_results[first_controller]["metrics"]["q_des"]
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

        ax.set_ylabel(f"{strain_name}")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(title)

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_tracking.pdf", dpi=200, bbox_inches="tight")
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

        ax.set_ylabel(f"Error {strain_name}")
        ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(f"{title}: Tracking Errors")

    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_errors.pdf", dpi=200, bbox_inches="tight")
    plt.show()


def print_metrics_table(all_results: dict, strain_names: list, title: str):
    """Print a formatted metrics table."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    print(
        f"{'Controller':<30} {strain_names[0]:>15} {strain_names[1]:>15} {strain_names[2]:>15}"
    )
    print("-" * 80)

    for name, results in all_results.items():
        rmse = results["metrics"]["rmse"]
        print(f"{name:<30} {rmse[0]:>15.4f} {rmse[1]:>15.4f} {rmse[2]:>15.4f}")


def main():
    # =========================================================================
    # Setup
    # =========================================================================
    robot, num_dofs = create_robot()
    num_segments = 1

    print(f"Number of DOFs: {num_dofs}")
    print(f"Number of actuators: {robot.num_actuators}")

    # Time parameters
    t0, t1_slow, t1_fast = 0.0, 30.0, 15.0

    # Create shared PID control (same gains for all controllers)
    pid_control = create_pid_control(num_dofs, num_segments)

    # Strain indices to analyze: kappa_y (1), kappa_z (2), sigma_x (3)
    strain_indices = [1, 2, 3]
    strain_names = [r"$\kappa_y$", r"$\kappa_z$", r"$\sigma_x$"]

    # Define colors
    tracker_colors = {
        "Computed Torque": "#E24A33",  # Red-orange
        "Feedforward Compensation": "#348ABD",  # Blue
        "Mixed State Feedback": "#988ED5",  # Purple
    }

    regulator_colors = {
        "PID (model-free)": "#E24A33",  # Red-orange
        "Potential Compensation": "#348ABD",  # Blue
        "Potential Cancellation": "#988ED5",  # Purple
        "Gravity Cancellation": "#8EBA42",  # Green
    }

    # =========================================================================
    # PART 1: Slow Trajectory - Compare Trackers and Regulators
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: SLOW TRAJECTORY (Quasi-static)")
    print("=" * 80)

    # Create slow trajectory
    reference_slow = create_smooth_trajectory(
        num_dofs, t0, t1_slow, frequency_scale=1.0
    )

    # Define all controllers for slow trajectory
    all_controllers_slow = {
        # Trackers
        "Computed Torque": ComputedTorqueTracker,
        "Feedforward Compensation": FeedforwardCompensationTracker,
        "Mixed State Feedback": MixedStateFeedbackTracker,
        # Regulators
        "PID (model-free)": PIDController,
        "Potential Compensation": PotentialCompensationRegulator,
        "Potential Cancellation": PotentialCancellationRegulator,
        "Gravity Cancellation": GravityCancellationRegulator,
    }

    slow_results = {}

    for name, ControllerClass in all_controllers_slow.items():
        print(f"\nRunning slow trajectory with {name}...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            controller = ControllerClass(
                robot=robot,
                reference_trajectory=reference_slow,
                pid_control=pid_control,
            )

        results = run_simulation(robot, controller, num_dofs, t0, t1_slow)
        metrics = compute_metrics(results, reference_slow.x_des_fn, strain_indices)
        results["metrics"] = metrics
        slow_results[name] = results
        print(f"  Completed. RMSE: {metrics['rmse']}")

    # Print metrics
    print_metrics_table(slow_results, strain_names, "Slow Trajectory: RMSE Comparison")

    # Plot trackers on slow trajectory
    tracker_slow_results = {
        k: v for k, v in slow_results.items() if k in tracker_colors
    }
    plot_trajectory_comparison(
        tracker_slow_results,
        "Slow Trajectory: Tracking Controllers",
        "trajectory_slow_trackers",
        strain_indices,
        strain_names,
        tracker_colors,
    )

    # Plot regulators on slow trajectory
    regulator_slow_results = {
        k: v for k, v in slow_results.items() if k in regulator_colors
    }
    plot_trajectory_comparison(
        regulator_slow_results,
        "Slow Trajectory: Regulators (quasi-static)",
        "trajectory_slow_regulators",
        strain_indices,
        strain_names,
        regulator_colors,
    )

    # =========================================================================
    # PART 2: Fast Trajectory - Compare Trackers Only
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: FAST TRAJECTORY (Dynamic)")
    print("=" * 80)

    # Create fast trajectory
    reference_fast = create_smooth_trajectory(
        num_dofs, t0, t1_fast, frequency_scale=3.0
    )

    # Define trackers for fast trajectory
    trackers_fast = {
        "Computed Torque": ComputedTorqueTracker,
        "Feedforward Compensation": FeedforwardCompensationTracker,
        "Mixed State Feedback": MixedStateFeedbackTracker,
    }

    fast_results = {}

    for name, ControllerClass in trackers_fast.items():
        print(f"\nRunning fast trajectory with {name}...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            controller = ControllerClass(
                robot=robot,
                reference_trajectory=reference_fast,
                pid_control=pid_control,
            )

        results = run_simulation(robot, controller, num_dofs, t0, t1_fast)
        metrics = compute_metrics(results, reference_fast.x_des_fn, strain_indices)
        results["metrics"] = metrics
        fast_results[name] = results
        print(f"  Completed. RMSE: {metrics['rmse']}")

    # Print metrics
    print_metrics_table(
        fast_results, strain_names, "Fast Trajectory: RMSE Comparison (Trackers)"
    )

    # Plot trackers on fast trajectory
    plot_trajectory_comparison(
        fast_results,
        "Fast Trajectory: Tracking Controllers",
        "trajectory_fast_trackers",
        strain_indices,
        strain_names,
        tracker_colors,
    )

    # =========================================================================
    # PART 3: Summary Bar Chart
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Slow trajectory - all controllers
    ax1 = axes[0]
    controller_names = list(slow_results.keys())
    x = jnp.arange(len(controller_names))
    width = 0.25

    for i, strain_name in enumerate(strain_names):
        rmse_values = [
            slow_results[name]["metrics"]["rmse"][i] for name in controller_names
        ]
        ax1.bar(x + i * width, rmse_values, width, label=strain_name)

    ax1.set_xlabel("Controller")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Slow Trajectory: All Controllers")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(controller_names, rotation=30, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Fast trajectory - trackers only
    ax2 = axes[1]
    tracker_names = list(fast_results.keys())
    x = jnp.arange(len(tracker_names))

    for i, strain_name in enumerate(strain_names):
        rmse_values = [
            fast_results[name]["metrics"]["rmse"][i] for name in tracker_names
        ]
        ax2.bar(x + i * width, rmse_values, width, label=strain_name)

    ax2.set_xlabel("Controller")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Fast Trajectory: Tracking Controllers")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(tracker_names, rotation=30, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("trajectory_tracking_rmse_summary.pdf", dpi=200, bbox_inches="tight")
    plt.show()

    if Open3DRenderer is None:
        print("Open3DRenderer unavailable. Install open3d to view the animation.")
    else:
        renderer = Open3DRenderer(robot, num_points=50)
        slow_render_name = "Computed Torque"
        if slow_render_name not in slow_results:
            slow_render_name = next(iter(slow_results.keys()))
        slow_render_results = slow_results[slow_render_name]
        renderer.render_sequence(
            ts=slow_render_results["t"],
            q_ts=slow_render_results["q"],
            playback_speed=1.0,
            window_name=f"Slow Trajectory ({slow_render_name})",
        )

        fast_render_name = "Computed Torque"
        if fast_render_name not in fast_results:
            fast_render_name = next(iter(fast_results.keys()))
        fast_render_results = fast_results[fast_render_name]
        renderer.render_sequence(
            ts=fast_render_results["t"],
            q_ts=fast_render_results["q"],
            playback_speed=1.0,
            window_name=f"Fast Trajectory ({fast_render_name})",
        )

    print("\n" + "=" * 80)
    print("Plots saved:")
    print("  - trajectory_slow_trackers_tracking.pdf")
    print("  - trajectory_slow_trackers_errors.pdf")
    print("  - trajectory_slow_regulators_tracking.pdf")
    print("  - trajectory_slow_regulators_errors.pdf")
    print("  - trajectory_fast_trackers_tracking.pdf")
    print("  - trajectory_fast_trackers_errors.pdf")
    print("  - trajectory_tracking_rmse_summary.pdf")
    print("=" * 80)


if __name__ == "__main__":
    main()
