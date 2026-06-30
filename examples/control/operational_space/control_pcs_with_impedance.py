"""
Operational-Space Impedance Control Example for PCS (Piecewise Constant Strain) Soft Robot.

This example demonstrates how to use the OperationalSpaceImpedanceControlTracker
to control a two-segment PCS soft robot in operational space. It supports both
end-effector position tracking and full end-effector pose tracking, with
trajectory primitives shared by the operational-space examples in this folder.

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

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import Array
from trajectory_primitives import (
    ORIENTATION_PRIMITIVES,
    POSITION_PRIMITIVES,
    SURFACES,
    TRACKING_MODES,
    TaskSpaceTrajectoryConfig,
    make_end_effector_task_selector,
    make_operational_space_reference_trajectory,
)

from soromox.control import OperationalSpaceImpedanceControlTracker
from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.rendering import Open3DRenderer
from soromox.systems import PCS, PCSParams, SystemState

jax.config.update("jax_enable_x64", True)  # Double precision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", choices=TRACKING_MODES, default="position")
    parser.add_argument(
        "--position-primitive",
        choices=POSITION_PRIMITIVES,
        default="figure-eight",
    )
    parser.add_argument(
        "--orientation-primitive",
        choices=ORIENTATION_PRIMITIVES,
        default="fixed",
    )
    parser.add_argument("--surface", choices=SURFACES, default="plane")
    parser.add_argument("--t1", type=float, default=10.0, help="Final time [s].")
    parser.add_argument(
        "--period", type=float, default=4.0, help="Trajectory period [s]."
    )
    parser.add_argument(
        "--output-prefix",
        default="control_pcs_with_impedance",
        help="Prefix for generated PDF plots.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show plots.")
    parser.add_argument(
        "--no-render", action="store_true", help="Skip optional Open3D rendering."
    )
    return parser.parse_args()


def _output_path(prefix: str, suffix: str) -> str:
    path = Path(prefix)
    suffix = suffix.replace("_", "-")
    return str(path.with_name(f"{path.name}_{suffix}.pdf"))


def _position_indices(osd: OperationalSpaceDynamics) -> Array:
    pos_start = osd.n_orientation_dim
    pos_dim = 2 if osd.is_planar else 3
    return jnp.arange(pos_start, pos_start + pos_dim)


def _plot_tracking_results(
    args: argparse.Namespace,
    osd: OperationalSpaceDynamics,
    robot: PCS,
    t_traj: Array,
    x_traj_full: Array,
    x_des_traj_full: Array,
    u_traj: Array,
) -> tuple[str, Array, Array | None]:
    pos_indices = _position_indices(osd)
    x_traj_pos = x_traj_full[:, pos_indices]
    x_des_traj_pos = x_des_traj_full[:, pos_indices]
    tracking_error_pos = x_des_traj_pos - x_traj_pos

    pose_error = None
    if args.tracking == "pose":
        pose_error = jax.vmap(osd.compute_pose_error)(x_traj_full, x_des_traj_full)

    n_rows = 5 if args.tracking == "pose" else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.2 * n_rows), sharex=True)
    colors = plt.cm.tab10.colors

    axis_labels = ["x", "y", "z"][: x_traj_pos.shape[1]]

    ax = axes[0]
    for i, label in enumerate(axis_labels):
        color = colors[i % len(colors)]
        ax.plot(
            t_traj, x_traj_pos[:, i], color=color, linewidth=2, label=f"$p_{label}$"
        )
        ax.plot(
            t_traj,
            x_des_traj_pos[:, i],
            "--",
            color=color,
            linewidth=1.5,
            alpha=0.7,
            label=f"$p_{{{label},des}}$",
        )
    ax.set_ylabel("Position [m]")
    ax.set_title("End-Effector Position Tracking")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, label in enumerate(axis_labels):
        color = colors[i % len(colors)]
        ax.plot(
            t_traj,
            tracking_error_pos[:, i] * 1000.0,
            color=color,
            linewidth=2,
            label=f"$e_{label}$",
        )
    ax.set_ylabel("Position Error [mm]")
    ax.set_title("Position Tracking Error")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    control_axis_index = 2
    if args.tracking == "pose":
        assert pose_error is not None
        orientation_dim = osd.n_orientation_dim
        orientation_error = pose_error[:, : osd.n_angular_velocity_dim]
        orientation_labels = ["x", "y", "z"][:orientation_dim]

        ax = axes[2]
        for i, label in enumerate(orientation_labels):
            color = colors[i % len(colors)]
            ax.plot(
                t_traj,
                x_traj_full[:, i],
                color=color,
                linewidth=2,
                label=f"$r_{label}$",
            )
            ax.plot(
                t_traj,
                x_des_traj_full[:, i],
                "--",
                color=color,
                linewidth=1.5,
                alpha=0.7,
                label=f"$r_{{{label},des}}$",
            )
        ax.set_ylabel("Orientation")
        ax.set_title("End-Effector Orientation Tracking")
        ax.legend(loc="upper right", ncol=2)
        ax.grid(True, alpha=0.3)

        ax = axes[3]
        error_labels = ["x", "y", "z"][: orientation_error.shape[1]]
        for i, label in enumerate(error_labels):
            color = colors[i % len(colors)]
            ax.plot(
                t_traj,
                jnp.rad2deg(orientation_error[:, i]),
                color=color,
                linewidth=2,
                label=f"$e_{{r,{label}}}$",
            )
        ax.set_ylabel("Orientation Error [deg]")
        ax.set_title("Geometric Orientation Error")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        control_axis_index = 4

    ax = axes[control_axis_index]
    num_actuators_to_plot = min(robot.num_actuators, 6)
    for i in range(num_actuators_to_plot):
        color = colors[i % len(colors)]
        ax.plot(t_traj, u_traj[:, i], color=color, label=f"$u_{i}$", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Control Input")
    ax.set_title("Actuator Inputs")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output = _output_path(
        args.output_prefix,
        f"{args.tracking}_{args.position_primitive}_{args.orientation_primitive}_results",
    )
    plt.savefig(output, dpi=200)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()

    return output, tracking_error_pos, pose_error


def _plot_configuration_results(
    args: argparse.Namespace,
    num_dofs: int,
    t_traj: Array,
    q_traj: Array,
    qd_traj: Array,
) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = plt.cm.tab10.colors
    num_dofs_to_plot = min(num_dofs, 6)

    ax = axes[0]
    for i in range(num_dofs_to_plot):
        color = colors[i % len(colors)]
        ax.plot(t_traj, q_traj[:, i], color=color, label=f"$q_{i}$", linewidth=2)
    ax.set_ylabel("Configuration")
    ax.set_title("Configuration Space Evolution")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(num_dofs_to_plot):
        color = colors[i % len(colors)]
        ax.plot(
            t_traj, qd_traj[:, i], color=color, label=f"$\\dot{{q}}_{i}$", linewidth=2
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity")
    ax.set_title("Configuration Space Velocities")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output = _output_path(
        args.output_prefix,
        f"{args.tracking}_{args.position_primitive}_{args.orientation_primitive}_config",
    )
    plt.savefig(output, dpi=200)
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()
    return output


def main() -> None:
    args = parse_args()

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
        gravity=jnp.array([0.0, 0.0, -9.81]),
        young_modulus=2e3 * jnp.ones((num_segments,)),
        shear_modulus=1e3 * jnp.ones((num_segments,)),
        damping_matrix=damping_matrix,
        reference_strain=jnp.tile(
            jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), num_segments
        ),
    )

    robot = PCS(params=params)
    num_dofs = robot.num_active_strains
    total_length = float(jnp.sum(segment_lengths))

    print(f"Number of DOFs: {num_dofs}")
    print(f"Number of actuators: {robot.num_actuators}")
    print(f"Total robot length: {total_length:.3f} m")

    # =========================================================================
    # Operational Space Definition
    # =========================================================================
    s_ps = jnp.array([total_length])
    task_selector = make_end_effector_task_selector(args.tracking, n_velocity_dim=6)

    osd = OperationalSpaceDynamics(
        robot=robot,
        s_ps=s_ps,
        task_selector=task_selector,
    )

    print(f"Tracking mode: {args.tracking}")
    print(f"Position primitive: {args.position_primitive}")
    print(f"Orientation primitive: {args.orientation_primitive}")
    print(f"Surface: {args.surface}")
    print(f"Operational space dimension (velocity): {osd.n_operational_space}")
    print(f"Operational space dimension (pose): {osd.n_pose_operational_space}")
    print(f"Full pose dimension: {osd.n_points * osd.n_pose_dim}")

    # =========================================================================
    # Reference Trajectory Definition
    # =========================================================================
    t0, t1 = 0.0, args.t1
    ts = jnp.linspace(t0, t1, 1000)

    q0 = jnp.zeros((num_dofs,))
    x0_full = osd.operational_space_poses(q0)
    print(f"Initial end-effector pose (full): {x0_full}")

    trajectory_config = TaskSpaceTrajectoryConfig(
        tracking=args.tracking,
        position_primitive=args.position_primitive,
        orientation_primitive=args.orientation_primitive,
        surface=args.surface,
        period=args.period,
    )
    reference_trajectory = make_operational_space_reference_trajectory(
        osd=osd,
        q0=q0,
        ts=ts,
        config=trajectory_config,
    )
    assert reference_trajectory.x_des_fn is not None

    # =========================================================================
    # Impedance Controller Setup
    # =========================================================================
    K_x = 5e0 * jnp.ones((osd.n_operational_space,))
    D_x = 1e0 * jnp.ones((osd.n_operational_space,))

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
    qd0 = jnp.zeros((num_dofs,))
    y0 = jnp.concatenate([q0, qd0])

    initial_state = SystemState(
        t=jnp.array(t0),
        y=y0,
        u=jnp.zeros((robot.num_actuators,)),
        control_state=None,
    )

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

    t_traj = trajectory.t
    q_traj, qd_traj = jnp.split(trajectory.y, 2, axis=1)
    assert trajectory.u is not None
    u_traj = trajectory.u

    x_traj_full = jax.vmap(osd.operational_space_poses)(q_traj)
    x_des_traj_full = jax.vmap(reference_trajectory.x_des_fn)(t_traj)

    print(f"Simulation completed. {len(t_traj)} time steps saved.")

    # =========================================================================
    # Plotting Results
    # =========================================================================
    tracking_output, tracking_error_pos, pose_error = _plot_tracking_results(
        args,
        osd,
        robot,
        t_traj,
        x_traj_full,
        x_des_traj_full,
        u_traj,
    )
    config_output = _plot_configuration_results(args, num_dofs, t_traj, q_traj, qd_traj)

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    pos_rmse = jnp.sqrt(jnp.mean(tracking_error_pos**2, axis=0))
    print(f"\nPosition RMSE tracking error: {pos_rmse * 1000} mm")
    print(f"Mean position RMSE: {jnp.mean(pos_rmse) * 1000:.3f} mm")

    if pose_error is not None:
        orientation_error = pose_error[:, : osd.n_angular_velocity_dim]
        orient_rmse = jnp.sqrt(jnp.mean(orientation_error**2, axis=0))
        print(f"Orientation RMSE tracking error: {jnp.rad2deg(orient_rmse)} deg")
        print(f"Mean orientation RMSE: {jnp.rad2deg(jnp.mean(orient_rmse)):.3f} deg")

    print("\nPlots saved to:")
    print(f"  - {tracking_output}")
    print(f"  - {config_output}")

    if args.no_render:
        return

    if Open3DRenderer is None:
        print("\nOpen3DRenderer unavailable. Install open3d to view the animation.")
    else:
        pos_indices = _position_indices(osd)
        target_radius = float(jnp.mean(params.radius)) * 0.5
        target_positions = jnp.asarray(x_des_traj_full[:, pos_indices])[None, :, :]
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
