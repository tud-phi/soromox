"""Simulate and render a spatial articulated soft robot."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # double precision

import jax.numpy as jnp
import matplotlib.pyplot as plt

from soromox.rendering import (
    CameraConfig,
    MatplotlibRenderer,
    Open3DRenderer,
    ViserRenderer,
    get_color_theme,
)
from soromox.systems import (
    ArticulatedSoftRobot,
    ArticulatedSoftRobotParams,
    SystemState,
)


def build_robot() -> ArticulatedSoftRobot:
    """Build a small spatial serial chain with compliant joints."""
    p_tip = jnp.array(
        [
            [0.0, 0.0, 0.30],
            [0.24, 0.0, 0.10],
            [0.20, 0.12, 0.04],
            [0.16, -0.08, 0.02],
        ]
    )
    p_com = 0.5 * p_tip
    lengths = jnp.linalg.norm(p_tip, axis=1)
    masses = jnp.array([1.1, 0.85, 0.65, 0.45])
    radii = jnp.array([0.026, 0.022, 0.019, 0.016])
    transverse_inertia = masses * lengths**2 / 12.0
    axial_inertia = 0.5 * masses * radii**2
    I_com = jax.vmap(jnp.diag)(
        jnp.stack([transverse_inertia, transverse_inertia, axial_inertia], axis=1)
    )

    params = ArticulatedSoftRobotParams(
        joint_screw=jnp.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        tip_position=p_tip,
        center_of_mass_position=p_com,
        mass=masses,
        center_of_mass_inertia=I_com,
        gravity=jnp.array([0.0, 0.0, -9.81]),
        joint_stiffness=jnp.diag(jnp.array([18.0, 13.0, 8.0, 4.5])),
        joint_damping=jnp.diag(jnp.array([1.4, 1.1, 0.75, 0.45])),
        joint_rest_configuration=jnp.zeros((4,)),
        parent_to_joint_transform=jnp.broadcast_to(jnp.eye(4), (4, 4, 4)),
        radius=radii,
    )
    return ArticulatedSoftRobot(params=params)


def simulate(
    robot: ArticulatedSoftRobot,
    *,
    t1: float = 2.0,
    solver_dt: float = 1e-3,
    save_dt: float = 0.02,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Roll out the articulated robot from a displaced compliant-joint state."""
    q0 = jnp.array([0.35, -0.45, 0.28, -0.20])
    qd0 = jnp.zeros_like(q0)
    u = jnp.zeros((robot.num_actuators,))

    initial_state = SystemState(t=0.0, y=jnp.concatenate([q0, qd0]))
    trajectory = robot.rollout_to(
        initial_state=initial_state,
        u=u,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        max_steps=None,
    )
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    return trajectory.t, q_ts, qd_ts


def end_effector_positions(robot: ArticulatedSoftRobot, q_ts: jax.Array) -> jax.Array:
    """Return the end-effector translation for each saved configuration."""
    g_tip_ts = jax.vmap(robot.forward_kinematics_tips)(q_ts)
    return g_tip_ts[:, -1, :3, 3]


def plot_results(
    robot: ArticulatedSoftRobot,
    ts: jax.Array,
    q_ts: jax.Array,
    qd_ts: jax.Array,
) -> None:
    """Plot joint states, end-effector path, and energies."""
    p_ee_ts = end_effector_positions(robot, q_ts)

    plt.figure()
    for joint_idx in range(robot.num_dofs):
        plt.plot(ts, q_ts[:, joint_idx], label=f"q{joint_idx + 1} [rad]")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint configuration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(ts, p_ee_ts[:, 0], label="End-effector x [m]")
    plt.plot(ts, p_ee_ts[:, 1], label="End-effector y [m]")
    plt.plot(ts, p_ee_ts[:, 2], label="End-effector z [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position [m]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    points = ax.scatter(
        p_ee_ts[:, 0],
        p_ee_ts[:, 1],
        p_ee_ts[:, 2],
        c=ts,
        cmap="viridis",
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("End-effector trajectory")
    ax.axis("equal")
    fig.colorbar(points, ax=ax, label="Time [s]")
    plt.tight_layout()
    plt.show()

    U_ts = jax.vmap(robot.potential_energy)(q_ts)
    T_ts = jax.vmap(robot.kinetic_energy)(q_ts, qd_ts)

    plt.figure()
    plt.plot(ts, U_ts, label="Potential energy")
    plt.plot(ts, T_ts, label="Kinetic energy")
    plt.plot(ts, U_ts + T_ts, label="Total energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def render_robot(
    robot: ArticulatedSoftRobot,
    ts: jax.Array,
    q_ts: jax.Array,
    *,
    backend: str = "matplotlib",
    record_path: Path | None = None,
) -> None:
    """Render the simulated trajectory with the selected backend."""
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)

    camera = CameraConfig(
        position=(1.1, -1.3, 0.9),
        look_at=(0.2, 0.0, 0.25),
        fov=45.0,
    )
    open3d_camera = CameraConfig(
        position=(4.0, -4.5, 3.0),
        look_at=(0.25, 0.05, 0.35),
        up=(0.0, 0.0, 1.0),
    )

    if backend in {"matplotlib", "all"}:
        renderer = MatplotlibRenderer(
            robot,
            num_points=80,
            color_config=get_color_theme("soromox:paper"),
            line_width=5.0,
        )
        if record_path is None:
            renderer.animate(
                ts=ts,
                q_ts=q_ts,
                interval=60,
                mode="slider",
                camera_config=camera,
            )
        else:
            renderer.render_sequence(
                ts=ts,
                q_ts=q_ts,
                interval=60,
                record_path=str(record_path),
                camera_config=camera,
                show=True,
            )

    if backend in {"open3d", "all"}:
        if Open3DRenderer is None:
            print("Open3DRenderer is unavailable. Install the optional Open3D extras.")
        else:
            renderer = Open3DRenderer(robot, num_points=80)
            renderer.render_sequence(
                ts,
                q_ts,
                playback_speed=1.0,
                loop=record_path is None,
                record_path=None if record_path is None else str(record_path),
                camera_config=open3d_camera,
                window_name="Articulated Soft Robot",
            )

    if backend in {"viser", "all"}:
        if ViserRenderer is None:
            print("ViserRenderer is unavailable. Install the optional Viser extras.")
        else:
            renderer = ViserRenderer(
                robot,
                num_points=80,
            )
            renderer.render_sequence(
                ts,
                q_ts,
                playback_speed=1.0,
                loop=True,
                autoplay=True,
                plot_configurations=True,
                robot_name="ArticulatedSoftRobot",
                record_path=None if record_path is None else str(record_path),
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Simulate and render an articulated soft robot."
    )
    parser.add_argument(
        "--render",
        choices=("none", "matplotlib", "open3d", "viser", "all"),
        default="matplotlib",
        help="Rendering backend to launch after simulation.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip Matplotlib diagnostic plots.",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help="Optional animation output path, e.g. videos/articulated_soft_robot.mp4.",
    )
    parser.add_argument("--t1", type=float, default=2.0, help="Final time [s].")
    return parser.parse_args()


def main() -> None:
    """Run the example."""
    args = parse_args()
    robot = build_robot()
    ts, q_ts, qd_ts = simulate(robot, t1=args.t1)

    print("Final configuration:")
    print(q_ts[-1])
    print("Final end-effector position:")
    print(end_effector_positions(robot, q_ts)[-1])

    if not args.no_plots:
        plot_results(robot, ts, q_ts, qd_ts)
    if args.render != "none":
        render_robot(robot, ts, q_ts, backend=args.render, record_path=args.record)


if __name__ == "__main__":
    main()
