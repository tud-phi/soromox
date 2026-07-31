"""Simulate and render the McKibben-actuated UMArm."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from soromox.rendering import MatplotlibRenderer, UMArmViserRenderer
from soromox.systems import (
    McKibbenActuatedUMArm,
    SystemState,
)

PSI_TO_PA = 6894.75729
DEFAULT_VIDEO_PATH = Path("videos") / "mckibben_umarm.mp4"
UMARM_Z_DOWN_BASE_POSE = jnp.array(
    [2**-0.5, 0.0, 2**-0.5, 0.0, 0.0, 0.0, 1.2],
    dtype=jnp.float64,
)


def _repo_params_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "assets"
        / "robot_parameters"
        / "mckibben_umarm"
        / "reference_parameters.npz"
    )


def build_robot(params_path: Path | None = None) -> McKibbenActuatedUMArm:
    """Build the UMArm from repo-local or explicit cached parameters."""
    source_path = _repo_params_path() if params_path is None else params_path
    robot = McKibbenActuatedUMArm.from_cached_parameters(source_path)
    return robot.update_params(base_pose=UMARM_Z_DOWN_BASE_POSE)


def pressure_input(robot: McKibbenActuatedUMArm) -> jax.Array:
    """Return a small constant pressure input in Pa."""
    pressures = jnp.zeros((robot.num_actuators,))
    return pressures.at[jnp.array([0, 9, 16])].set(
        jnp.array([12.0, 9.0, 7.0]) * PSI_TO_PA
    )


def simulate(
    robot: McKibbenActuatedUMArm,
    *,
    t1: float = 1.0,
    solver_dt: float = 1e-3,
    save_dt: float = 0.02,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Roll out the UMArm with a constant pressure input."""
    q0 = jnp.zeros((robot.num_dofs,))
    q0 = q0.at[0].set(0.25)
    q0 = q0.at[4].set(-0.20)
    qd0 = jnp.zeros_like(q0)
    pressures = pressure_input(robot)
    trajectory = robot.rollout_to(
        initial_state=SystemState(t=0.0, y=jnp.concatenate([q0, qd0])),
        u=pressures,
        t1=t1,
        solver_dt=solver_dt,
        save_dt=save_dt,
        max_steps=None,
    )
    q_ts, qd_ts = jnp.split(trajectory.y, 2, axis=1)
    return trajectory.t, q_ts, qd_ts, pressures


def end_effector_positions(robot: McKibbenActuatedUMArm, q_ts: jax.Array) -> jax.Array:
    """Return end-effector positions over a configuration sequence."""
    tips = jax.vmap(robot.forward_kinematics_tips)(q_ts)
    return tips[:, -1, :3, 3]


def plot_results(
    robot: McKibbenActuatedUMArm,
    ts: jax.Array,
    q_ts: jax.Array,
    qd_ts: jax.Array,
) -> None:
    """Plot selected joint states, end-effector motion, and energies."""
    p_ee_ts = end_effector_positions(robot, q_ts)

    plt.figure()
    for joint_idx in (0, 1, 4, 5, 8, 9):
        plt.plot(ts, q_ts[:, joint_idx], label=f"q{joint_idx}")
    plt.xlabel("Time [s]")
    plt.ylabel("Joint angle [rad]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(ts, p_ee_ts[:, 0], label="x [m]")
    plt.plot(ts, p_ee_ts[:, 1], label="y [m]")
    plt.plot(ts, p_ee_ts[:, 2], label="z [m]")
    plt.xlabel("Time [s]")
    plt.ylabel("End-effector position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    kinetic = jax.vmap(robot.kinetic_energy)(q_ts, qd_ts)
    potential = jax.vmap(robot.potential_energy)(q_ts)
    plt.figure()
    plt.plot(ts, kinetic, label="Kinetic energy")
    plt.plot(ts, potential, label="Potential energy")
    plt.plot(ts, kinetic + potential, label="Total energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def render_robot(
    robot: McKibbenActuatedUMArm,
    ts: jax.Array,
    q_ts: jax.Array,
    pressures: jax.Array,
    *,
    backend: str,
    record_path: Path | None = None,
) -> None:
    """Render the UMArm trajectory with the selected backend."""
    if record_path is not None:
        record_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "matplotlib":
        renderer = MatplotlibRenderer(robot, num_points=80)
        if record_path is None:
            renderer.animate(ts=ts, q_ts=q_ts, mode="slider", interval=60)
        else:
            renderer.render_sequence(
                ts=ts,
                q_ts=q_ts,
                interval=60,
                record_path=str(record_path),
                show=False,
            )
        return

    if backend == "viser":
        if UMArmViserRenderer is None:
            print("UMArmViserRenderer is unavailable. Install the Viser extras.")
            return
        renderer = UMArmViserRenderer(
            robot,
            num_points=80,
            backbone_style="discrete",
            actuator_color_mode="pressure",
        )
        renderer.render_sequence(
            ts=ts,
            q_ts=q_ts,
            pressures=pressures,
            actuator_color_mode="pressure",
            playback_speed=1.0,
            loop=True,
            autoplay=True,
            record_path=None if record_path is None else str(record_path),
            robot_name="McKibbenActuatedUMArm",
        )
        return

    raise ValueError(f"Unknown renderer backend: {backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the McKibben UMArm.")
    parser.add_argument(
        "--render",
        choices=("none", "matplotlib", "viser"),
        default="viser",
        help="Renderer to launch after simulation.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plots.")
    parser.add_argument(
        "--record",
        type=Path,
        nargs="?",
        const=DEFAULT_VIDEO_PATH,
        default=None,
        help=(
            "Optional video path. If provided without a path, saves to "
            f"{DEFAULT_VIDEO_PATH}."
        ),
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help="Optional path to a UMArm .npz parameter file.",
    )
    parser.add_argument("--t1", type=float, default=1.0, help="Final time [s].")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = build_robot(args.params)
    ts, q_ts, qd_ts, pressures = simulate(robot, t1=args.t1)

    print("Final configuration:")
    print(q_ts[-1])
    print("Final end-effector position:")
    print(end_effector_positions(robot, q_ts)[-1])

    if not args.no_plots:
        plot_results(robot, ts, q_ts, qd_ts)
    if args.render != "none":
        render_robot(
            robot, ts, q_ts, pressures, backend=args.render, record_path=args.record
        )


if __name__ == "__main__":
    main()
