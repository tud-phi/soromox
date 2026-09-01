"""Generate the full-vs-partial feedback-linearization comparison data."""

# ruff: noqa: E402

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from soromox.control import (
    OperationalSpaceImpedanceControlTracker,
    ReferenceTrajectory,
)
from soromox.coordinate_transformations import OperationalSpaceDynamics
from soromox.systems import PCS, LinkSpec, SystemState
from soromox.utils.geometry.rotations import (
    rotation_matrix_to_rotation_vector,
    rotation_vector_to_rotation_matrix,
)

CASE_DIR = Path(__file__).parent.parent
DEFAULT_DATA_OUTPUT = (
    CASE_DIR / "data" / "impedance_feedback_linearization_comparison.npz"
)
CONTROLLER_MODES = ("full", "partial")


def build_robot() -> PCS:
    """Construct the fully actuated two-segment PCS benchmark robot."""
    num_segments = 2
    return PCS.from_links(
        [
            LinkSpec.circular(
                length=0.1,
                radius=0.02,
                density=1070.0,
                young_modulus=2e3,
                shear_modulus=1e3,
                material_damping_coefficient=362.0,
                reference_strain=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            for _ in range(num_segments)
        ],
        base_pose=jnp.array([0.5, -0.5, 0.5, 0.5, 0.0, 0.0, 0.0]),
        gravity=jnp.array([0.0, 0.0, 9.81]),
    )


def minimum_jerk_ramp(t: jax.Array, duration: float = 0.25) -> jax.Array:
    """Return a C2-continuous ramp from zero to one."""
    phase = jnp.clip(t / duration, 0.0, 1.0)
    return phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)


def make_reference(
    osd: OperationalSpaceDynamics,
    x0: jax.Array,
    t1: float,
) -> tuple[ReferenceTrajectory, object]:
    """Create a smooth, changing-axis full-pose reference."""
    R0 = rotation_vector_to_rotation_matrix(x0[:3])

    def x_des_fn(t):
        ramp = minimum_jerk_ramp(t)
        phase = 2.0 * jnp.pi * t / 1.5
        relative_orientation = ramp * jnp.array(
            [
                0.12 * jnp.sin(phase),
                0.08 * jnp.sin(2.0 * phase),
                0.06 * (jnp.cos(phase) - 1.0),
            ]
        )
        R_des = rotation_vector_to_rotation_matrix(relative_orientation) @ R0
        orientation_des = rotation_matrix_to_rotation_vector(R_des)
        position_des = x0[3:] + ramp * jnp.array(
            [
                0.005 * (jnp.cos(phase) - 1.0),
                0.012 * jnp.sin(phase),
                0.008 * jnp.sin(2.0 * phase),
            ]
        )
        return jnp.concatenate([orientation_des, position_des])

    reference = ReferenceTrajectory(
        ts=jnp.linspace(0.0, t1, 101),
        x_des_fn=x_des_fn,
        rotation_representation=osd.rotation_representation,
        n_points=osd.n_points,
        is_planar=osd.is_planar,
    )
    return reference, x_des_fn


def run_controller(
    mode: str,
    robot: PCS,
    osd: OperationalSpaceDynamics,
    reference: ReferenceTrajectory,
    initial_state: SystemState,
    K_x: jax.Array,
    D_x: jax.Array,
    t1: float,
) -> tuple[object, float]:
    """Run one controller mode with the shared benchmark settings."""
    controller = OperationalSpaceImpedanceControlTracker(
        operational_space_dynamics=osd,
        reference_trajectory=reference,
        K_x=K_x,
        D_x=D_x,
        feedback_linearization=mode,
    )
    start = time.perf_counter()
    trajectory = robot.rollout_closed_loop_to(
        initial_state=initial_state,
        controller=controller,
        t1=t1,
        solver_dt=2e-5,
        save_dt=0.005,
    )
    jax.block_until_ready(trajectory.y)
    return trajectory, time.perf_counter() - start


def extract_results(
    trajectory,
    osd: OperationalSpaceDynamics,
    x0: jax.Array,
    x_des_fn,
) -> dict[str, np.ndarray]:
    """Convert a rollout into plotting coordinates and geometric errors."""
    q_traj, _ = jnp.split(trajectory.y, 2, axis=1)
    pose = jax.vmap(osd.operational_space_poses)(q_traj)
    pose_desired = jax.vmap(x_des_fn)(trajectory.t)
    pose_error = jax.vmap(osd.compute_task_pose_error)(pose, pose_desired)

    R0 = rotation_vector_to_rotation_matrix(x0[:3])

    def relative_orientation(single_pose):
        R = rotation_vector_to_rotation_matrix(single_pose[:3])
        return rotation_matrix_to_rotation_vector(R @ R0.T)

    orientation = jax.vmap(relative_orientation)(pose)
    orientation_desired = jax.vmap(relative_orientation)(pose_desired)

    return {
        "t": np.asarray(trajectory.t),
        "position_mm": np.asarray(1e3 * (pose[:, 3:] - x0[3:])),
        "position_desired_mm": np.asarray(1e3 * (pose_desired[:, 3:] - x0[3:])),
        "position_error_mm": np.asarray(
            1e3 * jnp.linalg.norm(pose_error[:, 3:], axis=1)
        ),
        "orientation_deg": np.asarray(jnp.rad2deg(orientation)),
        "orientation_desired_deg": np.asarray(jnp.rad2deg(orientation_desired)),
        "orientation_error_deg": np.asarray(
            jnp.rad2deg(jnp.linalg.norm(pose_error[:, :3], axis=1))
        ),
    }


def print_metrics(results: dict[str, dict[str, np.ndarray]]) -> None:
    """Print matched RMS and maximum error metrics."""
    for mode in ("full", "partial"):
        position_error = results[mode]["position_error_mm"]
        orientation_error = results[mode]["orientation_error_deg"]
        print(
            f"{mode:>7}: "
            f"position RMS/max = "
            f"{np.sqrt(np.mean(position_error**2)):.8g}/"
            f"{np.max(position_error):.8g} mm; "
            f"orientation RMS/max = "
            f"{np.sqrt(np.mean(orientation_error**2)):.8g}/"
            f"{np.max(orientation_error):.8g} deg"
        )


def save_results(
    path: Path,
    results: dict[str, dict[str, np.ndarray]],
    elapsed_seconds: dict[str, float],
    *,
    force: bool = False,
) -> None:
    """Save both controller rollouts and their provenance to one NPZ file."""
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "controller_modes": np.asarray(CONTROLLER_MODES),
        "solver_dt": np.asarray(2e-5),
        "save_dt": np.asarray(0.005),
        "t1": np.asarray(1.0),
        "natural_frequency": np.asarray(30.0),
        "damping_ratio": np.asarray(1.0),
    }
    for mode in CONTROLLER_MODES:
        payload[f"{mode}_elapsed_seconds"] = np.asarray(elapsed_seconds[mode])
        for key, value in results[mode].items():
            payload[f"{mode}_{key}"] = value
    np.savez_compressed(path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_OUTPUT,
        help="NPZ destination (default: case-local data directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing NPZ artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot = build_robot()
    q0 = jnp.zeros((robot.num_active_strains,))
    osd = OperationalSpaceDynamics(
        robot=robot,
        s_ps=jnp.array([0.2]),
        task_selector=jnp.ones((6,), dtype=bool),
    )
    x0 = osd.operational_space_poses(q0)
    t1 = 1.0
    reference, x_des_fn = make_reference(osd, x0, t1)

    modal_inertia = jnp.diag(osd.inertia_matrix(q0))
    natural_frequency = 30.0
    damping_ratio = 1.0
    K_x = modal_inertia * natural_frequency**2
    D_x = 2.0 * damping_ratio * modal_inertia * natural_frequency
    initial_state = SystemState(
        t=jnp.array(0.0),
        y=jnp.concatenate([q0, jnp.zeros_like(q0)]),
        u=jnp.zeros((robot.num_actuators,)),
        control_state=None,
    )

    results: dict[str, dict[str, np.ndarray]] = {}
    elapsed_seconds: dict[str, float] = {}
    for mode in CONTROLLER_MODES:
        trajectory, elapsed = run_controller(
            mode,
            robot,
            osd,
            reference,
            initial_state,
            K_x,
            D_x,
            t1,
        )
        results[mode] = extract_results(trajectory, osd, x0, x_des_fn)
        elapsed_seconds[mode] = elapsed
        print(f"{mode.capitalize()} simulation completed in {elapsed:.1f} s")

    print_metrics(results)
    save_results(args.output, results, elapsed_seconds, force=args.force)
    print(f"Comparison data written to {args.output}")


if __name__ == "__main__":
    main()
