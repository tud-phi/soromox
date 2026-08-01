import sys
from pathlib import Path

import jax.numpy as jnp
from numpy.testing import assert_allclose

MODULE_DIR = (
    Path(__file__).resolve().parents[4]
    / "paper_results"
    / "secVc_model_based_control"
    / "operational_space_impedance_control"
    / "code"
)

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import operational_space_impedance_common  # noqa: E402
import trajectory_primitives  # noqa: E402


def test_position_task_preserves_origin_main_gains():
    problem = operational_space_impedance_common.build_problem(
        trajectory_primitives.TaskSpaceTrajectoryConfig(tracking="position")
    )
    K_x, D_x = operational_space_impedance_common._compute_impedance_gains(problem)
    Lambda_diag = jnp.diag(problem.osd.inertia_matrix(problem.q0))

    assert_allclose(K_x, jnp.ones_like(K_x))
    assert_allclose(D_x, 2.0 * jnp.sqrt(K_x * Lambda_diag))


def test_pose_task_matches_rotational_and_mean_positional_bandwidth():
    problem = operational_space_impedance_common.build_problem(
        trajectory_primitives.TaskSpaceTrajectoryConfig(tracking="pose")
    )
    K_x, D_x = operational_space_impedance_common._compute_impedance_gains(problem)
    Lambda_diag = jnp.diag(problem.osd.inertia_matrix(problem.q0))
    n_angular = problem.osd.n_angular_velocity_dim

    K_rot = K_x[:n_angular]
    K_pos = K_x[n_angular:]
    Lambda_rot = Lambda_diag[:n_angular]
    Lambda_pos = Lambda_diag[n_angular:]
    target_omega_n = jnp.sqrt(
        operational_space_impedance_common.POSITION_STIFFNESS / jnp.mean(Lambda_pos)
    )

    assert_allclose(
        K_pos,
        operational_space_impedance_common.POSITION_STIFFNESS * jnp.ones_like(K_pos),
    )
    assert_allclose(jnp.sqrt(K_rot / Lambda_rot), target_omega_n)
    assert_allclose(D_x, 2.0 * jnp.sqrt(K_x * Lambda_diag))
