# ruff: noqa: E402
"""Reference checks against the sibling umarm_soromox checkout."""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from soromox.systems import McKibbenActuatedUMArm, McKibbenActuatedUMArmParams

ASSET_PARAMS_PATH = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "robot_parameters"
    / "mckibben_umarm"
    / "reference_parameters.npz"
)
_UMARM_REFERENCE_ROOT_TRANSLATION = np.array([0.0, 0.0, 1.2])
_UMARM_REFERENCE_TO_SOROMOX_ROTATION = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
)


def params_with_prior_reference_base_pose(
    params: McKibbenActuatedUMArmParams,
) -> McKibbenActuatedUMArmParams:
    parent_to_joint = np.asarray(params.parent_to_joint_transform).copy()
    soromox_to_reference = _UMARM_REFERENCE_TO_SOROMOX_ROTATION.T
    parent_to_joint[0, :3, :3] = soromox_to_reference @ parent_to_joint[0, :3, :3]
    parent_to_joint[0, :3, 3] = _UMARM_REFERENCE_ROOT_TRANSLATION
    return params.replace(parent_to_joint_transform=jnp.asarray(parent_to_joint))


def _reference_robot():
    external_package = Path(__file__).resolve().parents[3] / "umarm_soromox"
    if not external_package.exists():
        pytest.skip("external umarm_soromox checkout is unavailable")
    external_src = external_package / "src"
    if str(external_src) not in sys.path:
        sys.path.insert(0, str(external_src))
    try:
        from umarm_soromox.mckibben import McKibbenArticulatedUMArm as Reference
    except Exception as exc:  # pragma: no cover - depends on local checkout
        pytest.skip(f"external umarm_soromox import failed: {exc}")
    return Reference.build("original")


def _rk4_rollout(robot, y0, pressures, *, dt: float, n_steps: int):
    y = y0
    states = []
    for _ in range(n_steps):

        def f(y_):
            return robot.forward_dynamics(jnp.array(0.0), y_, (pressures, None))

        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(y)
    return jnp.stack(states)


def test_prior_reference_base_pose_rotation_for_comparison() -> None:
    params = McKibbenActuatedUMArmParams.from_cached_npz(ASSET_PARAMS_PATH)
    reference_pose_params = params_with_prior_reference_base_pose(params)
    reference_pose_robot = McKibbenActuatedUMArm(reference_pose_params)
    q = jnp.zeros((reference_pose_robot.num_dofs,))
    joint_frames = np.asarray(reference_pose_robot.forward_kinematics_joints(q))
    tip_frames = np.asarray(reference_pose_robot.forward_kinematics_tips(q))

    base_position = joint_frames[0, :3, 3]
    first_physical_tip = tip_frames[1, :3, 3]
    proximal_direction = first_physical_tip - base_position
    proximal_direction = proximal_direction / np.linalg.norm(proximal_direction)

    assert_allclose(base_position, np.array([0.0, 0.0, 1.2]), atol=1e-12)
    assert_allclose(proximal_direction, np.array([0.0, 0.0, -1.0]), atol=1e-12)


def test_matches_reference_umarm_soromox_checkout() -> None:
    params = McKibbenActuatedUMArmParams.from_cached_npz(ASSET_PARAMS_PATH)
    robot = McKibbenActuatedUMArm(params_with_prior_reference_base_pose(params))
    reference = _reference_robot()
    q = jnp.linspace(-0.12, 0.15, robot.num_dofs)
    qd = jnp.linspace(0.05, -0.04, robot.num_dofs)
    pressures = jnp.linspace(0.0, 6.0e4, robot.num_actuators)
    y = jnp.concatenate([q, qd])

    assert_allclose(
        robot.effective_lengths(q),
        reference.effective_lengths(q),
        rtol=1e-11,
        atol=1e-11,
    )
    assert_allclose(
        robot.actuator_forces(q, pressures),
        reference.actuator_forces(q, pressures),
        rtol=1e-11,
        atol=1e-8,
    )
    assert_allclose(
        robot.actuation_matrix(q),
        reference.actuation_matrix(q),
        rtol=1e-10,
        atol=1e-10,
    )
    assert_allclose(
        robot.inertia_matrix(q),
        reference.inertia_matrix(q),
        rtol=1e-10,
        atol=1e-10,
    )
    assert_allclose(
        robot.forward_dynamics(jnp.array(0.0), y, (pressures, None)),
        reference.forward_dynamics(jnp.array(0.0), y, (pressures, None)),
        rtol=1e-9,
        atol=1e-9,
    )

    ours = _rk4_rollout(robot, y, pressures, dt=1e-3, n_steps=5)
    expected = _rk4_rollout(reference, y, pressures, dt=1e-3, n_steps=5)
    assert_allclose(ours, expected, rtol=1e-9, atol=1e-9)
