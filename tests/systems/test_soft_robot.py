"""Unit tests for shared :class:`soromox.systems.SoftRobot` contracts."""

import jax
import equinox as eqx
import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose
from system_param_builders import pcs_params, planar_pcs_params

from soromox.systems import PCS, PCSStructure, PlanarPCS, PlanarPCSStructure

jax.config.update("jax_enable_x64", True)


def _planar_robot(*, floating_base: bool) -> PlanarPCS:
    params = planar_pcs_params(
        length=jnp.array([0.1]),
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1.0e5]),
        shear_modulus=jnp.array([4.0e4]),
        damping_matrix=jnp.eye(3),
        gravity=jnp.array([0.0, -9.81]),
    )
    return PlanarPCS(
        params=params,
        structure=PlanarPCSStructure(num_gauss_points=3),
        floating_base=floating_base,
        backend="jax",
    )


def _spatial_robot(*, floating_base: bool) -> PCS:
    params = pcs_params(
        length=jnp.array([0.1]),
        radius=jnp.array([0.01]),
        density=jnp.array([1000.0]),
        young_modulus=jnp.array([1.0e5]),
        shear_modulus=jnp.array([4.0e4]),
        damping_matrix=jnp.eye(6),
        gravity=jnp.array([0.0, 0.0, -9.81]),
    )
    return PCS(
        params=params,
        structure=PCSStructure(num_gauss_points=3),
        floating_base=floating_base,
        backend="jax",
    )


@pytest.mark.parametrize(
    ("factory", "base_coordinates", "base_velocities"),
    [(_planar_robot, 3, 3), (_spatial_robot, 7, 6)],
)
def test_dimensions_distinguish_storage_velocity_and_internal_coordinates(
    factory, base_coordinates: int, base_velocities: int
) -> None:
    fixed = factory(floating_base=False)
    floating = factory(floating_base=True)

    assert fixed.num_base_coordinates == 0
    assert fixed.num_base_velocities == 0
    assert fixed.num_coordinates == fixed.num_internal_dofs
    assert fixed.num_velocities == fixed.num_internal_dofs
    assert floating.num_base_coordinates == base_coordinates
    assert floating.num_base_velocities == base_velocities
    assert floating.num_coordinates == base_coordinates + floating.num_internal_dofs
    assert floating.num_velocities == base_velocities + floating.num_internal_dofs
    assert floating.state_size == (
        floating.num_coordinates
        + floating.num_velocities
        + floating.num_auxiliary_states
    )
    assert not hasattr(floating, "num_dofs")


def test_pack_and_split_helpers_broadcast_leading_axes() -> None:
    robot = _planar_robot(floating_base=True)
    q_internal = jnp.stack(
        [
            jnp.zeros((robot.num_internal_dofs,)),
            jnp.full((robot.num_internal_dofs,), 0.1),
        ]
    )
    q = robot.pack_configuration(q_internal, base_pose=jnp.array([0.2, 0.3, -0.1]))
    v = robot.pack_velocity(
        jnp.zeros((robot.num_internal_dofs,)),
        base_velocity=jnp.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]]),
    )
    y = robot.pack_state(q, v)

    q_base, q_internal_actual = robot.split_configuration(q)
    v_base, qd_internal_actual = robot.split_velocity(v)
    q_actual, v_actual, auxiliary = robot.split_state(y)
    assert q_base is not None
    assert v_base is not None
    assert q.shape == (2, robot.num_coordinates)
    assert v.shape == (2, robot.num_velocities)
    assert y.shape == (2, robot.state_size)
    assert_allclose(q_base, jnp.broadcast_to(jnp.array([0.2, 0.3, -0.1]), (2, 3)))
    assert_allclose(q_internal_actual, q_internal)
    assert_allclose(qd_internal_actual, 0.0)
    assert_allclose(q_actual, q)
    assert_allclose(v_actual, v)
    assert auxiliary.shape == (2, 0)


def test_pack_helpers_reject_wrong_base_usage_and_dimensions() -> None:
    fixed = _planar_robot(floating_base=False)
    floating = _planar_robot(floating_base=True)
    q_internal = jnp.zeros((fixed.num_internal_dofs,))

    with pytest.raises(ValueError, match="base_pose is required"):
        floating.pack_configuration(q_internal)
    with pytest.raises(ValueError, match="only valid for a floating-base robot"):
        fixed.pack_configuration(q_internal, base_pose=jnp.zeros((3,)))
    with pytest.raises(ValueError, match="trailing dimension"):
        floating.pack_configuration(q_internal, base_pose=jnp.zeros((4,)))
    with pytest.raises(ValueError, match="only valid for a floating-base robot"):
        fixed.pack_velocity(q_internal, base_velocity=jnp.zeros((3,)))


def test_spatial_normalization_validation_and_trace_safe_fallback() -> None:
    robot = _spatial_robot(floating_base=True)
    q_internal = jnp.zeros((robot.num_internal_dofs,))
    q = robot.pack_configuration(
        q_internal,
        base_pose=jnp.array([2.0, -0.2, 0.1, 0.3, 0.4, -0.5, 0.6]),
    )

    assert_allclose(jnp.linalg.norm(q[:4]), 1.0)
    assert_allclose(robot.normalize_configuration(-q)[:4], -q[:4])
    with pytest.raises(ValueError, match="nonzero norm"):
        robot.normalize_configuration(q.at[:4].set(0.0))
    with pytest.raises(ValueError, match="finite"):
        robot.normalize_configuration(q.at[0].set(jnp.inf))

    traced = eqx.filter_jit(robot.normalize_configuration)(q.at[:4].set(0.0))
    assert_allclose(traced[:4], jnp.array([1.0, 0.0, 0.0, 0.0]))


def test_configuration_derivative_matches_retraction_tangent() -> None:
    robot = _spatial_robot(floating_base=True)
    q = robot.pack_configuration(
        jnp.linspace(-0.02, 0.03, robot.num_internal_dofs),
        base_pose=jnp.array([0.9, 0.2, -0.1, 0.3, 0.1, -0.2, 0.4]),
    )
    v = robot.pack_velocity(
        jnp.linspace(0.03, -0.02, robot.num_internal_dofs),
        base_velocity=jnp.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1]),
    )
    epsilon = 1.0e-6
    finite_difference = (robot.retract_configuration(q, epsilon * v) - q) / epsilon
    assert_allclose(
        finite_difference,
        robot.configuration_derivative(q, v),
        rtol=2.0e-6,
        atol=2.0e-7,
    )


def test_fixed_mount_update_and_public_state_projection() -> None:
    floating = _spatial_robot(floating_base=True)
    q_internal = jnp.zeros((floating.num_internal_dofs,))
    base_pose = jnp.array([1.0, 0.0, 0.0, 0.0, 0.2, -0.1, 0.3])
    q = floating.pack_configuration(q_internal, base_pose=base_pose)
    v = floating.pack_velocity(jnp.zeros((floating.num_internal_dofs,)))
    y = floating.pack_state(q.at[:4].multiply(1.8), v)

    projected = floating.project_state(y)
    projected_q, projected_v, projected_auxiliary = floating.split_state(projected)
    assert_allclose(jnp.linalg.norm(projected_q[:4]), 1.0)
    assert_allclose(projected_q[4:], q[4:])
    assert_allclose(projected_v, v)
    assert projected_auxiliary.shape == (0,)
    assert_allclose(
        floating.base_transform_from_configuration(q)[:3, 3], base_pose[4:]
    )

    fixed = _spatial_robot(floating_base=False).with_fixed_base_pose(base_pose)
    assert_allclose(fixed.fixed_base_pose, base_pose)
    assert_allclose(
        fixed.base_transform_from_configuration(q_internal)[:3, 3], base_pose[4:]
    )
    with pytest.raises(ValueError, match="has no fixed base pose"):
        floating.with_fixed_base_pose(base_pose)
