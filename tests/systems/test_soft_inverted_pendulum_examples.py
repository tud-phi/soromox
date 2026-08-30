"""Tests for the shared GVS soft inverted-pendulum examples."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from examples.simulation.gvs.soft_inverted_pendulum import (
    AFFINE_CURVATURE_MATRIX,
    SoftCartPendulumGVS,
    SoftInvertedPendulumConfig,
    initial_configuration,
    make_cart_pendulum,
    make_fixed_pendulum,
    simulate_open_loop,
    split_pendulum_state,
    tip_torque_generalized_force,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def config() -> SoftInvertedPendulumConfig:
    """Return the shared paper parameters."""
    return SoftInvertedPendulumConfig()


@pytest.fixture(scope="module")
def fixed_robot(config):
    """Return the fixed-base example model."""
    return make_fixed_pendulum(config)


@pytest.fixture(scope="module")
def cart_robot(config):
    """Return the passive-cart example model."""
    return make_cart_pendulum(config)


def test_models_share_the_affine_curvature_soft_link(config, fixed_robot, cart_robot):
    assert fixed_robot.K.shape == (2, 2)
    assert cart_robot.K.shape == (3, 3)
    assert fixed_robot.num_actuators == 0
    assert cart_robot.num_actuators == 0
    assert_allclose(fixed_robot.K, config.stiffness)
    assert_allclose(fixed_robot.D_active, config.damping)
    assert_allclose(cart_robot.K[1:, 1:], fixed_robot.K)
    assert_allclose(cart_robot.D_active[1:, 1:], fixed_robot.D_active)
    assert_allclose(cart_robot.K[0], 0.0)
    assert_allclose(cart_robot.D_active[0], 0.0)
    assert_allclose(fixed_robot.params.link.density, [config.density])
    assert_allclose(cart_robot.params.link.density, fixed_robot.params.link.density)
    assert_allclose(config.stiffness, AFFINE_CURVATURE_MATRIX)
    assert_allclose(config.damping, 0.1 * AFFINE_CURVATURE_MATRIX)


def test_initial_state_and_tip_torque_preserve_the_passive_cart_row():
    assert_allclose(initial_configuration(cart=False), [np.pi / 4, -np.pi / 4])
    assert_allclose(initial_configuration(cart=True), [0.0, np.pi / 4, -np.pi / 4])
    assert_allclose(tip_torque_generalized_force(2.0, cart=False), [2.0, 1.0])
    assert_allclose(tip_torque_generalized_force(2.0, cart=True), [0.0, 2.0, 1.0])


def test_affine_curvature_tip_orientation_matches_paper_coordinates(fixed_robot):
    q = jnp.array([0.3, -0.2])
    tip = fixed_robot.forward_kinematics(q, fixed_robot.length)
    relative = jnp.linalg.inv(fixed_robot.base_transform) @ tip
    tip_angle = jnp.arctan2(relative[1, 0], relative[0, 0])
    assert_allclose(tip_angle, q[0] + 0.5 * q[1], atol=1e-10)


def test_prismatic_cart_translates_the_complete_backbone(cart_robot):
    s_points = jnp.linspace(0.0, cart_robot.length, 7)
    at_origin = cart_robot.forward_kinematics_abscissa_batched(
        jnp.zeros((3,)), s_points
    )
    translated = cart_robot.forward_kinematics_abscissa_batched(
        jnp.array([0.35, 0.0, 0.0]), s_points
    )
    displacement = translated[:, :3, 3] - at_origin[:, :3, 3]
    assert_allclose(
        displacement,
        jnp.broadcast_to(jnp.array([0.0, 0.35, 0.0]), displacement.shape),
        atol=1e-10,
    )


def test_cart_mass_is_added_once_without_changing_coriolis_or_gravity(
    config, cart_robot
):
    q = jnp.array([0.1, 0.2, -0.15])
    qd = jnp.array([0.05, -0.1, 0.07])
    inertia, coriolis_velocity, gravity = cart_robot.dynamics_terms(q, qd)
    base_inertia, base_coriolis_velocity, base_gravity = super(
        SoftCartPendulumGVS, cart_robot
    ).dynamics_terms(q, qd)
    expected_delta = jnp.zeros((3, 3)).at[0, 0].set(config.cart_mass)
    assert_allclose(inertia - base_inertia, expected_delta, atol=1e-10)
    assert_allclose(coriolis_velocity, base_coriolis_velocity, atol=1e-10)
    assert_allclose(gravity, base_gravity, atol=1e-10)
    assert_allclose(cart_robot.inertia_matrix(q), inertia, atol=1e-9)


@pytest.mark.parametrize("cart", [False, True])
def test_short_open_loop_rollout_is_finite(config, cart):
    robot = make_cart_pendulum(config) if cart else make_fixed_pendulum(config)
    trajectory = simulate_open_loop(
        robot,
        cart=cart,
        t1=0.03,
        solver_dt=1e-3,
        save_dt=0.01,
    )
    num_coordinates = 3 if cart else 2
    assert trajectory.y.shape == (4, 2 * num_coordinates)
    assert jnp.all(jnp.isfinite(trajectory.y))
    q_ts, v_ts = split_pendulum_state(trajectory.y, cart=cart)
    total_energy = jax.vmap(robot.potential_energy)(q_ts) + jax.vmap(
        robot.kinetic_energy
    )(q_ts, v_ts)
    assert total_energy[-1] <= total_energy[0] + 1e-7
